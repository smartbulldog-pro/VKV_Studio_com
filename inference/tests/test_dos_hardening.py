"""
Iter 4 (DoS / resource exhaustion / billing abuse) regression tests.
====================================================================
Covers:
  - TTS budget: atomic reserve/refund, no TOCTOU overshoot under concurrency,
    atomic file write (no leftover .tmp, valid JSON).
  - ContentLengthLimitMiddleware: rejects oversized declared bodies (413) before
    reaching the app; passes small/chunked/non-http through untouched.
  - _content_length_over helper.
  - ChirpTTS refunds the reservation when the Google call fails.

Run from inference/:
    ./.venv/Scripts/python.exe -m unittest tests.test_dos_hardening -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402
import tts  # noqa: E402
from tts_budget import TTSBudget  # noqa: E402

_CLOCK = lambda: datetime(2026, 7, 15, tzinfo=timezone.utc)


class TestBudgetReserve(unittest.TestCase):
    def _budget(self, cap=1000):
        path = Path(tempfile.mkdtemp()) / "budget.json"
        return TTSBudget(path, cap, clock=_CLOCK), path

    def test_reserve_within_cap(self) -> None:
        b, _ = self._budget(1000)
        self.assertTrue(b.try_reserve(400))
        self.assertEqual(b.usage(), 400)

    def test_reserve_over_cap_refused_and_no_charge(self) -> None:
        b, _ = self._budget(1000)
        self.assertTrue(b.try_reserve(900))
        self.assertFalse(b.try_reserve(200))     # 900+200 > 1000
        self.assertEqual(b.usage(), 900)          # not charged for the refused one

    def test_refund(self) -> None:
        b, _ = self._budget(1000)
        b.try_reserve(500)
        b.refund(200)
        self.assertEqual(b.usage(), 300)

    def test_reserve_boundary_exact(self) -> None:
        b, _ = self._budget(1000)
        self.assertTrue(b.try_reserve(1000))      # exactly the cap is allowed
        self.assertFalse(b.try_reserve(1))

    def test_concurrent_reserves_never_overshoot(self) -> None:
        # 20 threads each reserve 100 against a cap of 1000 → at most 10 succeed,
        # and usage must never exceed the cap (TOCTOU regression).
        b, _ = self._budget(1000)
        results: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            r = b.try_reserve(100)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        granted = sum(1 for r in results if r)
        self.assertLessEqual(b.usage(), 1000, "usage must never exceed cap")
        self.assertEqual(b.usage(), granted * 100)
        self.assertLessEqual(granted, 10)

    def test_atomic_write_leaves_no_tmp_and_valid_json(self) -> None:
        b, path = self._budget(1000)
        b.try_reserve(123)
        self.assertFalse(path.with_name(path.name + ".tmp").exists(), "no leftover .tmp")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["chars"], 123)

    def test_cap_zero_disabled_always_reserves(self) -> None:
        b, _ = self._budget(0)
        self.assertTrue(b.try_reserve(10_000_000))


class _FakeEncoding:
    MP3 = "MP3"; LINEAR16 = "LINEAR16"; OGG_OPUS = "OGG_OPUS"


class _FakeBudget:
    def __init__(self, exceed: bool) -> None:
        self._exceed = exceed
        self.reserved: list[int] = []
        self.refunded: list[int] = []

    def try_reserve(self, n):
        if self._exceed:
            return False
        self.reserved.append(n)
        return True

    def refund(self, n):
        self.refunded.append(n)

    def usage(self):
        return 0

    @property
    def cap(self):
        return 900_000


class TestChirpRefundOnFailure(unittest.TestCase):
    def test_reservation_refunded_when_google_fails(self) -> None:
        budget = _FakeBudget(exceed=False)
        eng = tts.ChirpTTS(budget=budget)

        def _boom(clean_text, lang):
            raise RuntimeError("google down")

        eng._synthesize_google = _boom  # type: ignore[assignment]

        class _StubEdge:
            def synthesize(self, text, lang):
                return b"EDGE"

        eng._fallback = _StubEdge()

        out = eng.synthesize("hello", "en")
        self.assertEqual(out, b"EDGE")
        self.assertEqual(budget.reserved, [5])   # reserved 5 chars ("hello")
        self.assertEqual(budget.refunded, [5])   # refunded since Google failed


class TestContentLengthHelper(unittest.TestCase):
    class _Req:
        def __init__(self, cl):
            self.headers = {"content-length": cl} if cl is not None else {}
            # mimic Starlette headers.get
            self.headers = type("H", (), {"get": lambda s, k, d=None: (
                {"content-length": cl} if cl is not None else {}).get(k, d)})()

    def test_over(self) -> None:
        self.assertTrue(main._content_length_over(self._Req("999"), 100))

    def test_under(self) -> None:
        self.assertFalse(main._content_length_over(self._Req("50"), 100))

    def test_missing(self) -> None:
        self.assertFalse(main._content_length_over(self._Req(None), 100))

    def test_invalid(self) -> None:
        self.assertFalse(main._content_length_over(self._Req("not-a-number"), 100))


class TestContentLengthMiddleware(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_oversized(self) -> None:
        async def inner(scope, receive, send):
            raise AssertionError("inner app must not be reached for oversized body")

        mw = main.ContentLengthLimitMiddleware(inner, max_bytes=100)
        sent: list = []
        scope = {"type": "http", "headers": [(b"content-length", b"999")]}

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(msg):
            sent.append(msg)

        await mw(scope, receive, send)
        self.assertEqual(sent[0]["type"], "http.response.start")
        self.assertEqual(sent[0]["status"], 413)

    async def test_passes_small_body(self) -> None:
        reached = {}

        async def inner(scope, receive, send):
            reached["yes"] = True

        mw = main.ContentLengthLimitMiddleware(inner, max_bytes=1000)
        scope = {"type": "http", "headers": [(b"content-length", b"10")]}
        await mw(scope, None, None)
        self.assertTrue(reached.get("yes"))

    async def test_passes_chunked_no_content_length(self) -> None:
        reached = {}

        async def inner(scope, receive, send):
            reached["yes"] = True

        mw = main.ContentLengthLimitMiddleware(inner, max_bytes=100)
        scope = {"type": "http", "headers": []}  # no content-length
        await mw(scope, None, None)
        self.assertTrue(reached.get("yes"))

    async def test_non_http_scope_passes(self) -> None:
        reached = {}

        async def inner(scope, receive, send):
            reached["yes"] = True

        mw = main.ContentLengthLimitMiddleware(inner, max_bytes=100)
        await mw({"type": "lifespan"}, None, None)
        self.assertTrue(reached.get("yes"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
