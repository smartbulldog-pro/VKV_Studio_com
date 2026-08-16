"""
require_generation_access + _sweep_rate_limit_store — the only billable path.
============================================================================
Generation is the one thing on this backend that costs money per use, so the
gate in front of it is the money path: it decides who may spend a GPU second and
caps how many. It had no direct test. These lock the behaviours that, when they
broke before, silently handed everyone an unlimited claim on the box:

  * anonymous callers are metered by IP (not waved through) when auth is optional,
    and refused outright when auth is required;
  * the quota is a sliding window — expired hits stop counting;
  * an exhausted quota is a 429 with Retry-After, not a 500 or a free pass;
  * the periodic sweep judges a generation-quota key against its 5-hour window,
    NOT the 60-second burst window — evicting it early is exactly the bug that
    made "five messages per five hours" bind nobody.

The async gate is driven with asyncio.run (no pytest-asyncio in this suite), and
its collaborators (authenticated_owner, get_client_ip) plus the quota constants
are monkeypatched on the main module, which is where the function resolves them.

Run from inference/:
    ./.venv/Scripts/python.exe -m pytest tests/test_generation_quota.py -q
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402
from fastapi import HTTPException  # noqa: E402


class _QuotaTestBase(unittest.TestCase):
    """Snapshot and restore every main-module global the gate reads."""

    _PATCHED = (
        "authenticated_owner",
        "get_client_ip",
        "GEN_REQUIRE_AUTH",
        "GEN_QUOTA_MAX",
        "GEN_QUOTA_WINDOW_SEC",
    )

    def setUp(self) -> None:
        self._saved = {name: getattr(main, name) for name in self._PATCHED}
        main._rate_limit_store.clear()

        self._owner: "str | None" = None
        outer = self

        async def _fake_owner(request, *args, **kwargs):  # noqa: ANN001
            return outer._owner

        main.authenticated_owner = _fake_owner
        main.get_client_ip = lambda request: "1.2.3.4"  # noqa: ARG005
        main.GEN_REQUIRE_AUTH = False
        main.GEN_QUOTA_MAX = 5
        main.GEN_QUOTA_WINDOW_SEC = 5 * 3600.0
        self._req = object()  # never inspected — both collaborators are patched

    def tearDown(self) -> None:
        for name, val in self._saved.items():
            setattr(main, name, val)
        main._rate_limit_store.clear()

    def _call(self) -> str:
        return asyncio.run(main.require_generation_access(self._req))

    def _key(self, subject: str) -> str:
        return f"{main.GEN_QUOTA_KEY_PREFIX}{subject}"

    def _hits(self, subject: str) -> int:
        return len(main._rate_limit_store.get(self._key(subject), []))


class TestGenerationGate(_QuotaTestBase):
    def test_anonymous_refused_when_auth_required(self) -> None:
        self._owner = None
        main.GEN_REQUIRE_AUTH = True
        with self.assertRaises(HTTPException) as cm:
            self._call()
        self.assertEqual(cm.exception.status_code, 401)

    def test_anonymous_metered_by_ip_when_auth_optional(self) -> None:
        self._owner = None
        main.GEN_REQUIRE_AUTH = False
        owner = self._call()
        self.assertEqual(owner, "")  # anonymous owner is the empty string
        self.assertEqual(self._hits("ip:1.2.3.4"), 1)  # a hit was recorded, not waved through

    def test_signed_in_metered_by_account_not_ip(self) -> None:
        self._owner = "alice@example.com"
        owner = self._call()
        self.assertEqual(owner, "alice@example.com")
        self.assertEqual(self._hits("alice@example.com"), 1)
        self.assertEqual(self._hits("ip:1.2.3.4"), 0)  # the IP bucket stays untouched

    def test_two_subjects_have_independent_buckets(self) -> None:
        self._owner = "bob@example.com"
        self._call()
        self._owner = "eve@example.com"
        self._call()
        self.assertEqual(self._hits("bob@example.com"), 1)
        self.assertEqual(self._hits("eve@example.com"), 1)

    def test_quota_of_zero_disables_metering(self) -> None:
        self._owner = "carol@example.com"
        main.GEN_QUOTA_MAX = 0  # 0 disables the quota
        key = self._key("carol@example.com")
        main._rate_limit_store[key] = [time.time()] * 50  # pretend a flood already happened
        owner = self._call()
        self.assertEqual(owner, "carol@example.com")
        self.assertEqual(len(main._rate_limit_store[key]), 50)  # nothing appended, nothing pruned

    def test_exhausted_quota_is_429_with_retry_after(self) -> None:
        self._owner = "dave@example.com"
        main.GEN_QUOTA_MAX = 5
        now = time.time()
        key = self._key("dave@example.com")
        main._rate_limit_store[key] = [now - 60 * i for i in range(5)]  # 5 recent hits
        with self.assertRaises(HTTPException) as cm:
            self._call()
        self.assertEqual(cm.exception.status_code, 429)
        self.assertIn("Retry-After", cm.exception.headers or {})
        self.assertGreater(int(cm.exception.headers["Retry-After"]), 0)
        self.assertIn("5", str(cm.exception.detail))  # tells the visitor the allowance
        self.assertEqual(len(main._rate_limit_store[key]), 5)  # the blocked call adds nothing

    def test_expired_hits_are_pruned_and_do_not_count(self) -> None:
        self._owner = "frank@example.com"
        main.GEN_QUOTA_MAX = 5
        main.GEN_QUOTA_WINDOW_SEC = 5 * 3600.0
        now = time.time()
        key = self._key("frank@example.com")
        # 5 hits, all OLDER than the window → pruned, so the request is allowed.
        main._rate_limit_store[key] = [now - (5 * 3600 + 100) - i for i in range(5)]
        owner = self._call()
        self.assertEqual(owner, "frank@example.com")
        self.assertEqual(len(main._rate_limit_store[key]), 1)  # pruned to the one fresh hit


class TestSweepWindows(_QuotaTestBase):
    def test_sweep_keeps_gen_quota_key_but_evicts_stale_burst_key(self) -> None:
        # This is the "unlimited GPU budget" regression guard. A generation-quota
        # key ten minutes old must survive (its window is five hours); a plain
        # per-IP burst key ten minutes old must be evicted (its window is 60s).
        now = time.time()
        ten_min_ago = now - 600
        gen_key = self._key("ip:9.9.9.9")  # genquota\x00-prefixed
        burst_key = "5.5.5.5"  # unprefixed per-IP burst key
        main._rate_limit_store[gen_key] = [ten_min_ago]
        main._rate_limit_store[burst_key] = [ten_min_ago]

        main._sweep_rate_limit_store()

        self.assertIn(gen_key, main._rate_limit_store)  # judged against 5h window → survives
        self.assertNotIn(burst_key, main._rate_limit_store)  # judged against 60s window → evicted


if __name__ == "__main__":
    unittest.main()
