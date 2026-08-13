"""
Iter 6 (info-disclosure + hardening) regression tests.
======================================================
  - /api/docs + /openapi.json disabled by default (ENABLE_DOCS off).
  - _actual_tts_backend() reports the engine really in use, not the config value.
  - SecurityHeadersMiddleware injects X-Content-Type-Options: nosniff.
  - Config flags (ENABLE_DOCS, RELOAD) default off.

Run from inference/:
    ./.venv/Scripts/python.exe -m unittest tests.test_infodisclosure -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import main  # noqa: E402


class TestDocsDisabled(unittest.TestCase):
    def test_docs_url_none_by_default(self) -> None:
        self.assertIsNone(main.app.docs_url)

    def test_openapi_url_none_by_default(self) -> None:
        self.assertIsNone(main.app.openapi_url)

    def test_no_openapi_route(self) -> None:
        paths = {r.path for r in main.app.routes if hasattr(r, "path")}
        self.assertNotIn("/openapi.json", paths)
        self.assertNotIn("/api/docs", paths)


class TestConfigFlags(unittest.TestCase):
    def test_enable_docs_off_by_default(self) -> None:
        self.assertFalse(config.ENABLE_DOCS)

    def test_reload_off_by_default(self) -> None:
        self.assertFalse(config.RELOAD)


class _ChirpTTS:  # class name is what _actual_tts_backend maps on
    pass


class _EdgeTTSEngine:
    pass


class TestActualTtsBackend(unittest.TestCase):
    def _with_engine(self, engine):
        orig = main.tts_engine
        main.tts_engine = engine
        try:
            return main._actual_tts_backend()
        finally:
            main.tts_engine = orig

    def test_reports_edge_when_edge(self) -> None:
        eng = _EdgeTTSEngine()
        eng.__class__.__name__ = "EdgeTTSEngine"
        self.assertEqual(self._with_engine(eng), "edge")

    def test_reports_chirp_when_chirp(self) -> None:
        eng = _ChirpTTS()
        eng.__class__.__name__ = "ChirpTTS"
        self.assertEqual(self._with_engine(eng), "chirp3")

    def test_none_engine(self) -> None:
        self.assertEqual(self._with_engine(None), "none")


class TestSecurityHeaders(unittest.IsolatedAsyncioTestCase):
    async def test_nosniff_injected(self) -> None:
        async def inner(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        mw = main.SecurityHeadersMiddleware(inner)
        sent: list = []

        async def send(msg):
            sent.append(msg)

        await mw({"type": "http"}, None, send)
        start = next(m for m in sent if m["type"] == "http.response.start")
        header_keys = {k.lower(): v for k, v in start["headers"]}
        self.assertEqual(header_keys.get(b"x-content-type-options"), b"nosniff")
        self.assertIn(b"referrer-policy", header_keys)

    async def test_non_http_passthrough(self) -> None:
        reached = {}

        async def inner(scope, receive, send):
            reached["yes"] = True

        mw = main.SecurityHeadersMiddleware(inner)
        await mw({"type": "lifespan"}, None, None)
        self.assertTrue(reached.get("yes"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
