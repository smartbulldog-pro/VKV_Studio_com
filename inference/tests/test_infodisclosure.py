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

    # ── HSTS ────────────────────────────────────────────────────────────────
    # This host answers plain HTTP with real JSON, and a client that reaches
    # api.vkvstudio.com without visiting the apex first has no HSTS pin from
    # anywhere. It has to assert its own — but only over TLS, because RFC 6797
    # requires a browser to ignore the header on a plain-HTTP response, and
    # behind the tunnel the only truthful source for the original scheme is
    # X-Forwarded-Proto.

    async def _headers_for(self, scope: dict) -> dict:
        async def inner(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        sent: list = []

        async def send(msg):
            sent.append(msg)

        await main.SecurityHeadersMiddleware(inner)(scope, None, send)
        start = next(m for m in sent if m["type"] == "http.response.start")
        return {k.lower(): v for k, v in start["headers"]}

    async def test_hsts_sent_when_forwarded_proto_is_https(self) -> None:
        h = await self._headers_for(
            {"type": "http", "scheme": "http", "headers": [(b"x-forwarded-proto", b"https")]}
        )
        self.assertEqual(
            h.get(b"strict-transport-security"), b"max-age=31536000; includeSubDomains"
        )

    async def test_hsts_absent_over_plain_http(self) -> None:
        h = await self._headers_for(
            {"type": "http", "scheme": "http", "headers": [(b"x-forwarded-proto", b"http")]}
        )
        self.assertNotIn(b"strict-transport-security", h)

    async def test_hsts_uses_first_value_of_a_forwarded_chain(self) -> None:
        # Proxy chains append; the client-facing hop is the first entry, and it
        # is the only one that says how the browser actually connected.
        h = await self._headers_for(
            {"type": "http", "scheme": "http", "headers": [(b"x-forwarded-proto", b"https, http")]}
        )
        self.assertIn(b"strict-transport-security", h)

    async def test_hsts_falls_back_to_asgi_scheme_without_the_header(self) -> None:
        # Direct origin access, no proxy in front.
        self.assertIn(
            b"strict-transport-security",
            await self._headers_for({"type": "http", "scheme": "https", "headers": []}),
        )
        self.assertNotIn(
            b"strict-transport-security",
            await self._headers_for({"type": "http", "scheme": "http", "headers": []}),
        )



class TestHealthMethods(unittest.TestCase):
    """The liveness endpoint must answer the verb monitors actually send.

    FastAPI's APIRoute does not add HEAD to a GET route the way a plain
    Starlette Route does, so `@app.get("/api/health")` answered HEAD with 405 —
    visible in the service log next to the 200s. A monitor probing with HEAD
    would have reported the service down while it was serving normally.
    """

    def _methods_for(self, path: str) -> set:
        for route in main.app.routes:
            if getattr(route, "path", None) == path:
                return set(getattr(route, "methods", set()) or set())
        self.fail(f"no route registered for {path}")

    def test_health_answers_get_and_head(self) -> None:
        self.assertEqual({"GET", "HEAD"}, self._methods_for("/api/health"))

if __name__ == "__main__":
    unittest.main(verbosity=2)
