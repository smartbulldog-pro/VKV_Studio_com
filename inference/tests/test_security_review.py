"""Regression tests for the 2026-07-08 pre-deploy security review fixes.

Covers the unit-testable new guards: cross-site CSRF rejection, capability-token
conversation ownership, secret-stripped child env, loopback startup assertion,
chunked-body 411, and conversation retention purge.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest
from unittest import mock

import main
import log_safety
import chat_storage


class _FakeClient:
    def __init__(self, host: str) -> None:
        self.host = host


class _FakeHeaders(dict):
    """Case-insensitive .get, like Starlette's Headers."""
    def get(self, key, default=None):  # type: ignore[override]
        return super().get(key.lower(), default)


class _FakeRequest:
    def __init__(self, headers: dict | None = None, host: str = "1.2.3.4") -> None:
        self.headers = _FakeHeaders({k.lower(): v for k, v in (headers or {}).items()})
        self.client = _FakeClient(host)


class TestCrossSiteGuard(unittest.TestCase):
    def test_cross_site_rejected(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as cm:
            main._reject_cross_site(_FakeRequest({"sec-fetch-site": "cross-site"}))
        self.assertEqual(cm.exception.status_code, 403)

    def test_same_and_missing_allowed(self):
        # None of these should raise.
        for site in ("same-origin", "same-site", "none", ""):
            main._reject_cross_site(_FakeRequest({"sec-fetch-site": site}))
        main._reject_cross_site(_FakeRequest({}))  # header absent (non-browser client)


class TestGoogleAuthOwner(unittest.TestCase):
    """Conversation ownership is a VERIFIED Google account; anonymous = None (no server
    history). IP is NEVER an ownership fallback. verify_oauth2_token is mocked."""

    def test_anonymous_is_none(self):
        # No Authorization header → anonymous → None (no server-side history).
        self.assertIsNone(main.get_authenticated_owner(_FakeRequest({})))

    def test_no_client_id_configured_is_none(self):
        # Even with a Bearer token, if the server has no configured client id, fail closed.
        with mock.patch.object(main, "GOOGLE_OAUTH_CLIENT_ID", ""):
            self.assertIsNone(main.get_authenticated_owner(_FakeRequest({"authorization": "Bearer a.b.c"})))

    def test_valid_token_returns_sub(self):
        with mock.patch.object(main, "GOOGLE_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com"), \
             mock.patch.object(main, "_GOOGLE_AUTH_AVAILABLE", True), \
             mock.patch.object(main, "_google_id_token") as gid:
            gid.verify_oauth2_token.return_value = {"sub": "1234567890", "email": "x@y.z"}
            owner = main.get_authenticated_owner(_FakeRequest({"authorization": "Bearer good.jwt.token"}))
        self.assertEqual(owner, "sub:1234567890")

    def test_audience_is_enforced(self):
        # The client id MUST be passed as `audience=` — else any Google token passes.
        with mock.patch.object(main, "GOOGLE_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com"), \
             mock.patch.object(main, "_GOOGLE_AUTH_AVAILABLE", True), \
             mock.patch.object(main, "_google_id_token") as gid:
            gid.verify_oauth2_token.return_value = {"sub": "s"}
            main.get_authenticated_owner(_FakeRequest({"authorization": "Bearer t"}))
            _, kwargs = gid.verify_oauth2_token.call_args
            self.assertEqual(kwargs.get("audience"), "cid.apps.googleusercontent.com")

    def test_verification_failure_is_none(self):
        # A raised ValueError (bad sig / expired / wrong aud) → None, never a leak.
        with mock.patch.object(main, "GOOGLE_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com"), \
             mock.patch.object(main, "_GOOGLE_AUTH_AVAILABLE", True), \
             mock.patch.object(main, "_google_id_token") as gid:
            gid.verify_oauth2_token.side_effect = ValueError("bad token")
            self.assertIsNone(main.get_authenticated_owner(_FakeRequest({"authorization": "Bearer t"})))

    def test_oversized_token_rejected(self):
        with mock.patch.object(main, "GOOGLE_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com"):
            huge = "Bearer " + "x" * (main._MAX_BEARER_LEN + 100)
            self.assertIsNone(main.get_authenticated_owner(_FakeRequest({"authorization": huge})))

    def test_beacon_explicit_token(self):
        # sendBeacon path: token from the body via explicit_token, verified identically.
        with mock.patch.object(main, "GOOGLE_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com"), \
             mock.patch.object(main, "_GOOGLE_AUTH_AVAILABLE", True), \
             mock.patch.object(main, "_google_id_token") as gid:
            gid.verify_oauth2_token.return_value = {"sub": "42"}
            self.assertEqual(main.get_authenticated_owner(_FakeRequest({}), "body.jwt.token"), "sub:42")


class TestChildEnv(unittest.TestCase):
    def test_secrets_stripped_path_kept(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sek", "GOOGLE_API_KEY": "g",
                                          "SYNAPSE_GOOGLE_TTS_CREDENTIALS": "/x", "PATH": "/usr/bin"}):
            env = log_safety.child_env()
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("GOOGLE_API_KEY", env)
        self.assertNotIn("SYNAPSE_GOOGLE_TTS_CREDENTIALS", env)
        self.assertIn("PATH", env)


class TestLoopbackAssertion(unittest.TestCase):
    def test_loopback_values(self):
        self.assertTrue(main._is_loopback("127.0.0.1"))
        self.assertTrue(main._is_loopback("::1"))
        self.assertTrue(main._is_loopback("localhost"))
        self.assertFalse(main._is_loopback("0.0.0.0"))
        self.assertFalse(main._is_loopback("10.0.0.5"))

    def test_startup_refuses_public_internal_host(self):
        with mock.patch.object(main, "LLM_SERVER_HOST", "0.0.0.0"):
            with self.assertRaises(SystemExit):
                main._security_startup_checks()

    def test_startup_refuses_wildcard_cors(self):
        with mock.patch.object(main, "CORS_ORIGINS", ["*"]):
            with self.assertRaises(SystemExit):
                main._security_startup_checks()


class TestChunkedBody411(unittest.TestCase):
    def _run(self, method: str, headers: list) -> int:
        mw = main.ContentLengthLimitMiddleware(app=None, max_bytes=1000)
        status = {}

        async def send(msg):
            if msg["type"] == "http.response.start":
                status["code"] = msg["status"]

        async def fake_app(scope, receive, send):  # passthrough → 200 sentinel
            status["code"] = 200

        mw.app = fake_app
        scope = {"type": "http", "method": method, "headers": headers}
        asyncio.run(mw(scope, None, send))
        return status["code"]

    def test_post_without_content_length_is_411(self):
        self.assertEqual(self._run("POST", []), 411)

    def test_post_with_content_length_passes(self):
        self.assertEqual(self._run("POST", [(b"content-length", b"50")]), 200)

    def test_oversized_content_length_is_413(self):
        self.assertEqual(self._run("POST", [(b"content-length", b"999999")]), 413)

    def test_get_without_length_passes(self):
        self.assertEqual(self._run("GET", []), 200)


class TestRetentionPurge(unittest.TestCase):
    def test_purge_old_removes_stale_keeps_fresh(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        cs = chat_storage.ChatStorage(path)
        cs.init()
        now = int(time.time() * 1000)
        cs.save_conversation("ip1", {"id": "old", "title": "x", "createdAt": now,
                                     "updatedAt": now - 40 * 86_400_000,
                                     "messages": [{"id": "m1", "role": "user", "content": "hi", "timestamp": now}]})
        cs.save_conversation("ip1", {"id": "fresh", "title": "y", "createdAt": now, "updatedAt": now,
                                     "messages": [{"id": "m2", "role": "user", "content": "yo", "timestamp": now}]})
        self.assertEqual(cs.purge_old(30), 1)
        self.assertEqual([c["id"] for c in cs.list_conversations("ip1")], ["fresh"])

    def test_purge_disabled_when_zero(self):
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        cs = chat_storage.ChatStorage(path)
        cs.init()
        self.assertEqual(cs.purge_old(0), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
