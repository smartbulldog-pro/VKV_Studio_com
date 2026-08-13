"""
Iter 5 (auth/CORS/validation) regression tests.
===============================================
Covers:
  - ConversationMessageItem.role / timestamp and ConversationSaveRequest.createdAt/
    updatedAt bounds — reject values that would otherwise become an uncaught 500
    (DB CHECK / OverflowError).
  - _require_json() CSRF guard: non-JSON content types are rejected (415).
  - CORS config is an explicit allowlist (never "*"), entries stripped.

Pure unit tests — no TestClient, no DB writes. Run from inference/:
    ./.venv/Scripts/python.exe -m unittest tests.test_validation_security -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402
from pydantic import ValidationError  # noqa: E402


class _FakeHeaders:
    def __init__(self, d):
        self._d = {k.lower(): v for k, v in d.items()}

    def get(self, k, default=None):
        return self._d.get(k.lower(), default)


class _FakeReq:
    def __init__(self, headers):
        self.headers = _FakeHeaders(headers)


# ── Model validation (finding #3) ────────────────────────────────────────────

class TestMessageRole(unittest.TestCase):
    def test_valid_roles_accepted(self) -> None:
        for role in ("user", "assistant", "system"):
            m = main.ConversationMessageItem(role=role, content="x")
            self.assertEqual(m.role, role)

    def test_invalid_role_rejected(self) -> None:
        # Would violate the SQLite CHECK(role IN ...) → 500 without this guard.
        for bad in ("hacker", "admin", "User", "", "system ", "root"):
            with self.assertRaises(ValidationError):
                main.ConversationMessageItem(role=bad, content="x")

    def test_content_length_bounded(self) -> None:
        with self.assertRaises(ValidationError):
            main.ConversationMessageItem(role="user", content="x" * 100_000)


class TestTimestampBounds(unittest.TestCase):
    def test_message_timestamp_overflow_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            main.ConversationMessageItem(role="user", content="x", timestamp=10**30)

    def test_message_timestamp_negative_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            main.ConversationMessageItem(role="user", content="x", timestamp=-1)

    def test_conversation_createdat_overflow_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            main.ConversationSaveRequest(id="abc", createdAt=10**30)

    def test_conversation_updatedat_overflow_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            main.ConversationSaveRequest(id="abc", updatedAt=10**30)

    def test_valid_millisecond_timestamp_ok(self) -> None:
        m = main.ConversationSaveRequest(id="abc", createdAt=1_750_000_000_000,
                                         updatedAt=1_750_000_000_000)
        self.assertGreater(m.createdAt, 0)


# ── CSRF: Content-Type enforcement (finding #1) ──────────────────────────────

class TestRequireJson(unittest.TestCase):
    def test_rejects_text_plain(self) -> None:
        # The simple-request CSRF vector: text/plain body sent cross-site.
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            main._require_json(_FakeReq({"content-type": "text/plain"}))
        self.assertEqual(ctx.exception.status_code, 415)

    def test_rejects_form_urlencoded(self) -> None:
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):
            main._require_json(_FakeReq({"content-type": "application/x-www-form-urlencoded"}))

    def test_rejects_missing_content_type(self) -> None:
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):
            main._require_json(_FakeReq({}))

    def test_accepts_application_json(self) -> None:
        main._require_json(_FakeReq({"content-type": "application/json"}))  # no raise

    def test_accepts_application_json_with_charset(self) -> None:
        main._require_json(_FakeReq({"content-type": "application/json; charset=utf-8"}))

    def test_accepts_case_insensitive(self) -> None:
        main._require_json(_FakeReq({"content-type": "Application/JSON"}))


# ── CORS config (finding #4) ─────────────────────────────────────────────────

class TestCorsConfig(unittest.TestCase):
    def test_not_wildcard(self) -> None:
        self.assertNotIn("*", main.CORS_ORIGINS)

    def test_entries_stripped(self) -> None:
        for origin in main.CORS_ORIGINS:
            self.assertEqual(origin, origin.strip())
            self.assertNotEqual(origin, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
