"""
Iter 3 (SSRF/injection) regression tests — IP validation, log sanitization, id charset.
========================================================================================
Covers:
  - get_client_ip validates proxy-header candidates as real IPs (no garbage /
    log-injection / spoofed non-IP strings returned).
  - log_safety.oneline() strips CR/LF/TAB so ids/transcripts can't forge log lines.
  - ConversationSaveRequest.id rejects control chars (accepts UUIDs).

Run from inference/:
    ./.venv/Scripts/python.exe -m unittest tests.test_request_security -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402  (lazy LLM — safe to import headless)
from log_safety import oneline  # noqa: E402


class _FakeClient:
    def __init__(self, host: str) -> None:
        self.host = host


class _FakeHeaders:
    def __init__(self, d: dict) -> None:
        self._d = {k.lower(): v for k, v in d.items()}

    def get(self, k: str, default=None):
        return self._d.get(k.lower(), default)


class _FakeReq:
    def __init__(self, host: str, headers: dict | None = None) -> None:
        self.client = _FakeClient(host)
        self.headers = _FakeHeaders(headers or {})


class TestGetClientIP(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_trust = main.TRUST_PROXY_HEADERS
        self._orig_proxies = main.TRUSTED_PROXIES

    def tearDown(self) -> None:
        main.TRUST_PROXY_HEADERS = self._orig_trust
        main.TRUSTED_PROXIES = self._orig_proxies

    def test_no_trust_returns_peer(self) -> None:
        main.TRUST_PROXY_HEADERS = False
        main.TRUSTED_PROXIES = {"127.0.0.1"}
        # Even with a spoofed header, untrusted mode ignores it.
        req = _FakeReq("203.0.113.9", {"x-forwarded-for": "1.2.3.4"})
        self.assertEqual(main.get_client_ip(req), "203.0.113.9")

    def test_valid_xff_returned_when_trusted(self) -> None:
        main.TRUST_PROXY_HEADERS = True
        main.TRUSTED_PROXIES = {"127.0.0.1"}
        req = _FakeReq("127.0.0.1", {"x-forwarded-for": "9.9.9.9, 127.0.0.1"})
        self.assertEqual(main.get_client_ip(req), "9.9.9.9")

    def test_garbage_xff_not_returned(self) -> None:
        # An injection/garbage value must NEVER be returned as the client IP.
        main.TRUST_PROXY_HEADERS = True
        main.TRUSTED_PROXIES = {"127.0.0.1"}
        req = _FakeReq("127.0.0.1", {
            "x-forwarded-for": "not-an-ip\n12:00 FAKE LOG LINE, 127.0.0.1",
        })
        result = main.get_client_ip(req)
        self.assertEqual(result, "127.0.0.1")  # fell back to peer
        self.assertNotIn("\n", result)
        self.assertNotIn("FAKE", result)

    def test_valid_cf_connecting_ip(self) -> None:
        main.TRUST_PROXY_HEADERS = True
        main.TRUSTED_PROXIES = {"127.0.0.1"}
        req = _FakeReq("127.0.0.1", {"cf-connecting-ip": "8.8.8.8"})
        self.assertEqual(main.get_client_ip(req), "8.8.8.8")

    def test_invalid_cf_ip_falls_through_to_peer(self) -> None:
        main.TRUST_PROXY_HEADERS = True
        main.TRUSTED_PROXIES = {"127.0.0.1"}
        req = _FakeReq("127.0.0.1", {"cf-connecting-ip": "evil\nstring"})
        self.assertEqual(main.get_client_ip(req), "127.0.0.1")

    def test_ipv6_valid(self) -> None:
        main.TRUST_PROXY_HEADERS = True
        main.TRUSTED_PROXIES = {"127.0.0.1"}
        req = _FakeReq("127.0.0.1", {"x-forwarded-for": "2001:db8::1, 127.0.0.1"})
        self.assertEqual(main.get_client_ip(req), "2001:db8::1")


class TestOneline(unittest.TestCase):
    def test_strips_newlines(self) -> None:
        out = oneline("id\n12:00 fake ERROR forged")
        self.assertNotIn("\n", out)
        self.assertIn("\\n", out)

    def test_strips_cr_and_tab(self) -> None:
        out = oneline("a\r\tb")
        self.assertNotIn("\r", out)
        self.assertNotIn("\t", out)

    def test_truncates(self) -> None:
        out = oneline("x" * 500, limit=64)
        self.assertLessEqual(len(out), 64 + 1)  # +ellipsis


class TestConversationIdCharset(unittest.TestCase):
    def test_uuid_accepted(self) -> None:
        m = main.ConversationSaveRequest(id="4b2f1c9a-1e2d-4a3b-9c8d-0f1e2d3c4b5a")
        self.assertTrue(m.id)

    def test_newline_id_rejected(self) -> None:
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            main.ConversationSaveRequest(id="x\n12:00 FAKE LOG")

    def test_space_and_symbols_rejected(self) -> None:
        from pydantic import ValidationError
        for bad in ("has space", "semi;colon", "quote\"x", "slash/x"):
            with self.assertRaises(ValidationError):
                main.ConversationSaveRequest(id=bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
