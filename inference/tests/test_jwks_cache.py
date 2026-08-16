"""
JWKS certificate caching on the Google ID-token path.
=====================================================
`google.oauth2.id_token.verify_oauth2_token` fetches Google's signing
certificates through the transport it is handed, and `_fetch_certs` in the
installed library is three lines that do a plain GET with no caching — checked
by reading the package, not assumed. The code here used to carry a comment
claiming the library cached them, so every authenticated request paid an HTTPS
round-trip to Google before it could do anything else.

Verification fails CLOSED, so the risk of caching is the mirror image of the
risk of not caching: a stale key set locks people out. Hence the TTL comes from
Google's own Cache-Control header and is clamped, and only successful responses
are stored.

Run from inference/:
    ./.venv/Scripts/python.exe -m pytest tests/test_jwks_cache.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402

CERTS_URL = "https://www.googleapis.com/oauth2/v1/certs"


class _Resp:
    def __init__(self, status: int = 200, cache_control: str | None = None) -> None:
        self.status = status
        self.headers = {"Cache-Control": cache_control} if cache_control else {}
        self.data = b"{}"


class _Transport:
    """Counts how many times the wrapped transport actually reaches the network."""

    def __init__(self, response_factory=None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._factory = response_factory or (lambda: _Resp(cache_control="max-age=3600"))

    def __call__(self, url, method="GET", **kwargs):
        self.calls.append((url, method))
        return self._factory()


class TestCertCaching(unittest.TestCase):
    def test_repeated_verifications_hit_the_network_once(self) -> None:
        t = _Transport()
        req = main._CachingCertsRequest(t)
        first = req(CERTS_URL)
        for _ in range(9):
            req(CERTS_URL)
        self.assertEqual(1, len(t.calls), "the certificate fetch is still per-request")
        self.assertIs(first, req(CERTS_URL), "a cache hit must return the same response")

    def test_an_expired_entry_is_refetched(self) -> None:
        t = _Transport()
        req = main._CachingCertsRequest(t)
        req(CERTS_URL)
        # Age the entry past its TTL without waiting an hour.
        with req._lock:
            expires, resp = req._cache[CERTS_URL]
            req._cache[CERTS_URL] = (expires - 10_000, resp)
        req(CERTS_URL)
        self.assertEqual(2, len(t.calls))

    def test_a_failed_fetch_is_never_cached(self) -> None:
        # Remembering a 500 as if it were the key set would lock every account
        # out until the TTL expired.
        t = _Transport(lambda: _Resp(status=500))
        req = main._CachingCertsRequest(t)
        req(CERTS_URL)
        req(CERTS_URL)
        self.assertEqual(2, len(t.calls))

    def test_non_get_requests_are_passed_straight_through(self) -> None:
        t = _Transport()
        req = main._CachingCertsRequest(t)
        req(CERTS_URL, method="POST")
        req(CERTS_URL, method="POST")
        self.assertEqual(2, len(t.calls))

    def test_separate_urls_do_not_share_an_entry(self) -> None:
        t = _Transport()
        req = main._CachingCertsRequest(t)
        req(CERTS_URL)
        req("https://www.googleapis.com/oauth2/v3/certs")
        self.assertEqual(2, len(t.calls))


class TestTtl(unittest.TestCase):
    def test_uses_googles_own_max_age(self) -> None:
        ttl = main._CachingCertsRequest._ttl_from(
            _Resp(cache_control="public, max-age=19845, must-revalidate")
        )
        self.assertEqual(19845.0, ttl)

    def test_clamps_an_absurdly_long_max_age(self) -> None:
        # Never outlive a key rotation by much, whatever a header claims.
        self.assertEqual(
            main._CachingCertsRequest._MAX_TTL,
            main._CachingCertsRequest._ttl_from(_Resp(cache_control="max-age=999999")),
        )

    def test_clamps_an_absurdly_short_max_age(self) -> None:
        # Otherwise a max-age=1 restores the per-request fetch this exists to stop.
        self.assertEqual(
            main._CachingCertsRequest._MIN_TTL,
            main._CachingCertsRequest._ttl_from(_Resp(cache_control="max-age=1")),
        )

    def test_falls_back_to_a_default_without_a_header(self) -> None:
        self.assertEqual(
            main._CachingCertsRequest._DEFAULT_TTL,
            main._CachingCertsRequest._ttl_from(_Resp()),
        )

    def test_unparseable_header_does_not_raise(self) -> None:
        self.assertEqual(
            main._CachingCertsRequest._DEFAULT_TTL,
            main._CachingCertsRequest._ttl_from(_Resp(cache_control="no-store, private")),
        )


class TestWiring(unittest.TestCase):
    def test_the_verifier_is_handed_the_caching_transport(self) -> None:
        # A cache nothing is wired to is decoration.
        if not main._GOOGLE_AUTH_AVAILABLE:
            self.skipTest("google-auth not installed")
        self.assertIsInstance(main._google_request, main._CachingCertsRequest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
