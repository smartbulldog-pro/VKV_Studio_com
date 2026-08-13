"""
Security & cost-control tests for the Google Chirp 3 HD TTS backend.
===================================================================
No network / no real key: the Google client is mocked and the budget uses a
temp file with an injected clock. Covers:
  - per-request character cap logic,
  - monthly budget counter (add/reset/boundary/disabled),
  - budget-cap routes synthesis to free Edge (Google NOT called),
  - the credentials PATH is scrubbed from logs and the key CONTENT never leaks,
  - creds-inside-repo warning,
  - graceful fallback never drops the request.

Run from inference/:
    ./.venv/Scripts/python.exe -m unittest tests.test_tts_security -v
"""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tts  # noqa: E402
from tts_budget import TTSBudget  # noqa: E402


# ── Fakes ────────────────────────────────────────────────────────────────────

class _FakeEncoding:
    MP3 = "MP3"
    LINEAR16 = "LINEAR16"
    OGG_OPUS = "OGG_OPUS"


class _FakeResp:
    def __init__(self, content: bytes) -> None:
        self.audio_content = content


class _FakeTTSModule:
    AudioEncoding = _FakeEncoding

    def SynthesisInput(self, text):  # noqa: N802
        return {"text": text}

    def VoiceSelectionParams(self, language_code, name):  # noqa: N802
        return {"language_code": language_code, "name": name}

    def AudioConfig(self, audio_encoding, sample_rate_hertz):  # noqa: N802
        return {"enc": audio_encoding, "sr": sample_rate_hertz}


class _FakeClient:
    def __init__(self, content: bytes = b"GOOGLE_AUDIO") -> None:
        self._content = content
        self.calls = 0

    def synthesize_speech(self, input, voice, audio_config):  # noqa: A002
        self.calls += 1
        return _FakeResp(self._content)


class _FakeBudget:
    """Deterministic budget stand-in (reserve/refund interface)."""
    def __init__(self, exceed: bool, cap: int = 900_000, used: int = 0) -> None:
        self._exceed = exceed
        self._cap = cap
        self._used = used
        self.added: list[int] = []
        self.refunded: list[int] = []

    def try_reserve(self, n):
        if self._exceed:
            return False
        self.added.append(n)
        self._used += n
        return True

    def refund(self, n):
        self.refunded.append(n)
        self._used = max(0, self._used - n)

    def would_exceed(self, n):
        return self._exceed

    def usage(self):
        return self._used

    @property
    def cap(self):
        return self._cap

    def add(self, n):
        self.added.append(n)
        self._used += n
        return self._used


def _wire_fake_google(engine: "tts.ChirpTTS", content: bytes = b"GOOGLE_AUDIO") -> _FakeClient:
    client = _FakeClient(content)
    engine._client = client
    engine._texttospeech = _FakeTTSModule()
    return client


class _ListHandler(logging.Handler):
    """Capture emitted log messages (formatted) for leak assertions."""
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


# ── Per-request character cap ────────────────────────────────────────────────

class TestCharCap(unittest.TestCase):
    def test_within_cap(self) -> None:
        self.assertTrue(tts.within_char_cap("x" * tts.TTS_MAX_CHARS))

    def test_over_cap(self) -> None:
        self.assertFalse(tts.within_char_cap("x" * (tts.TTS_MAX_CHARS + 1)))

    def test_empty(self) -> None:
        self.assertTrue(tts.within_char_cap(""))


# ── Monthly budget counter ───────────────────────────────────────────────────

class TestBudget(unittest.TestCase):
    def _budget(self, cap=1000, month="2026-07"):
        tmp = Path(tempfile.mkdtemp()) / "budget.json"
        clock = lambda: datetime(int(month[:4]), int(month[5:]), 15, tzinfo=timezone.utc)
        return TTSBudget(tmp, cap, clock=clock), tmp

    def test_fresh_usage_zero(self) -> None:
        b, _ = self._budget()
        self.assertEqual(b.usage(), 0)
        self.assertEqual(b.remaining(), 1000)

    def test_add_and_persist(self) -> None:
        b, path = self._budget()
        b.add(300)
        self.assertEqual(b.usage(), 300)
        self.assertEqual(b.remaining(), 700)
        # New instance over the same file sees persisted value.
        b2 = TTSBudget(path, 1000, clock=b._clock)
        self.assertEqual(b2.usage(), 300)

    def test_would_exceed_boundary(self) -> None:
        b, _ = self._budget(cap=1000)
        b.add(900)
        self.assertFalse(b.would_exceed(100))   # 900+100 == 1000, not over
        self.assertTrue(b.would_exceed(101))     # 900+101 > 1000

    def test_month_rollover_resets(self) -> None:
        b, path = self._budget(cap=1000, month="2026-07")
        b.add(500)
        self.assertEqual(b.usage(), 500)
        # Same file, next month → counter resets to 0.
        b_aug = TTSBudget(path, 1000,
                          clock=lambda: datetime(2026, 8, 1, tzinfo=timezone.utc))
        self.assertEqual(b_aug.usage(), 0)

    def test_cap_zero_disables(self) -> None:
        b, _ = self._budget(cap=0)
        self.assertFalse(b.would_exceed(10_000_000))
        b.add(10_000_000)
        self.assertFalse(b.would_exceed(1))


# ── Budget cap routes to Edge (Google NOT called) ────────────────────────────

class TestBudgetRoutesToEdge(unittest.TestCase):
    def test_over_budget_uses_edge_not_google(self) -> None:
        eng = tts.ChirpTTS(budget=_FakeBudget(exceed=True))
        google = _wire_fake_google(eng)

        class _StubEdge:
            def synthesize(self, text, lang):
                return b"EDGE_AUDIO"

        eng._fallback = _StubEdge()

        out = eng.synthesize("hello there", "en")
        self.assertEqual(out, b"EDGE_AUDIO")
        self.assertEqual(google.calls, 0, "Google must NOT be called once over budget")

    def test_under_budget_calls_google_and_counts(self) -> None:
        budget = _FakeBudget(exceed=False)
        eng = tts.ChirpTTS(budget=budget)
        google = _wire_fake_google(eng, content=b"GOOGLE_AUDIO")

        out = eng.synthesize("hello", "en")
        self.assertEqual(out, b"GOOGLE_AUDIO")
        self.assertEqual(google.calls, 1)
        # Counted exactly the cleaned char count sent to Google ("hello" → 5).
        self.assertEqual(budget.added, [5])


# ── Secret hygiene: path scrubbed, content never leaks ───────────────────────

class TestSecretHygiene(unittest.TestCase):
    def test_scrub_redacts_path(self) -> None:
        orig = tts.GOOGLE_TTS_CREDENTIALS
        tts.GOOGLE_TTS_CREDENTIALS = "/home/user/secret/key.json"
        try:
            scrubbed = tts._scrub_secret("boom reading /home/user/secret/key.json now")
            self.assertNotIn("/home/user/secret/key.json", scrubbed)
            self.assertIn("<tts-creds-path>", scrubbed)
        finally:
            tts.GOOGLE_TTS_CREDENTIALS = orig

    def test_failure_log_scrubs_path_and_never_logs_key_content(self) -> None:
        # Create a real creds file with sentinel CONTENT that must never appear.
        tmp = Path(tempfile.mkdtemp()) / "sa.json"
        secret_content = "SUPER_SECRET_PRIVATE_KEY_ABC123"
        tmp.write_text('{"private_key": "%s"}' % secret_content, encoding="utf-8")

        handler = _ListHandler()
        log = logging.getLogger("synapse.tts")
        log.addHandler(handler)
        prev_level = log.level
        log.setLevel(logging.DEBUG)

        orig = tts.GOOGLE_TTS_CREDENTIALS
        tts.GOOGLE_TTS_CREDENTIALS = str(tmp)
        try:
            eng = tts.ChirpTTS(budget=_FakeBudget(exceed=False))

            # Google path raises, its message mentions the PATH (as a real SDK error might).
            def _boom(clean_text, lang):
                raise RuntimeError(f"auth failed for {tmp}")

            eng._synthesize_google = _boom  # type: ignore[assignment]

            class _StubEdge:
                def synthesize(self, text, lang):
                    return b"EDGE_AUDIO"

            eng._fallback = _StubEdge()

            out = eng.synthesize("hello", "en")
            self.assertEqual(out, b"EDGE_AUDIO")  # request did not fail

            joined = "\n".join(handler.messages)
            # The key file CONTENT must never appear in any log line.
            self.assertNotIn(secret_content, joined)
            # The raw creds path must be scrubbed in the failure log.
            self.assertNotIn(str(tmp), joined)
            self.assertIn("<tts-creds-path>", joined)
        finally:
            tts.GOOGLE_TTS_CREDENTIALS = orig
            log.removeHandler(handler)
            log.setLevel(prev_level)


# ── Credentials-location check ───────────────────────────────────────────────

class TestCredentialsLocation(unittest.TestCase):
    def test_none_when_unset(self) -> None:
        orig = tts.GOOGLE_TTS_CREDENTIALS
        tts.GOOGLE_TTS_CREDENTIALS = ""
        try:
            self.assertIsNone(tts.check_credentials_location())
        finally:
            tts.GOOGLE_TTS_CREDENTIALS = orig

    def test_warns_when_inside_repo(self) -> None:
        repo = Path(tempfile.mkdtemp())
        inside = repo / "inference" / "key.json"
        inside.parent.mkdir(parents=True)
        inside.write_text("{}", encoding="utf-8")
        orig = tts.GOOGLE_TTS_CREDENTIALS
        tts.GOOGLE_TTS_CREDENTIALS = str(inside)
        try:
            warning = tts.check_credentials_location(repo_root=str(repo))
            self.assertIsNotNone(warning)
            self.assertIn("INSIDE the repo", warning)
            self.assertIn("ROTATE", warning)
        finally:
            tts.GOOGLE_TTS_CREDENTIALS = orig

    def test_ok_when_outside_repo(self) -> None:
        repo = Path(tempfile.mkdtemp())
        (repo / "inference").mkdir(parents=True)
        outside = Path(tempfile.mkdtemp()) / "key.json"
        outside.write_text("{}", encoding="utf-8")
        orig = tts.GOOGLE_TTS_CREDENTIALS
        tts.GOOGLE_TTS_CREDENTIALS = str(outside)
        try:
            self.assertIsNone(tts.check_credentials_location(repo_root=str(repo)))
        finally:
            tts.GOOGLE_TTS_CREDENTIALS = orig


# ── Graceful fallback never drops the request ────────────────────────────────

class TestGracefulFallback(unittest.TestCase):
    def test_no_creds_synth_falls_back(self) -> None:
        # No client wired + no creds → _ensure_loaded import/call fails → Edge.
        eng = tts.ChirpTTS(budget=_FakeBudget(exceed=False))

        def _boom(clean_text, lang):
            raise RuntimeError("no credentials")

        eng._synthesize_google = _boom  # type: ignore[assignment]

        class _StubEdge:
            def synthesize(self, text, lang):
                return b"EDGE_AUDIO"

        eng._fallback = _StubEdge()
        self.assertEqual(eng.synthesize("hello", "en"), b"EDGE_AUDIO")


if __name__ == "__main__":
    unittest.main(verbosity=2)
