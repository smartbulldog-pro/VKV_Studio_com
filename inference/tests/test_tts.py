"""
Unit tests for TTS backend consolidation (Google Chirp 3 HD + Edge fallback).
============================================================================
No network, no Google package required: the Google client is mocked and the
lazy `from google.cloud import texttospeech` is bypassed by injecting fakes.

Run from inference/:
    ./.venv/Scripts/python.exe -m unittest tests.test_tts -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tts  # noqa: E402


# ── Fakes for the Google Cloud TTS client (no package/network needed) ────────

class _FakeEncoding:
    MP3 = "MP3"
    LINEAR16 = "LINEAR16"
    OGG_OPUS = "OGG_OPUS"


class _FakeResp:
    def __init__(self, content: bytes) -> None:
        self.audio_content = content


class _FakeTTSModule:
    """Stand-in for the `texttospeech` module, recording what it was handed."""
    AudioEncoding = _FakeEncoding

    def __init__(self) -> None:
        self.last_voice = None
        self.last_audio_config = None

    def SynthesisInput(self, text):  # noqa: N802 (match Google's API casing)
        return {"text": text}

    def VoiceSelectionParams(self, language_code, name):  # noqa: N802
        self.last_voice = {"language_code": language_code, "name": name}
        return self.last_voice

    def AudioConfig(self, audio_encoding, sample_rate_hertz):  # noqa: N802
        self.last_audio_config = {
            "audio_encoding": audio_encoding,
            "sample_rate_hertz": sample_rate_hertz,
        }
        return self.last_audio_config


class _FakeClient:
    def __init__(self, content: bytes = b"GOOGLE_AUDIO") -> None:
        self._content = content
        self.calls = 0

    def synthesize_speech(self, input, voice, audio_config):  # noqa: A002
        self.calls += 1
        return _FakeResp(self._content)


def _wire_fake_chirp(engine: "tts.ChirpTTS", content: bytes = b"GOOGLE_AUDIO"):
    """Inject fakes so _ensure_loaded() short-circuits (never imports google)."""
    fake_mod = _FakeTTSModule()
    engine._client = _FakeClient(content)
    engine._texttospeech = fake_mod
    return engine._client, fake_mod


# ── Backend selection ────────────────────────────────────────────────────────

class TestBackendSelection(unittest.TestCase):
    def test_edge_explicit(self) -> None:
        self.assertIsInstance(tts.create_tts_engine("edge"), tts.EdgeTTSEngine)

    def test_chirp_with_credentials(self) -> None:
        orig = tts.ChirpTTS.credentials_available
        tts.ChirpTTS.credentials_available = staticmethod(lambda: True)
        try:
            self.assertIsInstance(tts.create_tts_engine("chirp3"), tts.ChirpTTS)
        finally:
            tts.ChirpTTS.credentials_available = staticmethod(orig)

    def test_chirp_without_credentials_falls_back_to_edge(self) -> None:
        orig = tts.ChirpTTS.credentials_available
        tts.ChirpTTS.credentials_available = staticmethod(lambda: False)
        try:
            eng = tts.create_tts_engine("chirp3")
            self.assertIsInstance(eng, tts.EdgeTTSEngine)
        finally:
            tts.ChirpTTS.credentials_available = staticmethod(orig)

    def test_google_cloud_alias(self) -> None:
        orig = tts.ChirpTTS.credentials_available
        tts.ChirpTTS.credentials_available = staticmethod(lambda: True)
        try:
            self.assertIsInstance(tts.create_tts_engine("google_cloud"), tts.ChirpTTS)
        finally:
            tts.ChirpTTS.credentials_available = staticmethod(orig)

    def test_unknown_backend_falls_back_to_edge(self) -> None:
        # Removed backends (xtts/silero) must no longer raise — they degrade to Edge.
        for name in ("xtts", "silero", "bogus"):
            self.assertIsInstance(tts.create_tts_engine(name), tts.EdgeTTSEngine, name)

    def test_env_default_used_when_no_arg(self) -> None:
        orig_backend = tts.TTS_BACKEND
        orig_creds = tts.ChirpTTS.credentials_available
        tts.TTS_BACKEND = "edge"
        try:
            self.assertIsInstance(tts.create_tts_engine(), tts.EdgeTTSEngine)
        finally:
            tts.TTS_BACKEND = orig_backend
            tts.ChirpTTS.credentials_available = staticmethod(orig_creds)


# ── Language → voice mapping ─────────────────────────────────────────────────

class TestVoiceMapping(unittest.TestCase):
    def test_russian_voice(self) -> None:
        eng = tts.ChirpTTS()
        _, fake_mod = _wire_fake_chirp(eng)
        out = eng._synthesize_google("привет", "ru")
        self.assertEqual(out, b"GOOGLE_AUDIO")
        self.assertEqual(fake_mod.last_voice["language_code"], "ru-RU")
        self.assertEqual(fake_mod.last_voice["name"], tts.CHIRP_TTS_VOICE_RU)

    def test_english_voice(self) -> None:
        eng = tts.ChirpTTS()
        _, fake_mod = _wire_fake_chirp(eng)
        eng._synthesize_google("hello", "en")
        self.assertEqual(fake_mod.last_voice["language_code"], "en-US")
        self.assertEqual(fake_mod.last_voice["name"], tts.CHIRP_TTS_VOICE_EN)

    def test_encoding_matches_format(self) -> None:
        eng = tts.ChirpTTS()
        eng._format = "wav"
        _, fake_mod = _wire_fake_chirp(eng)
        eng._synthesize_google("hi", "en")
        self.assertEqual(fake_mod.last_audio_config["audio_encoding"], "LINEAR16")


# ── Graceful fallback on synth failure ───────────────────────────────────────

class TestGracefulFallback(unittest.TestCase):
    def test_synth_failure_falls_back_to_edge(self) -> None:
        eng = tts.ChirpTTS()

        # Make the Google path explode.
        def _boom(clean_text, lang):
            raise RuntimeError("invalid credentials")

        eng._synthesize_google = _boom  # type: ignore[assignment]

        # Stub the Edge fallback so we don't hit the network.
        class _StubEdge:
            def synthesize(self, text, lang):
                return b"EDGE_AUDIO"

        eng._fallback = _StubEdge()  # type: ignore[assignment]

        out = eng.synthesize("some **markdown** text", "en")
        self.assertEqual(out, b"EDGE_AUDIO")

    def test_empty_text_returns_empty(self) -> None:
        eng = tts.ChirpTTS()
        # Whitespace/markdown-only → cleaned to empty → no synth, no fallback needed.
        self.assertEqual(eng.synthesize("   ", "en"), b"")


# ── Output format contract (what main.py reads for Content-Type) ─────────────

class TestOutputFormat(unittest.TestCase):
    def test_edge_is_mp3(self) -> None:
        self.assertEqual(tts.EdgeTTSEngine().output_format, "mp3")

    def test_chirp_default_is_mp3(self) -> None:
        self.assertEqual(tts.ChirpTTS().output_format, "mp3")

    def test_chirp_respects_configured_format(self) -> None:
        eng = tts.ChirpTTS()
        eng._format = "ogg"
        self.assertEqual(eng.output_format, "ogg")

    def test_base_default_is_wav(self) -> None:
        # Any future engine that doesn't override output_format defaults to wav.
        class _Bare(tts.TTSEngine):
            def synthesize(self, text, lang):
                return b""

            @property
            def is_loaded(self):
                return True

            def unload(self):
                pass

        self.assertEqual(_Bare().output_format, "wav")


# ── Removed backends are gone ────────────────────────────────────────────────

class TestRemovedBackends(unittest.TestCase):
    def test_classes_removed(self) -> None:
        for gone in ("XTTSEngine", "SileroTTSEngine", "GoogleCloudTTSEngine"):
            self.assertFalse(hasattr(tts, gone), f"{gone} should be removed")

    def test_silero_helpers_removed(self) -> None:
        for gone in ("_preprocess_for_silero", "_number_to_russian_words",
                     "_transliterate_to_cyrillic"):
            self.assertFalse(hasattr(tts, gone), f"{gone} should be removed")

    def test_no_coqui_or_torch_tts_imports(self) -> None:
        # Check for actual CODE tokens, not the word "silero" in the changelog note.
        src = (Path(__file__).resolve().parent.parent / "tts.py").read_text(encoding="utf-8")
        self.assertNotIn("from TTS", src)          # Coqui import
        self.assertNotIn("import torch", src)       # Silero/torch dependency
        self.assertNotIn("snakers4", src)           # Silero torch.hub repo
        self.assertNotIn("apply_tts", src)          # Silero synthesis call
        self.assertNotIn("silero_tts", src)         # Silero model name


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ── Mixed-language segmentation ───────────────────────────────────────────────

class TestMergeUnspeakable:
    """
    Guards the defect that silently disabled dual-voice TTS in production.

    `_split_mixed_text` emits the gaps BETWEEN adjacent English words as Russian
    segments, so an ordinary bilingual sentence produced a segment that was just
    ", ". Edge TTS rejects that with "No audio was received", which raised,
    aborted the whole mixed render, and fell back to a single voice. The server
    log showed that warning on essentially every bilingual answer — the feature
    was never actually running, and nothing failed loudly enough to notice.
    """

    def test_production_sentence_has_an_unspeakable_segment(self):
        # The exact shape that was failing: two English names separated by a
        # comma, so the splitter hands ", " to the synthesiser on its own.
        from tts import _split_mixed_text
        text = "Три инструмента: Tokenizer Profiler, Prompt Architect и Embedding Explorer."
        segments = _split_mixed_text(text)
        assert any(not any(c.isalnum() for c in s) for s, _ in segments), (
            "the regression this guards against is gone from the splitter; "
            "re-check whether _merge_unspeakable is still needed"
        )

    def test_merge_leaves_nothing_unspeakable(self):
        from tts import _split_mixed_text, _merge_unspeakable
        text = "Три инструмента: Tokenizer Profiler, Prompt Architect и Embedding Explorer."
        merged = _merge_unspeakable(_split_mixed_text(text))
        assert merged, "merging must not empty the segment list"
        for seg, _lang in merged:
            assert any(c.isalnum() for c in seg), f"{seg!r} has nothing to pronounce"

    def test_merge_loses_no_text(self):
        # Punctuation is folded into the previous segment, never dropped, so the
        # spoken text is character-for-character what was asked for.
        from tts import _split_mixed_text, _merge_unspeakable
        text = "Я Synapse, ассистент VKVstudio. React, Vue и Svelte."
        segments = _split_mixed_text(text)
        assert "".join(s for s, _ in _merge_unspeakable(segments)) == "".join(
            s for s, _ in segments
        )

    def test_punctuation_joins_the_preceding_voice(self):
        from tts import _merge_unspeakable
        merged = _merge_unspeakable([("React", "en"), (", ", "ru"), ("Vue", "en")])
        assert merged == [("React, ", "en"), ("Vue", "en")]

    def test_leading_unspeakable_fragment_is_dropped(self):
        # Nothing to attach it to, and it carries no words.
        from tts import _merge_unspeakable
        assert _merge_unspeakable([("— ", "ru"), ("Hello", "en")]) == [("Hello", "en")]

    def test_digits_count_as_speakable(self):
        # "2026" is pronounceable; it must not be folded away as punctuation.
        from tts import _merge_unspeakable
        assert _merge_unspeakable([("2026", "ru")]) == [("2026", "ru")]

    def test_empty_and_whitespace_segments_are_skipped(self):
        from tts import _merge_unspeakable
        assert _merge_unspeakable([("", "ru"), ("   ", "en"), ("да", "ru")]) == [("да", "ru")]
