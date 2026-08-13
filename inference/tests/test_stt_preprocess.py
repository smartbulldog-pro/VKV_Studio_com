"""
Unit tests for GemmaSTT audio preprocessing (pure DSP — no model, no GPU).
=========================================================================
Run from the inference/ directory:

    ./.venv/Scripts/python.exe -m unittest discover -s tests -v
    # or a single module:
    ./.venv/Scripts/python.exe -m unittest tests.test_stt_preprocess -v

These exercise the signal path only (decode → mono → resample → normalize →
chunk → WAV encode → language guess). The model call is gated separately and is
NOT invoked here. Pure stdlib + numpy; safe to run while the GPU is training.
"""

from __future__ import annotations

import io
import sys
import unittest
from unittest import mock
import wave
from pathlib import Path

import numpy as np

# Allow `import stt` when tests are run from inside inference/tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stt  # noqa: E402
from config import (  # noqa: E402
    GEMMA_STT_MAX_SEGMENT_SEC,
    GEMMA_STT_SAMPLE_RATE,
)


class TestToMono(unittest.TestCase):
    def test_passthrough_1d(self) -> None:
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(stt.to_mono(x), x)

    def test_average_channels(self) -> None:
        # (channels, n) planar layout, as PyAV yields
        stereo = np.array([[1.0, 3.0, 5.0], [3.0, 5.0, 7.0]], dtype=np.float32)
        np.testing.assert_array_equal(stt.to_mono(stereo), np.array([2.0, 4.0, 6.0]))

    def test_rejects_3d(self) -> None:
        with self.assertRaises(ValueError):
            stt.to_mono(np.zeros((2, 2, 2), dtype=np.float32))


class TestResample(unittest.TestCase):
    def test_downsample_length(self) -> None:
        y = stt.resample_linear(np.zeros(48_000, dtype=np.float32), 48_000, 16_000)
        self.assertEqual(len(y), 16_000)

    def test_upsample_length(self) -> None:
        y = stt.resample_linear(np.zeros(8_000, dtype=np.float32), 8_000, 16_000)
        self.assertEqual(len(y), 16_000)

    def test_identity_when_same_rate(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(stt.resample_linear(x, 16_000, 16_000), x)

    def test_empty(self) -> None:
        self.assertEqual(len(stt.resample_linear(np.zeros(0, dtype=np.float32), 44_100, 16_000)), 0)

    def test_preserves_dc_level(self) -> None:
        # A constant signal must stay (approximately) constant after resampling.
        const = np.full(44_100, 0.25, dtype=np.float32)
        y = stt.resample_linear(const, 44_100, 16_000)
        self.assertTrue(np.allclose(y, 0.25, atol=1e-4))


class TestNormalize(unittest.TestCase):
    def test_scales_down_clipping_signal(self) -> None:
        y = stt.normalize_peak(np.array([2.0, -4.0, 1.0], dtype=np.float32))
        self.assertLessEqual(float(np.max(np.abs(y))), 0.98 + 1e-6)

    def test_leaves_quiet_signal_untouched(self) -> None:
        quiet = np.array([0.1, -0.2, 0.05], dtype=np.float32)
        np.testing.assert_allclose(stt.normalize_peak(quiet), quiet)

    def test_within_unit_range(self) -> None:
        rng = np.random.default_rng(0)
        loud = (rng.standard_normal(1000) * 10).astype(np.float32)
        y = stt.normalize_peak(loud)
        self.assertLessEqual(float(np.max(np.abs(y))), 1.0)

    def test_empty(self) -> None:
        self.assertEqual(len(stt.normalize_peak(np.zeros(0, dtype=np.float32))), 0)


class TestChunking(unittest.TestCase):
    SR = GEMMA_STT_SAMPLE_RATE

    def test_short_audio_single_chunk(self) -> None:
        sig = np.zeros(5 * self.SR, dtype=np.float32)
        chunks = stt.chunk_audio(sig, self.SR, GEMMA_STT_MAX_SEGMENT_SEC, 0.5)
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0].end_sec, 5.0)

    def test_every_chunk_within_limit(self) -> None:
        sig = np.arange(65 * self.SR, dtype=np.float32)
        chunks = stt.chunk_audio(sig, self.SR, 30, 0.5)
        for c in chunks:
            self.assertLessEqual(c.end_sec - c.start_sec, 30 + 1e-6)
            self.assertLessEqual(len(c.samples), 30 * self.SR)

    def test_covers_full_signal(self) -> None:
        sig = np.arange(65 * self.SR, dtype=np.float32)
        chunks = stt.chunk_audio(sig, self.SR, 30, 0.5)
        self.assertEqual(chunks[0].start_sec, 0.0)
        self.assertAlmostEqual(chunks[-1].end_sec, 65.0)

    def test_overlap_present(self) -> None:
        sig = np.arange(65 * self.SR, dtype=np.float32)
        chunks = stt.chunk_audio(sig, self.SR, 30, 0.5)
        # Second chunk starts 0.5 s before the first chunk's end.
        self.assertAlmostEqual(chunks[1].start_sec, chunks[0].end_sec - 0.5, places=3)

    def test_no_overlap_when_zero(self) -> None:
        sig = np.arange(60 * self.SR, dtype=np.float32)
        chunks = stt.chunk_audio(sig, self.SR, 30, 0.0)
        self.assertEqual(len(chunks), 2)
        self.assertAlmostEqual(chunks[1].start_sec, 30.0)

    def test_pathological_overlap_does_not_hang(self) -> None:
        # overlap >= chunk length must not create an infinite loop.
        sig = np.arange(90 * self.SR, dtype=np.float32)
        chunks = stt.chunk_audio(sig, self.SR, 30, 40)
        self.assertGreater(len(chunks), 0)
        self.assertAlmostEqual(chunks[-1].end_sec, 90.0)

    def test_empty_audio(self) -> None:
        self.assertEqual(stt.chunk_audio(np.zeros(0, dtype=np.float32), self.SR), [])


class TestWavEncode(unittest.TestCase):
    def test_roundtrip_shape_and_rate(self) -> None:
        sig = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        raw = stt.float_to_wav_bytes(sig, 16_000)
        with wave.open(io.BytesIO(raw), "rb") as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getframerate(), 16_000)
            self.assertEqual(w.getnframes(), 5)

    def test_riff_header(self) -> None:
        raw = stt.float_to_wav_bytes(np.zeros(10, dtype=np.float32), 16_000)
        self.assertEqual(raw[:4], b"RIFF")
        self.assertEqual(raw[8:12], b"WAVE")


class TestLanguageDetect(unittest.TestCase):
    def test_russian(self) -> None:
        self.assertEqual(stt.detect_language("Привет, как дела")[0], "ru")

    def test_english(self) -> None:
        self.assertEqual(stt.detect_language("Hello there friend")[0], "en")

    def test_empty(self) -> None:
        self.assertEqual(stt.detect_language(""), ("en", 0.0))

    def test_confidence_in_range(self) -> None:
        _, prob = stt.detect_language("Привет hello мир world")
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


class TestDecodeAndPreprocess(unittest.TestCase):
    """End-to-end through PyAV decode on a tiny synthetic WAV (no model)."""

    @staticmethod
    def _make_wav(seconds: float, sr: int, channels: int) -> bytes:
        n = int(sr * seconds)
        t = np.linspace(0, seconds, n, endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        if channels == 2:
            inter = np.stack([tone, tone], axis=1)
        else:
            inter = tone.reshape(-1, 1)
        pcm16 = (np.clip(inter, -1, 1) * 32767).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm16.tobytes())
        return buf.getvalue()

    def test_stereo_44k_to_mono_16k(self) -> None:
        data = self._make_wav(0.3, 44_100, 2)
        chunks, dur = stt.GemmaSTT().preprocess(data)
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(dur, 0.3, delta=0.02)
        self.assertAlmostEqual(len(chunks[0].samples), int(0.3 * 16_000), delta=80)

    def test_preprocess_rejects_overlong_audio(self) -> None:
        """DoS guard: decoded audio longer than the cap is rejected before chunking,
        so a tiny compressed file that unpacks to hours can't monopolize the STT server.
        (Replaces a stale test that asserted transcribe() was an unimplemented stub —
        voice STT has since been implemented, commit 66e5a25.)"""
        engine = stt.GemmaSTT()
        with mock.patch.object(stt, "GEMMA_STT_MAX_DURATION_SEC", 1):
            over = self._make_wav(2.0, 16_000, 1)   # 2s > 1s cap
            with self.assertRaises(ValueError):
                engine.preprocess(over)
            under = self._make_wav(0.5, 16_000, 1)  # 0.5s < 1s cap → fine
            chunks, dur = engine.preprocess(under)
            self.assertEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
