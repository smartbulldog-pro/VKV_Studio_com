"""
Optional: generate reference clips for the WER harness via Edge TTS.
====================================================================
Synthesizes each `reference` string in a manifest into eval/clips/<file>. This
produces CLEAN synthetic speech — a lower bound on real-world WER, but a fair,
reproducible A/B substrate for Whisper vs Gemma (both see identical audio).

DO NOT run this while heavy GPU/CPU training is going on unless you're fine with
a little network + light CPU — Edge TTS is a *network* call (Microsoft), not local
compute, so it's light on this machine. Still, it's OPT-IN: nothing runs it
automatically.

Usage (from inference/):
    ./.venv/Scripts/python.exe -m eval.gen_clips --manifest eval/manifest.json

Requires `edge-tts` (already used by the TTS backend). Voices match the studio's
Edge TTS setup (multilingual neural). Output is 16 kHz mono WAV so it matches
Gemma's input contract directly (Edge outputs 24 kHz MP3 → we transcode via PyAV).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

_INFERENCE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_INFERENCE_DIR))

# Edge multilingual voice handles both RU and EN.
VOICE = "en-US-AvaMultilingualNeural"


async def _synth_one(text: str, out_path: Path) -> None:
    import edge_tts  # type: ignore[import-untyped]

    communicate = edge_tts.Communicate(text, VOICE)
    mp3 = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3.extend(chunk["data"])
    # Transcode MP3 → 16 kHz mono WAV via PyAV so clips match the STT input contract.
    _mp3_to_wav16k(bytes(mp3), out_path)


def _mp3_to_wav16k(mp3_bytes: bytes, out_path: Path) -> None:
    import wave

    import numpy as np

    import stt  # reuse the same decode/DSP path the backend uses

    raw, src_sr = stt._decode_audio(mp3_bytes)   # (channels, n) @ native sr
    mono = stt.to_mono(raw)
    mono16 = stt.normalize_peak(stt.resample_linear(mono, src_sr, 16_000))
    pcm16 = (np.clip(mono16, -1, 1) * 32767).astype("<i2")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(pcm16.tobytes())


async def _main_async(manifest: Path) -> None:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    base = manifest.parent
    for c in data.get("clips", []):
        out = base / c["file"]
        if out.exists():
            print(f"skip (exists): {out}")
            continue
        print(f"synth [{c['lang']}] -> {out}")
        await _synth_one(c["reference"], out)
    print("done.")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Generate WER reference clips via Edge TTS.")
    ap.add_argument("--manifest", default="eval/manifest.json", type=Path)
    args = ap.parse_args(argv)
    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest} (copy manifest.example.json first)")
        return 2
    asyncio.run(_main_async(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
