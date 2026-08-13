# STT WER eval — Whisper vs Gemma-4 native audio

Gate **(d)** from the STT migration: the Gemma backend may only become the default
once its **RU/EN WER is at or below Whisper's**. This harness measures that.

## Layout

| File | Purpose |
|---|---|
| `wer.py` | Pure-Python WER metric (word-level Levenshtein). No deps. |
| `wer_harness.py` | Runs backends over a manifest → RU-WER / EN-WER table. |
| `gen_clips.py` | *Optional* — synth reference clips via Edge TTS (network, opt-in). |
| `manifest.example.json` | Template; copy to `manifest.json` and fill in. |
| `clips/` | Audio clips (gitignored — local only). |

## Quick start

```bash
cd inference
cp eval/manifest.example.json eval/manifest.json      # then edit references/files

# 1. validate the manifest, load nothing (safe anytime):
./.venv/Scripts/python.exe -m eval.wer_harness --manifest eval/manifest.json --dry-run

# 2. (optional) generate clean clips from the references (Edge TTS = network, light CPU):
./.venv/Scripts/python.exe -m eval.gen_clips --manifest eval/manifest.json

# 3. score Whisper (loads Whisper — CPU heavy; run when the GPU/CPU is free):
./.venv/Scripts/python.exe -m eval.wer_harness --manifest eval/manifest.json --backends whisper

# 4. once the audio-mmproj GGUF exists + `llama-server --mmproj` is running,
#    score both and compare:
./.venv/Scripts/python.exe -m eval.wer_harness --manifest eval/manifest.json --backends whisper gemma
```

Example output:

```
backend        RU-WER     EN-WER
-------------------------------
whisper          4.20%      2.10%
gemma           PENDING    PENDING     ← until audio-mmproj GGUF is exported
```

## Status of the Gemma side

- ✅ Audio preprocessing (decode → mono → 16 kHz → normalize → ≤30 s chunks) is
  implemented and unit-tested (`tests/test_stt_preprocess.py`).
- ✅ mtmd HTTP client (llama-server `/v1/chat/completions`, `input_audio`) is fully
  written.
- ⛔ **Blocked on the main session:** export the **audio-mmproj GGUF** and build/run
  `llama-mtmd-cli` / `llama-server --mmproj`. Until the GGUF at
  `config.GEMMA_STT_MMPROJ_PATH` exists, `GemmaSTT.transcribe()` raises
  `NotImplementedError` and the harness reports `PENDING` for that backend.

## Notes

- Clips are **synthetic** (Edge TTS) by default → a clean-speech lower bound. Add
  real/noisy public-domain clips for a robustness read (Whisper's known strength).
- WER normalization: lowercase, NFKC, punctuation stripped, whitespace collapsed
  (`ё`/`е` kept distinct). Micro-averaged across a language's clips.
- Do **not** run heavy backends while models are training — stick to `--dry-run`
  and the unit tests, which use only tiny synthetic signals.
```
