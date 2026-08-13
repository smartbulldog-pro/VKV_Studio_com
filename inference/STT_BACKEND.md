# STT backend — Gemma-4 native audio (self-hosted, Google-only) ✅ LIVE

The Synapse voice pipeline is **audio → STT → LLM → TTS**. STT is Google-only and
**self-hosted**: `GemmaSTT` transcribes speech with **Gemma 4's own native audio
encoder** via llama.cpp mtmd — no cloud STT, no per-minute billing, no third party.
Whisper (faster-whisper) was removed 2026-07-06; Gemma STT went **live 2026-07-06**.

## How it works

Gemma 4 is natively multimodal (text + vision + **audio**). `GemmaSTT` (`stt.py`)
spawns its own `llama-server --mmproj` child and calls its OpenAI-compatible
`/v1/chat/completions` with an `input_audio` part:

```
mic/audio ─► preprocess (PyAV decode → mono → 16 kHz → normalize → ≤30 s chunks)
          ─► POST /v1/chat/completions  [ text: "transcribe…"  +  input_audio: <wav> ]
          ─► transcript ─► LLM router ─► Chirp 3 TTS
```

- **Models (both from ggml-org, non-gated):**
  - text: **`gemma-4-E4B-it-Q4_K_M.gguf`** — the **BASE** model, *not* the synapse
    fine-tune (see "Two findings"). Q4 is plenty for ASR.
  - audio mmproj: **`mmproj-gemma-4-E4B-it-bf16.gguf`** — **BF16 is required** for
    Gemma-4 audio (F16/Q8 are unreliable — llama.cpp #24118).
- **Runs on CPU** by default (`GEMMA_STT_GPU_LAYERS=0`) so it never competes with the
  LLM router for the 12 GB GPU. A base-E4B-Q4 + mmproj on GPU would collide with the
  router's E4B. CPU latency is fine: **~1.5 s per short clip** (audio encode + short decode).
- Lifecycle mirrors the LLM router / embeddings: lazy spawn on first request, health-wait,
  persistent, `unload()` on shutdown. Config: `config.py` `GEMMA_STT_*`
  (`MODEL_PATH`, `MMPROJ_PATH`, `HOST`, `PORT` 8093, `GPU_LAYERS`, `CTX_SIZE`, `PROMPT`).

Verified end-to-end through `/api/voice` and `/api/voice/stream` (RU+EN: transcript →
LLM → Chirp 3 TTS audio out). The existing mic UI works with **no frontend change**.

## Two findings that made it work

1. **Use the BASE gemma-4 model, not the synapse fine-tune.** The persona fine-tune
   ignores "transcribe verbatim" and answers in character (verified: one good
   transcript, then hallucinations). The base model does clean, deterministic ASR.
2. **Send the ASR instruction BEFORE the audio (text-first).** With the audio first,
   the model tends to *reply* to the speech instead of transcribing it.

## Getting the model files (gitignored — download on a fresh machine / deploy)

```bash
cd inference/models
curl -L -o gemma-4-E4B-it-Q4_K_M.gguf \
  https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf
curl -L -o mmproj-gemma-4-E4B-it-bf16.gguf \
  https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/mmproj-gemma-4-E4B-it-bf16.gguf
```

Requires a llama.cpp build **after PR #24118** (Gemma-4 audio fix). The bundled
`inference/llama-bin` (build 9587) is new enough — the audio server logs
`init_audio: audio input is in experimental stage` and transcribes correctly.
An E2B mmproj also exists (`ggml-org/gemma-4-E2B-it-GGUF`) if a smaller STT model is
ever wanted.

## Rejected alternatives (owner's call)

- **Google Cloud Speech-to-Text** — works (verified via a TTS→WAV→STT round-trip on the
  existing service account), but its 60-min/month free tier is a dealbreaker.
- **Web Speech API** (browser `SpeechRecognition`) — free + Google, but Chrome/Edge only
  (no Firefox/Safari), and not self-hosted.

Self-hosted Gemma audio is free, unlimited, Google-native, and works for every browser
(the audio is handled server-side).

## Limits & notes

- **≤30 s per segment** (Gemma's audio contract); longer audio is chunked with ~0.5 s overlap.
- Q4 + synthetic TTS audio can mis-hear a word or two; real speech is cleaner, and Q8
  (`gemma-4-E4B-it-Q8_0.gguf`) is a drop-in quality bump if wanted (`SYNAPSE_GEMMA_STT_MODEL`).
- `eval/wer_harness.py` (gemma-only) can score RU/EN WER against reference clips.

## TTS is unaffected

An LLM emits **text**, not audio — Gemma can *understand* speech but not *speak*. Voice-out
is always a separate **TTS** stage (Chirp 3 HD, Edge fallback). Only the STT (hear) side is Gemma.
