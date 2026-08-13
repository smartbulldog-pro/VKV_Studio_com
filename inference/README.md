# Synapse Inference Server

> Voice AI assistant for VKVstudio — custom FastAPI server without Ollama.

## Architecture

```
Browser (Astro/Svelte)                    Inference Server (this)
──────────────────────                    ──────────────────────────
 🎙️ MediaRecorder                         FastAPI (Python)
 │                                        │
 └─ POST /api/voice ──────────────────►   ├─ 1. faster-whisper (STT)
                                          │     audio → text
                                          ├─ 2. llama-cpp-python (LLM)
                                          │     text → Synapse response
                                          ├─ 3. TTS engine
 🔊 Audio playback   ◄────────────────   │     response → audio
                                          └─ Return audio/wav
```

## Models

| Component | Model | Size | Location |
|-----------|-------|------|----------|
| LLM | Synapse (Gemma 4 E2B, Q8_0, QLoRA fine-tuned) | 4.63 GB | `models/synapse-q8.gguf` |
| STT | faster-whisper `medium` | ~1.5 GB | `models/faster-whisper-medium/` |
| TTS (dev) | Edge TTS (dual-voice: SvetlanaNeural + AvaMultilingual) | 0 GB (API) | Cloud (Microsoft) |
| TTS (prod) | Google Cloud Neural2 | API (0 GB) | Cloud |

## Setup

### 1. Create virtual environment

```bash
cd inference/
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
# For CUDA GPU (recommended):
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
pip install -r requirements.txt

# For CPU only:
pip install -r requirements.txt
```

### 3. (Optional) Add voice reference for cloning

Place a 6-second WAV file of the desired voice at:
```
inference/voices/synapse-voice.wav
```
XTTS v2 will clone this voice for all responses. Without it, a default voice is used.

### 4. Run

```bash
# Development (with auto-reload):
python main.py

# Or via uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server starts at `http://localhost:8000`.
API docs at `http://localhost:8000/api/docs`.

## API Reference

### `GET /api/health`
Health check. Returns model load status.

```json
{
  "status": "ok",
  "llm_loaded": true,
  "stt_loaded": true,
  "tts_loaded": false,
  "tts_backend": "xtts"
}
```

### `POST /api/chat`
Text chat with Synapse.

**Request:**
```json
{
  "message": "Расскажи про стек VKVstudio",
  "history": [
    {"role": "user", "content": "Привет!"},
    {"role": "assistant", "content": "Соединение установлено! Чем могу помочь?"}
  ]
}
```

**Response:**
```json
{
  "response": "Стек VKVstudio — это Astro + Svelte 5 + vanilla CSS...",
  "language": "ru"
}
```

### `POST /api/voice`
Full voice pipeline: audio in → text → AI → audio out.

**Request:** `multipart/form-data` with `audio` file (WAV/WebM/OGG).

**Response:** `audio/wav` binary.

Custom response headers:
- `X-Synapse-Transcript` — transcribed user speech
- `X-Synapse-Response` — Synapse text response
- `X-Synapse-Language` — detected language ("ru" or "en")

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNAPSE_TTS_BACKEND` | `edge` | TTS engine: `edge` (multilingual) \| `silero` (offline) \| `xtts` (dev) \| `google_cloud` (prod) |
| `SYNAPSE_GPU_LAYERS` | `-1` | GPU layers for LLM (-1 = all, 0 = CPU only) |
| `SYNAPSE_WHISPER_MODEL` | `medium` | Whisper model size: tiny/base/small/medium/large-v3 |
| `SYNAPSE_PORT` | `8000` | Server port |
| `SYNAPSE_CORS_ORIGINS` | `http://localhost:4173,...` | Allowed CORS origins |
| `SYNAPSE_RATE_LIMIT` | `10` | Max requests per minute per IP |
| `SYNAPSE_CTX_SIZE` | `4096` | LLM context window size |
| `GOOGLE_APPLICATION_CREDENTIALS` | — | Path to GCP service account key (prod only) |

## Deployment (VPS)

### Minimum VPS requirements (CPU only):
- 4 vCPU / 16 GB RAM / 40 GB SSD
- ~$15-25/month (Hetzner CPX31)

### Recommended (with GPU):
- NVIDIA T4 (16 GB VRAM) — fits all three models
- ~$50-80/month (Vast.ai, RunPod)

### Production architecture:
```
Cloudflare Pages (free)     →  Static Astro site
  └─ DNS: api.vkvstudio.com →  Hetzner VPS
                                 ├─ Nginx (reverse proxy, SSL)
                                 └─ Synapse Inference Server
                                      ├─ Gemma Q8 (llama-cpp)
                                      ├─ faster-whisper
                                      └─ Google Cloud TTS (API)
```
