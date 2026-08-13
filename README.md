# VKVstudio — Premium Web Engineering Studio

> AI-powered web engineering studio by Valery. Built with **Astro 6 + Svelte 5 + GSAP + Gemma 4 E2B**.

## Quick Start

```bash
cd c:/projects/VKVstudio
pnpm run dev           # Frontend → http://localhost:4173
cd inference && .\.venv\Scripts\python.exe main.py  # Backend → http://localhost:8000
```

## Architecture

```
Frontend (Astro 6 + Svelte 5)     Inference Server (FastAPI)
─────────────────────────          ─────────────────────────
🌐 Static site (Cloudflare)        🧠 Gemma 4 E2B Q8_0 (llama-cpp-python)
🎨 Vanilla CSS + GSAP              🎙️ faster-whisper STT (CPU)
🔊 Audio Engine (Web API)          🔊 Edge TTS (multilingual)
💬 SynapseTerminal (SSE stream)    📡 SSE streaming endpoints
📝 IndexedDB chat history          🔒 Rate limiting + CORS
```

## Synapse AI Assistant

Neural terminal with voice I/O, SSE streaming, and chat history.

- **LLM**: Gemma 4 E2B, QLoRA fine-tuned, Q8_0 GGUF (4.63 GB)
- **STT**: faster-whisper (small, CPU, int8)
- **TTS**: Edge TTS (AvaMultilingualNeural — 100+ languages)
- **Frontend**: Svelte 5 runes, GSAP animations, IndexedDB persistence
- **Chat History**: Multiple conversations, rename, delete, auto-save

## System Docs

All project documentation lives in `.system/`:

| File | Purpose |
|------|---------|
| `checkpoint.md` | Current status & session recovery |
| `architecture_rules.md` | Tech stack, file structure, rules |
| `design_system.md` | Colors, typography, animations |
| `known_bugs.md` | Bugs to fix before deploy |
| `synapse_character.md` | AI character sheet |

## Session Recovery

```
Tell any model:
"Read c:/projects/VKVstudio/.system/checkpoint.md and continue from the current step."
```
