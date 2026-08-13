# TTS backend — consolidated on Google (Chirp 3: HD) → Edge fallback

The voice-out stage of the Synapse pipeline. Consolidated around Google for GEAR
and to shed local-model cruft.

## Final stack

```
PRIMARY   Google Cloud TTS "Chirp 3: HD"   ($30 / 1M chars, free 1M chars/month)
   │      Google-native (GEAR). Needs a service-account JSON.
   ▼ (no creds / API error → automatic, per-request)
FALLBACK  Microsoft Edge TTS                (free, no key; dev default)
```

**✅ ACTIVATED + LIVE-VERIFIED 2026-07-06:** Chirp 3 HD confirmed on RU + EN via
`POST /api/tts` (200, `audio/mpeg`, valid MP3, ~21–24 KB; server log shows the real
Google client initialized, no Edge fallback). Activation is a single env —
`SYNAPSE_GOOGLE_TTS_CREDENTIALS` → the service-account JSON path (stored outside the
repo tree; only the PATH is ever read, never the key content).

Removed (2026-07-05): **Coqui XTTS v2**, **Silero**, and the old **Neural2** Google
engine — folded into `ChirpTTS`. `scipy` (XTTS-only) dropped from requirements.
(2026-07-06: **faster-whisper** also removed — STT moved to Gemma-4 native audio.)
`torch` is no longer a direct requirement; it remains only as a transitive dep and
supplies the CUDA runtime DLLs `llm.py` puts on PATH for the llama-server children.

## Selecting a backend

| env | values | default |
|---|---|---|
| `SYNAPSE_TTS_BACKEND` | `chirp3` \| `edge` | `chirp3` |
| `SYNAPSE_TTS_VOICE_RU` | Chirp3-HD voice | `ru-RU-Chirp3-HD-Aoede` |
| `SYNAPSE_TTS_VOICE_EN` | Chirp3-HD voice | `en-US-Chirp3-HD-Aoede` |
| `SYNAPSE_TTS_FORMAT` | `mp3` \| `wav` \| `ogg` | `mp3` |
| `SYNAPSE_TTS_SAMPLE_RATE` | Hz | `24000` |
| `SYNAPSE_GOOGLE_TTS_CREDENTIALS` | path to service-account JSON | — |

`chirp3` is the default, **but if no Google credentials are found the factory logs a
WARNING and returns Edge** — so voice works in dev with zero setup (same philosophy as
the frontend mock when the backend is down). Credentials = `SYNAPSE_GOOGLE_TTS_CREDENTIALS`
or `GOOGLE_APPLICATION_CREDENTIALS` pointing at an existing JSON file.

## Getting a Google key (for the owner)

1. Google Cloud Console → create/select a project (billing enabled — you have it).
2. Enable **Cloud Text-to-Speech API** for that project.
3. IAM & Admin → Service Accounts → create one → grant **Cloud Text-to-Speech User**.
4. That service account → Keys → **Add key → JSON** → download the file.
5. Put it somewhere private (NOT in the repo — `.env`/keys are gitignored) and set
   `SYNAPSE_GOOGLE_TTS_CREDENTIALS=/abs/path/to/key.json` before starting the server.
6. `pip install google-cloud-texttospeech` in `inference/.venv` (in requirements.txt).

Free tier is 1M chars/month for Chirp3-HD; beyond that $30/1M. At our traffic it's
effectively free.

## Security & cost controls

Defense in depth so the service-account key can't leak and nobody can run up your
Google bill through the public endpoints.

### Env variables

| env | what it protects | default |
|---|---|---|
| `SYNAPSE_GOOGLE_TTS_CREDENTIALS` | path to the service-account JSON (content never read/logged) | — |
| `SYNAPSE_TTS_MAX_CHARS` | per-request character cap; `/api/tts` returns **413** over it (no Google call); voice paths truncate | `2000` |
| `SYNAPSE_TTS_RATE_LIMIT` | stricter per-IP req/min on the paid TTS/voice endpoints (stacked on the global limit) | `5` |
| `SYNAPSE_TTS_MONTHLY_CHAR_CAP` | in-app monthly char budget; on reach → free Edge TTS, **$0** | `900000` |
| `SYNAPSE_TTS_BUDGET_FILE` | where the monthly counter persists (auto-resets each calendar month) | `inference/tts_char_budget.json` |

### The key never leaks

- Only the **path** is read (from env) and may be logged; the **content** (the key) is
  never read into a string, logged, or returned in any response/traceback.
- Google client errors are caught, **scrubbed** (path redacted), logged server-side, and
  the caller only ever sees a generic message. No `/config`, `/env`, or `/debug` endpoint
  exists.
- **Store the key OUTSIDE the repo tree** and set `SYNAPSE_GOOGLE_TTS_CREDENTIALS` to that
  external path. On startup the server logs a WARNING if the key sits inside the repo
  ("move outside & rotate"). **Rotate the key on any suspicion of exposure.**

### Verify the key is not in git

```bash
git ls-files | grep -iE "service-account|\.sa\.json|credential|secret|\.pem$|\.key$"
# → expect NO output. If a key shows up: remove it, ROTATE it in GCP, then scrub history.
```
`inference/.gitignore` blocks `*.sa.json`, `*service-account*.json`, `*credentials*.json`,
`gcp-*.json`, `.secrets/`, `*.pem`, `*.key`, and the runtime budget file. (A root-level
copy of these rules is recommended too, but the key should live outside the repo anyway.
→ main session / owner if a global rule is wanted.)

### Cost cap = practical $ limit

The monthly budget (`SYNAPSE_TTS_MONTHLY_CHAR_CAP`, default `900000`) is set **below**
Google's free 1M/month. Each successful Chirp synth adds the characters actually sent; a
call that *would* cross the cap is not sent to Google at all — it silently uses free Edge
TTS, with a WARNING (and an 80%-of-cap alert earlier). So the backend physically cannot
leave the free tier → **$0 billed**. Reset is automatic on the 1st of each month.

### Fail-safe

No creds / bad creds / quota / network error / over-budget — **all** degrade to Edge TTS
for that request. The request never fails because of the TTS stage.

## API contract (unchanged)

`/api/tts`, `/api/voice`, `/api/voice/stream` still return audio in the same format
and headers. `main.py` now derives Content-Type / `X-Synapse-Audio-Format` from the
engine's `output_format` property (both Chirp default and the Edge fallback are **MP3**),
instead of a class-name check — so the header always matches the bytes regardless of
which backend produced them.

## Robustness

- **Factory-level fallback**: `chirp3` without creds → Edge (warning logged).
- **Per-request fallback**: if a Chirp synth call raises (bad creds surfaced on first
  call, network, quota), `ChirpTTS.synthesize()` catches it and re-synthesizes via Edge
  for that request — a transient Google error can't drop a reply. Edge emits MP3, which
  matches the default `output_format`, so keep `SYNAPSE_TTS_FORMAT=mp3` for seamless
  fallback (Edge cannot emit wav/ogg).

## TTS is always a separate step

An LLM emits **text**, not audio — so voice-out is its own engine. It pairs with the
(future) Gemma-native STT on the *hear* side (see `STT_BACKEND.md`) but stays independent:
Gemma folds in STT, never TTS.

## Tests

`tests/test_tts.py` (stdlib unittest, Google client mocked — no network/package needed):
backend selection by env, language→voice mapping, encoding↔format, graceful Edge
fallback on failure, `output_format` contract, and that XTTS/Silero/Neural2 are gone.
