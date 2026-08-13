# Pre-deploy security review — inference backend (2026-07-08)

Five independent auditors reviewed the whole `inference/` backend along five dimensions
(network/exposure, CORS/CSRF/rate-limit/authz, input/injection/DoS, secret/PII leakage,
deploy hardening) ahead of exposing it publicly (Oracle/Hetzner ARM). Standing rule:
backend never trusts the frontend. This records findings + remediation status.

## BLOCKING — exploitable-today (fixed this pass)
| # | Finding | Fix |
|---|---|---|
| B1 | **CSRF on `/api/voice`, `/api/voice/stream`, `/api/session/close`, `/api/conversations/beacon`** — multipart & text/plain are CORS-safelisted → no preflight → any cross-origin page can drive the paid STT→LLM→TTS pipeline or plant a conversation into a visitor's IP-scoped store. | Added `_reject_cross_site()` (rejects `Sec-Fetch-Site: cross-site`) on all non-JSON state-changing routes; browsers can't forge that header. |
| B2 | **Chunked-encoding bypass of the body-size cap** — `ContentLengthLimitMiddleware` only checks the `Content-Length` header; a chunked POST with no header streams unbounded into RAM before Pydantic validates. | Middleware now rejects body-bearing methods that omit `Content-Length` (411). |
| B3 | **Unbounded decoded-audio duration** — 10 MB compressed can decode to hours → thousands of serial STT chunks tie up the shared CPU STT server. | Hard cap on decoded duration in `stt.py` preprocess (`SYNAPSE_STT_MAX_DURATION_S`, default 120s). |
| B4 | **Rate-limit: cheap reads share the expensive-LLM bucket** (global 10/min) — normal UI trips 429s AND a co-tenant can DoS the shared bucket with free reads. | Conversation CRUD moved to its own generous `conversations` scope; chat/voice/tts keep the tight global budget. |

## MEDIUM — real, partially addressed / needs frontend or ops follow-up
| # | Finding | Status |
|---|---|---|
| M1 | **IP-only authorization → shared-IP IDOR** (CGNAT/wifi/VPN co-tenants can read/delete each other's chat history; recycled IPs inherit access). | **RESOLVED end-to-end.** Backend prefers an opaque client-id capability (header, or body field for the header-less beacon) as the ownership key, IP fallback (`get_owner_key`). Frontend `src/lib/client-id.ts` generates+persists a random id and sends `X-Synapse-Client-Id` on all conversation calls (+ `clientId` in the beacon body). |
| M2 | **Indefinite plaintext chat store + raw IP, no retention.** | Added a startup + periodic purge of conversations older than `SYNAPSE_CHAT_RETENTION_DAYS` (default 30) and `chmod 600` on the DB file. |
| M3 | **`CF-Connecting-IP` / direct-exposure trust has no code backstop.** | `SERVER_HOST` now defaults to `127.0.0.1` (was `0.0.0.0`) so a forgotten firewall rule can't expose the app directly; set the env explicitly for a direct bind. Ops must still firewall the internal ports + validate the proxy path. |

## LOW / hardening (fixed this pass)
- **Internal llama-server children** (LLM/embed/STT) had no guard against a config-drift bind to a public interface → added a startup assertion that refuses non-loopback `*_SERVER_HOST`.
- **CORS `*`+credentials** only warned → now hard-exits on startup.
- **Secrets inherited by child processes** via `os.environ.copy()` → children now spawned with known secret env vars stripped.
- **Unsanitized log sites**: STT transcript in the streaming voice endpoint and the beacon `title` now go through `_oneline`; four lower-level exception logs (`llm.py`, `embeddings.py`, `stt.py`) now go through `safe_err()`.
- **Prod CORS default** ships localhost dev origins if the env is unset → startup now warns loudly when localhost origins are active.

## DEPLOY-PORT BLOCKER (partially addressed in code; still needs ARM hardware)
- **The backend was Windows+CUDA-only.** Code-side fixes applied (testable on Windows, safe):
  `llm.py` binary path is now **platform-aware** (`llama-server.exe` on Windows, `llama-server`
  ELF elsewhere — no code change needed on ARM, just drop the binary in `llama-bin/`); the unused
  in-process **`llama-cpp-python` dep removed** from `requirements.txt` (the app uses the
  `llama-server` subprocess, confirmed no in-process imports; that 525 MB win_amd64 wheel wouldn't
  resolve on ARM anyway). **STILL REQUIRED before ARM deploy (needs the target host):** build/obtain
  an ARM64-Linux `llama-server` binary + place it in `llama-bin/`; generate an ARM-native lockfile
  (the current `requirements.lock` pins Windows-absolute-path + `+cu124` wheels — regenerate on the
  target, or maintain `requirements.linux-arm64.lock`); verify end-to-end on real/emulated ARM.

## Verified CORRECT (do not regress)
SQL fully parameterized + cross-owner write guard (with tests); RAG is 100% server-sourced and format-string-safe; no SSRF; no command injection; no ReDoS; SSE UTF-8 pinned; rate-limit store memory-bounded; body-size layering; `X-Forwarded-For` walk validates IPs from the right; TTS/tokenizer secrets never echoed; error responses generic (no tracebacks); response headers `quote()`-encoded (no CRLF split); `/api/docs`+`openapi.json` off by default; `reload` off by default; `.env` gitignored; corpus deny-list clean (no age/health/location).
