"""
Synapse Inference Server — FastAPI Application
================================================
Voice AI assistant: audio in → Whisper STT → Synapse LLM → TTS → audio out.
Also provides a text chat fallback endpoint.

Run (prod): uvicorn main:app --host 0.0.0.0 --port 8000
Run (dev):  SYNAPSE_RELOAD=1 SYNAPSE_ENABLE_DOCS=1 python main.py
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import re
import threading
import time
import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file (API keys etc.)
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Iterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

from config import (
    CHAT_RETENTION_DAYS,
    CORS_ORIGINS,
    EMBED_MAX_CHARS,
    EMBED_MAX_TEXTS,
    EMBED_MAX_TOTAL_CHARS,
    EMBED_NATIVE_DIMS,
    EMBED_SERVER_HOST,
    ENABLE_DOCS,
    GEMMA_STT_HOST,
    GOOGLE_OAUTH_CLIENT_ID,
    LLM_SERVER_HOST,
    MAX_AUDIO_SIZE_MB,
    MAX_BEACON_BODY_BYTES,
    MAX_INPUT_LENGTH,
    MAX_REQUEST_BYTES,
    MAX_SAVE_BODY_BYTES,
    RAG_CACHE_PATH,
    RAG_CORPUS_PATH,
    RAG_EMBED_TAG,
    RAG_ENABLED,
    RAG_MAX_CONTEXT_CHARS,
    RAG_MIN_SCORE,
    RAG_REL_MARGIN,
    RAG_TOP_K,
    RATE_LIMIT_CONVERSATIONS_RPM,
    GEN_QUOTA_MAX,
    GEN_QUOTA_WINDOW_SEC,
    GEN_REQUIRE_AUTH,
    TOKENIZE_ALLOWED_MODELS,
    TOKENIZE_BUDGET_FILE,
    TOKENIZE_DAILY_CALL_CAP,
    RATE_LIMIT_RPM,
    SERVER_HOST,
    SERVER_PORT,
    TRUSTED_PROXIES,
    TRUST_PROXY_HEADERS,
    TTS_MAX_CHARS,
    TTS_RATE_LIMIT_RPM,
    VOICE_LLM_VARIANT,
)
from llm import SynapseLLM
from router import route, VALID_VARIANTS

# Voice turns are pinned to one model (default E4B). Validated once at import.
_VOICE_VARIANT = VOICE_LLM_VARIANT if VOICE_LLM_VARIANT in VALID_VARIANTS else "e4b"
from chat_storage import ChatStorage, ConversationOwnershipError


# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-14s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("synapse.api")


# ── Global engine instances ──────────────────────────────────────────────────
synapse_llm = SynapseLLM()
# SYNAPSE_DB_PATH lets the database live outside the source tree. Production
# needs that: the service runs with its own code mounted read-only (systemd
# ProtectSystem=strict, ReadWritePaths pointing at a separate state dir), so
# an RCE cannot rewrite the code it is executing — and a database sitting
# next to main.py would have forced that protection open. Defaults to the
# in-tree path so local dev is unchanged.
chat_storage = ChatStorage(
    os.getenv("SYNAPSE_DB_PATH") or (Path(__file__).parent / "synapse_chats.db")
)
try:
    from stt import get_stt
    stt_engine = get_stt()  # GemmaSTT (Gemma-4 native audio; Whisper removed 2026-07-06)
except ImportError:
    stt_engine = None
    logger.warning("STT backend not installed — /api/voice disabled")

try:
    from tts import create_tts_engine
    tts_engine = create_tts_engine()
except ImportError:
    tts_engine = None
    logger.warning("TTS not installed — /api/voice disabled")

try:
    from embeddings import EmbeddingEngine
    embedding_engine = EmbeddingEngine()  # EmbeddingGemma, lazy-spawns on first /api/embed
except ImportError:
    embedding_engine = None
    logger.warning("Embeddings not available — /api/embed disabled")

# RAG grounding: reuses the embedding engine (CPU EmbeddingGemma) to retrieve from
# the studio corpus and inject it into chat turns. Off unless SYNAPSE_RAG_ENABLED.
retriever = None
if RAG_ENABLED and embedding_engine is not None:
    try:
        from rag.retriever import Retriever
        retriever = Retriever(
            embed_fn=lambda texts, task: embedding_engine.embed(texts, task),
            corpus_path=RAG_CORPUS_PATH,
            cache_path=RAG_CACHE_PATH,
            embed_tag=RAG_EMBED_TAG,
            top_k=RAG_TOP_K,
            min_score=RAG_MIN_SCORE,
            rel_margin=RAG_REL_MARGIN,
            max_context_chars=RAG_MAX_CONTEXT_CHARS,
        )
        logger.info("RAG grounding ENABLED (top_k=%d, min_score=%.2f)", RAG_TOP_K, RAG_MIN_SCORE)
    except Exception as e:
        retriever = None
        logger.warning("RAG grounding requested but failed to init: %s", e)
elif RAG_ENABLED:
    logger.warning("RAG grounding requested but embedding engine unavailable — disabled")


def _ground_message(message: str) -> tuple[str, list[dict]]:
    """If RAG is on, return (grounded_message, sources); else the raw message + []."""
    if retriever is None:
        return message, []
    try:
        wrapped, sources = retriever.ground(message)
    except Exception as e:  # never let grounding break a chat
        logger.warning("RAG grounding failed (%s) — sending raw message", _safe_err(e))
        return message, []
    if wrapped is None:
        return message, []
    logger.info("RAG grounded message with %d source(s): %s",
                len(sources), ", ".join(s["key"] for s in sources))
    return wrapped, sources

# ── API Clients for Exact Tokenization ───────────────────────────────────────
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic and anthropic_api_key else None
gemini_client = genai.Client(api_key=google_api_key) if genai and google_api_key else None


# Secret-safe exception rendering for logs (see log_safety.py). Client responses
# already use static generic strings; this hardens server-side log hygiene.
from tts_budget import TTSBudget
from log_safety import safe_err as _safe_err, oneline as _oneline


# ── Trusted client IP resolution ─────────────────────────────────────────────
# Behind the planned Nginx/Cloudflare reverse proxy, request.client.host is
# the PROXY's address for every request. Used unconditionally, that would
# (a) collapse rate limiting to a single global bucket, and (b) let every
# user list/read/delete every other user's saved conversations (they're
# scoped by client IP). We only trust proxy headers when the immediate TCP
# peer is a configured, trusted proxy — otherwise a client could simply send
# its own spoofed headers directly.
#
# IMPORTANT: X-Forwarded-For is APPENDED to by each proxy hop (Nginx's
# $proxy_add_x_forwarded_for, RFC 7239 intent, Cloudflare's own behavior),
# so the RIGHTMOST entries are the ones added by real, trusted infrastructure
# while the LEFTMOST entry is whatever the original client put there —
# entirely attacker-controlled. Reading the leftmost entry (as a naive
# `.split(",")[0]` does) lets any client set
# `X-Forwarded-For: <victim-ip>` and impersonate that IP for rate-limiting
# and conversation scoping. We therefore either use CF-Connecting-IP (which
# Cloudflare overwrites, never appends to, so it can't be spoofed through
# Cloudflare) or walk X-Forwarded-For from the right.
def _is_valid_ip(value: str) -> bool:
    """True if `value` is a syntactically valid IPv4/IPv6 address."""
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _content_length_over(request: Request, max_bytes: int) -> bool:
    """True if the request's declared Content-Length exceeds max_bytes.

    Lets us reject an oversized upload BEFORE reading/spooling the whole body.
    A missing/invalid Content-Length returns False (falls through to the post-read
    length check, which still bounds the actual bytes).
    """
    cl = request.headers.get("content-length")
    if cl is None:
        return False
    try:
        return int(cl) > max_bytes
    except ValueError:
        return False


def _require_json(request: Request) -> None:
    """Enforce Content-Type: application/json on state-changing JSON endpoints (415 else).

    CSRF defense for this cookieless API: a cross-site page can send a CORS "simple
    request" (text/plain / form-encoded) with NO preflight, so the CORS allowlist never
    blocks it. Requiring application/json forces any cross-origin caller through a
    preflight (application/json is not CORS-safelisted), which our origin allowlist then
    rejects. The real frontend already sends application/json on these routes. Does NOT
    apply to /api/voice* (multipart uploads) or the beacon (text/plain by necessity).
    """
    ctype = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    if ctype != "application/json":
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")


def _reject_cross_site(request: Request) -> None:
    """CSRF guard for state-changing routes that CANNOT use the application/json trick —
    /api/voice* (multipart is CORS-safelisted) and the beacon/session routes (text/plain
    / no body). A cross-origin page can fire those with NO preflight, so the CORS
    allowlist never sees them; without this guard any site a victim visits could drive
    the paid STT→LLM→TTS pipeline or plant a conversation into the victim's store.

    `Sec-Fetch-Site` is a browser-set, JS-unforgeable Fetch-Metadata header: a genuine
    cross-site CSRF request always carries `cross-site`, so rejecting it closes the hole.
    A missing header (older browser / non-browser client like curl) is allowed — those
    aren't CSRF vectors (no ambient victim session to ride). Same-origin/same-site/none
    (address-bar navigations) pass.
    """
    site = (request.headers.get("sec-fetch-site") or "").strip().lower()
    if site == "cross-site":
        raise HTTPException(status_code=403, detail="Cross-site request rejected")


def get_client_ip(request: Request) -> str:
    """Resolve the real client IP, honoring proxy headers only from trusted proxies.

    Every candidate taken from a proxy header is validated as a real IP address
    before use: a proxy-supplied value that isn't a valid IP is never returned, so
    it can't pollute the rate-limit / conversation-scoping key or forge log lines
    (the value flows into log statements and DB keys). This does NOT by itself stop
    a single-hop attacker from claiming a *valid* victim IP — the deployment must
    firewall the origin so only the trusted proxy can reach it (see config.py).
    """
    peer = request.client.host if request.client else "unknown"
    if TRUST_PROXY_HEADERS and peer in TRUSTED_PROXIES:
        # 1. Cloudflare always overwrites (never appends to) this header with
        #    the true client IP, stripping any client-supplied value first.
        cf_ip = (request.headers.get("cf-connecting-ip") or "").strip()
        if cf_ip and _is_valid_ip(cf_ip):
            return cf_ip

        # 2. Fall back to X-Forwarded-For, read from the right: drop trailing
        #    entries that are themselves trusted proxies, and take the first
        #    remaining (rightmost non-trusted) entry that is a VALID IP as the
        #    real client. A non-IP entry (garbage/injection) stops the walk —
        #    we never return an unvalidated string.
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            entries = [e.strip() for e in forwarded.split(",") if e.strip()]
            for entry in reversed(entries):
                if entry in TRUSTED_PROXIES:
                    continue
                if _is_valid_ip(entry):
                    return entry
                break  # first non-trusted entry isn't a valid IP → don't trust the header
    # 3. No trusted proxy headers available — use the immediate TCP peer.
    return peer


# Conversation OWNERSHIP is by a verified Google account, NEVER the IP (IP is shared
# via NAT/carriers, reassigned, spoofable — a cross-tenant leak). Anonymous users get
# NO server-side history at all (their history lives only in the browser's IndexedDB).
# The IP is still used, but ONLY for rate-limiting (see get_client_ip).
class _CachingCertsRequest:
    """A google-auth transport that remembers Google's signing certificates.

    The comment this replaces said the JWKS fetch was "cached by the lib". It is
    not. `google.oauth2.id_token.verify_oauth2_token` calls `_fetch_certs`, and
    `_fetch_certs` is three lines that do a plain GET every single time — read
    from the installed package, not assumed. So every authenticated request made
    a fresh HTTPS round-trip to Google before it could do anything else: every
    conversation sync, and now every signed-in generation call. On a four-core
    ARM box behind a tunnel that is real latency, and it puts Google's
    availability in front of the owner's own history.

    Verification fails closed, so a stale cache would lock people out rather
    than let anyone in. That is why the TTL comes from Google's own
    `Cache-Control: max-age` — the header exists precisely so clients refresh
    before a key rotates — and is clamped rather than trusted blindly.
    """

    _MIN_TTL = 300.0      # 5 min: never hammer the endpoint on a bad header
    _MAX_TTL = 6 * 3600.0  # 6 h: never outlive a rotation by much
    _DEFAULT_TTL = 3600.0  # 1 h: used when the header is missing or unparseable

    def __init__(self, inner) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, object]] = {}

    @classmethod
    def _ttl_from(cls, response) -> float:
        raw = ""
        try:
            raw = (response.headers or {}).get("Cache-Control", "") or ""
        except Exception:  # noqa: BLE001 — a header we cannot read is just a default TTL
            return cls._DEFAULT_TTL
        m = re.search(r"max-age\s*=\s*(\d+)", raw, re.I)
        if not m:
            return cls._DEFAULT_TTL
        try:
            return min(cls._MAX_TTL, max(cls._MIN_TTL, float(m.group(1))))
        except ValueError:
            return cls._DEFAULT_TTL

    def __call__(self, url, method="GET", **kwargs):
        # Only GETs are cacheable, and only successful ones are stored — an
        # error response must not be remembered as if it were the key set.
        if method != "GET":
            return self._inner(url, method=method, **kwargs)
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(url)
            if hit is not None and hit[0] > now:
                return hit[1]
        response = self._inner(url, method=method, **kwargs)
        if getattr(response, "status", None) == 200:
            with self._lock:
                self._cache[url] = (now + self._ttl_from(response), response)
        return response


try:
    from google.oauth2 import id_token as _google_id_token
    from google.auth.transport import requests as _google_requests
    # Reused HTTP session, wrapped so the certificate fetch is actually cached.
    _google_request = _CachingCertsRequest(_google_requests.Request())
    _GOOGLE_AUTH_AVAILABLE = True
except ImportError:  # google-auth not installed → auth disabled, app runs anonymous/local-only
    _GOOGLE_AUTH_AVAILABLE = False

_MAX_BEARER_LEN = 4096  # a GIS ID token is ~1 KB; cap defensively before any crypto work


def _verify_google_token(token: str) -> "str | None":
    """Verify a Google ID token and return the owner key "sub:<google-sub>", or None.
    Fail-CLOSED on every error (bad signature, wrong audience, expired, malformed, JWKS
    fetch failure). NEVER logs the token or the decoded claims.

    SECURITY: `audience=GOOGLE_OAUTH_CLIENT_ID` MUST be passed — without it, a Google ID
    token minted for ANY OAuth client (incl. one an attacker registers) would pass the
    signature/issuer/expiry checks and be accepted as a valid owner.
    """
    if not token or not GOOGLE_OAUTH_CLIENT_ID or not _GOOGLE_AUTH_AVAILABLE:
        return None
    if len(token) > _MAX_BEARER_LEN:
        return None
    try:
        claims = _google_id_token.verify_oauth2_token(
            token, _google_request, audience=GOOGLE_OAUTH_CLIENT_ID
        )
    except Exception:  # noqa: BLE001 — any verification failure ⇒ anonymous, and never leak the token
        return None
    sub = claims.get("sub")
    if not sub or not isinstance(sub, str):
        return None
    # Extract ONLY the stable `sub`; discard email/name/picture so no PII reaches storage or logs.
    return "sub:" + sub


def get_authenticated_owner(request: Request, explicit_token: "str | None" = None) -> "str | None":
    """Owner key from a verified Google ID token — from `Authorization: Bearer <jwt>`, or
    `explicit_token` for the header-less sendBeacon path (token carried in the body).
    Returns None for anonymous requests (which get NO server-side history — hard-gated by
    the callers). This is the ownership key ONLY; rate-limiting stays on the IP.
    """
    token = explicit_token
    if not token:
        auth = request.headers.get("authorization") or ""
        if auth[:7].lower() == "bearer " and len(auth) <= _MAX_BEARER_LEN + 8:
            token = auth[7:].strip()
    if not token:
        return None
    return _verify_google_token(token)


async def authenticated_owner(request: Request, explicit_token: "str | None" = None) -> "str | None":
    """Async wrapper for `get_authenticated_owner`. `verify_oauth2_token` does a SYNCHRONOUS,
    network-capable JWKS cert fetch on a cache miss — running it inline on the event loop would
    stall EVERY request (chat/voice/tts) for that HTTPS round-trip. Offload to a thread, matching
    the rest of this file's "never block the loop" discipline (LLM/STT/TTS all use to_thread)."""
    # Deliberately NOT asyncio.to_thread: that uses the process-wide default
    # executor, which a burst of concurrent inference requests can pin for
    # minutes. Signing in and listing your own conversations must not depend on
    # whether someone else is currently flooding /api/chat.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _AUTH_EXECUTOR, get_authenticated_owner, request, explicit_token
    )


# ── Inference admission control ──────────────────────────────────────────────
# Rate limiting bounds requests-per-minute-per-IP; it does NOT bound how many
# requests are in flight at once. That gap is exploitable here.
#
# The heavy path is already serialised downstream — SynapseLLM._lifecycle_lock
# means only one generation runs at a time. Extra callers do not go faster; they
# sit blocked. The problem is WHERE they sit: every `asyncio.to_thread` call in
# this process shares the default ThreadPoolExecutor, sized min(32, cpu+4) —
# about 8 threads on the 4-core ARM production box. authenticated_owner() uses
# that same pool. So ~8 concurrent /api/chat requests from ONE IP, comfortably
# inside the 10/min limit, pin every worker for up to LLM_SERVER_REQUEST_TIMEOUT
# (180s) and stall the entire API, including authenticated conversation routes
# belonging to other people.
#
# Two independent fixes, because either alone leaves a hole:
#   1. Bound the heavy path explicitly and SHED load with 503 rather than
#      queueing. Queueing is what converts "slow" into "everything is down".
#   2. Give authentication its own small executor so a chat flood can never
#      starve it, no matter what happens to the default pool.
_INFERENCE_SLOTS = max(1, int(os.getenv("SYNAPSE_MAX_CONCURRENT_INFERENCE", "2")))
_inference_semaphore: asyncio.Semaphore | None = None
_AUTH_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="synapse-auth")


def _get_inference_semaphore() -> asyncio.Semaphore:
    """Created lazily so it binds to the running loop, not import order."""
    global _inference_semaphore
    if _inference_semaphore is None:
        _inference_semaphore = asyncio.Semaphore(_INFERENCE_SLOTS)
    return _inference_semaphore


@asynccontextmanager
async def _admit_inference(what: str):
    """Hold one inference slot for the whole request, or refuse it outright.

    The tiny acquire timeout is deliberate: if no slot frees almost immediately
    the server is already saturated, and making the caller wait 180s is strictly
    worse for them AND for everyone else than an honest, instant 503.
    """
    sem = _get_inference_semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=0.25)
    except asyncio.TimeoutError:
        logger.warning("Shedding %s — all %d inference slots busy", what, _INFERENCE_SLOTS)
        raise HTTPException(
            status_code=503,
            detail="The model is busy right now. Please try again in a moment.",
            headers={"Retry-After": "5"},
        ) from None
    try:
        yield
    finally:
        sem.release()


# ── Rate limiter (simple in-memory, dev-grade) ───────────────────────────────
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_WINDOW_SEC = 60.0
_RATE_LIMIT_SWEEP_INTERVAL_SEC = 300.0  # 5 min
# Hard bound on distinct keys so a flood of rotating source IPs (trivial with IPv6)
# can't grow the in-memory store without limit between sweeps (memory DoS).
_MAX_RATE_LIMIT_KEYS = int(os.getenv("SYNAPSE_MAX_RL_KEYS", "20000"))


def check_rate_limit(client_ip: str, scope: str = "global", limit: int | None = None) -> None:
    """Raise 429 if client exceeds `limit` requests per minute within `scope`.

    Scopes get independent counters (keyed by scope+IP), so a stricter per-endpoint
    limit (e.g. the paid TTS/voice endpoints) stacks on top of the global one without
    interfering. `limit` defaults to the global RATE_LIMIT_RPM.
    """
    if limit is None:
        limit = RATE_LIMIT_RPM
    now = time.time()
    key = f"{scope}\x00{client_ip}"

    # Clean old entries
    _rate_limit_store[key] = [
        ts for ts in _rate_limit_store[key]
        if now - ts < _RATE_LIMIT_WINDOW_SEC
    ]

    if len(_rate_limit_store[key]) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {limit} requests per minute.",
        )

    _rate_limit_store[key].append(now)

    # Bound total distinct keys (memory DoS via rotating IPs). Evict oldest-inserted
    # keys (dicts preserve insertion order) until back under the cap; never evict the
    # key we just touched.
    #
    # GENERATION-QUOTA KEYS ARE NEVER EVICTED HERE. They live in the same dict, and
    # evicting one does not free memory so much as hand back an allowance: flooding
    # this endpoint from rotating IPs — trivial over IPv6, which is the whole reason
    # the cap exists — would push the quota entries out and reset the five-messages-
    # per-five-hours limit for everybody, including the flooder. A cheap request
    # would then buy back the expensive one, which is the wrong way round.
    #
    # Leaving them is safe: one short list per subject, removed by the sweep as soon
    # as its window expires (that sweep is already prefix-aware). Even a very large
    # number of distinct callers within one window costs a few tens of MB on a box
    # with 23 GB, and unlike the burst counters these entries are the thing being
    # protected rather than a cache.
    if len(_rate_limit_store) > _MAX_RATE_LIMIT_KEYS:
        for old_key in list(_rate_limit_store.keys()):
            if len(_rate_limit_store) <= _MAX_RATE_LIMIT_KEYS:
                break
            if old_key != key and not old_key.startswith(GEN_QUOTA_KEY_PREFIX):
                del _rate_limit_store[old_key]


GEN_QUOTA_KEY_PREFIX = "genquota\x00"


async def require_generation_access(request: Request) -> str:
    """Gate every endpoint that makes the model produce tokens. Returns the owner.

    Generation is the only thing here that costs money per use, so it is the only
    thing that requires an account. Anonymous visitors are not shut out of the
    site: the frontend answers them from its mock and labels it as a mock, which
    is honest and costs nothing. What they do not get is an unmetered claim on a
    GPU that somebody pays for.

    Two guards, because they stop different failures. Sign-in makes every billable
    token attributable to an account instead of to a shared IP. The quota then
    bounds what one account can spend: RATE_LIMIT_RPM is a burst guard measured
    per MINUTE and would happily allow ten messages a minute for six hours, which
    is precisely the shape that drains a budget without ever tripping it.

    The 429 carries Retry-After and says in words when the allowance returns —
    "Rate limit exceeded" tells a visitor nothing about what to do next.

    THE QUOTA APPLIES TO ANONYMOUS CALLERS TOO, keyed by IP. It did not, and the
    gap made the whole control decorative: with GEN_REQUIRE_AUTH off — which is
    the state whenever generation is not on a metered GPU — an anonymous caller
    returned here BEFORE the quota ran, so "five messages per five hours" bound
    nobody. Signing in was the only way to become rate-limited, which is exactly
    backwards. An IP is a weaker identity than an account and a shared NAT shares
    the allowance; that is the correct trade for a demo, and far better than the
    unlimited claim on the box it replaces.
    """
    owner = await authenticated_owner(request)

    if owner is None:
        if GEN_REQUIRE_AUTH:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Sign in to talk to the real model. The assistant runs on a metered "
                    "GPU, so live answers are for signed-in visitors; without an account "
                    "you get the scripted demo instead."
                ),
            )
        # Anonymous but allowed: metered by IP rather than waved through.
        quota_subject = f"ip:{get_client_ip(request)}"
        owner = ""
    else:
        quota_subject = owner

    if GEN_QUOTA_MAX <= 0:  # 0 disables the quota
        return owner

    now = time.time()
    key = f"{GEN_QUOTA_KEY_PREFIX}{quota_subject}"
    hits = [ts for ts in _rate_limit_store[key] if now - ts < GEN_QUOTA_WINDOW_SEC]
    _rate_limit_store[key] = hits

    if len(hits) >= GEN_QUOTA_MAX:
        # The oldest hit still inside the window is the one whose expiry frees a slot.
        retry_after = max(1, int(GEN_QUOTA_WINDOW_SEC - (now - min(hits))))
        hours, minutes = divmod(retry_after // 60, 60)
        when = f"{hours}h {minutes}m" if hours else f"{minutes}m"
        raise HTTPException(
            status_code=429,
            detail=(
                f"You've used all {GEN_QUOTA_MAX} messages for now. The assistant runs "
                f"on a GPU paid for out of pocket, so it is rationed. Your next message "
                f"unlocks in about {when}."
            ),
            headers={"Retry-After": str(retry_after)},
        )

    _rate_limit_store[key].append(now)
    return owner


def _sweep_rate_limit_store() -> None:
    """Drop IP keys with no timestamps left inside the window.

    check_rate_limit() only prunes the list for an IP when THAT IP makes
    another request — an IP that stops making requests permanently leaves
    its key in the dict forever, growing unbounded. This periodic sweep
    (called from the event loop, so no lock needed) evicts idle keys.
    """
    now = time.time()
    # Two window lengths share this store. Sweeping generation-quota keys against
    # the 60-second burst window would evict a five-hour allowance minutes after
    # it was spent — handing everyone an unlimited GPU budget, silently.
    stale_ips = [
        ip for ip, timestamps in _rate_limit_store.items()
        if not any(
            now - ts < (
                GEN_QUOTA_WINDOW_SEC if ip.startswith(GEN_QUOTA_KEY_PREFIX)
                else _RATE_LIMIT_WINDOW_SEC
            )
            for ts in timestamps
        )
    ]
    for ip in stale_ips:
        del _rate_limit_store[ip]
    if stale_ips:
        logger.debug("Rate limit store sweep: evicted %d idle IP(s)", len(stale_ips))


async def _rate_limit_sweep_loop() -> None:
    """Background task: periodically evict idle IPs from the rate-limit store."""
    while True:
        await asyncio.sleep(_RATE_LIMIT_SWEEP_INTERVAL_SEC)
        try:
            _sweep_rate_limit_store()
        except Exception:
            logger.exception("Rate limit sweep failed")


_RETENTION_SWEEP_INTERVAL_SEC = 86_400.0  # 24h


async def _retention_sweep_loop() -> None:
    """Background task: enforce the chat retention window on a running server.

    The startup purge alone was not enough, and the gap was user-visible: the
    privacy policy promises conversations are "automatically purged 30 days
    after they were last updated", but a purge that only runs at boot means a
    process up for months never deletes anything. That made the policy a
    promise the code did not keep — a trust problem before it is a technical
    one. Runs off the event loop so a large DELETE can't stall requests.
    """
    while True:
        await asyncio.sleep(_RETENTION_SWEEP_INTERVAL_SEC)
        try:
            await asyncio.to_thread(chat_storage.purge_old, CHAT_RETENTION_DAYS)
            logger.info("Retention sweep: purged conversations older than %d days", CHAT_RETENTION_DAYS)
        except Exception:
            logger.exception("Retention sweep failed")


# ── Startup security checks ──────────────────────────────────────────────────

def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.strip().lower() in ("localhost", "")


def _security_startup_checks() -> None:
    """Fail-closed guards run at startup, before serving a single request."""
    # 1. A wildcard CORS origin + allow_credentials=True makes the API credentialed-
    #    readable from ANY site. This is never intentional in prod — hard-stop.
    if "*" in CORS_ORIGINS:
        raise SystemExit(
            "FATAL: '*' in SYNAPSE_CORS_ORIGINS with credentials enabled — any origin "
            "could read authenticated responses. Set an explicit origin allowlist."
        )
    # 2. The internal llama-server children (LLM router / EmbeddingGemma / GemmaSTT)
    #    are unauthenticated, unrate-limited model endpoints. They MUST stay on
    #    loopback — a config-drift bind to 0.0.0.0 would expose a free GPU/CPU
    #    inference endpoint to the internet. Refuse to start if any is non-loopback.
    for name, host in (("SYNAPSE_LLM_SERVER_HOST", LLM_SERVER_HOST),
                       ("SYNAPSE_EMBED_HOST", EMBED_SERVER_HOST),
                       ("SYNAPSE_GEMMA_STT_HOST", GEMMA_STT_HOST)):
        if not _is_loopback(host):
            raise SystemExit(
                f"FATAL: {name}={host!r} is not loopback. The internal model servers "
                "must bind 127.0.0.1 only (they have no auth/rate-limit). Unset it or "
                "use 127.0.0.1, and firewall those ports from the public interface."
            )
    # 3. Warn (don't fail) if dev localhost origins are live — an easy thing to forget
    #    to override before a public deploy.
    localhost_origins = [o for o in CORS_ORIGINS if "localhost" in o or "127.0.0.1" in o]
    if localhost_origins:
        logger.warning(
            "CORS allowlist includes localhost dev origins %s — override "
            "SYNAPSE_CORS_ORIGINS with prod-only origins before going public.",
            localhost_origins,
        )


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup/shutdown lifecycle."""
    logger.info("═══════════════════════════════════════════════")
    logger.info("  Synapse Inference Server starting...")
    logger.info("  Endpoints: /api/voice, /api/chat, /api/health, /api/conversations, /api/tokenize/count")
    logger.info("═══════════════════════════════════════════════")
    chat_storage.init()
    _security_startup_checks()
    with suppress(Exception):  # retention hygiene must never block startup
        chat_storage.purge_old(CHAT_RETENTION_DAYS)
    # Security: warn (don't fail) if the Google TTS key is kept inside the repo tree.
    try:
        from tts import check_credentials_location
        _cred_warning = check_credentials_location()
        if _cred_warning:
            logger.warning(_cred_warning)
    except Exception:  # noqa: BLE001 — never let a security check block startup
        logger.debug("TTS credentials-location check skipped", exc_info=True)
    sweep_task = asyncio.create_task(_rate_limit_sweep_loop())
    retention_task = asyncio.create_task(_retention_sweep_loop())
    yield
    # Cleanup on shutdown
    for _task in (sweep_task, retention_task):
        _task.cancel()
        with suppress(asyncio.CancelledError):
            await _task
    synapse_llm.unload()
    if stt_engine: stt_engine.unload()
    if tts_engine: tts_engine.unload()
    if embedding_engine: embedding_engine.unload()
    logger.info("Synapse Inference Server stopped.")


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Synapse Voice AI",
    description="Voice AI assistant for VKVstudio — Gemma 4 (E2B/E4B router) + Gemma audio STT + Google Chirp 3 TTS",
    version="1.0.0",
    lifespan=lifespan,
    # Docs + OpenAPI schema expose the full route/model map. Disabled by default
    # (secure); enable in dev with SYNAPSE_ENABLE_DOCS=1. openapi_url must be gated
    # too — otherwise /openapi.json leaks the schema even with Swagger UI off.
    docs_url="/api/docs" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
    redoc_url=None,
)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Answer a malformed request with a STATIC 422 — never the caller's own body.

    FastAPI's default validation handler serializes Pydantic's `input` field,
    which is the raw request body, straight back into the response. Two problems,
    both demonstrated: it reflects whatever was sent (a canary posted to
    /api/chat and /api/embed came back verbatim — a 1:1 echo bounded only by the
    12 MB request cap), and validation runs as a dependency BEFORE the handler
    body where `check_rate_limit` lives, so those echoes are unmetered. A static
    body removes the reflection and the amplification in one line each.

    Safe for the frontend: the client throws on any non-2xx via `httpError`
    (synapse-client.ts) BEFORE it ever reads a body, so it sees the 422 status
    and never the `detail`. Nothing in src/ parses a 422 payload.
    """
    return JSONResponse(status_code=422, content={"detail": "Invalid request"})


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[
        "X-Synapse-Transcript",
        "X-Synapse-Response",
        "X-Synapse-Language",
        "X-Synapse-Audio-Format",
        "X-Synapse-Model",
        # Retry-After is set on every 429 this server returns, and the browser
        # hides any response header not named here from cross-origin JS. Without
        # it the client could see the 429 but not the "how long" — so a visitor
        # who had spent the 5-messages-per-5-hours allowance was told the server
        # was busy and should be retried in a moment, which was both false and
        # an invitation to hammer the box. The number exists; let the page read it.
        "Retry-After",
    ],
)


class ContentLengthLimitMiddleware:
    """Pure-ASGI outer guard: reject a request whose declared Content-Length exceeds
    MAX_REQUEST_BYTES with a 413, BEFORE any endpoint buffers/parses the body.

    Pure ASGI (not BaseHTTPMiddleware) so it never buffers the body and can't break
    SSE/streaming responses — it only inspects request headers and either short-circuits
    or passes the request through untouched.

    Two guards:
      - declared Content-Length > MAX_REQUEST_BYTES -> 413.
      - a body-bearing method (POST/PUT/PATCH) with NO Content-Length -> 411. Without
        this, a chunked/streamed request bypasses the size cap entirely and FastAPI
        buffers the whole (unbounded) body into RAM before any Pydantic validator runs
        — a trivial unauthenticated memory-exhaustion DoS. The real frontend always
        sends Content-Length (fetch with a body, sendBeacon), so 411 costs it nothing.
    """

    _BODY_METHODS = {b"POST", b"PUT", b"PATCH"}

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") == "http":
            method = scope.get("method", "").encode() if isinstance(scope.get("method"), str) else scope.get("method", b"")
            has_length = False
            for name, value in scope.get("headers", []):
                if name == b"content-length":
                    has_length = True
                    try:
                        if int(value) > self.max_bytes:
                            await self._reject(send, 413, b"Payload too large")
                            return
                    except ValueError:
                        pass
                    break
            if method in self._BODY_METHODS and not has_length:
                await self._reject(send, 411, b"Length Required")
                return
        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(send, status: int, detail: bytes) -> None:
        body = b'{"detail":"' + detail + b'"}'
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


class SecurityHeadersMiddleware:
    """Pure-ASGI: add baseline security headers to every response.

    X-Content-Type-Options: nosniff matters for the raw audio bytes from
    /api/voice & /api/tts (stops content-sniffing edge cases). Referrer-Policy is
    cheap defense in depth. Header injection only — never touches the body, so it's
    safe for SSE/streaming responses.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {k.lower() for k, _ in headers}
                if b"x-content-type-options" not in existing:
                    headers.append((b"x-content-type-options", b"nosniff"))
                if b"referrer-policy" not in existing:
                    headers.append((b"referrer-policy", b"no-referrer"))
                # This API answers on its own subdomain (api.vkvstudio.com).
                # Nothing here is a page: it is JSON and audio meant for the
                # site's own JavaScript. Without this a crawler that finds the
                # subdomain could index /api/health and friends, and the API
                # would start showing up in search results next to the actual
                # site. Sent on EVERY response, including 404s, so there is no
                # crawlable surface at all.
                if b"x-robots-tag" not in existing:
                    headers.append((b"x-robots-tag", b"noindex, nofollow, noarchive"))
                # HSTS, but only on a request that actually arrived over TLS.
                # RFC 6797 requires a UA to IGNORE this header on a plain-HTTP
                # response, so sending it unconditionally is noise; sending it
                # never is worse. The site's own _headers file already pins the
                # apex with includeSubDomains, which covers this host — but only
                # for a client that visited vkvstudio.com first in the same
                # browser. Anything that reaches api.vkvstudio.com directly had
                # no pin at all, and this host answers plain HTTP with real
                # JSON. This makes the API assert its own policy.
                #
                # Behind the Cloudflare tunnel the origin sees the hop as http,
                # so the real scheme is X-Forwarded-Proto. Falls back to the ASGI
                # scheme when the header is absent (direct origin access, tests).
                # NOT a redirect: that belongs to "Always Use HTTPS" on the
                # Cloudflare zone, where it can be enforced before the request
                # ever leaves the edge.
                fwd = b""
                for k, v in scope.get("headers", []):
                    if k.lower() == b"x-forwarded-proto":
                        fwd = v.split(b",")[0].strip().lower()
                        break
                scheme = fwd.decode("ascii", "replace") if fwd else scope.get("scheme", "")
                if scheme == "https" and b"strict-transport-security" not in existing:
                    headers.append(
                        (b"strict-transport-security", b"max-age=31536000; includeSubDomains")
                    )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)


# Added AFTER CORS so it wraps outermost (runs first): an oversized body is rejected
# before it can be buffered by any inner handler.
app.add_middleware(ContentLengthLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)
app.add_middleware(SecurityHeadersMiddleware)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse({"detail": "Invalid request"}, status_code=422)


# ── Request/Response Models ──────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant)$")
    content: str = Field(..., max_length=MAX_INPUT_LENGTH)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    # Optional client override of the router's model pick ("e2b" | "e4b"). When
    # None (the normal case) the backend router chooses from the message itself.
    # An invalid value is ignored and the router decides.
    model: str | None = Field(default=None, max_length=8)


class ChatResponse(BaseModel):
    response: str
    language: str
    # Which fine-tune actually answered ("e2b" | "e4b") — drives the per-message
    # model badge in the UI. Mirrored in the X-Synapse-Model response header.
    model: str


class HealthResponse(BaseModel):
    status: str
    llm_loaded: bool
    stt_loaded: bool
    tts_loaded: bool
    tts_backend: str


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=EMBED_MAX_TEXTS)
    # EmbeddingGemma is prompt-conditioned: 'query' for a search query, 'document'
    # for corpus text. Wrong prefix measurably hurts retrieval.
    task: str = Field(default="document", pattern=r"^(query|document)$")
    # Registry id → output width (Matryoshka). Unknown → native 768.
    model: str = Field(default="embeddinggemma-300m", max_length=64)


# Registry model id → embedding width. The "-256" variant is a Matryoshka
# truncation of the native 768, served by the same model.
_EMBED_MODEL_DIMS = {"embeddinggemma-300m": EMBED_NATIVE_DIMS, "embeddinggemma-300m-256": 256}


class TokenizeRequest(BaseModel):
    model: str = Field(..., max_length=64)
    text: str = Field(..., max_length=MAX_INPUT_LENGTH)


# ── Endpoints ────────────────────────────────────────────────────────────────

# HEAD as well as GET, and the distinction is FastAPI's, not Starlette's:
# a plain Starlette Route adds HEAD to any GET route automatically, but
# FastAPI's APIRoute does not. So `HEAD /api/health` answered 405 — which is
# how most uptime monitors probe a liveness endpoint, and would have read as
# the service being down the day one was pointed at it. Found in the service
# log, where a HEAD from the edge sat next to the GETs returning 200.
@app.api_route("/api/health", methods=["GET", "HEAD"], response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe — cheap check that the process is up and serving.

    Always returns 200 (attribute reads only, no model work) so an
    orchestrator restarts the pod only if the process is truly hung/dead.
    Use /api/ready to decide whether to ROUTE TRAFFIC to this instance.
    """
    return HealthResponse(
        status="ok",
        llm_loaded=synapse_llm.is_loaded,
        stt_loaded=stt_engine.is_loaded if stt_engine else False,
        tts_loaded=tts_engine.is_loaded if tts_engine else False,
        tts_backend=_actual_tts_backend(),
    )


def _actual_tts_backend() -> str:
    """Report the TTS engine ACTUALLY in use, not the configured value.

    create_tts_engine() silently falls back to Edge when Google creds are missing,
    so the raw config value can lie (say "chirp3" while Edge is really serving). A
    health probe must report the truth.
    """
    name = type(tts_engine).__name__ if tts_engine else "none"
    return {"ChirpTTS": "chirp3", "EdgeTTSEngine": "edge"}.get(name, name)


@app.get("/api/ready")
async def readiness_check() -> JSONResponse:
    """Readiness probe — honestly reports whether the core LLM is loaded
    and able to serve /api/chat and /api/voice.

    Returns 503 while the model is cold (it lazy-loads on first request),
    so an orchestrator/load-balancer won't route traffic to a pod that
    would have to pay the cold-load cost on the user's first request.
    STT/TTS are also lazy-loaded and reported, but only the LLM gates
    readiness since text chat is the primary path.
    """
    llm_ready = synapse_llm.is_loaded
    body = {
        "ready": llm_ready,
        "llm_loaded": llm_ready,
        "stt_loaded": stt_engine.is_loaded if stt_engine else False,
        "tts_loaded": tts_engine.is_loaded if tts_engine else False,
    }
    return JSONResponse(body, status_code=200 if llm_ready else 503)


def map_model_name(model_id: str) -> str:
    mapping = {
        # Claude — NO substitution for the 4.7+ generation. Anthropic's docs are
        # explicit that "Claude 4.7 and later models use a newer tokenizer" and
        # that the same text yields ~30% MORE tokens than on earlier models.
        # Fable 5 and Opus 4.8 used to fall back to claude-sonnet-4-6-20250514,
        # which is PRE-4.7 — so every count came back roughly a third low while
        # the UI badged it as a verified exact count. That is worse than having
        # no number at all. Let these ids go through untouched: either Anthropic
        # answers with a real count from the right tokenizer, or the call fails
        # and the frontend degrades to its honest 'approx' estimate.
        "claude-sonnet-4.6": "claude-sonnet-4-6-20250514",
        # Gemini — the fictional "gemini-3.5-pro" id is long gone; Google never
        # shipped a 3.5 Pro and the frontend keys the row as gemini-3.1-pro.
        #
        # RESOLVED 2026-08-14, and the answer was the suffix. The note here used
        # to say it was UNVERIFIED whether Google's API wants the GA name or a
        # -preview one, and that an error was the acceptable outcome. Asking the
        # API itself settles it: ListModels on generativelanguage returns
        # `models/gemini-3.1-pro-preview` and no bare `gemini-3.1-pro` at all,
        # which is exactly why every Pro count came back 400 while both Flash
        # models answered. Verified against the live catalogue, not a docs page.
        #
        # Note the asymmetry: this is a TRANSPORT-level rename, mapping our id
        # onto the name Google's endpoint answers to. It is not the "closest
        # available" substitution that was removed from this table earlier —
        # that one silently counted a DIFFERENT model's tokens and reported the
        # result as exact. Renaming a model to itself is safe; swapping one
        # model for another is the lie worth refusing.
        "gemini-3.1-pro": "gemini-3.1-pro-preview",
        "gemini-3.5-flash": "gemini-3.5-flash",
        "gemma-4-e2b": "gemini-3.5-flash",  # same vocab family
    }
    return mapping.get(model_id, model_id)


TOKENIZE_PROVIDER_TIMEOUT = 15.0  # seconds — hard cap so a hung provider can't exhaust the thread pool

# Global daily cap on the relay to the owner's provider keys. Lazily built, and
# shared process-wide, exactly like the Chirp budget in tts.py — see
# config.TOKENIZE_DAILY_CALL_CAP for why this is global rather than per-IP.
_tokenize_budget: "TTSBudget | None" = None
_tokenize_budget_lock = threading.Lock()


def get_tokenize_budget() -> "TTSBudget":
    global _tokenize_budget
    if _tokenize_budget is None:
        with _tokenize_budget_lock:
            if _tokenize_budget is None:
                _tokenize_budget = TTSBudget(
                    TOKENIZE_BUDGET_FILE,
                    TOKENIZE_DAILY_CALL_CAP,
                    label="tokenize daily call",
                    period_fmt="%Y-%m-%d",
                )
    return _tokenize_budget


@app.post("/api/tokenize/count")
def count_tokens(request: Request, body: TokenizeRequest):
    """
    Проксирует подсчёт токенов к API провайдеров.
    Claude: POST /v1/messages/count_tokens (БЕСПЛАТНО)
    Gemini: POST /v1beta/models/{model}:countTokens
    """
    client_ip = get_client_ip(request)
    check_rate_limit(client_ip)
    _require_json(request)

    # ── Everything that can fail WITHOUT touching the network happens first ──
    # The order is the security property. An unknown model id must cost nothing
    # (or an attacker drains the day's allowance with junk and switches the
    # feature off for everyone), and it must reach nobody (or an
    # attacker-chosen string egresses on the owner's key). Validating before
    # the reservation is what makes both true at once, and it is why nothing
    # below this point is ever refunded.
    if body.model not in TOKENIZE_ALLOWED_MODELS:
        raise HTTPException(400, detail="Unknown model")
    is_claude = body.model.startswith("claude")
    if is_claude and not anthropic_client:
        raise HTTPException(503, detail="Anthropic client not configured")
    if not is_claude and not gemini_client:
        raise HTTPException(503, detail="Google GenAI client not configured")

    # Only now, with a request we are willing to pay for, spend from the cap.
    # Per-IP limiting above bounds one caller's rate; this bounds everyone's
    # total, which is the only thing IP rotation cannot walk around. NOT
    # refunded on failure: past this line the request has been handed to the
    # provider, and "the provider said no" is not evidence that nothing left
    # the box — a timeout and a 429 both mean it demonstrably did.
    budget = get_tokenize_budget()
    if not budget.try_reserve(1):
        logger.warning("Tokenize relay: daily cap of %d reached", budget.cap)
        raise HTTPException(
            status_code=429,
            detail=(
                "Exact token counts are unavailable for the rest of the day. "
                "The tokenizer falls back to its local estimate."
            ),
            headers={"Retry-After": "3600"},
        )

    try:
        if is_claude:
            result = anthropic_client.with_options(timeout=TOKENIZE_PROVIDER_TIMEOUT).messages.count_tokens(
                model=map_model_name(body.model),
                messages=[{"role": "user", "content": body.text}]
            )
            return {"totalTokens": result.input_tokens, "provider": "anthropic", "exact": True}

        else:
            result = gemini_client.models.count_tokens(
                model=map_model_name(body.model),
                contents=body.text,
                config=genai_types.CountTokensConfig(
                    http_options=genai_types.HttpOptions(timeout=int(TOKENIZE_PROVIDER_TIMEOUT * 1000))
                ) if genai_types else None,
            )
            return {"totalTokens": result.total_tokens, "provider": "google", "exact": True}

    # No refund on any path. The only failures reachable from here came back
    # FROM the provider, which means the request went TO the provider — the
    # thing the cap exists to count.
    except HTTPException:
        raise
    except Exception as e:
        error_str = str(e).lower()
        logger.warning("Tokenize API error for model=%r: %s", body.model, _safe_err(e))
        if "429" in error_str or "rate limit" in error_str:
            raise HTTPException(429, detail="Rate limit exceeded") from None
        elif "400" in error_str or "not found" in error_str:
            raise HTTPException(400, detail="Token counting failed") from None
        else:
            raise HTTPException(503, detail="Service Unavailable") from None


@app.post("/api/session/close")
async def session_close(request: Request):
    """
    Clean up on user closing Synapse terminal.
    Unloads STT and TTS (CPU models, safe to reload).
    LLM stays loaded — CUDA context cannot be safely re-created in-process.
    """
    _reject_cross_site(request)  # CSRF: no JSON content-type to gate this route
    # Rate-limited: previously unauthenticated + unlimited, so it could be spammed to
    # force STT/TTS unload→reload thrash. The unload race (concurrent unload nulling a
    # model mid-request) is separately defused by local-reference binding in the STT/TTS
    # engines, but we still throttle the trigger.
    client_ip = get_client_ip(request)
    check_rate_limit(client_ip)

    unloaded = []
    # NOTE: LLM stays loaded — del Llama + re-init crashes CUDA context
    # (known llama-cpp-python issue with GPU layers)
    if stt_engine and stt_engine.is_loaded:
        stt_engine.unload()
        unloaded.append("STT")
    if tts_engine and tts_engine.is_loaded:
        tts_engine.unload()
        unloaded.append("TTS")

    if unloaded:
        logger.info("Session closed — unloaded: %s (LLM kept warm)", ", ".join(unloaded))
    else:
        logger.info("Session closed — nothing to unload")

    return JSONResponse({"status": "ok", "unloaded": unloaded, "llm": "kept_warm"})


def _resolve_variant(body: ChatRequest) -> str:
    """Pick the model variant: honor a valid client override, else route by content."""
    override = (body.model or "").strip().lower()
    if override in VALID_VARIANTS:
        return override
    return route(body.message)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, body: ChatRequest, response: Response) -> ChatResponse:
    """
    Text chat with Synapse.

    Accepts a message and optional conversation history. The 2-model router picks
    E2B (simple) or E4B (complex) from the message; the chosen variant is returned
    both in the body (`model`) and the X-Synapse-Model header.
    """
    client_ip = get_client_ip(request)
    check_rate_limit(client_ip)
    _require_json(request)

    # Generation is metered and therefore gated: sign-in required, then a
    # per-account allowance. See require_generation_access.
    await require_generation_access(request)

    variant = _resolve_variant(body)  # route on the ORIGINAL message (before grounding)

    # One slot held across grounding AND generation: taking them separately
    # would let a caller pass admission, do the cheap half, then queue on the
    # expensive half anyway — which is the behaviour this guard exists to stop.
    async with _admit_inference("chat"):
        # RAG grounding (server-side): inject retrieved studio facts into the user turn.
        # Done AFTER routing so a large context block can't fool the router into E4B.
        grounded_message, _sources = await asyncio.to_thread(_ground_message, body.message)

        # Build message list for LLM
        messages = [{"role": m.role, "content": m.content} for m in body.history]
        messages.append({"role": "user", "content": grounded_message})

        try:
            response_text = await asyncio.to_thread(synapse_llm.generate, messages, variant)
        except Exception as e:
            logger.error("LLM generation failed: %s", _safe_err(e))
            raise HTTPException(status_code=500, detail="AI generation failed") from None

    # Detect language from the response (simple heuristic)
    lang = _detect_language(response_text)

    response.headers["X-Synapse-Model"] = variant
    return ChatResponse(response=response_text, language=lang, model=variant)


@app.post("/api/embed")
async def embed_endpoint(request: Request, body: EmbedRequest):
    """
    Server-side Google EmbeddingGemma embeddings for the Embedding Explorer.

    Body: {texts: string[], task: "query"|"document", model: id}. Returns
    {vectors, model, dims, backend:"server"}. Vectors are L2-normalized; the
    "-256" model id yields Matryoshka-truncated 256-dim vectors.
    """
    if not embedding_engine:
        raise HTTPException(status_code=503, detail="Embeddings not available")

    client_ip = get_client_ip(request)
    # Dedicated scope, separate from the global limit: the Explorer fires one call
    # per debounced keystroke, so it needs headroom the chat endpoint does not.
    # Lowered from 60/min: an embed call is not cheap in the way that comment
    # assumed — it holds one of two inference slots on a 4-core box for as long as
    # the batch takes, so sixty a minute from one caller is a denial of service
    # with a polite name. Twenty still exceeds anything a human typing can produce.
    check_rate_limit(client_ip, scope="embed", limit=20)
    _require_json(request)

    for t in body.texts:
        if len(t) > EMBED_MAX_CHARS:
            raise HTTPException(status_code=413, detail=f"Text too long. Max {EMBED_MAX_CHARS} chars per item.")

    # The per-item cap above bounds one text; this bounds the WORK. Without it the
    # maximum legal request is 256 x 4000 characters — a single call worth roughly
    # a minute of the whole machine, which is the cheapest takedown this API sells.
    total_chars = sum(len(t) for t in body.texts)
    if total_chars > EMBED_MAX_TOTAL_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Batch too large. Max {EMBED_MAX_TOTAL_CHARS} characters in total.",
        )

    dims = _EMBED_MODEL_DIMS.get(body.model, EMBED_NATIVE_DIMS)
    try:
        async with _admit_inference("embed"):
            vectors = await asyncio.to_thread(embedding_engine.embed, body.texts, body.task, dims)
    except HTTPException:
        raise  # a 503 from admission control must not be swallowed as a 500
    except Exception as e:
        logger.error("Embedding failed: %s", _safe_err(e))
        raise HTTPException(status_code=500, detail="Embedding failed") from None

    return {"vectors": vectors, "model": body.model, "dims": dims, "backend": "server"}


@app.post("/api/voice")
async def voice_endpoint(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (WAV, WebM, OGG)"),
) -> Response:
    """
    Voice-to-voice pipeline.

    1. Receive audio blob from browser (MediaRecorder)
    2. Transcribe with Gemma-4 native audio (STT)
    3. Generate response with Synapse (LLM)
    4. Synthesize speech (TTS)
    5. Return audio/wav response

    Custom headers:
      X-Synapse-Transcript: original user speech (transcribed)
      X-Synapse-Response: assistant's text response
      X-Synapse-Language: detected language
    """
    if not stt_engine or not tts_engine:
        raise HTTPException(status_code=503, detail="Voice pipeline not available (STT/TTS not installed)")

    _reject_cross_site(request)  # CSRF: multipart is CORS-safelisted → no preflight
    client_ip = get_client_ip(request)
    logger.info("Received voice request from %s (STT model might download on first run)", client_ip)

    check_rate_limit(client_ip)  # global limit
    check_rate_limit(client_ip, scope="tts", limit=TTS_RATE_LIMIT_RPM)  # stricter, paid pipeline

    # Generation is metered and therefore gated: sign-in required, then a
    # per-account allowance. See require_generation_access.
    await require_generation_access(request)

    # Validate file size — reject on declared Content-Length BEFORE reading/spooling
    # the whole upload, then re-check the actual bytes.
    max_bytes = MAX_AUDIO_SIZE_MB * 1024 * 1024
    if _content_length_over(request, max_bytes):
        raise HTTPException(status_code=413, detail=f"Audio too large. Max {MAX_AUDIO_SIZE_MB} MB.")
    audio_bytes = await audio.read()
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio too large. Max {MAX_AUDIO_SIZE_MB} MB.",
        )

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Voice is the most expensive route on the box — STT, then LLM, then TTS,
    # all serialised downstream. One slot covers the whole pipeline, TTS
    # included. (It did not, originally: the block closed before synthesis
    # while this comment already claimed otherwise. The comment was the more
    # dangerous half — a stated guarantee nobody re-checked.)
    async with _admit_inference("voice"):
        # ── Step 1: STT ──
        try:
            transcription = await asyncio.to_thread(stt_engine.transcribe, audio_bytes)
        except Exception as e:
            logger.error("STT failed: %s", _safe_err(e))
            raise HTTPException(status_code=500, detail="Speech recognition failed") from None

        user_text = transcription.text
        detected_lang = transcription.language

        if not user_text.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")

        logger.info(
            "STT: [%s] (%.1fs, %.0f%%) → %s",
            detected_lang, transcription.duration,
            transcription.language_prob * 100, _oneline(user_text, 80),
        )

        # ── Step 2: LLM ──
        try:
            # Ground the voice transcript too (voice is the flagship surface and is
            # pinned to E2B, the grounded model) — same server-side retrieval as text.
            grounded_text, _v_sources = await asyncio.to_thread(_ground_message, user_text)
            messages = [{"role": "user", "content": grounded_text}]
            response_text = await asyncio.to_thread(synapse_llm.generate, messages, _VOICE_VARIANT)
        except Exception as e:
            logger.error("LLM failed: %s", _safe_err(e))
            raise HTTPException(status_code=500, detail="AI generation failed") from None

        logger.info("LLM: → %s", _oneline(response_text, 80))

        # ── Step 3: TTS ──
        # Inside the slot, not after it. This block used to close one level up,
        # so synthesis ran with the slot already released — while the comment
        # above the block claimed "one slot covers the whole pipeline". An
        # adversarial review caught the contradiction; the comment was the
        # worse half of the bug, because it asserted a guarantee that was not
        # there.
        tts_text = _cap_tts_text(response_text, "voice")  # bound per-call cost
        try:
            audio_response = await asyncio.to_thread(
                tts_engine.synthesize,
                tts_text,
                lang=detected_lang,
            )
        except Exception as e:
            logger.error("TTS failed: %s", _safe_err(e))
            raise HTTPException(status_code=500, detail="Speech synthesis failed") from None

    # Determine audio format from the engine's declared output format.
    audio_format, media_type = _tts_format(tts_engine)
    import urllib.parse
    return Response(
        content=audio_response,
        media_type=media_type,
        headers={
            "X-Synapse-Transcript": urllib.parse.quote(user_text[:200]),
            "X-Synapse-Response": urllib.parse.quote(response_text[:200]),
            "X-Synapse-Language": urllib.parse.quote(detected_lang),
            "X-Synapse-Audio-Format": audio_format,
        },
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    # Outer guard: reject absurd bodies at validation time. The precise per-request
    # cap (TTS_MAX_CHARS) is enforced below with a 413 so we never call Google on
    # oversized input.
    text: str = Field(..., max_length=MAX_INPUT_LENGTH)

@app.post("/api/tts")
async def tts_endpoint(request: Request, body: TTSRequest) -> Response:
    """
    Text-to-Speech endpoint.
    Accepts text and returns synthesized audio.
    """
    if not tts_engine:
        raise HTTPException(status_code=503, detail="TTS not installed")

    client_ip = get_client_ip(request)
    check_rate_limit(client_ip)  # global limit
    check_rate_limit(client_ip, scope="tts", limit=TTS_RATE_LIMIT_RPM)  # stricter, paid endpoint
    _require_json(request)

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    # Hard per-request character cap — reject BEFORE any (paid) Google call.
    if len(text) > TTS_MAX_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Text too long. Max {TTS_MAX_CHARS} characters per request.",
        )

    detected_lang = _detect_language(text)

    async with _admit_inference("tts"):
        try:
            audio_response = await asyncio.to_thread(
                tts_engine.synthesize,
                text,
                lang=detected_lang,
            )
        except Exception as e:
            logger.error("TTS failed: %s", _safe_err(e))
            raise HTTPException(status_code=500, detail="Speech synthesis failed") from None

    audio_format, media_type = _tts_format(tts_engine)

    return Response(
        content=audio_response,
        media_type=media_type,
        headers={
            "X-Synapse-Audio-Format": audio_format,
        },
    )

def _detect_language(text: str) -> str:
    """Simple heuristic: if text contains Cyrillic characters, it's Russian."""
    cyrillic_count = sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")
    return "ru" if cyrillic_count > len(text) * 0.3 else "en"


def _cap_tts_text(text: str, context: str) -> str:
    """Truncate text sent to the (paid) TTS engine to TTS_MAX_CHARS.

    For voice paths the TTS input is the model's OWN response (not user input), so we
    truncate + log rather than reject \u2014 bounding per-call cost without dropping the
    reply. Public /api/tts (user-supplied text) rejects with 413 instead.
    """
    if len(text) <= TTS_MAX_CHARS:
        return text
    logger.warning(
        "%s: TTS text %d chars exceeds cap %d \u2014 truncating before synthesis",
        context, len(text), TTS_MAX_CHARS,
    )
    return text[:TTS_MAX_CHARS]


_AUDIO_MEDIA_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg"}


def _tts_format(engine: object) -> tuple[str, str]:
    """Return (format_token, media_type) for a TTS engine's output.

    Uses the engine's declared `output_format` ("mp3"/"wav"/"ogg"); falls back to
    "wav" for anything unexpected so the header/Content-Type stay in sync with the
    actual bytes regardless of which backend (Chirp/Edge) produced them.
    """
    fmt = getattr(engine, "output_format", "wav")
    if fmt not in _AUDIO_MEDIA_TYPES:
        fmt = "wav"
    return fmt, _AUDIO_MEDIA_TYPES[fmt]


# ── SSE Streaming Endpoints ─────────────────────────────────────────────────

def _sse_event(event: str, data: str) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {data}\n\n"


async def _consume_in_thread(gen_factory: Callable[[], Iterator[Any]]) -> AsyncGenerator[Any, None]:
    """Run a blocking (synchronous) generator in a worker thread and yield its
    items on the calling asyncio event loop.

    `synapse_llm.generate_stream()` is a plain sync generator — each `next()`
    call runs real CPU/GPU token-generation work. Iterating it directly with
    `for token in gen: ... await asyncio.sleep(0)` runs that work ON the event
    loop thread; `sleep(0)` only yields BETWEEN tokens, so the loop is still
    blocked (health checks, other requests, everything) for the duration of
    each token. Here the generator instead runs to completion in a background
    thread, pushing items into an asyncio.Queue; the event loop only ever
    awaits `queue.get()`, so it stays free to serve other requests throughout.

    Residual risk: if the consumer stops iterating early (client disconnect),
    the worker thread keeps running the sync generator to completion in the
    background (still holding SynapseLLM._inference_lock for its full
    duration) — llama.cpp has no cooperative-cancellation hook to interrupt
    mid-generation, so this is unavoidable short of killing the thread.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def _worker() -> None:
        try:
            for item in gen_factory():
                loop.call_soon_threadsafe(queue.put_nowait, ("item", item))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", _DONE))

    thread = threading.Thread(target=_worker, daemon=True, name="synapse-stream-worker")
    thread.start()

    while True:
        kind, payload = await queue.get()
        if kind == "item":
            yield payload
        elif kind == "error":
            raise payload
        else:
            return


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: Request, body: ChatRequest) -> StreamingResponse:
    """
    Streaming text chat with Synapse via SSE.

    Events:
      event: token   data: <text>     — each generated token
      event: done    data:            — generation complete
      event: error   data: <message>  — on failure
    """
    client_ip = get_client_ip(request)
    check_rate_limit(client_ip)
    _require_json(request)

    # Generation is metered and therefore gated: sign-in required, then a
    # per-account allowance. See require_generation_access.
    await require_generation_access(request)

    variant = _resolve_variant(body)  # route on the ORIGINAL message (before grounding)

    # SSE cannot use the `_admit_inference` context manager: the slot has to be
    # held for the whole life of the generator, which outlives this function,
    # and a 503 is only possible BEFORE the first byte goes out. So acquire here
    # by hand, and release in the generator's finally.
    _sem = _get_inference_semaphore()
    try:
        await asyncio.wait_for(_sem.acquire(), timeout=0.25)
    except asyncio.TimeoutError:
        logger.warning("Shedding chat stream — all %d inference slots busy", _INFERENCE_SLOTS)
        raise HTTPException(
            status_code=503,
            detail="The model is busy right now. Please try again in a moment.",
            headers={"Retry-After": "5"},
        ) from None

    try:
        grounded_message, _sources = await asyncio.to_thread(_ground_message, body.message)
    except BaseException:
        _sem.release()  # never strand a slot if grounding dies before the stream starts
        raise

    messages = [{"role": m.role, "content": m.content} for m in body.history]
    messages.append({"role": "user", "content": grounded_message})

    async def generate_sse() -> AsyncGenerator[str, None]:
        try:
            # Announce which model is answering BEFORE the first token so the UI
            # can render the per-message badge immediately (mirrors X-Synapse-Model).
            yield _sse_event("model", variant)

            # Stream tokens from LLM — generation runs in a worker thread
            # (see _consume_in_thread) so it never blocks the event loop.
            full_response = []
            async for token in _consume_in_thread(lambda: synapse_llm.generate_stream(messages, variant)):
                full_response.append(token)
                yield _sse_event("token", json.dumps(token, ensure_ascii=False))

            # Send language detection
            response_text = "".join(full_response)
            lang = _detect_language(response_text)
            yield _sse_event("language", lang)
            yield _sse_event("done", "")

        except Exception as e:
            logger.error("Chat stream failed: %s", _safe_err(e))
            yield _sse_event("error", "Internal error during generation")
        finally:
            # Runs on normal completion, on error, AND when the client hangs up
            # (the generator is closed) — an abandoned stream must not leak the
            # slot, or a few dropped connections would wedge the server shut.
            _sem.release()

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/voice/stream")
async def voice_stream_endpoint(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (WAV, WebM, OGG)"),
) -> StreamingResponse:
    """
    Streaming voice-to-voice pipeline via SSE.

    Events:
      event: transcript  data: {"text": "...", "lang": "ru", "duration": 2.1}
      event: token       data: <text>        — each LLM token
      event: audio       data: <base64 WAV>  — synthesized audio
      event: done        data:               — pipeline complete
      event: error       data: <message>     — on failure
    """
    if not stt_engine or not tts_engine:
        raise HTTPException(status_code=503, detail="Voice pipeline not available")

    _reject_cross_site(request)  # CSRF: multipart is CORS-safelisted → no preflight
    client_ip = get_client_ip(request)
    logger.info("Received streaming voice request from %s", client_ip)
    check_rate_limit(client_ip)  # global limit
    check_rate_limit(client_ip, scope="tts", limit=TTS_RATE_LIMIT_RPM)  # stricter, paid pipeline

    # Generation is metered and therefore gated: sign-in required, then a
    # per-account allowance. See require_generation_access.
    await require_generation_access(request)

    max_bytes = MAX_AUDIO_SIZE_MB * 1024 * 1024
    if _content_length_over(request, max_bytes):
        raise HTTPException(status_code=413, detail=f"Audio too large. Max {MAX_AUDIO_SIZE_MB} MB.")
    audio_bytes = await audio.read()
    if len(audio_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Audio too large. Max {MAX_AUDIO_SIZE_MB} MB.")
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    import base64

    # The heaviest route on the box: STT, then LLM, then TTS, all chained and
    # all serialised downstream. It had NO concurrency bound at all — only a
    # per-minute rate limit, which caps how OFTEN a client may call but not how
    # many calls can be in flight at once. Five simultaneous requests all pass
    # the limiter and pin the process-wide thread pool, which is precisely the
    # exhaustion the admission control was introduced to stop. Found by an
    # adversarial review after the first pass claimed to have closed it.
    # Acquired by hand rather than with the context manager because the slot has
    # to outlive this function, and a 503 is only possible before the first byte.
    _sem = _get_inference_semaphore()
    try:
        await asyncio.wait_for(_sem.acquire(), timeout=0.25)
    except asyncio.TimeoutError:
        logger.warning("Shedding voice stream — all %d inference slots busy", _INFERENCE_SLOTS)
        raise HTTPException(
            status_code=503,
            detail="The model is busy right now. Please try again in a moment.",
            headers={"Retry-After": "5"},
        ) from None

    async def generate_sse() -> AsyncGenerator[str, None]:
        try:
            # ── Step 1: STT (blocking, but fast on CPU) ──
            transcription = await asyncio.to_thread(stt_engine.transcribe, audio_bytes)
            user_text = transcription.text
            detected_lang = transcription.language

            if not user_text.strip():
                yield _sse_event("error", "No speech detected")
                return

            logger.info(
                "STT: [%s] (%.1fs, %.0f%%) → %s",
                detected_lang, transcription.duration,
                transcription.language_prob * 100, _oneline(user_text, 80),
            )

            # Send transcript immediately
            yield _sse_event("transcript", json.dumps({
                "text": user_text,
                "lang": detected_lang,
                "duration": round(transcription.duration, 1),
                "confidence": round(transcription.language_prob, 2),
            }, ensure_ascii=False))

            # ── Step 2: LLM streaming — runs in a worker thread (see
            # _consume_in_thread) so token generation never blocks the loop ──
            grounded_text, _v_sources = await asyncio.to_thread(_ground_message, user_text)
            messages = [{"role": "user", "content": grounded_text}]
            full_response = []

            async for token in _consume_in_thread(lambda: synapse_llm.generate_stream(messages, _VOICE_VARIANT)):
                full_response.append(token)
                yield _sse_event("token", json.dumps(token, ensure_ascii=False))

            response_text = "".join(full_response)
            logger.info("LLM stream: → %s", _oneline(response_text, 80))

            # ── Step 3: TTS (synthesize full response) ──
            if tts_engine and response_text.strip():
                tts_text = _cap_tts_text(response_text, "voice/stream")  # bound per-call cost
                audio_wav = await asyncio.to_thread(
                    tts_engine.synthesize, tts_text, lang=detected_lang,
                )
                if audio_wav:
                    audio_b64 = base64.b64encode(audio_wav).decode("ascii")
                    yield _sse_event("audio", audio_b64)

            yield _sse_event("done", "")

        except Exception as e:
            logger.error("Voice stream failed: %s", _safe_err(e))
            yield _sse_event("error", "Internal error during generation")
        finally:
            # Also runs when the client hangs up and the generator is closed —
            # an abandoned stream must never strand a slot.
            _sem.release()

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Conversation History API ─────────────────────────────────────────────────

class ConversationMessageItem(BaseModel):
    """A single stored chat message. Bounds mirror ChatMessage/MAX_INPUT_LENGTH
    so a saved-conversation payload can't smuggle in an unbounded string."""
    id: str = Field(default="", max_length=128)
    # Constrain role to the DB's CHECK set (user|assistant|system). Without this a
    # bad role passes Pydantic but violates the SQLite CHECK → uncaught IntegrityError
    # → 500 (matches ChatMessage.role's constraint).
    role: str = Field(default="user", pattern=r"^(user|assistant|system)$")
    content: str = Field(default="", max_length=MAX_INPUT_LENGTH)
    # Bounded so an out-of-range value can't raise OverflowError on the SQLite bind.
    # 4102444800000 ms ≈ year 2100.
    timestamp: int = Field(default=0, ge=0, le=4102444800000)

    # "ignore" (not "allow"): unknown/extra fields are dropped rather than
    # retained. "allow" would let a client attach an arbitrarily large,
    # unbounded extra field to every one of up to 100 messages per
    # conversation with no length cap of its own — a body-size guard on the
    # endpoint (see MAX_SAVE_BODY_BYTES) limits total request size, but
    # dropping unknown fields here means nothing unbounded is ever persisted
    # to disk. If UI metadata genuinely needs to round-trip, add it as an
    # explicit, length-capped field instead.
    model_config = {"extra": "ignore"}


class ConversationSaveRequest(BaseModel):
    """Body for save/upsert conversation."""
    # Charset-restricted: the frontend uses crypto.randomUUID() (hex + hyphens).
    # Restricting to a safe id charset rejects control chars (log-injection) and
    # keeps ids clean as DB keys. UUIDs (36 chars) fit comfortably.
    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")
    title: str = Field(default="New Chat", max_length=200)
    # Bounded ints so an out-of-range value can't raise OverflowError on the SQLite bind.
    createdAt: int = Field(default=0, ge=0, le=4102444800000)
    updatedAt: int = Field(default=0, ge=0, le=4102444800000)
    messages: list[ConversationMessageItem] = Field(default_factory=list, max_length=100)


class ConversationRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


# ── Conversation storage: OFF the event loop ─────────────────────────────────
# Every one of these handlers is `async def`, and every one of them used to call
# chat_storage synchronously — so each DB round trip blocked the single event
# loop that also serves /api/chat, /api/health and every other request. On a
# four-core ARM box that is not theoretical: the retention purge already runs on
# a worker thread (`asyncio.to_thread(chat_storage.purge_old, ...)` in the
# lifespan), so a purge and a route could contend the same SQLite lock with the
# route holding the loop hostage while it waited.
#
# chat_storage opens a FRESH connection per call, runs WAL, and guards writes
# with its own threading.Lock, so it is safe to call from a worker thread — that
# is why this is a wrap rather than a rewrite. Same idiom the LLM, STT and TTS
# paths already use throughout this file.
@app.get("/api/conversations")
async def list_conversations(request: Request):
    """List all conversations for the signed-in account (no messages, lightweight).
    Anonymous → 401: anonymous users have no server-side history (local IndexedDB only)."""
    check_rate_limit(get_client_ip(request), scope="conversations", limit=RATE_LIMIT_CONVERSATIONS_RPM)
    owner = await authenticated_owner(request)
    if owner is None:
        raise HTTPException(status_code=401, detail="Sign in to sync conversations")
    convs = await asyncio.to_thread(chat_storage.list_conversations, owner)
    return JSONResponse({"conversations": convs})


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str, request: Request):
    """Fetch a single conversation with all messages (signed-in account only)."""
    check_rate_limit(get_client_ip(request), scope="conversations", limit=RATE_LIMIT_CONVERSATIONS_RPM)
    owner = await authenticated_owner(request)
    if owner is None:
        raise HTTPException(status_code=401, detail="Sign in to sync conversations")
    conv = await asyncio.to_thread(chat_storage.get_conversation, owner, conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return JSONResponse(conv)


@app.post("/api/conversations", status_code=200)
async def save_conversation(request: Request):
    """Create or update a conversation with messages."""
    client_ip = get_client_ip(request)
    check_rate_limit(client_ip, scope="conversations", limit=RATE_LIMIT_CONVERSATIONS_RPM)
    owner = await authenticated_owner(request)
    if owner is None:
        raise HTTPException(status_code=401, detail="Sign in to sync conversations")
    _require_json(request)

    # Reject oversized payloads before doing any parsing work — mirrors the
    # beacon endpoint's guard. FastAPI's usual `body: ConversationSaveRequest`
    # parameter would parse (and fully buffer) the body BEFORE Pydantic gets
    # a chance to validate it, so there's no hook to cap raw byte size ahead
    # of that; we take the request body manually instead so this check runs
    # first. Check the declared Content-Length first (cheap, avoids buffering
    # a huge body at all when the client is honest about size), then
    # re-check the actual bytes read in case Content-Length was absent/wrong.
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_SAVE_BODY_BYTES:
                return JSONResponse({"error": "Payload too large"}, status_code=413)
        except ValueError:
            pass

    raw = await request.body()
    if len(raw) > MAX_SAVE_BODY_BYTES:
        return JSONResponse({"error": "Payload too large"}, status_code=413)

    try:
        body = ConversationSaveRequest.model_validate_json(raw)
    except ValidationError:
        # Generic message — don't echo the parsed input back via e.errors().
        raise HTTPException(status_code=422, detail="Invalid conversation data") from None

    logger.info("SAVE conv id=%s title=%r msgs=%d from=%s", _oneline(body.id, 64), body.title, len(body.messages), client_ip)
    try:
        await asyncio.to_thread(chat_storage.save_conversation, owner, body.model_dump())
    except ConversationOwnershipError:
        logger.warning("Rejected cross-owner conversation write id=%s from=%s", _oneline(body.id, 64), client_ip)
        raise HTTPException(status_code=403, detail="Forbidden") from None
    except Exception as e:  # noqa: BLE001 — never leak a raw 500 on malformed data
        logger.warning("Conversation save failed id=%s: %s", _oneline(body.id, 64), _safe_err(e))
        raise HTTPException(status_code=422, detail="Invalid conversation data") from None
    return JSONResponse({"status": "ok", "id": body.id})


@app.post("/api/conversations/beacon", status_code=200)
async def save_conversation_beacon(request: Request):
    """Beacon endpoint for navigator.sendBeacon (text/plain, no CORS preflight).

    Used during page unload (Ctrl+R, tab close) when browser can't do
    a normal JSON POST with CORS preflight.
    """
    _reject_cross_site(request)  # CSRF: text/plain beacon is CORS-safelisted → no preflight
    client_ip = get_client_ip(request)
    check_rate_limit(client_ip, scope="conversations", limit=RATE_LIMIT_CONVERSATIONS_RPM)
    # owner is computed AFTER parsing the body (the capability id rides in the body here)

    # Reject oversized payloads before doing any parsing work. Check the
    # declared Content-Length first (cheap, avoids buffering a huge body at
    # all when the client is honest about size), then re-check the actual
    # bytes read in case Content-Length was absent/wrong.
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_BEACON_BODY_BYTES:
                return JSONResponse({"error": "Payload too large"}, status_code=413)
        except ValueError:
            pass

    # Bound before the try so the ownership handler below can name it even when
    # the body never parsed.
    conv_id = "?"
    try:
        raw = await request.body()
        if len(raw) > MAX_BEACON_BODY_BYTES:
            return JSONResponse({"error": "Payload too large"}, status_code=413)

        data = json.loads(raw)
        if not isinstance(data, dict):
            return JSONResponse({"error": "invalid body"}, status_code=400)

        # sendBeacon can't set headers → the Google ID token rides in the body,
        # verified identically to the Authorization: Bearer path. Anonymous → 401.
        # Popped BEFORE validation: it is transport, not conversation data, and
        # leaving it in would fail the model as an unexpected field.
        tok = data.pop("idToken", None)
        owner = await authenticated_owner(request, tok if isinstance(tok, str) else None)
        if owner is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        # SAME Pydantic model as the JSON route. This used to be a hand-written
        # subset of it, and the subset was missing the parts that matter: the id
        # charset was checked, but `createdAt`/`updatedAt` went to the SQLite bind
        # with no type and no bounds at all, and every message item — id, role,
        # content, timestamp — went through unvalidated.
        #
        # The worst of those was `updatedAt`. The JSON route bounds it at
        # 4102444800000 (about the year 2100); the beacon accepted anything, so a
        # conversation stamped far enough in the future was IMMUNE TO THE 30-DAY
        # RETENTION PURGE — a permanent record in a store whose whole policy is
        # that it does not keep one. An unexpected `role` was a second: it passes
        # a dict but violates the DB's CHECK constraint, which is an uncaught
        # IntegrityError and a 500.
        #
        # Two validators for one shape is how the second one ends up weaker. One
        # model, both paths.
        #
        # `messages` is truncated BEFORE validation rather than rejected, and only
        # that field: this runs during page unload, where a 422 reaches nobody and
        # simply loses the save. A stale tab holding an older bundle should still
        # get its last hundred messages stored.
        if isinstance(data.get("messages"), list):
            # Newest 100, not oldest: [:100] kept the FIRST hundred and dropped
            # everything the user had just written — the exact opposite of the
            # comment above and of chat_storage's own messages[-MAX:] policy.
            data["messages"] = data["messages"][-100:]
        try:
            beacon = ConversationSaveRequest.model_validate(data)
        except ValidationError:
            return JSONResponse({"error": "invalid conversation data"}, status_code=400)

        conv_id = beacon.id
        logger.info("BEACON save id=%s title=%s msgs=%d from=%s",
                     _oneline(conv_id, 64), _oneline(beacon.title, 200),
                     len(beacon.messages), client_ip)
        await asyncio.to_thread(
            chat_storage.save_conversation, owner, beacon.model_dump()
        )
        return JSONResponse({"status": "ok"})
    except ConversationOwnershipError:
        logger.warning("Rejected cross-owner beacon write id=%s from=%s", _oneline(conv_id, 64), client_ip)
        return JSONResponse({"error": "forbidden"}, status_code=403)
    except Exception as e:
        logger.error("Beacon save failed: %s", _safe_err(e))
        return JSONResponse({"error": "Invalid request"}, status_code=400)


@app.patch("/api/conversations/{conv_id}")
async def rename_conversation(conv_id: str, request: Request, body: ConversationRenameRequest):
    """Rename a conversation (signed-in account only)."""
    check_rate_limit(get_client_ip(request), scope="conversations", limit=RATE_LIMIT_CONVERSATIONS_RPM)
    owner = await authenticated_owner(request)
    if owner is None:
        raise HTTPException(status_code=401, detail="Sign in to sync conversations")
    _require_json(request)
    found = await asyncio.to_thread(chat_storage.rename_conversation, owner, conv_id, body.title)
    if not found:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return JSONResponse({"status": "ok"})


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str, request: Request):
    """Delete a conversation and all its messages (signed-in account only)."""
    check_rate_limit(get_client_ip(request), scope="conversations", limit=RATE_LIMIT_CONVERSATIONS_RPM)
    owner = await authenticated_owner(request)
    if owner is None:
        raise HTTPException(status_code=401, detail="Sign in to sync conversations")
    found = await asyncio.to_thread(chat_storage.delete_conversation, owner, conv_id)
    if not found:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return JSONResponse({"status": "ok"})


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import RELOAD
    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=RELOAD,          # OFF unless SYNAPSE_RELOAD=1 (dev) — prod-safe default
        log_level="info",
        server_header=False,    # don't advertise the uvicorn version banner
    )
