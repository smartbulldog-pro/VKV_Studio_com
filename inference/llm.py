"""
Synapse LLM — Gemma 4 (E2B ⇄ E4B) via a llama-server subprocess router
=======================================================================
Two equal-role fine-tunes (E2B "junior", E4B "mid") chosen per request by
router.route(). They can't both fit in 12 GB VRAM, so the engine keeps AT MOST
ONE resident at a time.

WHY A SUBPROCESS PER MODEL (not in-process llama-cpp-python):
llama.cpp's CUDA context CANNOT be safely re-created inside a single Python
process — unloading one Llama() and constructing another aborts the process with
a `ggml-cuda.cu: CUDA error`. Load-on-demand on ONE GPU therefore requires
tearing down the whole context, which only a process exit does cleanly. So each
variant runs as its own official `llama-server` child (OpenAI-compatible HTTP);
swapping = terminate the running child (OS frees its VRAM) + spawn the other.
One process = one CUDA context for that process's entire life. No re-creation.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import TypedDict

import requests  # type: ignore[import-untyped]

from config import (
    LLM_CONTEXT_SIZE,
    LLM_E2B_MODEL_PATH,
    LLM_E4B_MODEL_PATH,
    LLM_MAX_TOKENS,
    LLM_MIN_GEN_TOKENS,
    LLM_N_GPU_LAYERS,
    LLM_REMOTE_CREDENTIALS,
    LLM_REMOTE_TIMEOUT,
    LLM_REMOTE_URL,
    LLM_TOKEN_SAFETY_MARGIN,
    LLM_REPEAT_PENALTY,
    LLM_SERVER_HOST,
    LLM_SERVER_PORT,
    LLM_SERVER_REQUEST_TIMEOUT,
    LLM_SERVER_STARTUP_TIMEOUT,
    LLM_STOP_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_TOP_P,
    SYSTEM_PROMPT,
)
from router import VALID_VARIANTS, VARIANT_E2B, VARIANT_E4B
from log_safety import child_env, safe_err as _safe_err

logger = logging.getLogger("synapse.llm")

# ── Path to the llama.cpp server binary ──────────────────────────────────────
# Platform-aware: Windows uses the .exe, Linux/ARM (the prod deploy target) uses
# the extensionless ELF. The correct binary for the host still has to be present
# in llama-bin/ — this just stops the path from being hardcoded to Windows so the
# ARM port can drop in `llama-server` without a code change.
LLAMA_BIN_DIR = Path(__file__).resolve().parent / "llama-bin"
LLAMA_SERVER_EXE = LLAMA_BIN_DIR / ("llama-server.exe" if os.name == "nt" else "llama-server")


class ChatMessage(TypedDict):
    """OpenAI-compatible chat message format."""
    role: str       # "system" | "user" | "assistant"
    content: str


def _register_dll_directories() -> None:
    """
    Put llama.cpp's binaries and the CUDA runtime on PATH.

    The spawned `llama-server` child inherits this process's environment, so
    prepending these directories here is what lets the child's ggml-cuda.dll
    resolve the CUDA runtime DLLs (cublas/cudart, shipped with PyTorch) and its
    sibling ggml backends in llama-bin/. We no longer import llama-cpp-python in
    THIS process (the model runs in the child), but the PATH setup still matters.

    1. llama-bin/ — official llama.cpp CUDA binaries (llama-server.exe, ggml-*.dll)
    2. torch/lib/ — CUDA runtime DLLs (cublas, cudart, etc.)
    """
    # Fix OpenMP conflict: llama.cpp uses libomp140, PyTorch uses libiomp5md
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    is_windows = os.name == "nt"

    # Register llama.cpp binary directory.
    #
    # os.add_dll_directory exists ONLY on Windows — calling it on Linux raises
    # AttributeError, which _suppress(OSError) below does NOT catch, so this
    # used to kill the whole module at import time on the ARM box the moment
    # llama-bin/ appeared. (It was merely latent before that, because the
    # directory did not exist yet.) On Linux the equivalent concern is the
    # dynamic linker, not a DLL search path: a CMake-built llama-server sits
    # beside its libggml-*.so siblings, so export LD_LIBRARY_PATH instead —
    # the spawned child inherits it and resolves them wherever it is launched
    # from.
    if LLAMA_BIN_DIR.exists():
        if is_windows:
            with _suppress(OSError):
                os.add_dll_directory(str(LLAMA_BIN_DIR))
            os.environ["PATH"] = str(LLAMA_BIN_DIR) + os.pathsep + os.environ.get("PATH", "")
        else:
            os.environ["LD_LIBRARY_PATH"] = (
                str(LLAMA_BIN_DIR) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
            )
        logger.info("Registered llama.cpp binaries: %s", LLAMA_BIN_DIR)

    # Register PyTorch CUDA runtime (cublas, cudart, etc.) — Windows/CUDA only.
    # The ARM production box is CPU-only and deliberately has no torch, so skip
    # the whole block rather than logging a warning that means nothing there.
    if is_windows:
        try:
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isdir(torch_lib):
                with _suppress(OSError):
                    os.add_dll_directory(torch_lib)
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]
                logger.info("Registered CUDA runtime from PyTorch: %s", torch_lib)
        except ImportError:
            logger.warning("PyTorch not installed — CUDA DLLs may not be found")


class _suppress:
    """Tiny contextlib.suppress-alike (kept local to avoid an extra import)."""

    def __init__(self, *excs):  # noqa: D401
        self._excs = excs

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, self._excs)


# Populate PATH at import so any spawned llama-server inherits it.
_register_dll_directories()


class SynapseLLM:
    """
    Synapse LLM engine — a router over two `llama-server` child processes.

    Keeps at most one child alive (VRAM). `generate`/`generate_stream` take a
    `variant` ("e2b"|"e4b"); if the live child serves a different variant it is
    terminated (freeing VRAM on exit) and the requested one is spawned before the
    request is served.

    Concurrency: a single lock (_lifecycle_lock) serializes BOTH inference and
    swaps, so a swap can never terminate a child mid-request. Streaming holds the
    lock for the whole generator. Health/`is_loaded` reads are lock-free (they
    only touch a cheap `poll()` + a bool) so a health probe never blocks behind a
    model load or a long generation.
    """

    _PATHS = {VARIANT_E2B: LLM_E2B_MODEL_PATH, VARIANT_E4B: LLM_E4B_MODEL_PATH}

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._active: str | None = None      # variant of the live child
        self._ready: bool = False            # child passed /health
        self._logfile = None                 # child stdout/stderr sink
        self._load_errors: dict[str, str] = {}
        # Serializes inference AND swaps (a swap must not race a request).
        self._lifecycle_lock = threading.Lock()
        # Cached ID-token credentials for a private remote; see _auth_headers.
        self._remote_creds = None

    # ── HTTP base for the live child ─────────────────────────────────────────
    @property
    def _base_url(self) -> str:
        # A configured remote wins: generation happens on someone else's GPU and
        # this process never spawns a child. See SYNAPSE_LLM_REMOTE_URL in config.
        if LLM_REMOTE_URL:
            return LLM_REMOTE_URL
        return f"http://{LLM_SERVER_HOST}:{LLM_SERVER_PORT}"

    @property
    def _is_remote(self) -> bool:
        return bool(LLM_REMOTE_URL)

    @property
    def _request_timeout(self) -> float:
        # A local child is already warm by the time a request reaches it, so the
        # normal timeout fits. A scale-to-zero GPU service is not: the first
        # request after an idle period pays container start plus model load
        # before a single token comes back, and cutting that off would look
        # exactly like a broken backend.
        return LLM_REMOTE_TIMEOUT if self._is_remote else LLM_SERVER_REQUEST_TIMEOUT

    def _auth_headers(self) -> dict:
        """Authorization for a PRIVATE remote; empty dict for a local child.

        A Cloud Run service deployed without --allow-unauthenticated accepts a
        Google-signed ID token whose audience is the service URL — an access
        token is NOT enough. Minting one needs nothing beyond google-auth, which
        is already installed here because Chirp 3 TTS pulls it in.

        The credentials object is built once and cached: google-auth refreshes
        the token itself when it is close to expiry, so this neither mints one
        per request nor ever sends a stale one.

        Deliberately fails CLOSED. If a remote is configured but no credentials
        are, this raises rather than sending an unauthenticated request — a
        silent 403 from Cloud Run would look exactly like the model being
        unavailable, and the whole point of the private deployment is that an
        open GPU endpoint is a financial attack surface.
        """
        if not self._is_remote:
            return {}
        if not LLM_REMOTE_CREDENTIALS:
            raise RuntimeError(
                "SYNAPSE_LLM_REMOTE_URL is set but no service-account JSON was found "
                "(SYNAPSE_LLM_REMOTE_CREDENTIALS / SYNAPSE_GOOGLE_TTS_CREDENTIALS). "
                "Refusing to call the remote unauthenticated."
            )

        creds = self._remote_creds
        if creds is None:
            from google.oauth2 import service_account  # type: ignore[import-untyped]

            creds = service_account.IDTokenCredentials.from_service_account_file(
                LLM_REMOTE_CREDENTIALS, target_audience=LLM_REMOTE_URL
            )
            self._remote_creds = creds
            logger.info("Remote LLM auth: ID token for audience %s", LLM_REMOTE_URL)

        if not creds.valid:
            from google.auth.transport.requests import Request as _GoogleRequest  # type: ignore[import-untyped]

            creds.refresh(_GoogleRequest())
        return {"Authorization": f"Bearer {creds.token}"}

    # ── Lifecycle (caller must hold _lifecycle_lock) ─────────────────────────
    def _ensure_loaded(self, variant: str) -> None:
        """Ensure `variant`'s llama-server child is live + healthy, swapping if needed."""
        if self._is_remote:
            # There is no child to ensure. The remote serves whichever model its
            # image was built with, so variant swapping does not apply — the
            # router's pick is recorded for the response badge and otherwise
            # ignored. No health probe here on purpose: waking a scale-to-zero
            # GPU to answer "are you awake" is exactly the cost this design
            # avoids, and the generate call below will surface a real failure
            # anyway.
            if self._active != variant:
                logger.info(
                    "Remote LLM at %s serves a single model; routing pick %r is advisory",
                    LLM_REMOTE_URL, variant,
                )
            self._active = variant
            self._ready = True
            return

        if self._active == variant and self._ready and self._proc and self._proc.poll() is None:
            return

        if self._load_errors.get(variant):
            raise RuntimeError(
                f"Model '{variant}' failed to load previously: {self._load_errors[variant]}"
            )

        model_path = self._PATHS[variant]
        if not model_path.exists():
            err = f"Model file not found: {model_path}"
            self._load_errors[variant] = err
            raise FileNotFoundError(err)
        if not LLAMA_SERVER_EXE.exists():
            err = f"llama-server binary not found: {LLAMA_SERVER_EXE}"
            self._load_errors[variant] = err
            raise FileNotFoundError(err)

        # Free VRAM: stop the other child FIRST so its context is gone before we
        # spawn the new one (both together overflow 12 GB VRAM).
        if self._proc is not None:
            logger.info("Swapping LLM: stopping '%s' to make room for '%s'", self._active, variant)
            self._stop_child()

        self._spawn_child(variant, model_path)

    def _spawn_child(self, variant: str, model_path: Path) -> None:
        args = [
            str(LLAMA_SERVER_EXE),
            "--model", str(model_path),
            "--host", LLM_SERVER_HOST,
            "--port", str(LLM_SERVER_PORT),
            "--n-gpu-layers", str(LLM_N_GPU_LAYERS),
            "--ctx-size", str(LLM_CONTEXT_SIZE),
            "--parallel", "1",
            # Reuse KV cache across requests that share a prefix. Measured on
            # the ARM box 2026-08-09: a cold 1144-token RAG prompt costs 16.8s
            # of prefill at 68 tok/s, but a follow-up turn sharing that prefix
            # reprocesses only the new tokens — 0.4s. Every conversation turn
            # after the first is effectively free, and this flag extends the
            # same reuse to prompts that diverge partway (which is what a
            # changing RAG context does).
            "--cache-reuse", "256",
            # The slots monitoring endpoint is ENABLED by default and can
            # expose prompt content. Nothing here consumes it, and the whole
            # point of binding to loopback is that this process is not a
            # service anyone should be able to introspect. Off.
            "--no-slots",
        ]
        logger.info("Spawning llama-server '%s': %s (n_gpu_layers=%d, n_ctx=%d)",
                    variant, model_path.name, LLM_N_GPU_LAYERS, LLM_CONTEXT_SIZE)

        # Log to a file (never a PIPE — an unread pipe buffer would deadlock the
        # chatty server). cwd=llama-bin so the child resolves its sibling DLLs.
        # Logs go to SYNAPSE_LOG_DIR when set. On the production box that
        # points outside the code tree, so the service can run with its own
        # source read-only (systemd ProtectSystem=strict + a narrow
        # ReadWritePaths) — a compromised process should not be able to
        # rewrite the code it is executing. Defaults to the old in-tree
        # location so local dev is unchanged.
        log_dir = Path(os.getenv("SYNAPSE_LOG_DIR") or Path(__file__).resolve().parent)
        log_path = log_dir / f"llama-server.{variant}.log"
        try:
            self._logfile = open(log_path, "w", encoding="utf-8", errors="replace")
        except OSError:
            self._logfile = subprocess.DEVNULL  # type: ignore[assignment]

        creationflags = 0
        if os.name == "nt":
            # New process group so we can signal the child cleanly and it doesn't
            # share our console (also lets a hard kill target only the child).
            creationflags = subprocess.CREATE_NO_WINDOW

        try:
            self._proc = subprocess.Popen(
                args,
                cwd=str(LLAMA_BIN_DIR),
                stdout=self._logfile,
                stderr=subprocess.STDOUT,
                env=child_env(),  # secrets stripped — the child needs none
                creationflags=creationflags,
            )
        except Exception as e:
            self._load_errors[variant] = str(e)
            self._close_logfile()
            logger.error("Failed to spawn llama-server '%s': %s", variant, e)
            raise

        self._active = variant
        self._ready = False
        self._wait_until_healthy(variant, log_path)

    def _wait_until_healthy(self, variant: str, log_path: Path) -> None:
        """Poll /health until the child serves 200, or fail (dead process / timeout)."""
        deadline = time.monotonic() + LLM_SERVER_STARTUP_TIMEOUT
        url = f"{self._base_url}/health"
        while time.monotonic() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                self._close_logfile()
                self._active = None
                tail = _tail_file(log_path)
                err = f"llama-server '{variant}' exited during startup. Log tail:\n{tail}"
                self._load_errors[variant] = err
                raise RuntimeError(err)
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    self._ready = True
                    self._log_device_placement(variant, log_path)
                    return
            except requests.RequestException:
                pass  # not up yet
            time.sleep(0.5)

        # Timed out — tear the half-started child down.
        self._stop_child()
        err = f"llama-server '{variant}' did not become healthy within {LLM_SERVER_STARTUP_TIMEOUT:.0f}s"
        self._load_errors[variant] = err
        raise RuntimeError(err)

    def _log_device_placement(self, variant: str, log_path: Path) -> None:
        """Log GPU vs CPU honestly. llama-server SILENTLY falls back to CPU if the
        CUDA runtime DLLs (cudart64_*/cublas64_*/cublasLt64_*) aren't sitting beside
        ggml-cuda.dll in llama-bin — which ran the 8B model on the CPU unnoticed
        (slow + hot). Warn loudly instead of assuming GPU."""
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.info("llama-server '%s' healthy", variant)
            return
        on_gpu = ("CUDA0" in text) or ("CUDA : ARCHS" in text)
        if on_gpu:
            logger.info("llama-server '%s' healthy (GPU / CUDA)", variant)
        else:
            logger.warning(
                "llama-server '%s' healthy but running CPU-ONLY — no CUDA device detected. "
                "An 8B model on CPU is slow and runs the CPU hot. Ensure the CUDA runtime DLLs "
                "(cudart64_*.dll, cublas64_*.dll, cublasLt64_*.dll) sit next to ggml-cuda.dll in %s.",
                variant, LLAMA_BIN_DIR,
            )

    def _stop_child(self) -> None:
        """Terminate the live child and wait for exit so its VRAM is released."""
        proc = self._proc
        self._proc = None
        self._active = None
        self._ready = False
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("llama-server did not exit on terminate() — killing")
                proc.kill()
                with _suppress(Exception):
                    proc.wait(timeout=10)
            # Give the driver a beat to reclaim VRAM before a respawn allocates.
            time.sleep(0.4)
        self._close_logfile()

    def _close_logfile(self) -> None:
        if self._logfile not in (None, subprocess.DEVNULL):
            with _suppress(Exception):
                self._logfile.close()  # type: ignore[union-attr]
        self._logfile = None

    # ── Prompt assembly ──────────────────────────────────────────────────────
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Conservative (over-)estimate of a string's token count, for budgeting.

        Cyrillic tokenizes denser than Latin under this model's SentencePiece
        vocab, so count Cyrillic at ~2 chars/token and everything else at ~4 —
        deliberately biased to OVERESTIMATE so the generation budget stays inside
        the context window even when the estimate is wrong. A cheap heuristic
        beats a per-request /tokenize round-trip here; the safety margin absorbs
        the error. Plus a small per-message overhead for chat-template control
        tokens is added by the caller.
        """
        if not text:
            return 0
        cyr = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
        other = len(text) - cyr
        return int(cyr / 2 + other / 4) + 1

    # Per-message overhead (role marker + turn control tokens the chat template
    # injects around each message). Rough but on the safe side.
    _MSG_OVERHEAD = 8

    def _msgs_tokens(self, messages: list[ChatMessage]) -> int:
        return sum(self._estimate_tokens(m["content"]) + self._MSG_OVERHEAD for m in messages)

    def _build_messages(self, user_messages: list[ChatMessage], variant: str) -> list[ChatMessage]:
        """Prepend system prompt with a language + length hint (variant-aware),
        then trim the OLDEST history turns so the prompt leaves room for at least
        LLM_MIN_GEN_TOKENS of generation.

        E2B (simple asks) stays terse; E4B (complex asks) gets more room so a
        routed-as-hard question isn't clipped. The system prompt and the latest
        user turn are never dropped — only older history is shed — so the safety
        rules in the system prompt can never be evicted by an overlong context.
        """
        last_msg = user_messages[-1]["content"] if user_messages else ""
        # Ratio over LETTERS, not over the whole string — keep in step with
        # rag/retriever.py:detect_lang, which explains why. Counting spaces,
        # digits and punctuation as evidence for English meant a Russian
        # question naming a Latin-spelled product ("Ты используешь llama.cpp?")
        # could be told to answer in English, which is the one instruction in
        # this hint the model reliably obeys.
        cyrillic = sum(1 for c in last_msg if "Ѐ" <= c <= "ӿ")
        latin = sum(1 for c in last_msg if ("a" <= c <= "z") or ("A" <= c <= "Z"))
        letters = cyrillic + latin
        user_lang = "Russian" if letters and cyrillic > letters * 0.3 else "English"

        word_cap = 350 if variant == VARIANT_E4B else 150
        lang_hint = (
            f"\n\nIMPORTANT: The user is writing in {user_lang}. You MUST respond in "
            f"{user_lang} only. Keep your response concise (under {word_cap} words)."
        )
        system_msg: ChatMessage = {"role": "system", "content": SYSTEM_PROMPT + lang_hint}

        # Reserve = the most tokens the prompt (system + kept turns) may occupy and
        # still leave a minimum generation budget within the window.
        reserve = LLM_CONTEXT_SIZE - LLM_MIN_GEN_TOKENS - LLM_TOKEN_SAFETY_MARGIN
        kept = list(user_messages)
        # Drop oldest turns until the prompt fits the reserve, but always keep the
        # latest turn (kept[-1]) so the actual question survives.
        while len(kept) > 1 and self._msgs_tokens([system_msg, *kept]) > reserve:
            dropped = kept.pop(0)
            logger.debug("Trimmed oldest history turn to fit context budget (%d chars)",
                         len(dropped.get("content", "")))

        result: list[ChatMessage] = [system_msg, *kept]
        # Final guard: history trimming keeps the latest turn intact, but that turn
        # ALONE (a pathologically long raw message, or a huge history that trims to
        # one big turn) can still exceed the window and context-shift the system
        # prompt out the front. Cap the last turn's chars to whatever token budget
        # remains after the system prompt + earlier kept turns. RAG context is
        # already bounded, so this only bites abusive single messages.
        if result and result[-1]["role"] == "user":
            head_tokens = self._msgs_tokens(result[:-1])  # system + earlier kept turns
            last_budget = max(LLM_MIN_GEN_TOKENS, reserve - head_tokens)
            # tokens(content) <= chars/2 (all-Cyrillic worst case), so chars <=
            # last_budget*2 guarantees the last turn fits the remaining budget.
            max_last_chars = last_budget * 2 - self._MSG_OVERHEAD * 2
            content = result[-1]["content"]
            if len(content) > max_last_chars:
                logger.warning("Capping oversized last turn: %d -> %d chars (context budget)",
                               len(content), max_last_chars)
                result[-1] = {**result[-1], "content": content[:max(0, max_last_chars)].rstrip() + "\n…[truncated]"}
        return result

    def _budget_max_tokens(self, full_messages: list[ChatMessage]) -> int:
        """Effective max_tokens = window − estimated prompt − safety margin,
        floored at LLM_MIN_GEN_TOKENS and capped at the LLM_MAX_TOKENS ceiling.
        This is what stops a long prompt + a constant max_tokens from overflowing
        the 4096 window (and context-shifting the system prompt out the front)."""
        used = self._msgs_tokens(full_messages)
        budget = LLM_CONTEXT_SIZE - used - LLM_TOKEN_SAFETY_MARGIN
        return max(LLM_MIN_GEN_TOKENS, min(LLM_MAX_TOKENS, budget))

    def _payload(self, full_messages: list[ChatMessage], *, stream: bool) -> dict:
        return {
            "messages": full_messages,
            "max_tokens": self._budget_max_tokens(full_messages),
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "top_k": LLM_TOP_K,
            "repeat_penalty": LLM_REPEAT_PENALTY,
            "stop": LLM_STOP_TOKENS,
            "stream": stream,
        }

    # ── Inference ────────────────────────────────────────────────────────────
    def generate(self, messages: list[ChatMessage], variant: str = VARIANT_E2B) -> str:
        """Generate a complete response from the chosen model (blocking)."""
        variant = variant if variant in VALID_VARIANTS else VARIANT_E2B
        full_messages = self._build_messages(messages, variant)

        try:
            with self._lifecycle_lock:
                self._ensure_loaded(variant)
                r = requests.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=self._payload(full_messages, stream=False),
                    timeout=self._request_timeout,
                    headers=self._auth_headers(),
                )
                r.raise_for_status()
                data = r.json()
            content = data["choices"][0]["message"]["content"]
            return (content or "").strip()
        except Exception as e:
            logger.error("LLM generation failed: %s", _safe_err(e))
            raise

    def generate_stream(
        self, messages: list[ChatMessage], variant: str = VARIANT_E2B
    ) -> Generator[str, None, None]:
        """Stream response tokens from the chosen model.

        Holds the lifecycle lock for the whole generator so no swap can terminate
        the child mid-stream.
        """
        variant = variant if variant in VALID_VARIANTS else VARIANT_E2B
        full_messages = self._build_messages(messages, variant)

        with self._lifecycle_lock:
            try:
                self._ensure_loaded(variant)
                with requests.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=self._payload(full_messages, stream=True),
                    timeout=self._request_timeout,
                    headers=self._auth_headers(),
                    stream=True,
                ) as r:
                    r.raise_for_status()
                    # llama-server's SSE stream is UTF-8 but sends no charset on
                    # text/event-stream, so requests would default r.encoding to
                    # ISO-8859-1 and iter_lines(decode_unicode=True) would mangle
                    # Cyrillic (each 2-byte UTF-8 char → "Ð§"-style mojibake).
                    # Pin UTF-8 explicitly.
                    r.encoding = "utf-8"
                    for raw in r.iter_lines(decode_unicode=True):
                        if not raw or not raw.startswith("data:"):
                            continue
                        payload = raw[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token
            except Exception as e:
                logger.error("LLM stream failed: %s", _safe_err(e))
                raise

    # ── Introspection / teardown ─────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        """True if a child is live and healthy (lock-free — safe for health probes)."""
        proc = self._proc
        return bool(self._ready and proc is not None and proc.poll() is None)

    @property
    def active_variant(self) -> str | None:
        """Variant token of the live child, or None if cold."""
        return self._active if self.is_loaded else None

    def unload(self) -> None:
        """Terminate the live child and release its VRAM (clears sticky errors)."""
        with self._lifecycle_lock:
            if self._proc is not None:
                self._stop_child()
                logger.info("Synapse LLM child stopped")
            self._load_errors.clear()


def _tail_file(path: Path, n: int = 20) -> str:
    """Best-effort last-N-lines of a log file for error messages."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "(no log available)"
