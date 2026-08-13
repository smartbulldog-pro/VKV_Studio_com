"""
Synapse Embeddings — Google EmbeddingGemma via a persistent llama-server
========================================================================
Server-side embeddings for the Embedding Explorer (owner decision 2026-07-06:
"backend only" — no in-browser transformers.js download; single Google stack).

Runs Google's **EmbeddingGemma-300M** GGUF as its OWN persistent
`llama-server --embedding` child (OpenAI-compatible `/v1/embeddings`). Unlike the
LLM router (which swaps E2B⇄E4B on the GPU), this server is small and STAYS UP,
on **CPU by default** so it never competes with the LLM for the 12 GB GPU.

EmbeddingGemma is prompt-conditioned — callers pass a task ("query"|"document")
and this module applies the official prefix. Native output is 768-dim; a
Matryoshka "compact" width is served by truncating + re-normalizing.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import requests  # type: ignore[import-untyped]

from config import (
    EMBED_CTX_SIZE,
    EMBED_GPU_LAYERS,
    EMBED_MODEL_PATH,
    EMBED_NATIVE_DIMS,
    EMBED_SERVER_HOST,
    EMBED_SERVER_PORT,
    EMBED_SERVER_REQUEST_TIMEOUT,
    EMBED_SERVER_STARTUP_TIMEOUT,
)
# Reuse the LLM router's binary location + PATH setup (it registers the CUDA/ggml
# DLL dirs at import). The embedding child inherits that environment.
from llm import LLAMA_BIN_DIR, LLAMA_SERVER_EXE
from log_safety import child_env, safe_err as _safe_err

logger = logging.getLogger("synapse.embeddings")

# EmbeddingGemma's official retrieval prompt templates (model card /
# sentence-transformers). query vs document is asymmetric on purpose.
PROMPTS = {
    "query": lambda t: f"task: search result | query: {t}",
    "document": lambda t: f"title: none | text: {t}",
}


class EmbeddingEngine:
    """Persistent EmbeddingGemma `llama-server --embedding` child + a thin client."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._ready = False
        self._logfile = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()

    @property
    def _base_url(self) -> str:
        return f"http://{EMBED_SERVER_HOST}:{EMBED_SERVER_PORT}"

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._ready and self._proc and self._proc.poll() is None:
            return
        with self._load_lock:
            if self._ready and self._proc and self._proc.poll() is None:
                return
            if self._load_error:
                raise RuntimeError(f"Embedding server failed to load previously: {self._load_error}")
            if not EMBED_MODEL_PATH.exists():
                self._load_error = f"Embedding model not found: {EMBED_MODEL_PATH}"
                raise FileNotFoundError(self._load_error)
            if not LLAMA_SERVER_EXE.exists():
                self._load_error = f"llama-server binary not found: {LLAMA_SERVER_EXE}"
                raise FileNotFoundError(self._load_error)
            self._spawn()

    def _spawn(self) -> None:
        args = [
            str(LLAMA_SERVER_EXE),
            "--model", str(EMBED_MODEL_PATH),
            "--embedding",
            "--pooling", "mean",          # EmbeddingGemma uses mean pooling
            "--embd-normalize", "2",      # L2-normalize (the rest of the stack assumes unit vectors)
            "--host", EMBED_SERVER_HOST,
            "--port", str(EMBED_SERVER_PORT),
            "--ctx-size", str(EMBED_CTX_SIZE),
            "--n-gpu-layers", str(EMBED_GPU_LAYERS),
        ]
        logger.info("Spawning EmbeddingGemma llama-server: %s (gpu_layers=%d)",
                    EMBED_MODEL_PATH.name, EMBED_GPU_LAYERS)
        # Same rationale as llm.py: SYNAPSE_LOG_DIR keeps runtime logs out of
        # the code tree so production can mount the source read-only.
        log_dir = Path(os.getenv("SYNAPSE_LOG_DIR") or Path(__file__).resolve().parent)
        log_path = log_dir / "llama-server.embed.log"
        try:
            self._logfile = open(log_path, "w", encoding="utf-8", errors="replace")
        except OSError:
            self._logfile = subprocess.DEVNULL  # type: ignore[assignment]

        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self._proc = subprocess.Popen(
                args, cwd=str(LLAMA_BIN_DIR),
                stdout=self._logfile, stderr=subprocess.STDOUT,
                env=child_env(), creationflags=creationflags,
            )
        except Exception as e:
            self._load_error = str(e)
            self._close_log()
            logger.error("Failed to spawn embedding server: %s", _safe_err(e))
            raise

        deadline = time.monotonic() + EMBED_SERVER_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                self._close_log()
                self._proc = None
                self._load_error = "embedding llama-server exited during startup (see llama-server.embed.log)"
                raise RuntimeError(self._load_error)
            try:
                if requests.get(f"{self._base_url}/health", timeout=2).status_code == 200:
                    self._ready = True
                    logger.info("EmbeddingGemma server healthy on %s", self._base_url)
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        self.unload()
        self._load_error = f"embedding server not healthy within {EMBED_SERVER_STARTUP_TIMEOUT:.0f}s"
        raise RuntimeError(self._load_error)

    # ── inference ────────────────────────────────────────────────────────────
    def embed(self, texts: list[str], task: str = "document", dims: int | None = None) -> list[list[float]]:
        """Return L2-normalized EmbeddingGemma vectors for `texts`.

        `task` selects the query/document prompt prefix. `dims` (< 768) requests a
        Matryoshka truncation (re-normalized). Empty input → [].
        """
        if not texts:
            return []
        prefix = PROMPTS.get(task, PROMPTS["document"])
        prompted = [prefix(t) for t in texts]

        self._ensure_loaded()
        r = requests.post(
            f"{self._base_url}/v1/embeddings",
            json={"model": "embeddinggemma", "input": prompted},
            timeout=EMBED_SERVER_REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()["data"]
        # llama-server may reorder by "index" — sort to be safe, then extract.
        rows = sorted(data, key=lambda d: d.get("index", 0))
        vectors = [row["embedding"] for row in rows]

        if dims is not None and 0 < dims < EMBED_NATIVE_DIMS:
            vectors = [_truncate_normalize(v, dims) for v in vectors]
        return vectors

    @property
    def is_loaded(self) -> bool:
        proc = self._proc
        return bool(self._ready and proc is not None and proc.poll() is None)

    def _close_log(self) -> None:
        if self._logfile not in (None, subprocess.DEVNULL):
            try:
                self._logfile.close()  # type: ignore[union-attr]
            except Exception:
                pass
        self._logfile = None

    def unload(self) -> None:
        proc = self._proc
        self._proc = None
        self._ready = False
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._close_log()
        logger.info("EmbeddingGemma server stopped")


def _truncate_normalize(vec: list[float], dims: int) -> list[float]:
    """Matryoshka truncation to `dims` + L2 re-normalization."""
    sliced = vec[:dims]
    norm = math.sqrt(sum(x * x for x in sliced)) or 1.0
    return [x / norm for x in sliced]
