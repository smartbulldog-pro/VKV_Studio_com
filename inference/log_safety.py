"""
log_safety.py — secret-safe exception rendering for server logs.
================================================================
Provider SDK / model exceptions can embed request echoes, absolute local paths,
or (in pathological cases) auth material. Never log a raw exception verbatim:
route it through `safe_err()`, which returns "<ExcType>: <message>" with any
known secret value redacted and the whole string truncated.

Client responses already use static generic strings (never the raw exception),
so this only hardens SERVER-SIDE log hygiene — defense in depth.
"""

from __future__ import annotations

import os


def _load_secret_values() -> list[str]:
    """Secret env values that must never appear in a log line."""
    return [
        v for v in (
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("SYNAPSE_GOOGLE_TTS_CREDENTIALS"),
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
        if v
    ]


# Captured once at import (env is fixed for the process lifetime).
_SECRET_VALUES = _load_secret_values()

# Env var NAMES holding provider secrets. The local llama-server children need NONE
# of these; stripping them from the child environment shrinks the blast radius if the
# third-party binary ever dumps its env (crash handler, /proc/<pid>/environ on a
# shared host, a debug flag).
_SECRET_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "SYNAPSE_GOOGLE_TTS_CREDENTIALS",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


def child_env() -> dict:
    """A copy of the current environment with provider secrets removed, for spawning
    the llama-server subprocesses (they need PATH/CUDA vars but never the API keys)."""
    return {k: v for k, v in os.environ.items() if k not in _SECRET_ENV_KEYS}


def safe_err(exc: object, limit: int = 200, secrets: "list[str] | None" = None) -> str:
    """Render an exception for logs without leaking secrets.

    Args:
        exc: the exception (or any object) to render.
        limit: max characters of the rendered message (truncated with an ellipsis).
        secrets: override the redaction list (defaults to module-level env secrets);
                 mainly for tests.
    """
    secret_values = _SECRET_VALUES if secrets is None else secrets
    msg = f"{type(exc).__name__}: {exc}"
    for secret in secret_values:
        if secret and secret in msg:
            msg = msg.replace(secret, "<redacted>")
    if len(msg) > limit:
        msg = msg[:limit] + "…(truncated)"
    return msg


def oneline(value: object, limit: int = 200) -> str:
    """Sanitize a user/model-controlled value for a single-line log entry.

    Escapes CR/LF/TAB so an attacker can't inject newlines to forge additional log
    lines (log forging / SIEM poisoning), and truncates. Use for ANY request- or
    model-derived string interpolated into a log with %s (ids, transcripts, titles).
    """
    s = str(value).replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    if len(s) > limit:
        s = s[:limit] + "…"
    return s
