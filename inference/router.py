"""
Synapse LLM Router — E2B ⇄ E4B
==============================
Picks which fine-tune answers a request. EQUAL roles, not a quality ladder:

  "e2b" — the "junior": short, simple, conversational prompts. The default.
  "e4b" — the "mid":    long or technically-hard prompts.

The decision is a pure, dependency-free heuristic (no model, no network, no I/O)
so it costs nothing and is fully unit-testable with plain strings. Signals that
escalate a prompt simple → complex (E4B):

  • length      — char count ≥ ROUTER_CHAR_THRESHOLD
  • verbosity   — word count ≥ ROUTER_WORD_THRESHOLD
  • code        — a fenced ``` block or obvious code punctuation density
  • keywords    — a bilingual RU/EN list of "hard ask" markers

Any single signal is enough (logical OR) — we'd rather spend the bigger model on
a borderline-complex prompt than under-serve it. Everything is env-overridable via
config.ROUTER_* (see config.py).
"""

from __future__ import annotations

import re

from config import (
    ROUTER_CHAR_THRESHOLD,
    ROUTER_COMPLEX_KEYWORDS,
    ROUTER_DEFAULT_VARIANT,
    ROUTER_ENABLED,
    ROUTER_WORD_THRESHOLD,
)

# The two valid variant tokens. Kept here so callers (llm.py, main.py) share one
# source of truth and a typo can't silently create a third "variant".
VARIANT_E2B = "e2b"
VARIANT_E4B = "e4b"
VALID_VARIANTS = (VARIANT_E2B, VARIANT_E4B)

# Built-in bilingual "this is a hard/technical ask" markers. Used when
# ROUTER_COMPLEX_KEYWORDS (env override) is empty. Lowercase; matched as
# substrings/words case-insensitively. Deliberately conservative — these should
# signal genuine depth, not merely mention a topic.
_DEFAULT_COMPLEX_KEYWORDS: tuple[str, ...] = (
    # EN — reasoning / engineering depth. NOTE: deliberately NO bare interrogatives
    # ("why", "how") — a short casual "why X?" is E2B territory; only MULTI-WORD
    # depth markers ("step by step", "explain why") or strong engineering verbs escalate.
    "debug", "refactor", "architecture", "algorithm", "optimize", "optimise",
    "compare", "trade-off", "tradeoff", "prove", "derive", "analyze", "analyse",
    "step by step", "explain why", "explain how", "design a", "implement",
    "benchmark", "complexity", "regex", "typescript",
    "python", "rust", "async", "concurrency", "vulnerability", "exploit",
    # RU — те же намерения (без голого «почему»/«как» — их ловит только длина)
    "отладь", "отладка", "рефактор", "архитектур", "алгоритм", "оптимизир",
    "сравни", "компромисс", "докажи", "выведи", "проанализируй", "разбери",
    "по шагам", "объясни почему", "объясни как", "спроектируй", "реализуй",
    "сложность", "уязвимост", "асинхрон",
)

# Obvious code / structured-input punctuation. A high density of these in a short
# message still means "the user pasted code" → route to the stronger model.
_CODE_HINT = re.compile(r"```|;\s*\n|\{\s*\n|=>|::|</?\w+>|def |function |class |import ")


def _keywords() -> tuple[str, ...]:
    """Active keyword list — env override if provided, else the built-in bilingual set."""
    return tuple(ROUTER_COMPLEX_KEYWORDS) if ROUTER_COMPLEX_KEYWORDS else _DEFAULT_COMPLEX_KEYWORDS


def _normalize_variant(variant: str) -> str:
    """Clamp an arbitrary string to a valid variant, defaulting to E2B."""
    v = (variant or "").strip().lower()
    return v if v in VALID_VARIANTS else VARIANT_E2B


def is_complex(message: str) -> bool:
    """True if the heuristic judges `message` complex enough to warrant E4B.

    Pure function over the single (latest) user message. History is intentionally
    ignored: routing is per-turn and the current ask is what must be served well.
    """
    if not message:
        return False

    text = message.strip()
    if len(text) >= ROUTER_CHAR_THRESHOLD:
        return True

    if len(text.split()) >= ROUTER_WORD_THRESHOLD:
        return True

    if _CODE_HINT.search(text):
        return True

    lowered = text.lower()
    return any(kw in lowered for kw in _keywords())


def route(message: str) -> str:
    """Return the variant token ("e2b" | "e4b") that should answer `message`.

    When the router is disabled (ROUTER_ENABLED=false) every request goes to the
    configured default variant — no switching, no heuristic.
    """
    if not ROUTER_ENABLED:
        return _normalize_variant(ROUTER_DEFAULT_VARIANT)
    if is_complex(message):
        return VARIANT_E4B
    return _normalize_variant(ROUTER_DEFAULT_VARIANT)
