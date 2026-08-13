"""
Word Error Rate — pure-Python, dependency-free.
===============================================
WER = (S + D + I) / N_ref  computed via word-level Levenshtein alignment.
No jiwer/rapidfuzz needed — keeps the eval harness runnable anywhere with just
the stdlib. Text is normalized (lowercased, punctuation stripped, whitespace
collapsed) before comparison so scoring isn't dominated by trivial formatting.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Keep letters/digits/whitespace across scripts (Latin + Cyrillic); drop the rest.
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, NFKC-fold, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    # Treat ё/е as distinct (they are in ASR references); do NOT fold them.
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


@dataclass(frozen=True)
class WerResult:
    wer: float          # (S + D + I) / N_ref  (1.0 if ref empty and hyp non-empty)
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int
    hits: int

    def as_pct(self) -> float:
        return round(self.wer * 100, 2)


def word_error_rate(reference: str, hypothesis: str) -> WerResult:
    """Compute WER between a reference and hypothesis transcript."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    n, m = len(ref), len(hyp)

    if n == 0:
        # No reference words: any hypothesis words are pure insertions.
        return WerResult(
            wer=0.0 if m == 0 else 1.0,
            substitutions=0, deletions=0, insertions=m, ref_words=0, hits=0,
        )

    # Levenshtein DP with backtrace over words.
    # dp[i][j] = min edits to turn ref[:i] into hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # substitution
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                )

    # Backtrace to count S / D / I / hits.
    i, j = n, m
    s = d = ins = hits = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            hits += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return WerResult(
        wer=(s + d + ins) / n,
        substitutions=s, deletions=d, insertions=ins, ref_words=n, hits=hits,
    )


def corpus_wer(pairs: "list[tuple[str, str]]") -> WerResult:
    """Aggregate WER over many (reference, hypothesis) pairs (micro-average)."""
    s = d = ins = n = hits = 0
    for ref, hyp in pairs:
        r = word_error_rate(ref, hyp)
        s += r.substitutions
        d += r.deletions
        ins += r.insertions
        n += r.ref_words
        hits += r.hits
    return WerResult(
        wer=(s + d + ins) / n if n else 0.0,
        substitutions=s, deletions=d, insertions=ins, ref_words=n, hits=hits,
    )
