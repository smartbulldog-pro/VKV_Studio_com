"""
Synapse TTS — monthly character budget (cost cap).
==================================================
A tiny persistent counter of characters sent to Google Chirp 3 HD this calendar
month. When usage would exceed the cap we stop calling Google and fall back to
free Edge TTS — so the backend physically cannot leave Google's free 1M/month
tier and can never run up a bill.

Storage is a small JSON file (no DB coupling): {"month": "YYYY-MM", "chars": N}.
It resets automatically when the calendar month rolls over. Thread-safe.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger("synapse.tts.budget")


class TTSBudget:
    """Persistent monthly character budget with an 80% alert.

    Args:
        path: JSON file to persist the counter in.
        monthly_cap: max characters per calendar month. <= 0 disables the cap.
        clock: injectable UTC clock (for tests); defaults to real UTC now.
    """

    def __init__(
        self,
        path: str | Path,
        monthly_cap: int,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._path = Path(path)
        self._cap = int(monthly_cap)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._alerted_80 = False

    # ── internal ─────────────────────────────────────────────────────────────
    def _month_key(self) -> str:
        return self._clock().strftime("%Y-%m")

    def _load(self) -> dict:
        """Read the counter, auto-resetting if the stored month != current month."""
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            data = {}
        month = self._month_key()
        if not isinstance(data, dict) or data.get("month") != month:
            return {"month": month, "chars": 0}
        try:
            chars = int(data.get("chars", 0))
        except (TypeError, ValueError):
            chars = 0
        return {"month": month, "chars": max(0, chars)}

    def _save(self, data: dict) -> None:
        # Atomic write: serialize to a temp file then os.replace() so a crash or a
        # concurrent reader never sees a half-written / corrupt counter file.
        try:
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError as e:
            # Persistence failure must never break synthesis — just log it.
            logger.warning("TTS budget persist failed: %s", e)

    # ── public API ───────────────────────────────────────────────────────────
    @property
    def cap(self) -> int:
        return self._cap

    def usage(self) -> int:
        """Characters used so far this month."""
        with self._lock:
            return self._load()["chars"]

    def remaining(self) -> int:
        if self._cap <= 0:
            return 2**31  # effectively unlimited
        return max(0, self._cap - self.usage())

    def would_exceed(self, n: int) -> bool:
        """True if synthesizing `n` more chars would cross the monthly cap."""
        if self._cap <= 0:
            return False
        with self._lock:
            return self._load()["chars"] + max(0, int(n)) > self._cap

    def add(self, n: int) -> int:
        """Record `n` characters actually sent to Google; returns new total.

        Emits a one-shot WARNING when usage first crosses 80% of the cap.
        """
        n = max(0, int(n))
        with self._lock:
            data = self._load()
            data["chars"] += n
            self._save(data)
            used = data["chars"]
        self._maybe_alert_80(used)
        return used

    def try_reserve(self, n: int) -> bool:
        """Atomically reserve `n` chars against the cap. Returns True if reserved.

        This is the race-free replacement for a separate would_exceed()+add(): the
        cap check and the increment happen under ONE lock acquisition, so N concurrent
        requests can't all pass the check and then overshoot the cap. Reserve BEFORE
        calling Google; refund() if the call fails (nothing was actually billed).
        """
        n = max(0, int(n))
        with self._lock:
            data = self._load()
            if self._cap > 0 and data["chars"] + n > self._cap:
                return False
            data["chars"] += n
            self._save(data)
            used = data["chars"]
        self._maybe_alert_80(used)
        return True

    def refund(self, n: int) -> None:
        """Return a previously reserved `n` chars (e.g. the Google call failed)."""
        n = max(0, int(n))
        if n == 0:
            return
        with self._lock:
            data = self._load()
            data["chars"] = max(0, data["chars"] - n)
            self._save(data)

    def _maybe_alert_80(self, used: int) -> None:
        if self._cap > 0 and not self._alerted_80 and used >= 0.8 * self._cap:
            self._alerted_80 = True
            logger.warning(
                "TTS monthly char budget at %d/%d (>=80%%) — approaching free-tier limit",
                used, self._cap,
            )
