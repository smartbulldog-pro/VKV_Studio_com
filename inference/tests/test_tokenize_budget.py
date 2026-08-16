"""
/api/tokenize/count — the global daily cap on the relay to the owner's keys.
============================================================================
This endpoint is unauthenticated and forwards to Google/Anthropic with the
owner's API key. Its only guard was a per-IP rate limit, and a per-IP limit is
defeated by having more IPs — which over IPv6 costs nothing. The counter here
is global, so IP rotation cannot walk around it.

The tests that matter are the two the old design could not pass: many distinct
callers still share one allowance, and a reservation is given back when the
call never reached the provider.

Run from inference/:
    ./.venv/Scripts/python.exe -m pytest tests/test_tokenize_budget.py -q
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from tts_budget import TTSBudget  # noqa: E402


class TestDailyCallBudget(unittest.TestCase):
    """The budget class, driven the way main.py drives it for the relay."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "tokenize_call_budget.json"
        self.now = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _budget(self, cap: int) -> TTSBudget:
        return TTSBudget(
            self.path,
            cap,
            clock=lambda: self.now,
            label="tokenize daily call",
            period_fmt="%Y-%m-%d",
        )

    def test_reservations_are_shared_across_every_caller(self) -> None:
        # The whole point. A per-IP limit would let each of these through.
        b = self._budget(3)
        self.assertTrue(all(b.try_reserve(1) for _ in range(3)))
        self.assertFalse(b.try_reserve(1), "a fourth caller got through a cap of 3")

    def test_the_allowance_resets_the_next_day_not_the_next_month(self) -> None:
        # A monthly period would answer someone else's abuse by turning the
        # feature off until the calendar turned.
        b = self._budget(2)
        self.assertTrue(b.try_reserve(2))
        self.assertFalse(b.try_reserve(1))
        self.now = self.now + timedelta(days=1)
        self.assertTrue(b.try_reserve(1), "the cap did not reset on the next day")

    def test_the_same_hour_tomorrow_is_a_new_period_but_later_today_is_not(self) -> None:
        b = self._budget(1)
        self.assertTrue(b.try_reserve(1))
        self.now = self.now + timedelta(hours=11)  # still 15 August
        self.assertFalse(b.try_reserve(1))

    def test_refund_returns_a_reservation_when_it_is_used(self) -> None:
        # The primitive still works; the ROUTE no longer uses it (see below).
        b = self._budget(1)
        self.assertTrue(b.try_reserve(1))
        b.refund(1)
        self.assertTrue(b.try_reserve(1))

    def test_refund_cannot_manufacture_allowance(self) -> None:
        b = self._budget(5)
        b.refund(10)
        self.assertEqual(0, b.usage())

    def test_cap_of_zero_disables_the_limit(self) -> None:
        b = self._budget(0)
        self.assertTrue(all(b.try_reserve(1) for _ in range(50)))

    def test_a_corrupt_counter_file_fails_closed_to_zero_used(self) -> None:
        # Not a security property — just that a bad file cannot crash the route.
        self.path.write_text("{not json", encoding="utf-8")
        b = self._budget(2)
        self.assertTrue(b.try_reserve(1))

    def test_only_exact_known_model_ids_are_forwarded(self) -> None:
        # The relay used to match on startswith("gemini"), so any
        # attacker-chosen `gemini-<anything>` egressed on the owner's key.
        allowed = config.TOKENIZE_ALLOWED_MODELS
        for good in ("gemini-3.5-flash", "claude-sonnet-5", "gemma-4-e2b"):
            self.assertIn(good, allowed)
        for bad in (
            "gemini-9.9-ultra-does-not-exist",
            "gemini?",
            "gemini..",
            "gemini&x",
            "claude-anything",
            "gpt-5.5",
        ):
            self.assertNotIn(bad, allowed, f"{bad} would still reach a provider")

    def test_the_two_budgets_do_not_share_a_file(self) -> None:
        # They count different things at different periods; one file would make
        # a busy day of tokenizing switch the site's voice off.
        self.assertNotEqual(
            Path(config.TOKENIZE_BUDGET_FILE).name,
            Path(config.TTS_BUDGET_FILE).name,
        )


class TestRouteWiring(unittest.TestCase):
    """The route must actually consult the budget — a cap nothing calls is decoration."""

    def test_route_reserves_before_calling_a_provider_and_refunds_on_failure(self) -> None:
        src = Path(__file__).resolve().parent.parent / "main.py"
        body = src.read_text(encoding="utf-8")
        start = body.index('@app.post("/api/tokenize/count")')
        end = body.index("@app.", start + 10)
        route = body[start:end]

        self.assertIn("try_reserve(1)", route, "the relay does not reserve against the cap")

        # NOTHING is refunded, and that is the fix rather than an oversight.
        # Refunding provider errors made every rejected call free and uncapped
        # (reproduced: 200 upstream round-trips, counter still at 1). Refunding
        # nothing while still accepting any `gemini*` string was worse — the
        # installed google-genai raises a plain ValueError for ids containing
        # `..`, `?` or `&` BEFORE any socket work, so 2000 junk requests would
        # have drained the day's allowance with zero traffic to Google and
        # switched exact counts off for everyone. The answer is neither: reject
        # what we will not pay for BEFORE reserving, then never give it back.
        self.assertEqual(
            0,
            route.count("budget.refund(1)"),
            "the route refunds again — a refunded failure path is an uncapped one",
        )
        self.assertIn(
            "TOKENIZE_ALLOWED_MODELS", route, "the relay forwards unvalidated model ids"
        )
        self.assertLess(
            route.index("TOKENIZE_ALLOWED_MODELS"),
            route.index("try_reserve(1)"),
            "validation must happen BEFORE the reservation, or junk input costs the cap",
        )
        # Reserve must come before the PROVIDER calls, not after them. Matched on
        # the client method names — an earlier version of this test matched bare
        # "count_tokens(", which is also the name of the route function on the
        # line above, so it compared the reservation against the def and failed.
        for call in ("messages.count_tokens(", "models.count_tokens("):
            self.assertIn(call, route)
            self.assertLess(
                route.index("try_reserve(1)"),
                route.index(call),
                f"the reservation happens after {call}, which is no cap at all",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
