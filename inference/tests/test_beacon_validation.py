"""
The beacon path must be exactly as strict as the JSON path.
===========================================================
`POST /api/conversations` accepts two shapes: a normal JSON request validated by
`ConversationSaveRequest`, and a `text/plain` sendBeacon body used during page
unload, which the browser can send without a CORS preflight. The beacon path
used to parse the body by hand, and the hand-written version was a WEAKER subset
of the model it stood in for:

  * `createdAt` / `updatedAt` reached the SQLite bind with no type and no bounds.
    The JSON route caps them near the year 2100; the beacon accepted anything, so
    a conversation stamped far enough in the future was immune to the 30-day
    retention purge — a permanent record in a store whose entire policy is that
    it does not keep one.
  * Every message item went through unvalidated. An unexpected `role` passes a
    dict but violates the DB's CHECK constraint: an uncaught IntegrityError, a
    500, and a route that answers with a stack instead of a refusal.
  * `title` was sliced with `[:200]`, which does not fail on a list — it returns
    a shorter list.

Two validators for one shape is how the second one ends up weaker. There is one
model now, and these tests exist so it stays one.

Run from inference/:
    ./.venv/Scripts/python.exe -m pytest tests/test_beacon_validation.py -q
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

import main  # noqa: E402

MODEL = main.ConversationSaveRequest
# Roughly the year 2100 — the bound the JSON route has always had.
MAX_TS = 4102444800000

GOOD = {
    "id": "abc-123",
    "title": "Real chat",
    "createdAt": 1,
    "updatedAt": 2,
    "messages": [{"id": "m1", "role": "user", "content": "hi", "timestamp": 3}],
}


def rejects(payload: dict) -> bool:
    try:
        MODEL.model_validate(payload)
        return False
    except ValidationError:
        return True


class TestTheModelItself(unittest.TestCase):
    def test_a_legitimate_conversation_is_accepted(self) -> None:
        # Guard against "reject everything" passing the tests below.
        self.assertFalse(rejects(GOOD))

    def test_a_far_future_timestamp_cannot_buy_immunity_from_the_purge(self) -> None:
        self.assertTrue(rejects({**GOOD, "updatedAt": MAX_TS + 1}))
        self.assertTrue(rejects({**GOOD, "updatedAt": 99999999999999999}))
        self.assertFalse(rejects({**GOOD, "updatedAt": MAX_TS}), "the bound itself must pass")

    def test_negative_timestamps_are_refused(self) -> None:
        self.assertTrue(rejects({**GOOD, "createdAt": -1}))
        self.assertTrue(rejects({**GOOD, "updatedAt": -1}))

    def test_a_role_the_database_would_refuse_is_refused_here_first(self) -> None:
        # The DB CHECK is the backstop; reaching it means a 500 instead of a 400.
        for role in ("root", "system_", "", "USER", "assistant\n"):
            self.assertTrue(
                rejects({**GOOD, "messages": [{**GOOD["messages"][0], "role": role}]}),
                f"role {role!r} would reach the SQLite CHECK constraint",
            )
        for role in ("user", "assistant", "system"):
            self.assertFalse(rejects({**GOOD, "messages": [{**GOOD["messages"][0], "role": role}]}))

    def test_wrong_types_do_not_slip_through_a_slice(self) -> None:
        # `title[:200]` silently succeeds on a list — it returns a shorter list.
        # Validation does not.
        self.assertTrue(rejects({**GOOD, "title": ["x"]}))
        self.assertTrue(rejects({**GOOD, "messages": "not a list"}))
        self.assertTrue(rejects({**GOOD, "updatedAt": "tomorrow"}))

    def test_the_bound_survives_type_coercion(self) -> None:
        """The property that matters, stated as a property rather than a type check.

        Pydantic runs in lax mode, so a numeric STRING is coerced to an int
        rather than refused — which is fine and is not what the purge-immunity
        hole was about. What must hold is that coercion happens FIRST and the
        bound is applied to the result, so no encoding of a huge number gets a
        conversation past the retention cap.
        """
        for sneaky in ("99999999999999999", 9.9e17, MAX_TS + 1):
            self.assertTrue(
                rejects({**GOOD, "updatedAt": sneaky}),
                f"updatedAt={sneaky!r} got past the retention bound",
            )
        # And whatever does get through is an int inside the range.
        parsed = MODEL.model_validate({**GOOD, "updatedAt": "12345"})
        self.assertIsInstance(parsed.updatedAt, int)
        self.assertLessEqual(parsed.updatedAt, MAX_TS)

    def test_an_id_carrying_control_characters_is_refused(self) -> None:
        for bad in ("a\nb", "a b", "../etc/passwd", "a\x00b", ""):
            self.assertTrue(rejects({**GOOD, "id": bad}), f"id {bad!r} accepted")

    def test_the_message_list_is_bounded(self) -> None:
        many = [{"id": f"m{i}", "role": "user", "content": "x", "timestamp": 0} for i in range(101)]
        self.assertTrue(rejects({**GOOD, "messages": many}))


class TestTheRouteUsesIt(unittest.TestCase):
    """A model nothing calls is decoration — check the beacon actually validates."""

    def setUp(self) -> None:
        src = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
        start = src.index("async def save_conversation_beacon")
        end = src.index("\n@app.", start)
        self.route = src[start:end]

    def test_the_beacon_validates_with_the_shared_model(self) -> None:
        self.assertIn("ConversationSaveRequest.model_validate", self.route)

    def test_the_beacon_no_longer_hand_rolls_the_fields(self) -> None:
        # The exact shapes that were weaker than the model.
        for gone in ('data.get("createdAt"', 'data.get("updatedAt"', 'data.get("title", "New Chat")'):
            self.assertNotIn(
                gone,
                self.route,
                f"{gone} is back — that field reached the DB without the model's bounds",
            )

    def test_the_token_is_removed_before_validation(self) -> None:
        # It is transport, not conversation data; leaving it in fails the model.
        self.assertIn('data.pop("idToken"', self.route)
        self.assertLess(
            self.route.index('data.pop("idToken"'),
            self.route.index("ConversationSaveRequest.model_validate"),
            "the token must be popped before the body is validated",
        )

    def test_only_the_message_list_is_leniently_truncated(self) -> None:
        # Deliberate: this runs during unload, where a 422 reaches nobody. Every
        # OTHER field is rejected rather than trimmed. And the slice keeps the
        # NEWEST hundred (`[-100:]`), not the oldest — see test_hardening_batch3.
        self.assertIn('data["messages"][-100:]', self.route)
        self.assertNotIn(
            'data["messages"][:100]',
            self.route,
            "the beacon keeps the FIRST 100 messages, dropping the newest",
        )
        self.assertIsNone(
            re.search(r'\[:200\]\s*,?\s*$', self.route, re.M),
            "a field other than messages is still being silently truncated",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
