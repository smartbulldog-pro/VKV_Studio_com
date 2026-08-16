"""
Three small hardening fixes, each an audit finding a skeptic confirmed real.
============================================================================
All three were downgraded to low severity but reproduced, and all three have a
one-to-few-line fix. They are tested together because they share the beacon /
conversation-store surface.

  1. The beacon kept the OLDEST 100 messages, not the newest — `[:100]` where
     the comment above it and chat_storage's own policy both say "the last
     hundred". A user who wrote more than 100 messages had the ones they just
     typed dropped and the ones they opened with kept.

  2. `messages.id` is a GLOBAL primary key, and `ConversationMessageItem.id`
     defaults to "" — so `m.get("id", fallback)` never used its fallback and a
     blank id was stored under the shared key "". The next owner to save an
     id-less message hit `UNIQUE constraint failed` and lost their save.

  3. FastAPI's default validation handler reflects the raw request body back in
     the 422. A canary posted to /api/chat came back verbatim, unmetered by the
     rate limiter (validation runs before the handler body). A static handler
     removes both the reflection and the amplification.

Run from inference/:
    ./.venv/Scripts/python.exe -m pytest tests/test_hardening_batch3.py -q
"""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chat_storage as cs  # noqa: E402


class TestBeaconKeepsNewest(unittest.TestCase):
    def test_the_beacon_slice_takes_the_last_hundred(self) -> None:
        src = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
        start = src.index("async def save_conversation_beacon")
        end = src.index("\n@app.", start)
        route = src[start:end]
        self.assertIn('data["messages"][-100:]', route)
        self.assertNotIn(
            'data["messages"][:100]',
            route,
            "the beacon still keeps the FIRST 100 messages, dropping the newest",
        )


class TestMessageIdKeyspace(unittest.TestCase):
    """Two owners, blank message ids: neither may block the other's save."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.store = cs.ChatStorage(str(Path(self._dir.name) / "t.db"))
        self.store.init()

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _conv(self, conv_id: str, msg_id: str) -> dict:
        return {
            "id": conv_id,
            "title": "t",
            "createdAt": 0,
            "updatedAt": 0,
            "messages": [{"id": msg_id, "role": "user", "content": "hi", "timestamp": 0}],
        }

    def test_two_owners_can_both_save_a_message_with_no_id(self) -> None:
        # The exact squat: a blank id used to land on the global key "".
        self.store.save_conversation("sub:attacker", self._conv("a-conv", ""))
        # The victim's blank-id save must not collide with the attacker's.
        self.store.save_conversation("sub:victim", self._conv("v-conv", ""))
        self.assertEqual(1, len(self.store.list_conversations("sub:victim")))
        self.assertIsNotNone(self.store.get_conversation("sub:victim", "v-conv"))

    def test_blank_ids_within_one_conversation_do_not_collide(self) -> None:
        conv = {
            "id": "c1",
            "title": "t",
            "createdAt": 0,
            "updatedAt": 0,
            "messages": [
                {"id": "", "role": "user", "content": "one", "timestamp": 0},
                {"id": "", "role": "assistant", "content": "two", "timestamp": 1},
            ],
        }
        self.store.save_conversation("sub:x", conv)
        got = self.store.get_conversation("sub:x", "c1")
        self.assertEqual(2, len(got["messages"]))

    def test_an_explicit_id_still_round_trips(self) -> None:
        self.store.save_conversation("sub:x", self._conv("c1", "real-uuid-123"))
        got = self.store.get_conversation("sub:x", "c1")
        self.assertEqual("real-uuid-123", got["messages"][0]["id"])


class TestValidationHandlerIsStatic(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"fastapi TestClient unavailable: {e}")
        import main

        self.client = TestClient(main.app)

    def test_a_malformed_body_is_not_reflected(self) -> None:
        canary = "CANARY-do-not-echo-9c3f1a"
        r = self.client.post(
            "/api/chat",
            content=canary,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(422, r.status_code)
        self.assertNotIn(canary, r.text, "the request body was echoed into the 422")
        self.assertEqual({"detail": "Invalid request"}, r.json())

    def test_a_nested_canary_is_not_reflected(self) -> None:
        r = self.client.post("/api/embed", json={"texts": [{"pii": "CANARY-nested-7788"}]})
        self.assertEqual(422, r.status_code)
        self.assertNotIn("CANARY-nested-7788", r.text)


class TestHandlerIsRegistered(unittest.TestCase):
    def test_the_source_registers_a_validation_handler(self) -> None:
        src = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
        self.assertRegex(
            src,
            r"@app\.exception_handler\(\s*RequestValidationError\s*\)",
            "no RequestValidationError handler — FastAPI's default reflects the body",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
