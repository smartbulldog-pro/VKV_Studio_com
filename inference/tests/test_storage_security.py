"""
Iter 2 (storage/IDOR) regression tests — cross-tenant conversation isolation.
=============================================================================
Locks the fix for the cross-owner takeover in ChatStorage.save_conversation:
one client_ip must never overwrite/destroy a conversation owned by another.

Run from inference/:
    ./.venv/Scripts/python.exe -m unittest tests.test_storage_security -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chat_storage import ChatStorage, ConversationOwnershipError  # noqa: E402

IP_A = "10.0.0.1"
IP_B = "10.0.0.2"


def _conv(cid: str, title: str, content: str) -> dict:
    return {
        "id": cid,
        "title": title,
        "createdAt": 1,
        "updatedAt": 1,
        "messages": [{"id": f"{cid}-m0", "role": "user", "content": content, "timestamp": 1}],
    }


class StorageSecurityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db = Path(tempfile.mkdtemp()) / "chats.db"
        self.store = ChatStorage(self.db)
        self.store.init()

    def test_cross_owner_write_rejected_and_data_preserved(self) -> None:
        # Victim (IP-A) saves a conversation.
        self.store.save_conversation(IP_A, _conv("abc-123", "victim", "victim-secret"))

        # Attacker (IP-B) tries to overwrite the SAME id → must be refused.
        with self.assertRaises(ConversationOwnershipError):
            self.store.save_conversation(IP_B, _conv("abc-123", "pwned", "attacker-content"))

        # Victim's data is fully intact (title + messages unchanged).
        conv = self.store.get_conversation(IP_A, "abc-123")
        self.assertIsNotNone(conv)
        self.assertEqual(conv["title"], "victim")
        self.assertEqual(conv["messages"][0]["content"], "victim-secret")

        # Attacker still cannot read it either (owner scoping on read).
        self.assertIsNone(self.store.get_conversation(IP_B, "abc-123"))

    def test_same_owner_update_still_works(self) -> None:
        self.store.save_conversation(IP_A, _conv("x1", "first", "one"))
        # Same owner updates title + messages — allowed.
        self.store.save_conversation(IP_A, _conv("x1", "second", "two"))
        conv = self.store.get_conversation(IP_A, "x1")
        self.assertEqual(conv["title"], "second")
        self.assertEqual(conv["messages"][0]["content"], "two")

    def test_new_id_saves_normally(self) -> None:
        self.store.save_conversation(IP_B, _conv("new-1", "hello", "hi"))
        conv = self.store.get_conversation(IP_B, "new-1")
        self.assertIsNotNone(conv)
        self.assertEqual(conv["title"], "hello")

    def test_attacker_cannot_delete_victim_messages(self) -> None:
        # Regression for the DELETE-FROM-messages part of the exploit.
        self.store.save_conversation(IP_A, _conv("keep", "t", "important"))
        with self.assertRaises(ConversationOwnershipError):
            # Attacker sends an empty-messages payload for victim's id.
            self.store.save_conversation(IP_B, {
                "id": "keep", "title": "wipe", "createdAt": 1, "updatedAt": 2, "messages": [],
            })
        conv = self.store.get_conversation(IP_A, "keep")
        self.assertEqual(len(conv["messages"]), 1, "victim messages must not be deleted")
        self.assertEqual(conv["messages"][0]["content"], "important")

    def test_two_ips_independent_namespaces(self) -> None:
        # Distinct ids per owner coexist and stay isolated.
        self.store.save_conversation(IP_A, _conv("a-only", "A", "aaa"))
        self.store.save_conversation(IP_B, _conv("b-only", "B", "bbb"))
        self.assertIsNone(self.store.get_conversation(IP_B, "a-only"))
        self.assertIsNone(self.store.get_conversation(IP_A, "b-only"))
        self.assertEqual(len(self.store.list_conversations(IP_A)), 1)
        self.assertEqual(len(self.store.list_conversations(IP_B)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
