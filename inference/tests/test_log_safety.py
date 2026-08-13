"""
Iter 1 (secret leakage) regression tests — log_safety.safe_err().
=================================================================
Ensures exceptions logged server-side can't dump a secret value or an unbounded
blob. Client responses use static strings elsewhere; this locks the log channel.

Run from inference/:
    ./.venv/Scripts/python.exe -m unittest tests.test_log_safety -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from log_safety import safe_err  # noqa: E402


class TestSafeErr(unittest.TestCase):
    def test_prefixes_exception_type(self) -> None:
        out = safe_err(ValueError("boom"), secrets=[])
        self.assertTrue(out.startswith("ValueError: "))
        self.assertIn("boom", out)

    def test_redacts_known_secret(self) -> None:
        key = "sk-ant-SUPERSECRET-abc123"
        exc = RuntimeError(f"401 unauthorized for key {key} on request")
        out = safe_err(exc, secrets=[key])
        self.assertNotIn(key, out)
        self.assertIn("<redacted>", out)

    def test_redacts_credential_path(self) -> None:
        path = "/home/user/secrets/sa.json"
        exc = RuntimeError(f"could not open {path}")
        out = safe_err(exc, secrets=[path])
        self.assertNotIn(path, out)

    def test_truncates_long_message(self) -> None:
        exc = RuntimeError("x" * 5000)
        out = safe_err(exc, limit=200, secrets=[])
        # type prefix + 200 chars + ellipsis marker, but never the full 5000
        self.assertLessEqual(len(out), 200 + len("…(truncated)"))
        self.assertIn("(truncated)", out)

    def test_empty_secret_list_is_noop(self) -> None:
        out = safe_err(KeyError("missing"), secrets=[])
        self.assertIn("KeyError", out)

    def test_none_secret_entries_ignored(self) -> None:
        # A None/empty entry in the list must not blow up or match.
        out = safe_err(RuntimeError("plain message"), secrets=["", "nomatch"])
        self.assertIn("plain message", out)

    def test_redacts_multiple_secrets(self) -> None:
        exc = RuntimeError("key AAA and path BBB both leaked")
        out = safe_err(exc, secrets=["AAA", "BBB"])
        self.assertNotIn("AAA", out)
        self.assertNotIn("BBB", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
