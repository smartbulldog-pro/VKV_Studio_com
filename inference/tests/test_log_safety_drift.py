"""
The redaction list must not drift from the credentials the app actually reads.
=============================================================================
`log_safety._SECRET_ENV_KEYS` does two jobs: it names the values `safe_err()`
redacts from log lines, and it names the variables `child_env()` strips before
spawning llama-server. Both were maintained by hand, and both had fallen behind:
`SYNAPSE_GOOGLE_STT_CREDENTIALS` and `SYNAPSE_LLM_REMOTE_CREDENTIALS` are read by
config.py and appeared in neither, so a path identifying a key file could reach a
log and was handed to every child process.

A list maintained by hand drifts. This test reads config.py and fails when a
credential-shaped env var appears there that log_safety does not know about, so
the NEXT one cannot be forgotten the way these two were.

Run from inference/:
    ./.venv/Scripts/python.exe -m pytest tests/test_log_safety_drift.py -q
"""

from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import log_safety  # noqa: E402

# Words that mark an env var as naming or holding auth material. Deliberately
# broad: a false positive costs one line in the list, a false negative costs a
# leaked credential.
CREDENTIAL_WORDS = ("KEY", "TOKEN", "SECRET", "CREDENTIAL", "PASSWORD", "PASSWD")

# Env vars whose names match the words above but are demonstrably not secrets.
# Each needs a reason; "it looked fine" is not one.
ALLOWED_NON_SECRETS = {
    # Token BUDGETS and limits — integers about model context, not auth.
    "SYNAPSE_MAX_TOKENS",
    "SYNAPSE_MIN_GEN_TOKENS",
    "SYNAPSE_TOKEN_SAFETY_MARGIN",
    "SYNAPSE_TOKENIZE_DAILY_CAP",
    "SYNAPSE_TOKENIZE_BUDGET_FILE",
    # A comma-separated list of routing keywords.
    "SYNAPSE_ROUTER_KEYWORDS",
}


def _env_vars_config_reads() -> set:
    src = (Path(__file__).resolve().parent.parent / "config.py").read_text(encoding="utf-8")
    return set(re.findall(r'os\.getenv\(\s*"([A-Z0-9_]+)"', src))


class TestNoDrift(unittest.TestCase):
    def test_every_credential_env_var_is_known_to_log_safety(self) -> None:
        known = set(log_safety._SECRET_ENV_KEYS)
        suspicious = {
            name
            for name in _env_vars_config_reads()
            if any(w in name for w in CREDENTIAL_WORDS) and name not in ALLOWED_NON_SECRETS
        }
        missing = suspicious - known
        self.assertEqual(
            set(),
            missing,
            "config.py reads credential-shaped env vars that log_safety does not know "
            f"about: {sorted(missing)}. Add them to _SECRET_ENV_KEYS — that one tuple "
            "drives BOTH log redaction and child-process env stripping — or, if one is "
            "genuinely not a secret, add it to ALLOWED_NON_SECRETS with a reason.",
        )

    def test_the_two_that_were_missing_are_covered(self) -> None:
        # Named explicitly so a careless edit to the heuristic above cannot quietly
        # drop the exact pair this test was written for.
        for name in ("SYNAPSE_GOOGLE_STT_CREDENTIALS", "SYNAPSE_LLM_REMOTE_CREDENTIALS"):
            self.assertIn(name, log_safety._SECRET_ENV_KEYS)


class TestRedactionAndStripping(unittest.TestCase):
    """One tuple, two jobs — check both actually happen."""

    SENTINEL = "/etc/vkvstudio/UNIQUE-SENTINEL-VALUE.json"

    def _with_env(self, name: str):
        import importlib

        prev = os.environ.get(name)
        os.environ[name] = self.SENTINEL
        try:
            return importlib.reload(log_safety)
        finally:
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev

    def tearDown(self) -> None:
        import importlib

        importlib.reload(log_safety)

    def test_a_secret_value_never_survives_safe_err(self) -> None:
        for name in log_safety._SECRET_ENV_KEYS:
            mod = self._with_env(name)
            rendered = mod.safe_err(Exception(f"could not open {self.SENTINEL}"), limit=500)
            self.assertNotIn(
                self.SENTINEL, rendered, f"{name}'s value reached a log line verbatim"
            )

    def test_a_secret_var_is_never_handed_to_a_child_process(self) -> None:
        for name in log_safety._SECRET_ENV_KEYS:
            mod = self._with_env(name)
            self.assertNotIn(
                name, mod.child_env(), f"{name} was passed to a llama-server child"
            )

    def test_an_ordinary_env_var_is_left_alone(self) -> None:
        # The stripping must be surgical: children still need PATH and friends.
        self.assertIn("PATH", {k.upper(): v for k, v in log_safety.child_env().items()})


if __name__ == "__main__":
    unittest.main(verbosity=2)
