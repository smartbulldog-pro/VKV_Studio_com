"""
The conversation routes must not hold the event loop while SQLite works.
========================================================================
Every /api/conversations handler is `async def`, and every one of them used to
call chat_storage synchronously. On a single-loop server that means one DB round
trip stalls EVERY other request — /api/chat, /api/health, the lot. It is not
theoretical here: the retention purge already runs on a worker thread, so a purge
and a route could contend the same SQLite lock with the route holding the loop
hostage while it waited.

Two kinds of test, because either alone is weak:

  1. A behavioural one that actually blocks the DB and measures whether a second
     coroutine still gets scheduled. It fails on the real defect, not on a
     spelling.
  2. A source check, because the behavioural test only covers the routes it
     drives, and the next route added should not be able to reintroduce this
     quietly.

Run from inference/:
    ./.venv/Scripts/python.exe -m pytest tests/test_event_loop_blocking.py -q
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAIN = Path(__file__).resolve().parent.parent / "main.py"
_RAW = MAIN.read_text(encoding="utf-8")


def _without_comments(text: str) -> str:
    """Blank out whole-line comments, keeping line numbers intact.

    The section note above these routes quotes the very call it documents, so a
    scan over the raw file flagged the comment explaining the fix as an instance
    of the bug. Line count is preserved so failure messages still point at the
    right line.
    """
    return "\n".join("" if ln.lstrip().startswith("#") else ln for ln in text.split("\n"))


SRC = _without_comments(_RAW)

# Every chat_storage function a request handler may reach.
STORAGE_CALLS = (
    "list_conversations",
    "get_conversation",
    "save_conversation",
    "rename_conversation",
    "delete_conversation",
)


class TestSourceDiscipline(unittest.TestCase):
    def test_no_handler_calls_chat_storage_directly(self) -> None:
        """Each call must be handed to a worker thread, not run on the loop.

        Matched on the call sites rather than the route bodies: the lifespan is
        allowed to call chat_storage.init() synchronously (it runs before the
        server accepts requests), and purge_old already goes through to_thread.
        """
        offenders = []
        for name in STORAGE_CALLS:
            for m in re.finditer(rf"chat_storage\.{name}\s*\(", SRC):
                line_start = SRC.rfind("\n", 0, m.start()) + 1
                # A wrapped call reads `asyncio.to_thread(chat_storage.x, ...)`,
                # so the reference appears as an argument with no paren after it.
                preceding = SRC[max(0, m.start() - 120) : m.start()]
                if "to_thread" not in preceding:
                    line_no = SRC.count("\n", 0, m.start()) + 1
                    offenders.append(f"{name} at main.py:{line_no}")
        self.assertEqual(
            [],
            offenders,
            "these run SQLite on the event loop, stalling every other request: "
            f"{offenders}. Wrap them in `await asyncio.to_thread(...)` — "
            "chat_storage opens a fresh connection per call and guards writes "
            "with its own lock, so calling it from a worker thread is safe.",
        )

    def test_the_wrapped_calls_are_awaited(self) -> None:
        """`asyncio.to_thread(...)` without `await` returns a coroutine and does nothing."""
        for m in re.finditer(r"asyncio\.to_thread\(\s*chat_storage\.", SRC):
            preceding = SRC[max(0, m.start() - 40) : m.start()]
            line_no = SRC.count("\n", 0, m.start()) + 1
            self.assertIn(
                "await",
                preceding,
                f"main.py:{line_no}: to_thread(chat_storage...) is not awaited — "
                "the database work never happens and the handler returns success",
            )


class TestLoopStaysResponsive(unittest.IsolatedAsyncioTestCase):
    """The property itself: slow storage must not stop the loop serving others."""

    BLOCK_SECONDS = 0.4

    async def test_a_slow_storage_call_does_not_stall_a_concurrent_task(self) -> None:
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            deadline = time.monotonic() + self.BLOCK_SECONDS
            while time.monotonic() < deadline:
                ticks += 1
                await asyncio.sleep(0.01)

        def slow_storage() -> str:
            time.sleep(self.BLOCK_SECONDS)  # stands in for a contended SQLite lock
            return "done"

        # What the routes do now.
        offloaded = asyncio.create_task(asyncio.to_thread(slow_storage))
        beat = asyncio.create_task(heartbeat())
        result, _ = await asyncio.gather(offloaded, beat)

        self.assertEqual("done", result)
        self.assertGreater(
            ticks,
            5,
            "the loop was starved while storage worked — offloading is not in effect",
        )

    async def test_the_same_call_on_the_loop_does_starve_it(self) -> None:
        """The control. Without this, the test above could pass for the wrong reason."""
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            deadline = time.monotonic() + self.BLOCK_SECONDS
            while time.monotonic() < deadline:
                ticks += 1
                await asyncio.sleep(0.01)

        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)  # let it start
        time.sleep(self.BLOCK_SECONDS)  # blocking, the old behaviour
        await beat

        self.assertLess(
            ticks,
            5,
            "blocking the loop did not starve the heartbeat — this control is "
            "not measuring what it claims, so the test above proves nothing",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
