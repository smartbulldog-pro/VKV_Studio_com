"""Every expensive route must be under concurrency admission control.

This test exists because the first admission-control pass covered /api/chat and
/api/chat/stream and left /api/voice/stream, /api/tts and /api/embed unbounded —
while a comment in the source claimed the whole pipeline was covered. An
adversarial review caught it; nothing in CI would have.

It is a source-level check on purpose. The failure mode is *absence* — a route
that never acquires a slot — and absence is invisible to a functional test that
only ever fires one request at a time.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

MAIN = Path(__file__).resolve().parent.parent / "main.py"
SRC = MAIN.read_text(encoding="utf-8")

# Routes that run a model. Cheap metadata routes (health, ready, conversations)
# are deliberately absent — they must NOT hold an inference slot.
EXPENSIVE_ROUTES = [
    "/api/chat",
    "/api/chat/stream",
    "/api/voice",
    "/api/voice/stream",
    "/api/tts",
    "/api/embed",
]


def _route_body(path: str, verb: str = "post") -> str:
    """Source text of one route handler, up to the next @app decorator.

    Matched on `@app.post("/path"` without the closing paren, because several
    decorators carry extra arguments (`response_model=`, `status_code=`) — an
    exact-string match silently missed those.
    """
    marker = f'@app.{verb}("{path}"'
    start = SRC.find(marker)
    if start == -1:
        # Routes registered with @app.api_route(...) instead of a verb shorthand
        # — /api/health does that so it can answer HEAD as well as GET, which
        # FastAPI's APIRoute (unlike a plain Starlette Route) will not add on
        # its own. Look them up by path so this helper keeps finding them.
        start = SRC.index(f'@app.api_route("{path}"')
    nxt = SRC.find("@app.", start + 10)
    return SRC[start : nxt if nxt != -1 else len(SRC)]


@pytest.mark.parametrize("path", EXPENSIVE_ROUTES)
def test_expensive_route_acquires_an_inference_slot(path: str) -> None:
    body = _route_body(path)
    acquires = "_admit_inference(" in body or "_sem.acquire()" in body
    assert acquires, (
        f"{path} runs a model but never takes an inference slot. Rate limiting "
        f"bounds requests per minute, not requests in flight — without a slot a "
        f"handful of concurrent calls can pin the process-wide thread pool and "
        f"stall every other route, including authenticated ones."
    )


@pytest.mark.parametrize("path", ["/api/chat/stream", "/api/voice/stream"])
def test_streaming_routes_release_the_slot_in_a_finally(path: str) -> None:
    """A streaming handler acquires by hand, so the release must be in a finally.

    Without it, a client that simply closes the tab strands a slot forever and a
    few abandoned streams wedge the server shut.
    """
    body = _route_body(path)
    assert "_sem.acquire()" in body, f"{path} should acquire manually (slot outlives the handler)"
    assert re.search(r"finally:\s*\n(?:\s*#[^\n]*\n)*\s*_sem\.release\(\)", body), (
        f"{path} must release its slot in a finally block so a client disconnect "
        f"cannot leak it"
    )


def test_cheap_routes_do_not_hold_an_inference_slot() -> None:
    """Health checks must stay answerable while the model is saturated."""
    for path in ("/api/health", "/api/ready"):
        body = _route_body(path, verb="get")
        assert "_admit_inference(" not in body, (
            f"{path} must not take an inference slot — a saturated model would "
            f"then make the server look down to every monitor and load balancer."
        )
