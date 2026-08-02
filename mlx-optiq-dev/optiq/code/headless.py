"""Headless OptiQ Code — the `optiq code -p "goal"` path.

The same agent loop with auto-approve: no interactive gating, run to completion,
print a result, and set the exit code (0 iff the agent declared done). This is
both a real feature (scripting / CI / a `claude -p` analog) and the backbone of
the end-to-end tests (design §8, layers 2 & 6).

Ported from conjure/agent/headless.py, dropping the contract/propose params.
"""
from __future__ import annotations

from .loop import AgentResult, run_agent


async def run_headless(
    repo,
    goal: str,
    engine,
    *,
    verify_command: str | None = None,
    max_turns: int = 20,
    wall_s: float | None = None,
    max_tokens: int = 8192,
    max_tool_output: int | None = None,
    compact_at: int | None = None,
    log=None,
    _clock=None,
) -> AgentResult:
    """Run the loop non-interactively (auto-approve every tool) toward `goal`."""
    kw = {}
    if _clock is not None:
        kw["_clock"] = _clock
    if max_tool_output is not None:
        kw["max_tool_output"] = max_tool_output
    return await run_agent(
        engine=engine, repo=repo, goal=goal, verify_command=verify_command,
        log=log or (lambda _m: None), max_turns=max_turns, wall_s=wall_s,
        max_tokens=max_tokens, compact_at=compact_at, **kw,
    )
