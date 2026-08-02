"""The OptiQ Code agent loop.

Ported from conjure/tui/implement_loop.py, **stripped** of the done-gate,
contract/property machinery, anti-vacuous proof control, and repo-regression
oracle (all retired — verification was proven not to help). Completion is the
industry-standard one: the agent takes tool actions until it declares `done`,
hits `max_turns` / a wall-clock cap, or is stopped. The human reviews the diff.

What is KEPT is the operational robustness for weak models (the edge, design §3.3):
never emit an empty patch (salvage `git diff` on every exit path), edit-application
resilience, ANSI-strip/format robustness, stall detection + force-write, focused
context reset, and an append-only trajectory that a context reset can't erase.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from .approval import ToolCall
from .exec_env import as_executor
from .tools import (MUTATING, TOOLS, execute_tool, parse_pytest, truncate,
                    _DEFAULT_MAX_TOOL_OUTPUT, _SUBPROC_ENV)

_EXPLORE_LIMIT = 6          # read-only turns before we force an edit
_EDIT_FAIL_LIMIT = 2        # failed edit_file on one path before we nudge a rewrite
_DUP_CALL_LIMIT = 2        # identical read-only tool calls before we stop re-running them

_SYSTEM = """\
You are OptiQ Code, a coding agent working in a repository. Use the tools to make
changes and to inspect the project — do not narrate an action instead of taking
it. Keep edits minimal and focused, and keep the repo's existing tests passing.

You can run any shell command with the `bash` tool, and the user approves each
one before it runs. So DO the work yourself: if something needs a command run —
a script, a git or CLI operation, submitting or checking a job — run it with
`bash`. Never end by printing a command and telling the user to run it: if you
can run it, run it. Only ask the user to act when it genuinely requires something
you cannot do (credentials you don't have, a decision only they can make).

Make each edit small and targeted: use edit_file for a focused change rather than
rewriting a whole file with write_file. One tool call that emits too much text is
cut off at the output-token limit and cannot be applied, so split a large change
into several small edits rather than one big one.

Workflow:
  1. Explore only as much as you need (search / read_file), then edit
     (edit_file for a targeted change, write_file to create or rewrite a file).
  2. Run the tests (run_tests) to check your work.
  3. When you are finished — or when the user just asked a question — reply with a
     short final answer and stop. Do NOT call a tool in that turn: a plain reply
     with no tool call ends your turn, exactly like a chat. Calling done is
     optional (a way to signal completion after edits); it is never required, and
     you should never call it just to answer a question.

verify command: {verify}
"""


@dataclass
class AgentResult:
    succeeded: bool             # the agent declared done
    turns: int
    passed: int
    failed: int
    summary: str
    stop_reason: str = "max_turns"   # done | answered | max_turns | wall_clock | no_tools | truncated | error
    patch: str = ""             # git diff of what changed — never empty if code changed
    patch_bytes: int = 0
    messages: list = field(default_factory=list)   # full append-only trajectory
    turn_latencies: list = field(default_factory=list)   # wall-clock per model call (s)


# ─── helpers (robustness / grounding) ────────────────────────────────────────

_MAX_TRUNC = 3   # consecutive output-limit truncations before the loop gives up


def _valid_json(s) -> bool:
    """True if ``s`` parses as JSON — used to tell a complete tool call from one
    severed mid-generation at the output-token limit (whose arguments are a
    partial, unparseable JSON string)."""
    if not s:
        return False
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def _load_project_guidance(repo: str, max_chars: int = 8000) -> str:
    """Read a project's agent guidance so the user can steer the agent per-repo
    (conventions, edit style, gotchas) without editing OptiQ's own system prompt.
    ``AGENTS.md`` is the standard (and what ``/init`` writes); ``CLAUDE.md`` is
    honored as a fallback so a repo already set up for Claude Code works with no
    extra file. Checked at the repo root then ``.optiq/``/``.claude/``; first hit
    wins. Size-capped and best-effort — a missing or unreadable file just yields
    no guidance. This is the user-tunable layer: OptiQ's system prompt stays
    minimal and correct, project opinions live here."""
    import os
    for rel in ("AGENTS.md", ".optiq/AGENTS.md", "agents.md",
                "CLAUDE.md", ".claude/CLAUDE.md"):
        p = os.path.join(repo, rel)
        try:
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as fh:
                    return fh.read(max_chars).strip()
        except OSError:
            pass
    return ""


def _tool_calls_from_text(content: str):
    """Salvage a tool call a weak model wrote as TEXT instead of a structured
    call — a ```json {..}``` block or a bare {"name":..,"arguments":..} object
    (design §3.3 format robustness). Only used when the response has no
    structured tool_calls. Conservative: returns [] unless it clearly parses."""
    import re
    from types import SimpleNamespace
    if not content:
        return []
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    blob = m.group(1) if m else None
    if not blob:
        m = re.search(r'\{[^{}]*"name"\s*:[^{}]*"arguments"\s*:.*?\}\s*\}?', content, re.DOTALL)
        blob = m.group(0) if m else None
    if not blob:
        return []
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return []
    fn = obj.get("function") or {}
    name = obj.get("name") or fn.get("name")
    if not name or not isinstance(name, str):
        return []
    args = obj.get("arguments", fn.get("arguments", obj.get("parameters", {})))
    args_str = args if isinstance(args, str) else json.dumps(args)
    return [SimpleNamespace(id="healed-1", type="function",
                            function=SimpleNamespace(name=name, arguments=args_str))]

def _approx_tokens(messages) -> int:
    """Rough token count for the message window. ~4 chars/token is close enough
    to decide when to compact; the alternative is a tokenizer round-trip per
    turn, which is not worth it for a threshold check."""
    n = 0
    for m in messages:
        c = m.get("content")
        n += len(c) if isinstance(c, str) else 0
        tc = m.get("tool_calls")
        if tc:
            n += len(json.dumps(tc))
    return n // 4


_COMPACTED = "[compacted: earlier tool output dropped to free context]"
_COMPACTED_PREFIX = "[compacted:"


def _placeholder(call: dict | None, content: str) -> str:
    """What replaces a dropped tool result.

    Deliberately a *description*, not a summary. Summarizing would mean another
    call to the same weak local model at the exact moment context is tightest,
    and a wrong summary is worse than an empty one: it puts confident, false
    detail in front of the agent, where a placeholder is honestly blank.

    For a coding agent the dropped content is almost always re-fetchable -- file
    text, search hits, test output -- so naming the call that produced it is
    strictly more useful than paraphrasing it. The agent can just run the tool
    again. (That is the difference from a chat assistant, where compacted
    conversation is gone for good and a summary is the only option.)
    """
    lines = content.count("\n") + 1 if content else 0
    what = ""
    if call:
        fn = (call.get("function") or {})
        name = fn.get("name")
        if name:
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = None
            target = ""
            if isinstance(args, dict):
                for key in ("path", "file", "pattern", "query", "cmd", "command"):
                    if args.get(key):
                        target = f" {key}={str(args[key])[:60]!r}"
                        break
            what = f" from {name}{target},"
    return (f"{_COMPACTED_PREFIX} output{what} {lines} lines, {len(content)} bytes "
            f"— dropped to free context; run the tool again if you still need it]")


def _compact(messages: list, keep_recent: int = 8) -> tuple[list, int]:
    """Shrink the window by dropping the bulk of OLD tool output.

    An agent loop grows monotonically: every tool result is appended verbatim
    and re-sent on every subsequent turn. On a local model that ends one of two
    ways -- the prompt outgrows the served context, or a long generation on top
    of a near-resident model trips a Metal OOM and takes the server down. Both
    are what a user actually hits on a long task, and neither produced anything
    actionable before this existed.

    What gets dropped is old ``tool`` message *content* -- the bulk by a wide
    margin, and the least useful to keep verbatim once the agent has acted on
    it. The message *structure* is preserved exactly: same roles in the same
    order, so families that enforce strict alternation (Mistral, Devstral) stay
    valid, and every tool_call keeps its matching tool response.

    The system prompt, the goal, and the last ``keep_recent`` messages are never
    touched -- recent output is what the model is actually working from.

    Returns ``(new_messages, dropped)`` where ``dropped`` holds the *originals*
    of exactly the messages whose content was replaced. The caller archives
    those, so the full trajectory survives even though the model can no longer
    see it. An already-compacted message is skipped (its content is the short
    placeholder by then), so repeated compaction never archives anything twice.
    """
    if len(messages) <= keep_recent + 2:
        return messages, []
    head, tail = messages[:2], messages[-keep_recent:]
    middle = messages[2:-keep_recent]
    # tool_call_id -> the call that produced it, so the placeholder can name it.
    calls = {}
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            if isinstance(tc, dict) and tc.get("id"):
                calls[tc["id"]] = tc
    out, dropped = list(head), []
    for m in middle:
        if m.get("role") == "tool" and isinstance(m.get("content"), str) \
                and not m["content"].startswith(_COMPACTED_PREFIX):
            dropped.append(m)
            m = {**m, "content": _placeholder(calls.get(m.get("tool_call_id")),
                                              m["content"])}
        out.append(m)
    out.extend(tail)
    return out, dropped


def _restore(messages: list, archive: dict) -> list:
    """Put compacted tool output back, in place.

    Compaction preserves message structure exactly, so a tool result keeps its
    ``tool_call_id`` through the process and the original content can be slotted
    straight back. The result is the conversation as it actually happened:
    valid to replay when a session resumes, and complete when it is exported.
    """
    if not archive:
        return list(messages)
    out = []
    for m in messages:
        original = archive.get(m.get("tool_call_id"))
        if original is not None and isinstance(m.get("content"), str) \
                and m["content"].startswith(_COMPACTED_PREFIX):
            m = {**m, "content": original}
        out.append(m)
    return out


def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def _looks_repeated(cur: str, prev: str) -> bool:
    if not cur or not prev:
        return False
    return cur == prev or (len(cur) > 80 and cur[:140] == prev[:140])


def _solution_sig(repo) -> str:
    """Content hash of non-test source — tells whether a turn changed code."""
    import hashlib
    ex = as_executor(repo)
    if not ex.is_local:
        rc, out = ex.run(
            "find . -name '*.py' -not -name 'test_*' -not -name '*_test*' "
            "-not -path '*/__pycache__/*' -printf '%p %s %T@\\n' 2>/dev/null | sort", timeout=120)
        return hashlib.sha256(out.encode()).hexdigest() if rc == 0 else ""
    h = hashlib.sha256()
    try:
        for p in sorted(ex.root.rglob("*.py")):
            if "_test" in p.name or p.name.startswith("test_"):
                continue
            if any(part in (".git", ".pytest_cache", "__pycache__") for part in p.parts):
                continue
            try:
                h.update(p.name.encode()); h.update(p.read_bytes())
            except OSError:
                pass
    except OSError:
        pass
    return h.hexdigest()


def _ensure_git_baseline(repo) -> None:
    """Make sure there's a git baseline so `git diff` can salvage the agent's
    changes on any exit path. If the repo is not git-backed, initialize it and
    commit the current state as the baseline."""
    ex = as_executor(repo)
    rc, _ = ex.run("git rev-parse --is-inside-work-tree", timeout=15)
    if rc == 0:
        return
    ex.run("git init -q", timeout=30, env=_SUBPROC_ENV)
    ex.run("git add -A", timeout=120, env=_SUBPROC_ENV)
    ex.run('git -c user.email=code@optiq.local -c user.name="OptiQ Code" '
           'commit -q -m "optiq-code baseline" --no-verify', timeout=120, env=_SUBPROC_ENV)


def _salvage_patch(repo, paths=None) -> str:
    """The diff of what the AGENT changed this run.

    ``paths`` is the set of files the agent actually wrote/edited. Diffing only
    those is what makes the patch reflect the agent's work rather than the user's
    pre-existing uncommitted state: a project folder inside a monorepo routinely
    has dozens of unrelated dirty files, and an answer-only turn (a question)
    changes nothing at all — an unscoped ``git diff`` would report all of that as
    "the agent's changes". With ``paths`` empty the agent changed nothing, so the
    patch is empty. ``paths=None`` (legacy callers) falls back to the launch
    folder (``-- .``); still scoped off the whole monorepo, since git add/diff
    ignore the shell cwd."""
    import shlex
    ex = as_executor(repo)
    if paths is not None:
        specs = [p for p in paths if p]
        if not specs:
            return ""                       # the agent touched no files
        spec = " ".join(shlex.quote(p) for p in specs)
        ex.run(f"git add -A -N -- {spec}", timeout=60, env=_SUBPROC_ENV)
        rc, out = ex.run(f"git diff -- {spec}", timeout=120, env=_SUBPROC_ENV)
        return out if rc == 0 else ""
    ex.run("git add -A -N -- .", timeout=60, env=_SUBPROC_ENV)   # intent-to-add: new files show in diff
    rc, out = ex.run("git diff -- .", timeout=120, env=_SUBPROC_ENV)
    return out if rc == 0 else ""


def _run_tests_now(repo, verify_command, max_out):
    out = execute_tool("run_tests", {}, repo, verify_command, max_out)
    p, f = parse_pytest(out)
    return p, f, out


async def _direct(fn):
    return fn()


# ─── the loop ────────────────────────────────────────────────────────────────

async def run_agent(
    *,
    engine,
    repo,
    goal: str,
    verify_command: str | None = None,
    approve: Callable[[ToolCall], Awaitable[bool]] = None,
    log: Callable[[str], None] = lambda _m: None,
    run_sync: Callable[[Callable], Awaitable] = _direct,
    on_usage: Callable[[dict], None] = lambda _u: None,
    on_progress: Callable[[float, int], None] = lambda elapsed, tokens: None,
    on_reasoning: Callable[[str], None] = lambda _s: None,
    on_assistant_text: Callable[[str], None] = None,
    on_edit: Callable[[str, str, str], Awaitable[None]] = None,
    max_turns: int = 20,
    wall_s: float | None = None,
    repo_context: str = "",
    max_tokens: int = 8192,
    max_tool_output: int = _DEFAULT_MAX_TOOL_OUTPUT,
    compact_at: int | None = None,
    prior_messages: list | None = None,
    _clock: Callable[[], float] = time.monotonic,
) -> AgentResult:
    """Drive the agent loop and return an AgentResult (with the salvaged patch).

    ``prior_messages`` continues a resumed session: the new goal is appended to
    the loaded trajectory (which already carries the system prompt + prior turns)
    so the model keeps its context.

    ``compact_at`` is the approximate token count past which old tool output is
    compacted out of the window; callers get it from
    ``Config.resolve_compact_at(engine.context_window())``, so it tracks the
    model actually being served. ``None`` disables compaction, which is what
    happens when the server does not report a window and no explicit budget was
    configured -- better to leave the window alone than to invent a threshold.
    """
    if approve is None:
        async def approve(_call):  # default: auto-approve (headless)
            return True
    if on_edit is None:
        async def on_edit(_name, _path, _code):  # default: no reveal (headless)
            return None

    # Route blocking git/pytest through run_sync so the loop never freezes the
    # UI event loop when it drives the TUI (run_sync=_direct in headless).
    await run_sync(lambda: _ensure_git_baseline(repo))
    system = _SYSTEM.format(verify=verify_command or "pytest -q")
    guidance = _load_project_guidance(repo)
    if guidance:
        system += ("\n\n--- Project guidance (from AGENTS.md — the user's "
                   "conventions for this repo; follow it) ---\n" + guidance)
    user = f"GOAL:\n{goal}\n\nPROJECT CONTEXT:\n{repo_context[:4000]}"
    # history is append-only and always starts empty; it captures the message
    # window at each context reset and at the end (so a resumed prefix is not
    # duplicated when messages already begins with it).
    history: list = []
    # Original content of tool results that compaction replaced, keyed by
    # tool_call_id, so the saved trajectory can be restored to a COMPLETE and
    # still-valid conversation. Concatenating the originals ahead of the window
    # instead produces orphan tool messages and a system prompt in the middle,
    # which a resumed session then replays.
    archive: dict = {}
    if prior_messages:
        messages = list(prior_messages) + [{"role": "user", "content": user}]
    else:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]

    base_p, base_f, _ = await run_sync(
        lambda: _run_tests_now(repo, verify_command, max_tool_output))
    if base_p or base_f:            # quiet when the repo has no tests (0/0 is noise)
        log(f"baseline: {base_p} passed, {base_f} failed")
    touched: set = set()            # files the agent actually wrote/edited this run

    def say(text: str) -> None:
        """Deliver a harness message without breaking strict-alternation templates.

        Mistral and Devstral reject a ``user`` turn that directly follows tool
        results — after tool output they expect the assistant to speak. Qwen and
        Gemma tolerate it, which is why every harness nudge here used to be a
        bare user message. Attaching the text to the preceding tool result keeps
        the shape valid against any endpoint (including stock mlx-lm and vLLM,
        which OptiQ Code can also be pointed at) and invents no assistant turn.
        """
        prev = messages[-1] if messages else None
        if (isinstance(prev, dict) and prev.get("role") == "tool"
                and isinstance(prev.get("content"), str)):
            prev["content"] = f"{prev['content']}\n\n{text}" if prev["content"] else text
        else:
            messages.append({"role": "user", "content": text})

    final = AgentResult(False, 0, base_p, base_f, "did not converge")
    seen_tool_calls: dict[tuple[str, str], int] = {}
    no_tool_streak = 0
    trunc_streak = 0
    prev_norm = ""
    repeat_streak = 0
    best_passed = base_p
    stale_turns = 0
    explore_streak = 0
    wrote_yet = False
    edit_fails: dict[str, int] = {}
    turn_latencies: list = []
    start = _clock()
    # Cumulative output tokens across every model call this user-turn drives, so
    # the status line reads like Claude Code (elapsed + total tokens for the turn)
    # instead of resetting to the latest sub-step's count on each read/edit/test.
    _turn_tokens = [0]

    try:
        for turn in range(1, max_turns + 1):
            final.turns = turn
            if wall_s is not None and (_clock() - start) > wall_s:
                final.stop_reason = "wall_clock"
                final.summary = "wall-clock cap reached"
                break

            _t0 = _clock()
            # Compact before the call, not after a failure: the whole point is
            # to stay under the limit rather than recover from hitting it.
            if compact_at and _approx_tokens(messages) > compact_at:
                messages, dropped = _compact(messages)
                # Key the archive by tool_call_id, not position: compaction
                # preserves structure exactly, so the id identifies each
                # dropped message unambiguously after the window shifts.
                for _m in dropped:
                    if _m.get("tool_call_id"):
                        archive[_m["tool_call_id"]] = _m.get("content")
                if dropped:
                    log(f"  ⎿ compacted context: dropped {len(dropped)} old tool "
                        f"result(s), now ~{_approx_tokens(messages)} tokens")
            # Stream so the UI can show live progress (elapsed · tokens) instead
            # of a frozen screen while a local model generates a long response.
            # Throttle the callback so per-token deltas don't flood the UI.
            _last = [0.0]
            _cur = [0]              # this model call's running output-token count
            def _tok(ntok: int) -> None:
                _cur[0] = ntok
                now = _clock()
                if now - _last[0] >= 0.15:
                    _last[0] = now
                    # elapsed and tokens both cumulative for the whole user-turn
                    on_progress(now - start, _turn_tokens[0] + ntok)
            resp = await run_sync(lambda: engine.chat(
                messages, tools=TOOLS, max_tokens=max_tokens, on_token=_tok,
                on_reasoning=on_reasoning))
            _turn_tokens[0] += _cur[0]      # bank this call before the next sub-step
            turn_latencies.append(_clock() - _t0)
            _report_usage(engine, resp, on_usage)
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                # Format robustness: a weak model may write the tool call as text.
                tool_calls = _tool_calls_from_text(msg.content or "")

            # Output-limit truncation. A turn that finishes with reason "length"
            # has hit max_tokens; when it did so mid tool call the call arrives
            # with no usable arguments (unparseable partial JSON), so the server
            # drops it. Undetected this is indistinguishable from "made no tool
            # call", and the model retries the same oversized edit until the
            # no-tool cap trips — a silent multi-minute stall. Detect it, coach a
            # smaller edit, and cap the retries so it fails loud instead.
            finish_reason = getattr(resp.choices[0], "finish_reason", None)
            if finish_reason == "length" and (
                    not tool_calls
                    or not _valid_json(tool_calls[-1].function.arguments)):
                trunc_streak += 1
                if trunc_streak >= _MAX_TRUNC:
                    final.stop_reason = "truncated"
                    final.summary = ("edits kept exceeding the output-token limit "
                                     "— raise max_tokens or split the change into "
                                     "smaller steps")
                    log(f"stopped: tool call truncated at the {max_tokens}-token "
                        f"limit {trunc_streak}x — make smaller edits or raise "
                        "max_tokens (config: max_tokens)")
                    break
                log(f"  ⎿ response hit the {max_tokens}-token limit mid tool call "
                    f"— asking for a smaller edit (attempt {trunc_streak})")
                messages.append({"role": "assistant", "content": msg.content or ""})
                say("Your previous response was cut off at the output-token limit "
                    "before you finished a tool call, so nothing was applied. Make "
                    "a much SMALLER, targeted edit — change one function or a few "
                    "lines with edit_file — then continue with the next edit.")
                continue
            trunc_streak = 0

            messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name,
                                             "arguments": tc.function.arguments}}
                               for tc in tool_calls]})
            if msg.content:
                if on_assistant_text is not None:
                    on_assistant_text(msg.content.strip())   # full text, rendered
                else:
                    log(msg.content.strip()[:300])
            norm = _norm(msg.content)
            repeat_streak = repeat_streak + 1 if _looks_repeated(norm, prev_norm) else 0
            prev_norm = norm

            if not tool_calls:
                answer = (msg.content or "").strip()
                if answer:
                    # A final text answer with no tool call ends the turn — the
                    # model is finished, exactly like a chat reply (Claude-Code
                    # style). No `done` tool is required; the text (already shown
                    # above) is the response. This is what makes a plain question
                    # ("what does this repo do?") get an answer and stop, instead
                    # of being nagged to keep calling tools.
                    final.stop_reason = "answered"
                    final.succeeded = True
                    final.summary = answer
                    break
                # Empty content AND no tool call is a genuine stall (a weak model
                # that put everything in reasoning, or produced nothing) — coach
                # once, then give up rather than spin.
                no_tool_streak += 1
                if no_tool_streak >= 2:
                    final.stop_reason = "no_tools"
                    final.summary = "agent stopped without finishing"
                    break
                say("You returned nothing. Either use a tool (search/read_file/"
                    "edit_file/write_file/bash/run_tests) to make progress, or "
                    "reply with your answer to finish.")
                continue
            no_tool_streak = 0

            sig_before = _solution_sig(repo)
            turn_passed = None
            finished = False
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == "done":
                    p, f, out = await run_sync(lambda: _run_tests_now(repo, verify_command, max_tool_output))
                    final.passed, final.failed = p, f
                    final.succeeded = True
                    final.stop_reason = "done"
                    final.summary = args.get("summary", "done")
                    # The agent's answer/summary is the response — surface it
                    # (a question is answered here, not via a file change).
                    if (on_assistant_text is not None and final.summary
                            and final.summary != "done"):
                        on_assistant_text(final.summary)
                    if p or f:          # quiet the 0/0 line for no-test repos
                        log(f"done — {p} passed, {f} failed")
                    finished = True
                    break

                if name in MUTATING:
                    ok = await approve(ToolCall(name, args))
                    if not ok:
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": "REJECTED by the user. Do not retry this; try another approach."})
                        continue

                # Weak quants get stuck re-issuing one identical call (measured:
                # 18 consecutive `find . -name '*test*.py'` on a 4-bit 24B).
                # Re-running is pure turn burn when nothing has changed, so the
                # answer cannot differ.
                #
                # The test that matters is "has the repo changed since", NOT
                # whether the tool is mutating: the call that actually looped was
                # `bash`, which is in MUTATING because it needs approval, not
                # because repeating it is meaningful. Any edit clears the history
                # below, so a legitimate re-run after a change is never blocked.
                sig = (name, json.dumps(args, sort_keys=True, default=str))
                seen_tool_calls[sig] = seen_tool_calls.get(sig, 0) + 1
                if seen_tool_calls[sig] > _DUP_CALL_LIMIT:
                    log(f"  ⎿ skipped duplicate {name} (x{seen_tool_calls[sig]})")
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": (
                            f"[harness] You already ran this exact {name} call "
                            f"{seen_tool_calls[sig] - 1}x and nothing in the repo has "
                            "changed since, so the result would be identical and it was "
                            "not run again. Use what you already have: make an edit, or "
                            "call done."),
                    })
                    continue

                log(f"⏺ {ToolCall(name, args).summary(72)}")
                if name in ("write_file", "edit_file"):
                    _p = args.get("path") or args.get("file_path")
                    if _p:
                        touched.add(_p)     # for an agent-scoped diff at the end
                    # Reveal the code being written so the terminal shows the edit
                    # take shape (the model buffers the tool call, so this is the
                    # first the UI sees of it), instead of only "wrote N bytes".
                    await on_edit(name, args.get("path", ""),
                                  args.get("content") or args.get("new") or "")
                result = await run_sync(lambda: execute_tool(name, args, repo, verify_command, max_tool_output))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

                # Compact result line in the transcript (Claude-Code style marker).
                if name == "run_tests":
                    p, f = parse_pytest(result)
                    turn_passed = p
                    log(f"  ⎿ {p} passed, {f} failed")
                elif result.startswith("ERROR"):
                    log(f"  ⎿ {result.splitlines()[0][:80]}")
                elif name == "read_file":
                    log(f"  ⎿ {len(result.splitlines())} lines")
                elif name in ("bash", "git"):
                    # result is "$ cmd\n[exit N]\n<output>" — show the exit status
                    # and the first real output line, NOT the echoed command (the
                    # ⏺ line above already shows the command).
                    exit_tag, body = "", []
                    for ln in result.splitlines():
                        if ln.startswith("$ "):
                            continue
                        if ln.startswith("[exit"):
                            exit_tag = ln.strip("[]")
                            continue
                        if ln.strip():
                            body.append(ln)
                    snippet = f"  {body[0][:70]}" if body else ""
                    log(f"  ⎿ {exit_tag or 'ok'}{snippet}")
                else:
                    log(f"  ⎿ {(result.splitlines() or ['ok'])[0][:80]}")

                # Edit-application resilience: if edit_file keeps failing on the
                # same path, nudge a full-file rewrite (design §3.3).
                if name == "edit_file" and result.startswith("ERROR"):
                    path = args.get("path", "?")
                    edit_fails[path] = edit_fails.get(path, 0) + 1
                    if edit_fails[path] >= _EDIT_FAIL_LIMIT:
                        edit_fails[path] = 0
                        say(f"[harness] edit_file has failed {_EDIT_FAIL_LIMIT}x on {path}. "
                            f"Stop trying to patch it — read_file it, then use write_file to "
                            f"rewrite the whole file with your change applied.")
                elif name == "edit_file":
                    edit_fails.pop(args.get("path", "?"), None)

            if finished:
                break

            # ── grounding: keep a weak model from editing blind or looping ──
            turn_names = [tc.function.name for tc in tool_calls]
            code_changed = _solution_sig(repo) != sig_before
            ran_tests = "run_tests" in turn_names

            if code_changed:
                wrote_yet, explore_streak = True, 0
                # The repo moved, so every earlier result is potentially stale —
                # re-running a command that was a duplicate a moment ago is now
                # legitimate (re-reading an edited file, re-running the tests).
                seen_tool_calls.clear()
            elif not any(n in MUTATING for n in turn_names):
                explore_streak += 1
            if explore_streak >= _EXPLORE_LIMIT and not ran_tests:
                explore_streak = 0
                log(f"{_EXPLORE_LIMIT} turns of exploration with no edit — forcing a write")
                say((
                    f"You have spent {_EXPLORE_LIMIT} turns reading without changing any code. "
                    f"Stop exploring. Make your best edit now with edit_file or write_file"
                    f"{'' if wrote_yet else ' — you have not written anything yet'}, then call "
                    f"run_tests. An imperfect edit you can test beats more reading."))

            if code_changed and not ran_tests:
                p, f, out = await run_sync(lambda: _run_tests_now(repo, verify_command, max_tool_output))
                final.passed, final.failed = p, f
                turn_passed = p
                log(f"auto-verify after edit → {p} passed, {f} failed")
                if f == 0:
                    say(f"[harness] Ran the tests after your edit: all {p} pass. If the goal is complete, call done.")
                else:
                    say(truncate(
                        f"[harness] I ran the tests after your edit so you are not working blind: "
                        f"{p} passed, {f} failed. Look at the specific assertion failures below and "
                        f"make a targeted fix — do not rewrite the whole file from memory.\n{out}",
                        max_tool_output))

            # ── stall detection: progress, not just text ──
            if turn_passed is not None and turn_passed > best_passed:
                best_passed, stale_turns = turn_passed, 0
            else:
                stale_turns += 1
            if (stale_turns >= 4 or repeat_streak >= 2):
                log("stuck — nudging a small, specific change")
                say("[harness] You are not making progress. Make one small, specific change "
                    "with edit_file, then run_tests. Trace the failing case by hand first.")
                stale_turns, repeat_streak = 0, 0
    except Exception as e:  # noqa: BLE001 — never lose the patch to a crash
        final.stop_reason = "error"
        final.summary = f"{type(e).__name__}: {e}"
        log(f"error: {final.summary}")
    finally:
        final.patch = await run_sync(lambda: _salvage_patch(repo, touched))
        final.patch_bytes = len(final.patch.encode("utf-8", "replace"))
        # The trajectory is the conversation as it actually happened: the
        # final window with every compacted tool result restored in place.
        # Valid to replay on resume, and complete for export.
        history.extend(_restore(messages, archive))
        final.messages = history
        final.turn_latencies = turn_latencies

    return final


def _report_usage(engine, resp, on_usage) -> None:
    try:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return

        def _n(key: str) -> int:
            v = getattr(usage, key, None)
            if v is None and isinstance(usage, dict):
                v = usage.get(key)
            return int(v) if isinstance(v, (int, float)) else 0

        cost = getattr(usage, "cost", None)
        if cost is None and isinstance(usage, dict):
            cost = usage.get("cost")
        on_usage({
            "cost": float(cost) if isinstance(cost, (int, float)) else 0.0,
            "prompt_tokens": _n("prompt_tokens"),
            "completion_tokens": _n("completion_tokens"),
            "total_tokens": _n("total_tokens"),
        })
    except Exception:
        pass
