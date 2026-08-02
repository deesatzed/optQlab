"""`optiq code` command handlers.

Phase 1 wires the CLI surface and the session home; the TUI (Phase 4), the agent
loop (Phase 2), and the server client (Phase 3) fill in behind these entry points.
Kept import-light so `optiq --help` and `optiq code --help` never import Textual
or a model stack.
"""
from __future__ import annotations

import os
from pathlib import Path


def code_home() -> Path:
    """`~/.optiq/code` — session transcripts live under `sessions/`."""
    home = Path(os.environ.get("OPTIQ_HOME") or (Path.home() / ".optiq")) / "code"
    (home / "sessions").mkdir(parents=True, exist_ok=True)
    return home


def sessions_dir() -> Path:
    return code_home() / "sessions"


def run_code(
    *,
    path: str = ".",
    do_resume: bool = False,
    resume_id: str | None = None,
    print_goal: str | None = None,
    model: str | None = None,
    no_color: bool = False,
    auto_approve: bool = False,
) -> int:
    """Launch OptiQ Code in ``path``.

    Phase 1: prints the banner and reports the resolved surface. Interactive TUI,
    headless run, and session resume are implemented in later phases behind this
    same entry point.
    """
    from .banner import banner_text

    repo = Path(path).resolve()
    if not repo.is_dir():
        raise SystemExit(f"optiq code: not a directory: {repo}")

    # One resolution of every setting, one precedence order: flag > env >
    # repo config > user config > default (see optiq.code.config).
    from .config import load as load_config
    cfg, warnings = load_config(
        repo, model=model, color=(False if no_color else None))
    for w in warnings:
        print(f"  ! config: {w}")

    color = cfg.color and os.environ.get("NO_COLOR") is None
    print(banner_text(repo_name=repo.name, model=cfg.model, color=color))

    sessions_dir()  # ensure the session home exists

    if print_goal is not None:
        return _run_headless_cli(repo=repo, goal=print_goal, cfg=cfg)

    return _launch_tui(repo=repo, cfg=cfg, do_resume=do_resume, resume_id=resume_id,
                       auto_approve=auto_approve)


def _make_engine(handle, cfg):
    """Build the engine from the *repo-aware* resolved config.

    The engine also loads config on its own, but only without a repo -- so the
    repo tier (``<repo>/.optiq/code.json``) reaches it only if we pass every
    value in from the config resolved here (which did see the repo). base_url
    and model_id come off the ServerHandle because attach/spawn may have
    overridden them.
    """
    from .engine import OptiqEngine
    return OptiqEngine(
        model_id=handle.model_id, base_url=handle.base_url,
        api_key=cfg.api_key, reasoning_effort=cfg.reasoning_effort,
        context_window=cfg.context_window, request_timeout=cfg.request_timeout,
        max_retries=cfg.max_retries)


def _announce_compaction(cfg, engine, *, quiet: bool = False) -> int | None:
    """Resolve the compaction budget from the server's real window.

    Defaulting to a hardcoded token count would be wrong in both directions:
    too low on a model served at 128k (compacting away context that fits), too
    high on one the server capped to 4096 (compacting only after the prompt is
    already rejected). So ask the server what it will accept and take
    ``compact_headroom`` of it.

    ``quiet`` suppresses the stdout line: the interactive TUI takes over the
    screen right after this, so a pre-TUI print just sits as noise above the app
    (the same detail is available on demand via the TUI's ``/model`` command).
    Headless keeps the line -- a script/CI log wants it.
    """
    window = engine.context_window()
    budget = cfg.resolve_compact_at(window)
    if quiet:
        return budget
    if budget and not cfg.compact_at:
        print(f"  [context] server window {window} tokens; compacting old tool "
              f"output past ~{budget}")
    elif budget:
        print(f"  [context] compacting old tool output past ~{budget} (configured)")
    elif window is None:
        print("  [context] server did not report a window — compaction off "
              "(set compact_at to enable)")
    return budget


def _resolve_session(store, *, do_resume: bool, resume_id: str | None):
    """Return (session, resumed_bool). New session unless a resume was asked for
    and one exists."""
    if do_resume:
        if resume_id:
            sess = store.load(resume_id)
            if sess.path.is_file():
                return sess, True
            print(f"  no session '{resume_id}' in this repo — starting fresh.")
        else:
            latest = store.latest()
            if latest is not None:
                return store.load(latest.id), True
            print("  no previous session in this repo — starting fresh.")
    return store.new(), False


def _launch_tui(*, repo, cfg, do_resume: bool = False,
                resume_id: str | None = None, auto_approve: bool = False) -> int:
    """Interactive launch: attach/spawn a server, then run the Textual TUI."""
    from .approval import ApprovalMode
    from .server import NoModelError, ensure_server
    from .sessions import SessionStore
    from .tui import OptiqCodeApp

    try:
        handle = ensure_server(model=cfg.model, base_url=cfg.base_url)
    except NoModelError as e:
        print(f"\n{e}")
        return 2
    session, _ = _resolve_session(SessionStore(repo), do_resume=do_resume, resume_id=resume_id)
    engine = _make_engine(handle, cfg)
    compact_at = _announce_compaction(cfg, engine, quiet=True)
    try:
        OptiqCodeApp(repo, engine, model_name=handle.model_id, session=session,
                     mode=(ApprovalMode.AUTO if auto_approve else ApprovalMode.APPROVE),
                     max_turns=cfg.max_turns, max_tokens=cfg.max_tokens,
                     max_tool_output=cfg.max_tool_output,
                     compact_at=compact_at).run(mouse=cfg.mouse)
    finally:
        handle.close()
    return 0


def _run_headless_cli(*, repo, goal: str, cfg) -> int:
    """Headless `optiq code -p GOAL`: attach/spawn a server, run the loop with
    auto-approve, record the session, print the result + diff, exit 0 iff done."""
    import asyncio

    from .headless import run_headless
    from .server import NoModelError, ensure_server
    from .sessions import SessionStore

    try:
        handle = ensure_server(model=cfg.model, base_url=cfg.base_url)
    except NoModelError as e:
        print(f"\n{e}")
        return 2
    print(f"  [server] {handle.base_url}  model={handle.model_id}\n")
    session = SessionStore(repo).new()
    import json as _json
    session.append("meta", _json.dumps({"model": handle.model_id, "repo": str(repo)}))
    session.append("input", goal)
    try:
        engine = _make_engine(handle, cfg)
        compact_at = _announce_compaction(cfg, engine)
        result = asyncio.run(run_headless(
            repo, goal, engine, log=lambda m: print("  " + m),
            max_turns=cfg.max_turns, max_tokens=cfg.max_tokens,
            max_tool_output=cfg.max_tool_output,
            wall_s=cfg.wall_clock, compact_at=compact_at))
    finally:
        handle.close()
    session.save_messages(result.messages)

    print(f"\n  stop: {result.stop_reason}  turns: {result.turns}  "
          f"tests: {result.passed} passed / {result.failed} failed  ·  session {session.id}")
    # Like `claude -p`, don't dump the diff: the edits are applied to the working
    # tree (nothing committed) for you to review with git. Report what changed.
    if result.patch:
        n = sum(1 for ln in result.patch.splitlines() if ln.startswith("diff --git"))
        print(f"  changed {n} file{'s' if n != 1 else ''}, {result.patch_bytes} bytes "
              f"— review with `git diff` (nothing committed)")
    else:
        print("  (no changes)")
    return 0 if result.succeeded else 1


def run_export(*, session: str | None = None, output: str | None = None,
               repo: str = ".") -> int:
    """Export a session as HF Session-Traces JSONL (default: most recent)."""
    from .sessions import SessionStore
    from .trace_writer import write_stf

    store = SessionStore(repo)
    if session:
        sess = store.load(session)
        if not sess.path.is_file():
            print(f"no session '{session}' in this repo ({store.dir})")
            return 2
    else:
        latest = store.latest()
        if latest is None:
            print(f"no sessions in this repo ({store.dir})")
            return 2
        sess = store.load(latest.id)
    messages = sess.load_messages()
    if not messages:
        print(f"session {sess.id} has no recorded trajectory to export")
        return 2
    write_stf(output or "-", messages, session_id=sess.id,
              name=sess.last_goal()[:80] or None, model=sess.meta().get("model"))
    if output:
        print(f"wrote {output} ({len(messages)} messages, session {sess.id})")
    return 0


def run_config(*, path: str = ".") -> int:
    """`optiq code config` — every setting, its value, and where it came from.

    Settings resolve through four sources plus a default, so "why is it doing
    that" needs an answer that isn't grep. This prints the resolved value and
    the winning source for each field, then the file paths so it's clear which
    file to edit.
    """
    from .config import FIELDS, load, repo_config_path, user_config_path

    repo = Path(path).resolve()
    cfg, warnings = load(repo)
    for w in warnings:
        print(f"  ! {w}")

    width = max(len(f.name) for f in FIELDS)
    print("\n  setting" + " " * (width - 5) + "value                     from")
    print("  " + "-" * (width + 40))
    for f in FIELDS:
        val = getattr(cfg, f.name)
        shown = "(unset)" if val is None else str(val)
        print(f"  {f.name:<{width}}  {shown:<24}  {cfg.sources[f.name]}")

    print(f"\n  repo config:  {repo_config_path(repo)}"
          f"{'' if repo_config_path(repo).is_file() else '   (none)'}")
    print(f"  user config:  {user_config_path()}"
          f"{'' if user_config_path().is_file() else '   (none)'}")
    print("\n  precedence: flag > environment > repo config > user config > default")
    print("  environment: OPTIQ_CODE_<SETTING>, e.g. OPTIQ_CODE_MAX_TURNS=60\n")
    return 0
