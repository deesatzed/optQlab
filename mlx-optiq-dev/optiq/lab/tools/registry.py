"""Tool dispatch: route a tool name + JSON arguments to the right executor.

Used by the Lab chat orchestrator. Each executor must return a ``str``
result body that we can post back to the model as ``role=tool``.

Failures (bad JSON, unknown tool, executor exception) are converted to
human-readable strings that include enough detail for the model to
correct itself on a retry. This keeps the orchestrator simple: it just
loops on ``execute_tool`` and never has to handle exceptions itself.
"""
from __future__ import annotations

import json
import threading
import traceback
from typing import Any

from .web_search import execute_web_search
from ..sandbox import run_python, run_terminal


def _parse_arguments(arguments: str | dict | None) -> dict[str, Any]:
    """Tolerate both the JSON-string and pre-parsed-dict shapes."""
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    s = arguments.strip()
    if not s:
        return {}
    try:
        v = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"arguments is not valid JSON: {e}") from e
    if not isinstance(v, dict):
        raise ValueError("arguments must decode to a JSON object")
    return v


def _format_result_block(label: str, body: str, max_chars: int = 8000) -> str:
    """Trim a stdout/stderr block to ``max_chars`` and prefix the label."""
    body = body or ""
    if len(body) > max_chars:
        body = body[: max_chars - 50] + f"\n... [truncated, {len(body)} chars total]"
    return f"--- {label} ---\n{body}"


def _format_sandbox_result(kind: str, result) -> str:
    """Format a SandboxResult into a model-friendly string.

    The python sandbox can append an ``__IMAGES__:`` sentinel line to
    stdout with base64-encoded image data. We lift that sentinel out
    BEFORE truncation so a large stdout doesn't drop the image payload
    on the floor, then re-attach it after the formatted block.
    """
    parts: list[str] = []
    parts.append(f"sandbox: {result.sandbox_kind}  rc={result.returncode}")
    if result.timed_out:
        parts.append("(timed out)")
    if result.rejected_reason:
        parts.append(f"rejected: {result.rejected_reason}")

    stdout = result.stdout or ""
    sentinel_line = ""
    sentinel_marker = "__IMAGES__:"
    idx = stdout.find(sentinel_marker)
    if idx != -1:
        nl = stdout.find("\n", idx)
        nl = len(stdout) if nl == -1 else nl
        sentinel_line = stdout[idx:nl]
        stdout = (stdout[:idx] + stdout[nl:]).rstrip()

    parts.append(_format_result_block("stdout", stdout))
    parts.append(_format_result_block("stderr", result.stderr))
    if sentinel_line:
        # Append AFTER truncation; the orchestrator will lift and emit
        # this separately to the UI without showing it to the model.
        parts.append(sentinel_line)
    return "\n".join(parts)


def execute_tool(
    name: str,
    arguments: str | dict | None,
    *,
    cancel: threading.Event | None = None,
) -> str:
    """Run one tool call. Always returns a string for the role=tool reply.

    ``cancel``: optional threading.Event. When set mid-call, the underlying
    sandbox subprocess gets SIGKILL'd and the function returns a CANCELLED
    result. Web search ignores cancellation (network call is short).
    """
    try:
        args = _parse_arguments(arguments)
    except ValueError as e:
        return f"Error: {e}. Re-emit the tool call with valid JSON arguments."

    try:
        if name == "web_search":
            return execute_web_search(args)

        if name == "python":
            code = args.get("code")
            if not isinstance(code, str) or not code.strip():
                return "Error: python tool requires a `code` string argument."
            result = run_python(
                code, timeout=30.0, memory_limit_mb=1024, strict=True,
                cancel=cancel,
            )
            return _format_sandbox_result("python", result)

        if name == "terminal":
            command = args.get("command")
            if not isinstance(command, str) or not command.strip():
                return "Error: terminal tool requires a `command` string argument."
            result = run_terminal(
                command, timeout=60.0, memory_limit_mb=1024, strict=True,
                cancel=cancel,
            )
            return _format_sandbox_result("terminal", result)

        return (
            f"Error: unknown tool '{name}'. Available tools: "
            f"web_search, python, terminal."
        )
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return f"Tool '{name}' raised {e.__class__.__name__}: {e}\n{tb}"
