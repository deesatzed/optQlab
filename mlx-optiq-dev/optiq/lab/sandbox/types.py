"""Shared types + sandbox-kind detection for the sandbox modules."""
from __future__ import annotations

import shutil
from dataclasses import dataclass


_HAS_CONTAINER = shutil.which("container") is not None
_HAS_SANDBOX_EXEC = shutil.which("sandbox-exec") is not None


@dataclass
class SandboxResult:
    """Outcome of running code in the sandbox.

    ``rejected_reason`` is populated when an AST or blocked-command
    pre-check refused to run the code at all (returncode then -2).
    """

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    sandbox_kind: str  # "container", "sandbox-exec", "subprocess", or "rejected"
    rejected_reason: str | None = None


def detect_sandbox_kind() -> str:
    """Return the strongest sandbox we'd use right now."""
    if _HAS_CONTAINER:
        return "container"
    if _HAS_SANDBOX_EXEC:
        return "sandbox-exec"
    return "subprocess"


def has_container() -> bool:
    return _HAS_CONTAINER


def has_sandbox_exec() -> bool:
    return _HAS_SANDBOX_EXEC
