"""Unified sandbox for executing untrusted model-generated code.

Two execution surfaces, one isolation backend chain:

- ``python_sandbox.run_python(code, ...)`` — Python with AST safety pre-checks
- ``terminal_sandbox.run_terminal(command, ...)`` — bash with blocked-command
  pre-checks (no shell injection at the argument-position vs command-position
  layer)

Both share the three-tier isolation chain that we already had for the
HumanEval sandbox:

  1. ``apple/container`` Linux VM (Tahoe+, strongest)
  2. ``sandbox-exec`` macOS profile (deny network, scoped fs writes)
  3. ``subprocess`` with ``resource.setrlimit`` (last resort)

Public types live here so callers can ``from optiq.lab.sandbox import
run_python, run_terminal, SandboxResult, detect_sandbox_kind``.
"""
from __future__ import annotations

from .types import SandboxResult, detect_sandbox_kind
from .python_sandbox import run_python
from .terminal_sandbox import run_terminal

__all__ = [
    "SandboxResult",
    "detect_sandbox_kind",
    "run_python",
    "run_terminal",
]
