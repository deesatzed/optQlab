"""Sandboxed Python execution for agent / code-generation workflows.

Public re-export of the sandbox helpers used internally by the
HumanEval evaluator. Exposes a single function:

    from optiq.sandbox import run_python

    result = run_python(
        '''print("hello")\nx = 2 + 2\nprint(x)''',
        timeout=5.0,
        memory_limit_mb=512,
    )
    # SandboxResult(stdout='hello\\n4\\n', stderr='', returncode=0,
    #               timed_out=False, sandbox_kind='sandbox-exec')

The sandbox falls through three tiers automatically:

  1. ``apple/container`` — Linux-VM container (macOS Tahoe+); strongest.
  2. ``sandbox-exec`` — macOS native; no network, read-only FS outside
     a per-call temp dir. Default fallback.
  3. ``subprocess`` + ``setrlimit`` — universal; enforces wall-time
     and memory but not filesystem isolation.

Use ``detect_sandbox_kind()`` to inspect which tier the host supports
before running.
"""

from __future__ import annotations

from optiq.eval._sandbox import (
    SandboxResult,
    run_python,
    detect_sandbox_kind,
)

__all__ = ["SandboxResult", "run_python", "detect_sandbox_kind"]
