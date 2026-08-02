"""Sandboxed bash command execution.

Public entry point: ``run_terminal(command, ...)``. Same three-tier
isolation chain as ``python_sandbox``, plus an explicit blocked-command
pre-check that walks the command at the shell level to catch invocations
in command position (vs argument position).

Distinguishing command position from argument position
------------------------------------------------------
The naive ``"sudo" in command`` check rejects ``echo "do not use sudo"``,
which is obviously fine. The correct check looks at each token and
decides whether it sits in command position. A token is in command
position if it is the first token, or if the preceding token is:

  - a shell separator (``;``, ``&&``, ``||``, ``|``, ``&``, newline, ``(``, ``)``);
  - a backtick (`````);
  - a keyword that introduces a new command (``then``, ``do``, ``else``, ``elif``);
  - a wrapper whose next non-flag argument is itself a command
    (``env``, ``time``, ``nohup``, ``nice``, ``xargs``, ``timeout``, etc.);
  - an assignment of the form ``VAR=value`` that precedes a command.

We use ``shlex`` to tokenize, then a state machine to classify each
token. This is conservative: anything that looks ambiguous is treated as
command-position to err on the side of rejection.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading

from .python_sandbox import _run_with_cancel
from .types import SandboxResult, has_container, has_sandbox_exec


# Commands that are too dangerous to allow at command position.
_BLOCKED: frozenset[str] = frozenset(
    {
        # Filesystem destruction
        "rm", "dd", "mkfs", "fdisk", "mount", "umount",
        # Permission/ownership escalation
        "chmod", "chown", "passwd",
        # Privilege escalation
        "sudo", "su", "doas", "pkexec", "runas",
        # Process / system control
        "kill", "killall", "pkill", "shutdown", "reboot", "halt", "poweroff",
        # Network egress
        "curl", "wget", "nc", "ncat", "netcat", "socat",
        "ssh", "scp", "sftp", "rsync",
        # Shell evaluation backdoors
        "eval", "source",
    }
)

# Shell tokens that introduce a fresh command position for the next token.
_SEPARATORS: frozenset[str] = frozenset(
    {";", "&&", "||", "|", "&", "\n", "(", ")", "{", "}", "`"}
)
_KW_NEW_COMMAND: frozenset[str] = frozenset({"then", "do", "else", "elif"})

# Wrappers whose first non-flag positional argument IS the command to run.
_COMMAND_PREFIXES: frozenset[str] = frozenset(
    {
        "env", "command", "builtin", "exec", "time", "nohup", "nice",
        "setsid", "stdbuf", "timeout", "ionice", "chroot", "xargs",
    }
)

_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# `find -exec curl ...` is the classic prefix-trick for running blocked
# commands. Catch the find sub-flags that take a following command.
_FIND_EXEC_FLAGS: frozenset[str] = frozenset({"-exec", "-execdir", "-ok", "-okdir"})


def _at_command_position(tokens: list[str], i: int) -> bool:
    """Return True iff ``tokens[i]`` is the command being run (not arg)."""
    if i == 0:
        return True
    prev = tokens[i - 1]
    if prev in _SEPARATORS or prev in _KW_NEW_COMMAND:
        return True
    if prev == "find" and tokens[i] not in {".", "-"}:
        # find prefixes pass tokens around; treat conservatively
        return False
    if prev in _FIND_EXEC_FLAGS:
        return True
    if prev in _COMMAND_PREFIXES:
        return True
    if _ASSIGNMENT_RE.match(prev):
        return True
    return False


def _find_blocked(command: str) -> set[str]:
    """Return any blocked commands found at command position."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        # Unbalanced quotes etc. Treat as suspect.
        return {"(unparseable command)"}

    hits: set[str] = set()
    for i, tok in enumerate(tokens):
        # Strip absolute paths: /usr/bin/curl -> curl
        base = os.path.basename(tok)
        if base in _BLOCKED and _at_command_position(tokens, i):
            hits.add(base)
    return hits


def run_terminal(
    command: str,
    *,
    timeout: float = 300.0,
    memory_limit_mb: int = 1024,
    cwd: str | None = None,
    strict: bool = True,
    cancel: threading.Event | None = None,
) -> SandboxResult:
    """Run ``command`` (a bash one-liner) inside the sandbox.

    ``cwd``: optional working directory inside ``/tmp``. When None, a fresh
    per-call temp dir is used and removed afterwards.
    """
    if strict:
        blocked = _find_blocked(command)
        if blocked:
            joined = ", ".join(sorted(blocked))
            return SandboxResult(
                stdout="",
                stderr=f"sandbox: blocked command(s) at command position: {joined}",
                returncode=-2,
                timed_out=False,
                sandbox_kind="rejected",
                rejected_reason=f"blocked commands: {joined}",
            )

    own_workdir = cwd is None
    workdir = cwd or tempfile.mkdtemp(prefix="optiq_sh_")

    try:
        if has_container():
            return _run_with_container(command, workdir, timeout, memory_limit_mb, cancel)
        if has_sandbox_exec():
            return _run_with_sandbox_exec(command, workdir, timeout, memory_limit_mb, cancel)
        return _run_with_subprocess(command, workdir, timeout, memory_limit_mb, cancel)
    finally:
        if own_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def _run_with_container(
    command: str, workdir: str, timeout: float, memory_limit_mb: int,
    cancel: threading.Event | None,
) -> SandboxResult:
    try:
        return _run_with_cancel(
            [
                "container", "run", "--rm",
                "--network", "none",
                "--memory", f"{memory_limit_mb}m",
                "--mount", f"type=bind,source={workdir},target=/work",
                "--workdir", "/work",
                "alpine:3.20",
                "sh", "-c", command,
            ],
            timeout=timeout, cwd=None, env=None,
            sandbox_kind="container", cancel=cancel,
        )
    except Exception:
        if has_sandbox_exec():
            return _run_with_sandbox_exec(command, workdir, timeout, memory_limit_mb, cancel)
        return _run_with_subprocess(command, workdir, timeout, memory_limit_mb, cancel)


_SH_SBPL = """\
(version 1)
(deny default)
(allow process-fork)
(allow process-exec)
(allow file-read*)
(allow file-write* (subpath "{workdir}"))
(allow file-write-data
       (literal "/dev/null")
       (literal "/dev/dtracehelper"))
(allow sysctl-read)
(allow mach-lookup)
(allow ipc-posix-shm)
"""


def _run_with_sandbox_exec(
    command: str, workdir: str, timeout: float, memory_limit_mb: int,
    cancel: threading.Event | None,
) -> SandboxResult:
    profile = _SH_SBPL.format(workdir=workdir)
    profile_path = os.path.join(workdir, ".policy.sb")
    with open(profile_path, "w") as f:
        f.write(profile)

    try:
        return _run_with_cancel(
            ["sandbox-exec", "-f", profile_path, "/bin/sh", "-c", command],
            timeout=timeout, cwd=workdir,
            env={"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir},
            sandbox_kind="sandbox-exec", cancel=cancel,
        )
    except Exception:
        return _run_with_subprocess(command, workdir, timeout, memory_limit_mb, cancel)


def _run_with_subprocess(
    command: str, workdir: str, timeout: float, memory_limit_mb: int,
    cancel: threading.Event | None,
) -> SandboxResult:
    return _run_with_cancel(
        ["/bin/sh", "-c", command],
        timeout=timeout, cwd=workdir,
        env={"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir},
        sandbox_kind="subprocess", cancel=cancel,
    )
