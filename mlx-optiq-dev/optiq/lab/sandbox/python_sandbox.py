"""Sandboxed Python execution.

Public entry point: ``run_python(code, timeout, memory_limit_mb, strict)``.
The three-tier isolation chain is identical to the one OptiQ has used for
HumanEval since v0.1.0:

  1. ``apple/container`` Linux VM (strongest, requires the signed pkg).
  2. ``sandbox-exec`` macOS profile (default fallback on macOS).
  3. ``subprocess`` + ``resource.setrlimit`` (last resort, no fs/network
     isolation but enforces wall-time + memory).

When ``strict=True`` (default), code is first run through
``ast_safety.check_python``. If that refuses, the runner returns a
``rejected``-kind result without launching a subprocess.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time

from .ast_safety import check_python
from .types import SandboxResult, has_container, has_sandbox_exec


_POLL_INTERVAL = 0.05  # seconds between cancel-event polls
_IMAGE_EXTS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
}
_PER_IMAGE_MAX_BYTES = 2 * 1024 * 1024   # 2 MB per file
_TOTAL_IMAGES_MAX_BYTES = 6 * 1024 * 1024  # 6 MB total
_SENTINEL = "__IMAGES__:"


def _capture_new_images(workdir: str) -> list[dict[str, str]]:
    """Scan ``workdir`` for image files and return base64-encoded copies.

    The python sandbox builds a fresh workdir per call, so anything under
    it that was created by the user's code is new. Caps per-file and
    total size to keep tool replies bounded.
    """
    out: list[dict[str, str]] = []
    total = 0
    try:
        entries = sorted(os.listdir(workdir))
    except FileNotFoundError:
        return out
    for fn in entries:
        ext = os.path.splitext(fn)[1].lower()
        mime = _IMAGE_EXTS.get(ext)
        if not mime:
            continue
        path = os.path.join(workdir, fn)
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size == 0 or size > _PER_IMAGE_MAX_BYTES:
            continue
        if total + size > _TOTAL_IMAGES_MAX_BYTES:
            break
        try:
            with open(path, "rb") as f:
                data = f.read(_PER_IMAGE_MAX_BYTES + 1)
        except OSError:
            continue
        if len(data) > _PER_IMAGE_MAX_BYTES:
            continue
        out.append({
            "filename": fn,
            "mime": mime,
            "data_b64": base64.b64encode(data).decode("ascii"),
        })
        total += size
    return out


def _augment_with_images(result: SandboxResult, workdir: str) -> SandboxResult:
    """If the python sandbox produced image files, append an __IMAGES__
    sentinel line to stdout so the UI can render them inline. Idempotent;
    safe to call on a SandboxResult whose stdout already has one."""
    if _SENTINEL in (result.stdout or ""):
        return result
    images = _capture_new_images(workdir)
    if not images:
        return result
    sentinel_line = _SENTINEL + json.dumps(images)
    new_stdout = (result.stdout or "")
    if new_stdout and not new_stdout.endswith("\n"):
        new_stdout += "\n"
    new_stdout += sentinel_line + "\n"
    return SandboxResult(
        stdout=new_stdout, stderr=result.stderr,
        returncode=result.returncode, timed_out=result.timed_out,
        sandbox_kind=result.sandbox_kind,
        rejected_reason=result.rejected_reason,
    )


def _run_with_cancel(
    argv: list[str], *, timeout: float, cwd: str | None, env: dict | None,
    sandbox_kind: str, cancel: threading.Event | None,
) -> SandboxResult:
    """``subprocess.run`` replacement that supports an external cancel event.

    Spawns the child in its own process group (``start_new_session=True``)
    so we can SIGKILL the whole tree if the user aborts. Polls ``cancel``
    every ~50 ms; also enforces the timeout deadline ourselves rather than
    relying on subprocess.run's blocking variant, which is uninterruptible.

    Returns the same shape as the old code path so callers don't change.
    """
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
    except FileNotFoundError as e:
        return SandboxResult(
            stdout="", stderr=f"sandbox: failed to launch: {e}",
            returncode=-1, timed_out=False, sandbox_kind=sandbox_kind,
        )

    deadline = time.time() + timeout
    cancelled = False
    timed_out = False

    while True:
        if proc.poll() is not None:
            break
        if cancel is not None and cancel.is_set():
            cancelled = True
            break
        if time.time() > deadline:
            timed_out = True
            break
        time.sleep(_POLL_INTERVAL)

    if cancelled or timed_out:
        # Kill the whole process group so any child shells / subprocesses
        # die with the parent.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    try:
        stdout, stderr = proc.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=1)
        except Exception:
            stdout, stderr = "", ""

    if cancelled:
        return SandboxResult(
            stdout=stdout or "", stderr="CANCELLED",
            returncode=-3, timed_out=False, sandbox_kind=sandbox_kind,
        )
    if timed_out:
        return SandboxResult(
            stdout=stdout or "", stderr="TIMEOUT",
            returncode=-1, timed_out=True, sandbox_kind=sandbox_kind,
        )
    return SandboxResult(
        stdout=stdout or "", stderr=stderr or "",
        returncode=proc.returncode, timed_out=False, sandbox_kind=sandbox_kind,
    )


# sandbox-exec profile: deny network, deny most filesystem writes except
# the temp directory we hand to the script. Read access stays open so
# Python can find its standard library.
_SBPL_PROFILE = """\
(version 1)
(deny default)
(allow process-fork)
(allow process-exec)
(allow file-read*)
(allow file-write* (subpath "{tmp}"))
(allow file-write-data
       (literal "/dev/null")
       (literal "/dev/dtracehelper"))
(allow sysctl-read)
(allow mach-lookup)
(allow ipc-posix-shm)
"""


def run_python(
    code: str,
    *,
    timeout: float = 30.0,
    memory_limit_mb: int = 1024,
    strict: bool = True,
    cancel: threading.Event | None = None,
) -> SandboxResult:
    """Run ``code`` as a Python script with the strongest available
    sandbox. Returns a ``SandboxResult``.

    ``strict``: when True, refuse to run code that fails the AST safety
    check. When False, the check is skipped and the runner relies entirely
    on the process-level sandbox.
    """
    if strict:
        safe, reason = check_python(code)
        if not safe:
            return SandboxResult(
                stdout="",
                stderr=f"sandbox: {reason}",
                returncode=-2,
                timed_out=False,
                sandbox_kind="rejected",
                rejected_reason=reason,
            )

    # Resolve any symlinks (/var/folders is /private/var/folders on macOS),
    # otherwise the sandbox-exec subpath check rejects writes that arrive
    # under the realpath form.
    sb_root = os.path.realpath(tempfile.mkdtemp(prefix="optiq_sb_"))
    script_path = os.path.join(sb_root, "script.py")
    with open(script_path, "w") as f:
        f.write(code)

    try:
        if has_container():
            result = _run_with_container(script_path, sb_root, timeout, memory_limit_mb, cancel)
        elif has_sandbox_exec():
            result = _run_with_sandbox_exec(script_path, sb_root, timeout, memory_limit_mb, cancel)
        else:
            result = _run_with_subprocess(script_path, timeout, memory_limit_mb, cancel)
        # Pick up any matplotlib / Pillow output the user wrote to the
        # working dir. Done here (after run, before the finally cleanup)
        # so all three sandbox tiers benefit without per-tier branching.
        return _augment_with_images(result, sb_root)
    finally:
        shutil.rmtree(sb_root, ignore_errors=True)


def _run_with_container(
    script_path: str, sb_root: str, timeout: float, memory_limit_mb: int,
    cancel: threading.Event | None,
) -> SandboxResult:
    """Run inside an apple/container Linux VM with no network."""
    try:
        return _run_with_cancel(
            [
                "container", "run", "--rm",
                "--network", "none",
                "--memory", f"{memory_limit_mb}m",
                "--mount", f"type=bind,source={sb_root},target=/work",
                "--workdir", "/work",
                "python:3.11-slim",
                "python", "/work/script.py",
            ],
            timeout=timeout, cwd=None, env=None,
            sandbox_kind="container", cancel=cancel,
        )
    except Exception:
        if has_sandbox_exec():
            return _run_with_sandbox_exec(script_path, sb_root, timeout, memory_limit_mb, cancel)
        return _run_with_subprocess(script_path, timeout, memory_limit_mb, cancel)


def _run_with_sandbox_exec(
    script_path: str, sb_root: str, timeout: float, memory_limit_mb: int,
    cancel: threading.Event | None,
) -> SandboxResult:
    """Run with macOS sandbox-exec + a restrictive SBPL profile."""
    profile = _SBPL_PROFILE.format(tmp=sb_root)
    profile_path = os.path.join(sb_root, "policy.sb")
    with open(profile_path, "w") as f:
        f.write(profile)

    preamble = textwrap.dedent(f"""\
        import resource
        try:
            resource.setrlimit(resource.RLIMIT_AS,   ({memory_limit_mb} * 1024 * 1024,) * 2)
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_DATA, ({memory_limit_mb} * 1024 * 1024,) * 2)
        except (ValueError, OSError):
            pass
        import socket as _s
        _orig_socket = _s.socket
        def _no_socket(*a, **k):
            raise OSError("network access disabled in sandbox")
        _s.socket = _no_socket
    """)
    with open(script_path, "r") as f:
        user_code = f.read()
    with open(script_path, "w") as f:
        f.write(preamble + "\n" + user_code)

    try:
        return _run_with_cancel(
            ["sandbox-exec", "-f", profile_path, sys.executable, "-I", script_path],
            timeout=timeout, cwd=sb_root,
            env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8",
                 "HOME": sb_root, "TMPDIR": sb_root, "MPLBACKEND": "Agg",
                 "MPLCONFIGDIR": sb_root, "XDG_CACHE_HOME": sb_root,
                 "XDG_CONFIG_HOME": sb_root},
            sandbox_kind="sandbox-exec", cancel=cancel,
        )
    except Exception:
        return _run_with_subprocess(script_path, timeout, memory_limit_mb, cancel)


def _run_with_subprocess(
    script_path: str, timeout: float, memory_limit_mb: int,
    cancel: threading.Event | None,
) -> SandboxResult:
    """Last resort: subprocess with rlimit, no fs/network isolation."""
    preamble = textwrap.dedent(f"""\
        import resource
        try:
            resource.setrlimit(resource.RLIMIT_AS,   ({memory_limit_mb} * 1024 * 1024,) * 2)
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_DATA, ({memory_limit_mb} * 1024 * 1024,) * 2)
        except (ValueError, OSError):
            pass
    """)
    with open(script_path, "r") as f:
        user_code = f.read()
    if "import resource" not in user_code:
        with open(script_path, "w") as f:
            f.write(preamble + "\n" + user_code)

    sb_root = os.path.dirname(script_path)
    return _run_with_cancel(
        [sys.executable, "-I", script_path],
        timeout=timeout, cwd=sb_root,
        env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8",
             "HOME": sb_root, "TMPDIR": sb_root, "MPLBACKEND": "Agg",
             "MPLCONFIGDIR": sb_root, "XDG_CACHE_HOME": sb_root,
             "XDG_CONFIG_HOME": sb_root},
        sandbox_kind="subprocess", cancel=cancel,
    )
