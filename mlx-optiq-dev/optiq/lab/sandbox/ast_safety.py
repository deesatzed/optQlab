"""AST-level safety checks for Python code before sandbox execution.

Detects patterns that can defeat signal-based timeouts or escape the
restricted environment. Returns ``(safe, reason)`` so callers can either
short-circuit (refuse to run) or log the warning before running.

Detected patterns
-----------------
- ``signal.signal(...)`` calls, ``signal.SIGALRM`` references, ``signal.alarm``
  rebinding — the timeout enforcement uses SIGALRM in subprocess mode; code
  that catches or reroutes it can run indefinitely
- bare ``except:`` / ``except BaseException:`` / ``except KeyboardInterrupt:``
  / ``except SystemExit:`` — swallowing these can hide the timeout signal
- ``os.system``, ``os.popen``, ``os.exec*``, ``os.spawn*``, ``os.posix_spawn*``,
  ``subprocess.run/call/check_call/check_output/Popen/getoutput``,
  ``pty.spawn``, ``commands.*`` — shell escapes
- ``socket.socket`` — network access (the sandbox-exec profile denies network
  too, but the AST check gives an earlier failure with a clearer message)

The check is intentionally conservative. False positives are acceptable;
silent escapes are not.
"""
from __future__ import annotations

import ast


_SHELL_EXEC_NAMES = frozenset(
    {
        "os.system", "os.popen", "os.popen2", "os.popen3", "os.popen4",
        "os.execl", "os.execle", "os.execlp", "os.execlpe",
        "os.execv", "os.execve", "os.execvp", "os.execvpe",
        "os.spawnl", "os.spawnle", "os.spawnlp", "os.spawnlpe",
        "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe",
        "os.posix_spawn", "os.posix_spawnp",
        "subprocess.run", "subprocess.call", "subprocess.check_call",
        "subprocess.check_output", "subprocess.Popen", "subprocess.getoutput",
        "subprocess.getstatusoutput",
        "pty.spawn", "pty.fork",
        "commands.getoutput", "commands.getstatusoutput",
    }
)

_SIGNAL_NAMES = frozenset(
    {
        "signal.signal", "signal.alarm", "signal.setitimer", "signal.pthread_kill",
        "signal.SIGALRM", "signal.SIG_DFL", "signal.SIG_IGN",
    }
)

_NETWORK_NAMES = frozenset(
    {
        "socket.socket", "socket.create_connection",
        "urllib.request.urlopen", "urllib.urlopen",
        "http.client.HTTPConnection", "http.client.HTTPSConnection",
        "requests.get", "requests.post", "requests.request",
    }
)


def _qualified_name(node: ast.AST) -> str | None:
    """Resolve ``ast.Attribute`` and ``ast.Name`` chains to a dotted string.

    Returns None when the chain is dynamic (subscript, call, etc.).
    """
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def check_python(code: str) -> tuple[bool, str | None]:
    """Inspect ``code`` for sandbox-escape patterns.

    Returns ``(safe, reason)``. ``safe=True`` means no flagged pattern was
    found; ``safe=False`` returns a short reason string fit for stderr.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _qualified_name(node.func)
            if name in _SHELL_EXEC_NAMES:
                return False, f"shell-exec call '{name}' is not allowed in the sandbox"
            if name in _SIGNAL_NAMES:
                return False, f"signal call '{name}' is not allowed in the sandbox"
            if name in _NETWORK_NAMES:
                return False, f"network call '{name}' is not allowed in the sandbox"

        if isinstance(node, ast.Attribute):
            name = _qualified_name(node)
            if name in _SIGNAL_NAMES:
                return False, f"signal attribute '{name}' is not allowed in the sandbox"

        if isinstance(node, ast.ExceptHandler):
            t = node.type
            if t is None:
                return False, "bare 'except:' clause is not allowed (may swallow timeout)"
            handled = _qualified_name(t)
            if handled in {"BaseException", "KeyboardInterrupt", "SystemExit"}:
                return False, (
                    f"'except {handled}:' is not allowed "
                    "(may swallow timeout signal)"
                )

    return True, None
