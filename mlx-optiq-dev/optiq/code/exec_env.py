"""The execution-environment seam.

OptiQ Code has ONE agent loop (`optiq/code/loop.py`). What differs between the
interactive TUI, a headless `optiq code -p` run, and a repo inside a container is
not the loop — only *where the files live and where commands run*. That is this
module, and nothing else.

    LocalExecutor(repo)              # a repo on this machine
    DockerExecutor(container, cwd)   # a repo inside a running container

Ported from conjure/exec_env.py (kept verbatim except the branding).
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Executor(Protocol):
    """Everything the agent loop needs from the world. Deliberately tiny."""

    label: str
    is_local: bool
    root: Path | None

    def read_text(self, path: str) -> str: ...
    def write_text(self, path: str, content: str) -> None: ...
    def exists(self, path: str) -> bool: ...
    def rglob(self, pattern: str) -> list[str]: ...
    def run(self, cmd: str, timeout: int, env: dict | None = None) -> tuple[int, str]:
        """Run a shell command in the repo. Returns (returncode, stdout+stderr)."""
        ...


class LocalExecutor:
    """A repo on this machine."""

    is_local = True

    def __init__(self, repo: Path):
        self.repo = Path(repo)
        self.root = self.repo
        self.label = str(repo)

    def _p(self, path: str) -> Path:
        return (self.repo / path).resolve()

    def read_text(self, path: str) -> str:
        return self._p(path).read_text(errors="replace")

    def write_text(self, path: str, content: str) -> None:
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def exists(self, path: str) -> bool:
        return self._p(path).exists()

    def rglob(self, pattern: str) -> list[str]:
        return [str(p.relative_to(self.repo)) for p in sorted(self.repo.rglob(pattern))]

    def run(self, cmd: str, timeout: int, env: dict | None = None) -> tuple[int, str]:
        try:
            r = subprocess.run(cmd, shell=True, cwd=str(self.repo), capture_output=True,
                               text=True, timeout=timeout, env=env)
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return 124, "ERROR: command timed out"


class DockerExecutor:
    """A repo inside an already-running container. The caller starts and tears
    the container down; we only exec."""

    is_local = False
    root = None

    def __init__(self, container: str, workdir: str = "/testbed"):
        self.container = container
        self.workdir = workdir
        self.label = f"docker:{container}:{workdir}"

    def _abs(self, path: str) -> str:
        return path if path.startswith("/") else f"{self.workdir}/{path}"

    def run(self, cmd: str, timeout: int, env: dict | None = None) -> tuple[int, str]:
        argv = ["docker", "exec", "-w", self.workdir, self.container, "bash", "-lc", cmd]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return 124, "ERROR: command timed out"
        except OSError as e:
            return 125, f"ERROR: docker exec failed: {e}"

    def read_text(self, path: str) -> str:
        q = shlex.quote(self._abs(path))
        rc, out = self.run(f"cat -- {q}", timeout=60)
        if rc != 0:
            raise FileNotFoundError(path)
        return out

    def write_text(self, path: str, content: str) -> None:
        q = shlex.quote(self._abs(path))
        argv = ["docker", "exec", "-i", "-w", self.workdir, self.container, "bash", "-lc",
                f"mkdir -p \"$(dirname {q})\" && cat > {q}"]
        r = subprocess.run(argv, input=content, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise OSError(f"write failed: {r.stderr.strip()}")

    def exists(self, path: str) -> bool:
        rc, _ = self.run(f"test -e {shlex.quote(self._abs(path))}", timeout=30)
        return rc == 0

    def rglob(self, pattern: str) -> list[str]:
        rc, out = self.run(
            f"find . -name {shlex.quote(pattern)} -not -path '*/__pycache__/*' | sed 's|^\\./||'",
            timeout=120)
        if rc != 0:
            return []
        return sorted(x for x in out.splitlines() if x.strip())


def as_executor(target) -> Executor:
    """Coerce a Path/str repo into a LocalExecutor; pass an Executor through."""
    if isinstance(target, (str, Path)):
        return LocalExecutor(Path(target))
    return target
