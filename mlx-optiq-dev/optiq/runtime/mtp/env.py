"""Environment snapshots for reproducible MTPLX measurements."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _run(args: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            args,
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except Exception as exc:  # pragma: no cover - depends on host tools
        return f"ERROR: {exc}"


def _mlx_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import mlx.core as mx

        info["mlx"] = getattr(mx, "__version__", "unknown")
        info["default_device"] = str(mx.default_device())
        for attr in ("get_active_memory", "get_peak_memory"):
            fn = getattr(mx, attr, None)
            if fn is not None:
                try:
                    info[attr] = int(fn())
                except Exception as exc:  # pragma: no cover - host dependent
                    info[attr] = f"ERROR: {exc}"
    except Exception as exc:
        info["mlx_error"] = repr(exc)
    try:
        import mlx_lm

        info["mlx_lm"] = getattr(mlx_lm, "__version__", "unknown")
    except Exception as exc:
        info["mlx_lm_error"] = repr(exc)
    return info


@dataclass(frozen=True)
class EnvironmentSnapshot:
    project_root: str
    python_executable: str
    python_version: str
    platform: str
    git_branch: str
    git_status: str
    hf_path: str | None
    uv_path: str | None
    mlx: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "platform": self.platform,
            "git_branch": self.git_branch,
            "git_status": self.git_status,
            "hf_path": self.hf_path,
            "uv_path": self.uv_path,
            "mlx": self.mlx,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def collect_environment(project_root: Path | str = ".") -> EnvironmentSnapshot:
    root = Path(project_root).resolve()
    return EnvironmentSnapshot(
        project_root=str(root),
        python_executable=sys.executable,
        python_version=sys.version,
        platform=platform.platform(),
        git_branch=_run(["git", "branch", "--show-current"], cwd=root),
        git_status=_run(["git", "status", "--short", "--branch"], cwd=root),
        hf_path=shutil.which("hf"),
        uv_path=shutil.which("uv"),
        mlx=_mlx_info(),
    )
