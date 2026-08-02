"""Paths + runtime config for the Lab.

Single source of truth for where the Lab keeps state on disk. Mirrors the
``~/.optiq/`` convention the rest of optiq already uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LAB_PORT = 7860
DEFAULT_API_PORT = 8080


def _state_root() -> Path:
    """The user-state root. ``OPTIQ_HOME`` overrides for tests/dev."""
    override = os.environ.get("OPTIQ_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".optiq"


@dataclass(frozen=True)
class LabPaths:
    root: Path
    db: Path
    jobs_dir: Path     # per-job log files
    chats_dir: Path    # saved chat threads
    models_dir: Path   # local quants we've built
    cache_dir: Path    # transient files
    credentials_file: Path  # bootstrap marker so we know setup is done


def lab_paths() -> LabPaths:
    root = _state_root() / "lab"
    return LabPaths(
        root=root,
        db=root / "lab.db",
        jobs_dir=root / "jobs",
        chats_dir=root / "chats",
        models_dir=root / "models",
        cache_dir=root / "cache",
        credentials_file=root / ".bootstrap",
    )


def ensure_lab_dirs() -> LabPaths:
    """Create the on-disk layout. Safe to call repeatedly."""
    p = lab_paths()
    for d in (p.root, p.jobs_dir, p.chats_dir, p.models_dir, p.cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    return p
