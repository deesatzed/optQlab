"""Where OptiQ keeps its private sidecar weights, and how to find them.

OptiQ ships two weight files that are **not** part of a model's main weight set
and are meant to be read only by OptiQ:

* ``mtp.safetensors`` — the multi-token-prediction head (speculative decoding), and
* ``optiq_vision.safetensors`` — the bf16 vision/audio towers of a VLM base.

They were originally written at the repo root. mlx-lm ignores them because it
only globs ``model*.safetensors``. But any loader that globs ``*.safetensors``
— mlx-vlm, and the apps built on it (LM Studio, vMLX) — picks them up and then
fails a strict weight load, since the base model has no module for those
tensors ("Received N parameters not in model", reported on the 27B quant).

A non-recursive ``*.safetensors`` glob does not descend into subdirectories, so
placing these files under a subfolder makes them invisible to every such loader
while OptiQ still finds them by name. New models write the sidecars under
``optiq/``; models published before this change keep them at the root, so every
resolver here checks the subfolder first and falls back to the root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# The subfolder that hides OptiQ's sidecars from a `*.safetensors` glob.
SIDECAR_SUBDIR = "optiq"

MTP_BASENAME = "mtp.safetensors"
VISION_BASENAME = "optiq_vision.safetensors"


def canonical_rel(basename: str) -> str:
    """Repo-relative path a newly-written sidecar should be saved to."""
    return f"{SIDECAR_SUBDIR}/{basename}"


def candidates(model_dir: str | Path, basename: str) -> list[Path]:
    """Search order for an existing sidecar: subfolder first, then legacy root."""
    d = Path(model_dir)
    return [d / SIDECAR_SUBDIR / basename, d / basename]


def resolve(model_dir: str | Path, basename: str) -> Optional[Path]:
    """The existing sidecar path (subfolder preferred), or ``None``."""
    for c in candidates(model_dir, basename):
        if c.exists():
            return c
    return None


def exists(model_dir: str | Path, basename: str) -> bool:
    return resolve(model_dir, basename) is not None


def write_path(model_dir: str | Path, basename: str) -> Path:
    """Canonical path to write a sidecar to, creating the subfolder."""
    p = Path(model_dir) / SIDECAR_SUBDIR / basename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def local_model_dir(model_path: str | Path) -> Optional[str]:
    """The on-disk directory for ``model_path``, which may be a Hub repo id.

    Every sidecar lookup needs a *directory*. But the documented way to load an
    OptiQ model names it by repo id (``optiq serve --model
    mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit``), and three separate call sites
    -- the engine, ``optiq serve``, and the Lab -- each wrote their own
    ``os.path.isdir(model_path) and has_vision_sidecar(model_path)`` gate. A repo
    id is not a directory, so all three were False for every Hub-loaded model and
    vision was disabled with the sidecar sitting unread in the snapshot. Reported
    on the 26B against 0.3.2.

    Never fetches: by the time anyone asks, the weights are already downloaded.
    Returns ``None`` when there is no local copy.
    """
    import os

    if os.path.isdir(model_path):
        return str(model_path)
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(str(model_path), local_files_only=True)
    except Exception:
        return None
