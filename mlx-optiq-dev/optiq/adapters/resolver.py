"""Resolve an adapter identifier to a local directory on disk.

Adapter IDs accepted:

  * ``<owner>/<adapter-name>`` — HuggingFace repo id. Downloaded into
    ``~/.cache/optiq/adapters/<owner>--<name>/`` by default (override
    the base via the ``OPTIQ_ADAPTER_CACHE`` environment variable).
  * ``./adapters/local-run/`` — absolute or relative path to a directory
    containing ``adapter_config.json`` + ``adapters.safetensors`` (mlx-lm
    format) or ``adapter_model.safetensors`` (PEFT format).
  * ``none`` — sentinel disabling any currently-active adapter. Handled
    upstream in ``optiq/serve.py`` — this resolver raises on it so callers
    don't confuse disable-request with a real id.
"""

from __future__ import annotations

import os
from pathlib import Path


ADAPTER_CACHE_ENV = "OPTIQ_ADAPTER_CACHE"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "optiq" / "adapters"


def adapter_cache_dir() -> Path:
    """Return the base directory for cached HF adapters."""
    override = os.environ.get(ADAPTER_CACHE_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_CACHE_DIR


def _is_local_path(adapter_id: str) -> bool:
    # Treat anything with a path separator or starting with . / ~ as local.
    if adapter_id.startswith((".", "/", "~")):
        return True
    return os.sep in adapter_id and "/" not in adapter_id.rstrip("/")[1:]


def resolve_adapter_source(adapter_id: str) -> Path:
    """Return a local directory containing adapter files for ``adapter_id``.

    Downloads from HF if not already cached. Raises ``FileNotFoundError``
    if a path was given but doesn't exist, or ``ValueError`` on the
    ``none`` sentinel.
    """
    if not adapter_id or adapter_id.lower() == "none":
        raise ValueError(
            "adapter_id 'none' is a control sentinel; handle upstream"
        )

    adapter_id = adapter_id.strip()

    # Local path
    p = Path(adapter_id).expanduser()
    if p.exists() and p.is_dir():
        return p.resolve()

    # HF repo id: "owner/name" with exactly one slash and no path chars
    if "/" in adapter_id and not adapter_id.startswith(("./", "/", "~")):
        owner, _, name = adapter_id.partition("/")
        if owner and name:
            return _download_from_hf(adapter_id)

    raise FileNotFoundError(
        f"could not resolve adapter {adapter_id!r} — "
        f"pass an existing local directory or a HuggingFace repo id"
    )


def _download_from_hf(repo_id: str) -> Path:
    """Snapshot-download adapter files from HF. Returns the cache path.

    We only fetch the small files (adapter_config, safetensors, metadata) —
    using ``allow_patterns`` so we don't pull arbitrarily large junk.
    """
    from huggingface_hub import snapshot_download

    owner, name = repo_id.split("/", 1)
    target = adapter_cache_dir() / f"{owner}--{name}"
    target.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapters.safetensors",           # mlx-lm native save name
            "optiq_lora_config.json",
            "README.md",
            "*.json",
        ],
    )
    return target.resolve()
