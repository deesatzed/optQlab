"""In-process adapter registry for OptiQ serve.

Tracks which adapters have been loaded onto the base model, and exposes
a lightweight API for loading / listing / evicting them at runtime.

Current implementation: preload-one-adapter-at-startup semantics (matching
mlx-lm's `--adapter-path`). A live swap between different adapters per
request requires patching mlx-lm's LoRA layers into a "mounted" form
whose output can be toggled without re-loading the base; that's flagged
as the v0.0.8 work item.

Design intent documented here so the API shape stabilizes now:

  registry.load(adapter_id)       → AdapterInfo, ensures local files
  registry.list()                 → list of currently-known AdapterInfo
  registry.activate(adapter_id)   → apply adapter to model (mlx-lm's
                                    load_adapters for now; later this
                                    toggles a mounted LoRA path)
  registry.deactivate()           → remove adapter effects from model
  registry.unload(adapter_id)     → evict from registry
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .mount import (
    mount_adapter_on_model,
    prepare_model_for_mounted_lora,
    set_active_adapter,
    reset_active_adapter,
    unmount_adapter_from_model,
)
from .resolver import resolve_adapter_source


@dataclass
class AdapterInfo:
    """Metadata about a loaded adapter."""
    id: str                          # caller-supplied id (HF repo or local path)
    path: str                        # local dir on disk
    rank: Optional[int] = None
    target_modules: list[str] = field(default_factory=list)
    size_bytes: int = 0
    loaded_at: float = field(default_factory=time.time)
    optiq_sidecar: bool = False      # has optiq_lora_config.json
    # rank_distribution is an OptiQ-specific field; empty dict if this
    # adapter was trained with uniform rank.
    rank_distribution: dict = field(default_factory=dict)
    active: bool = False
    mounted_layers: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class AdapterRegistry:
    """Thread-safe in-memory registry."""

    def __init__(self, model=None):
        self._lock = threading.RLock()
        self._adapters: Dict[str, AdapterInfo] = {}
        self._active_id: Optional[str] = None
        self._model = model

    def bind_model(self, model) -> None:
        """Attach the serving model so activate() can mount adapters on it."""
        with self._lock:
            self._model = model

    def load(self, adapter_id: str) -> AdapterInfo:
        """Resolve + register an adapter (download if needed). Idempotent."""
        with self._lock:
            if adapter_id in self._adapters:
                return self._adapters[adapter_id]

            path = resolve_adapter_source(adapter_id)
            info = _inspect_adapter(adapter_id, path)
            self._adapters[adapter_id] = info
            return info

    def list(self) -> list[AdapterInfo]:
        with self._lock:
            return list(self._adapters.values())

    def get(self, adapter_id: str) -> Optional[AdapterInfo]:
        with self._lock:
            return self._adapters.get(adapter_id)

    def unload(self, adapter_id: str) -> bool:
        with self._lock:
            if self._active_id == adapter_id:
                self.deactivate()
            return self._adapters.pop(adapter_id, None) is not None

    def activate(self, adapter_id: str) -> AdapterInfo:
        """Mount the adapter on the bound model using the reversible
        mounted-LoRA path.

        Replaces target linear modules with ``MountedLoRALinear`` on
        first call, then registers the adapter's weights keyed by
        ``adapter_id``. Subsequent requests flip the active id via
        ``set_active_adapter`` (per-ContextVar) — no model reload.
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("registry has no bound model")
            if adapter_id not in self._adapters:
                self.load(adapter_id)
            info = self._adapters[adapter_id]

            # Idempotent mount — replaces linears with MountedLoRALinear
            # on first call and registers this adapter's weights.
            from pathlib import Path
            n_mounted = mount_adapter_on_model(
                self._model, adapter_id, Path(info.path)
            )
            if n_mounted == 0:
                raise RuntimeError(
                    f"failed to mount adapter {adapter_id!r}: no matching "
                    f"target modules (expected q_proj/v_proj). Check that "
                    f"the adapter was trained for this base model."
                )

            # Set as active on this context
            set_active_adapter(adapter_id)
            if self._active_id and self._active_id != adapter_id:
                prev = self._adapters.get(self._active_id)
                if prev:
                    prev.active = False
            self._active_id = adapter_id
            info.active = True
            info.mounted_layers = n_mounted
            return info

    def deactivate(self) -> None:
        """Clear the active adapter for the current context (does NOT
        unmount the adapter from the model; just stops the forward pass
        from routing through it)."""
        with self._lock:
            if self._active_id:
                info = self._adapters.get(self._active_id)
                if info:
                    info.active = False
            self._active_id = None
            set_active_adapter(None)

    def unmount(self, adapter_id: str) -> int:
        """Remove ``adapter_id`` from every MountedLoRALinear. Different
        from ``deactivate`` (which just stops routing) and from
        ``unload`` (which drops from registry but leaves model state).

        Returns the number of layers from which the adapter was removed.
        """
        with self._lock:
            if self._model is None:
                return 0
            n = unmount_adapter_from_model(self._model, adapter_id)
            if self._active_id == adapter_id:
                self.deactivate()
            return n

    def active_id(self) -> Optional[str]:
        with self._lock:
            return self._active_id


def _inspect_adapter(adapter_id: str, path: Path) -> AdapterInfo:
    """Read adapter metadata from disk. Supports both PEFT layout and the
    OptiQ sidecar."""
    rank: Optional[int] = None
    target_modules: list[str] = []
    rank_distribution: dict = {}
    optiq_sidecar = False

    optiq_path = path / "optiq_lora_config.json"
    peft_path = path / "adapter_config.json"

    if optiq_path.exists():
        optiq_sidecar = True
        try:
            cfg = json.loads(optiq_path.read_text())
            rank = cfg.get("rank")
            tm = cfg.get("target_modules")
            if isinstance(tm, (list, tuple)):
                target_modules = list(tm)
            applied = cfg.get("applied_ranks") or {}
            if applied:
                from collections import Counter
                rank_distribution = dict(Counter(applied.values()))
        except (ValueError, OSError):
            pass

    if peft_path.exists() and rank is None:
        try:
            cfg = json.loads(peft_path.read_text())
            rank = cfg.get("r")
            tm = cfg.get("target_modules")
            if isinstance(tm, (list, tuple)):
                target_modules = list(tm)
        except (ValueError, OSError):
            pass

    size_bytes = 0
    for f in path.glob("*.safetensors"):
        try:
            size_bytes += f.stat().st_size
        except OSError:
            pass

    return AdapterInfo(
        id=adapter_id,
        path=str(path),
        rank=rank,
        target_modules=target_modules,
        size_bytes=size_bytes,
        optiq_sidecar=optiq_sidecar,
        rank_distribution=rank_distribution,
    )
