"""Discover local OptiQ quants on disk + read their metadata.

Single source of truth used by Server, Models, and Fine-tune pages so
the same display (achieved BPW, MTP presence, source) appears
everywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class LocalQuant:
    name: str                 # parent dir name (what the user typed at quantize time)
    path: str                 # absolute path to the model dir (config.json lives here)
    achieved_bpw: float | None
    target_bpw: float | None
    bits: list[int] | None    # e.g. [4, 8] candidate bits
    has_mtp: bool             # mtp.safetensors sidecar present
    size_bytes: int           # total of all .safetensors

    @property
    def display_name(self) -> str:
        """Compact label for sidebar + dropdowns. New quants follow the
        community convention (<base>-OptiQ-<bits>bit) and need no
        cleanup. Older test/legacy dirs may have the Qwen_ / Gemma_
        path-sanitiser prefix — strip it for display."""
        n = self.name
        for v in ("Qwen_", "Gemma_", "Meta_", "Mistral_", "Llama_"):
            if n.startswith(v):
                n = n[len(v):]
                break
        return n

    @property
    def bits_label(self) -> str:
        """llama.cpp-style label: the dominant (lowest) candidate bit-width.

        A mix of 4- and 8-bit layers is still called '4-bit OptiQ' — same
        convention as llama.cpp Q4_K (where output / attention norms are
        higher precision but the model is named for the dominant weights).
        Falls back to the achieved BPW when no candidate-bits metadata
        is available."""
        if self.bits:
            return f"{min(self.bits)}-bit OptiQ"
        if self.achieved_bpw is not None:
            return f"{self.achieved_bpw:.1f} BPW"
        return "—"

    @property
    def bpw_detail(self) -> str:
        """Secondary detail like '(5.57 BPW avg)' for the careful reader."""
        if self.achieved_bpw is not None:
            return f"{self.achieved_bpw:.2f} BPW avg"
        if self.target_bpw is not None:
            return f"target {self.target_bpw:.1f} BPW"
        return ""

    @property
    def bpw_label(self) -> str:
        """Backwards-compat composite — primary + detail joined."""
        detail = self.bpw_detail
        return f"{self.bits_label} · {detail}" if detail else self.bits_label


def _is_drafter_name(name: str) -> bool:
    """Detect speculative-decoding drafter models by name.

    Drafters (Gemma-4 ``-assistant-bf16`` and similar) are meant to be
    paired with a target via ``optiq serve --drafter`` — they're never
    served on their own. Filtering them out of the model picker
    prevents the user from accidentally selecting one as the host
    model. The drafter UI accepts an HF id directly, so dropping these
    from the discovery list doesn't make them unreachable.
    """
    n = name.lower()
    return "-assistant" in n or n.endswith("-drafter")


def discover(root: Path) -> list[LocalQuant]:
    """List every OptiQ quant directory under ``root``. Drafter models
    (``-assistant`` / ``-drafter``) are skipped — see ``_is_drafter_name``."""
    if not root.is_dir():
        return []
    out: list[LocalQuant] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if _is_drafter_name(d.name):
            continue
        # Standard OptiQ layout puts the model under <name>/optiq_mixed/;
        # also accept a flat layout where config.json lives at the top.
        for model_dir in (d / "optiq_mixed", d):
            if (model_dir / "config.json").is_file():
                out.append(_read_one(name=d.name, model_dir=model_dir))
                break
    return out


def discover_hf_cache(
    cache_root: Path | None = None,
    *,
    org_filter: str | None = "mlx-community",
) -> list[LocalQuant]:
    """List MLX-loadable model snapshots in the HuggingFace hub cache.

    The HF cache stores each model under ``models--<org>--<name>/snapshots/<sha>/``
    with the actual files behind symlinks into a content-addressed blob store.
    For each repo we pick the snapshot with the newest mtime that contains a
    ``config.json`` and at least one ``.safetensors`` shard. Returns
    LocalQuant records pointing at the snapshot directory so the supervisor
    can use the path directly as a mlx-lm ``--model`` argument.

    ``org_filter`` defaults to ``"mlx-community"`` to skip PyTorch / GGUF
    repos that mlx-lm can't load anyway. Pass ``None`` to scan every repo.
    """
    if cache_root is None:
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_root.is_dir():
        return []

    out: list[LocalQuant] = []
    for repo_dir in sorted(cache_root.glob("models--*")):
        if not repo_dir.is_dir():
            continue
        # models--<org>--<name>  ->  (<org>, <name>)
        parts = repo_dir.name.removeprefix("models--").split("--", 1)
        if len(parts) != 2:
            continue
        org, name = parts
        if org_filter is not None and org != org_filter:
            continue
        if _is_drafter_name(name):
            # Skip -assistant / -drafter repos. They get used through
            # the Settings → Server drafter field, not as host models.
            continue

        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue

        best: tuple[float, Path] | None = None
        for snap in snapshots_dir.iterdir():
            if not snap.is_dir():
                continue
            if not (snap / "config.json").is_file():
                continue
            # Resolve the safetensors symlink to make sure the actual blob
            # is present (HF can cache the metadata without the weights).
            shards = list(snap.glob("*.safetensors"))
            if not shards:
                continue
            real_shards = [p for p in shards if p.resolve().is_file()]
            if not real_shards:
                continue
            mtime = snap.stat().st_mtime
            if best is None or mtime > best[0]:
                best = (mtime, snap)

        if best is None:
            continue
        snap = best[1]
        # Display name: keep the org/name shape so the user sees the same
        # string they would pass to optiq lab --model.
        display = f"{org}/{name}"
        out.append(_read_one(name=display, model_dir=snap))
    return out


def _read_one(name: str, model_dir: Path) -> LocalQuant:
    meta_path = model_dir / "optiq_metadata.json"
    achieved_bpw: float | None = None
    target_bpw: float | None = None
    bits: list[int] | None = None
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            achieved_bpw = _to_float(meta.get("achieved_bpw"))
            target_bpw = _to_float(meta.get("target_bpw"))
            cb = meta.get("candidate_bits")
            if isinstance(cb, list):
                bits = sorted({int(b) for b in cb})
        except Exception:
            pass

    from ..sidecar_layout import exists as _sidecar_exists
    has_mtp = _sidecar_exists(model_dir, "mtp.safetensors")
    size_bytes = sum(p.stat().st_size for p in model_dir.glob("*.safetensors"))

    return LocalQuant(
        name=name,
        path=str(model_dir),
        achieved_bpw=achieved_bpw,
        target_bpw=target_bpw,
        bits=bits,
        has_mtp=has_mtp,
        size_bytes=size_bytes,
    )


def _to_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
