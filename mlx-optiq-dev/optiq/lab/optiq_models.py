"""Curated list of published mlx-community OptiQ quants, fetched from HF.

Used by the Server page's "switch model" dropdown so users can boot any
published OptiQ quant without typing the full id. Cached for the
process lifetime — refreshes when the user clicks "Reload published
quants" in the UI.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterable


_CACHE: list["PublishedQuant"] | None = None
_CACHE_AT: float = 0.0
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_S = 15 * 60  # 15 min — list doesn't change often


@dataclass
class PublishedQuant:
    repo_id: str               # mlx-community/Qwen3.5-9B-OptiQ-4bit
    family: str | None         # Qwen3.5 / Qwen3.6 / Gemma-4 / etc.
    size_label: str | None     # 0.8B / 9B / 27B — heuristic from the id
    bits_label: str | None     # 4-bit / 5-bit
    downloads: int

    @property
    def display(self) -> str:
        parts = [self.repo_id.split("/", 1)[-1]]
        return " · ".join(parts)


def list_published(force_refresh: bool = False) -> list[PublishedQuant]:
    """Return all mlx-community models matching '*OptiQ*'. Cached."""
    global _CACHE, _CACHE_AT
    with _CACHE_LOCK:
        fresh = _CACHE is not None and (time.time() - _CACHE_AT) < _CACHE_TTL_S
        if not force_refresh and fresh:
            return list(_CACHE)
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Published quants are named e.g. mlx-community/Qwen3.5-9B-OptiQ-4bit
        # but the HF naming convention is actually 'OptiQ' (lowercase i)
        # in the repo name. The marketing brand uses 'OptiQ' — we accept
        # either casing.
        infos = list(api.list_models(
            author="mlx-community", search="OptiQ", limit=100,
        ))
    except Exception:
        infos = []

    out: list[PublishedQuant] = []
    for info in infos:
        # Different hf-hub versions: some expose `.modelId`, others `.id`.
        # Use getattr to survive both without AttributeError.
        rid = getattr(info, "modelId", None) or getattr(info, "id", None)
        if not rid or "optiq" not in rid.lower():
            continue
        out.append(PublishedQuant(
            repo_id=rid,
            family=_family_from_id(rid),
            size_label=_size_from_id(rid),
            bits_label=_bits_from_id(rid),
            downloads=int(getattr(info, "downloads", 0) or 0),
        ))
    # Sort by downloads desc — popular LLM quants float to the top,
    # legacy YOLO and one-off uploads sink to the bottom. Stable
    # secondary sort by id keeps the order deterministic across runs.
    out.sort(key=lambda q: (-q.downloads, q.repo_id))

    with _CACHE_LOCK:
        _CACHE = out
        _CACHE_AT = time.time()
    return list(out)


# ---------------------------------------------------------------------------


def _family_from_id(repo_id: str) -> str | None:
    name = repo_id.split("/", 1)[-1].lower()
    for needle in ("qwen3.5", "qwen3.6", "gemma-4", "minicpm5", "qwen3", "gemma", "minicpm", "llama"):
        if needle in name:
            label = needle.title().replace("Qwen3", "Qwen3").replace("Gemma-4", "Gemma-4")
            if label.lower().startswith("minicpm"):
                # Keep MiniCPM5 capitalization (not Minicpm5)
                label = label.replace("Minicpm", "MiniCPM")
            return label
    return None


def _size_from_id(repo_id: str) -> str | None:
    import re
    m = re.search(r"-(\d+(?:\.\d+)?[Bb])", repo_id)
    return m.group(1).upper() if m else None


def _bits_from_id(repo_id: str) -> str | None:
    import re
    m = re.search(r"Opt[iI]Q-(\d+)bit", repo_id)
    return f"{m.group(1)}-bit" if m else None


def _size_key(label: str | None) -> float:
    if not label:
        return 0.0
    try:
        return float(label.rstrip("Bb"))
    except ValueError:
        return 0.0
