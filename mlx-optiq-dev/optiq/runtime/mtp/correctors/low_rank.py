"""Low-rank residual hidden correctors for MTP recursive drift."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LowRankResidualCorrector:
    """Per-depth residual corrector: ``h_corr = h + (h - mean) A B``."""

    mean: np.ndarray
    in_proj: np.ndarray
    out_proj: np.ndarray
    rank: int
    kind: str = "c2_low_rank_residual"
    hidden_variant: str = "pre_norm"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.in_proj = np.asarray(self.in_proj, dtype=np.float32)
        self.out_proj = np.asarray(self.out_proj, dtype=np.float32)
        if self.mean.ndim != 2:
            raise ValueError("mean must have shape [depth, hidden_size]")
        if self.in_proj.ndim != 3:
            raise ValueError("in_proj must have shape [depth, hidden_size, rank]")
        if self.out_proj.ndim != 3:
            raise ValueError("out_proj must have shape [depth, rank, hidden_size]")
        if self.in_proj.shape[0] != self.mean.shape[0] or self.out_proj.shape[0] != self.mean.shape[0]:
            raise ValueError("all parameters must have the same depth dimension")
        if self.in_proj.shape[1] != self.mean.shape[1] or self.out_proj.shape[2] != self.mean.shape[1]:
            raise ValueError("hidden dimensions do not match")
        if self.in_proj.shape[2] != self.rank or self.out_proj.shape[1] != self.rank:
            raise ValueError("rank dimensions do not match")

    @property
    def depth_count(self) -> int:
        return int(self.mean.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.mean.shape[1])

    def apply_numpy(self, hidden: np.ndarray, *, depth: int) -> np.ndarray:
        if not 1 <= depth <= self.depth_count:
            raise ValueError(f"depth {depth} is outside 1..{self.depth_count}")
        arr = np.asarray(hidden, dtype=np.float32)
        centered = arr - self.mean[depth - 1]
        residual = (centered @ self.in_proj[depth - 1]) @ self.out_proj[depth - 1]
        return arr + residual

    def apply_mlx(self, hidden, *, depth: int):
        if not 1 <= depth <= self.depth_count:
            raise ValueError(f"depth {depth} is outside 1..{self.depth_count}")
        import mlx.core as mx

        mean = mx.array(self.mean[depth - 1]).astype(hidden.dtype)
        in_proj = mx.array(self.in_proj[depth - 1]).astype(hidden.dtype)
        out_proj = mx.array(self.out_proj[depth - 1]).astype(hidden.dtype)
        return hidden + ((hidden - mean) @ in_proj) @ out_proj

    def save(self, path: Path | str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.kind,
            "hidden_variant": self.hidden_variant,
            "depth_count": self.depth_count,
            "hidden_size": self.hidden_size,
            "rank": self.rank,
            "metadata": self.metadata,
        }
        np.savez_compressed(
            out,
            mean=self.mean,
            in_proj=self.in_proj,
            out_proj=self.out_proj,
            metadata_json=np.array(json.dumps(payload, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path | str) -> "LowRankResidualCorrector":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            return cls(
                mean=np.asarray(data["mean"], dtype=np.float32),
                in_proj=np.asarray(data["in_proj"], dtype=np.float32),
                out_proj=np.asarray(data["out_proj"], dtype=np.float32),
                rank=int(metadata["rank"]),
                kind=str(metadata.get("kind", "c2_low_rank_residual")),
                hidden_variant=str(metadata.get("hidden_variant", "pre_norm")),
                metadata=dict(metadata.get("metadata", {})),
            )


def fit_c2_low_rank(
    recursive_hidden: np.ndarray,
    target_hidden: np.ndarray,
    depths: np.ndarray,
    *,
    depth_count: int,
    rank: int,
    ridge: float = 1e-3,
    hidden_variant: str = "pre_norm",
    metadata: dict[str, Any] | None = None,
) -> LowRankResidualCorrector:
    if rank < 1:
        raise ValueError("rank must be >= 1")
    recursive_hidden = np.asarray(recursive_hidden, dtype=np.float32)
    target_hidden = np.asarray(target_hidden, dtype=np.float32)
    depths = np.asarray(depths, dtype=np.int64)
    if recursive_hidden.shape != target_hidden.shape:
        raise ValueError("recursive_hidden and target_hidden must have matching shapes")
    if recursive_hidden.ndim != 2:
        raise ValueError("hidden arrays must have shape [rows, hidden_size]")

    hidden_size = int(recursive_hidden.shape[1])
    mean = np.zeros((depth_count, hidden_size), dtype=np.float32)
    in_proj = np.zeros((depth_count, hidden_size, rank), dtype=np.float32)
    out_proj = np.zeros((depth_count, rank, hidden_size), dtype=np.float32)

    for depth in range(1, depth_count + 1):
        rows = depths == depth
        if not rows.any():
            continue
        x = recursive_hidden[rows]
        y = target_hidden[rows]
        residual = y - x
        x_mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        x_centered = x - x_mean
        mean[depth - 1] = x_mean
        if x.shape[0] < 2:
            continue

        _u, singular_values, vt = np.linalg.svd(x_centered.astype(np.float32), full_matrices=False)
        local_rank = min(rank, int(np.sum(singular_values > 1e-6)), x.shape[0] - 1)
        if local_rank < 1:
            continue
        components = vt[:local_rank].T.astype(np.float32)
        z = x_centered @ components
        gram = z.T @ z
        gram.flat[:: local_rank + 1] += float(ridge)
        coeff = np.linalg.solve(gram.astype(np.float64), (z.T @ residual).astype(np.float64)).astype(np.float32)
        in_proj[depth - 1, :, :local_rank] = components
        out_proj[depth - 1, :local_rank, :] = coeff

    return LowRankResidualCorrector(
        mean=mean,
        in_proj=in_proj,
        out_proj=out_proj,
        rank=rank,
        hidden_variant=hidden_variant,
        metadata=metadata or {},
    )
