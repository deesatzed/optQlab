"""Small offline hidden-state correctors for recursive MTP drift.

The first correctors are intentionally simple. C0 checks whether recursive MTP
hidden drift is mostly global statistics. C1 checks whether a per-depth,
per-channel affine map can move recursive hidden states toward target-forced
hidden states without touching the exact target verifier.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NoOpCorrector:
    """Corrector interface that intentionally leaves hidden states unchanged."""

    kind: str = "noop"

    def apply_numpy(self, hidden: np.ndarray, *, depth: int) -> np.ndarray:
        _ = depth
        return hidden

    def apply_mlx(self, hidden, *, depth: int):
        _ = depth
        return hidden


@dataclass
class DiagonalAffineCorrector:
    """Per-depth affine corrector: ``h_corr = scale_d * h + bias_d``."""

    scale: np.ndarray
    bias: np.ndarray
    kind: str = "diagonal_affine"
    hidden_variant: str = "pre_norm"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scale = np.asarray(self.scale, dtype=np.float32)
        self.bias = np.asarray(self.bias, dtype=np.float32)
        if self.scale.ndim != 2:
            raise ValueError("scale must have shape [depth, hidden_size]")
        if self.bias.shape != self.scale.shape:
            raise ValueError("bias must have the same shape as scale")

    @property
    def depth_count(self) -> int:
        return int(self.scale.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.scale.shape[1])

    def apply_numpy(self, hidden: np.ndarray, *, depth: int) -> np.ndarray:
        if not 1 <= depth <= self.depth_count:
            raise ValueError(f"depth {depth} is outside 1..{self.depth_count}")
        arr = np.asarray(hidden, dtype=np.float32)
        scale = self.scale[depth - 1]
        bias = self.bias[depth - 1]
        return arr * scale + bias

    def apply_mlx(self, hidden, *, depth: int):
        if not 1 <= depth <= self.depth_count:
            raise ValueError(f"depth {depth} is outside 1..{self.depth_count}")
        import mlx.core as mx

        scale = mx.array(self.scale[depth - 1]).astype(hidden.dtype)
        bias = mx.array(self.bias[depth - 1]).astype(hidden.dtype)
        return hidden * scale + bias

    def save(self, path: Path | str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.kind,
            "hidden_variant": self.hidden_variant,
            "depth_count": self.depth_count,
            "hidden_size": self.hidden_size,
            "metadata": self.metadata,
        }
        np.savez_compressed(
            out,
            scale=self.scale,
            bias=self.bias,
            metadata_json=np.array(json.dumps(payload, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path | str) -> "DiagonalAffineCorrector":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            return cls(
                scale=np.asarray(data["scale"], dtype=np.float32),
                bias=np.asarray(data["bias"], dtype=np.float32),
                kind=str(metadata.get("kind", "diagonal_affine")),
                hidden_variant=str(metadata.get("hidden_variant", "pre_norm")),
                metadata=dict(metadata.get("metadata", {})),
            )


def load_corrector(path: Path | str):
    """Load any supported corrector artifact."""

    with np.load(Path(path), allow_pickle=False) as data:
        keys = set(data.files)
    if {"scale", "bias"}.issubset(keys):
        return DiagonalAffineCorrector.load(path)
    if {"mean", "in_proj", "out_proj"}.issubset(keys):
        from .low_rank import LowRankResidualCorrector

        return LowRankResidualCorrector.load(path)
    raise ValueError(f"unsupported corrector artifact schema: {path}")


def deterministic_train_mask(
    prompt_ids: np.ndarray,
    window_indices: np.ndarray,
    *,
    train_fraction: float = 0.75,
    salt: str = "mtplx-corrector-v1",
) -> np.ndarray:
    """Split by prompt/window key, not row order."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if len(prompt_ids) != len(window_indices):
        raise ValueError("prompt_ids and window_indices must have the same length")
    mask = np.zeros(len(prompt_ids), dtype=bool)
    for idx, (prompt_id, window_index) in enumerate(zip(prompt_ids, window_indices, strict=True)):
        key = f"{salt}:{prompt_id}:{int(window_index)}".encode("utf-8")
        digest = hashlib.sha256(key).digest()
        value = int.from_bytes(digest[:8], "big") / float(1 << 64)
        mask[idx] = value < train_fraction
    return mask


def ensure_nonempty_split(
    train_mask: np.ndarray,
    prompt_ids: np.ndarray,
    window_indices: np.ndarray,
) -> np.ndarray:
    """Repair tiny smoke splits while preserving prompt/window grouping."""

    mask = np.asarray(train_mask, dtype=bool).copy()
    if mask.any() and (~mask).any():
        return mask

    keys = sorted({(str(pid), int(win)) for pid, win in zip(prompt_ids, window_indices, strict=True)})
    if len(keys) < 2:
        raise ValueError("need at least two prompt/window groups for a held-out split")
    heldout = set(keys[::4] or keys[-1:])
    if len(heldout) == len(keys):
        heldout = {keys[-1]}
    for idx, key in enumerate(zip(prompt_ids, window_indices, strict=True)):
        mask[idx] = (str(key[0]), int(key[1])) not in heldout
    return mask


def _fit_depthwise_affine(
    recursive_hidden: np.ndarray,
    target_hidden: np.ndarray,
    depths: np.ndarray,
    *,
    depth_count: int,
    eps: float,
    diagonal: bool,
    kind: str,
    hidden_variant: str,
    metadata: dict[str, Any] | None = None,
) -> DiagonalAffineCorrector:
    recursive_hidden = np.asarray(recursive_hidden, dtype=np.float32)
    target_hidden = np.asarray(target_hidden, dtype=np.float32)
    depths = np.asarray(depths, dtype=np.int64)
    if recursive_hidden.shape != target_hidden.shape:
        raise ValueError("recursive_hidden and target_hidden must have matching shapes")
    if recursive_hidden.ndim != 2:
        raise ValueError("hidden arrays must have shape [rows, hidden_size]")

    hidden_size = int(recursive_hidden.shape[1])
    scale = np.ones((depth_count, hidden_size), dtype=np.float32)
    bias = np.zeros((depth_count, hidden_size), dtype=np.float32)

    for depth in range(1, depth_count + 1):
        rows = depths == depth
        if not rows.any():
            continue
        x = recursive_hidden[rows]
        y = target_hidden[rows]
        if diagonal:
            x_mean = x.mean(axis=0)
            y_mean = y.mean(axis=0)
            centered_x = x - x_mean
            centered_y = y - y_mean
            var = np.mean(centered_x * centered_x, axis=0)
            cov = np.mean(centered_x * centered_y, axis=0)
            depth_scale = cov / (var + eps)
            depth_bias = y_mean - depth_scale * x_mean
        else:
            x_mean_scalar = float(x.mean())
            y_mean_scalar = float(y.mean())
            x_rms = float(np.sqrt(np.mean((x - x_mean_scalar) ** 2)))
            y_rms = float(np.sqrt(np.mean((y - y_mean_scalar) ** 2)))
            depth_scale = np.full(hidden_size, y_rms / (x_rms + eps), dtype=np.float32)
            depth_bias = np.full(hidden_size, y_mean_scalar - float(depth_scale[0]) * x_mean_scalar, dtype=np.float32)

        scale[depth - 1] = np.nan_to_num(depth_scale, nan=1.0, posinf=1.0, neginf=1.0)
        bias[depth - 1] = np.nan_to_num(depth_bias, nan=0.0, posinf=0.0, neginf=0.0)

    return DiagonalAffineCorrector(
        scale=scale,
        bias=bias,
        kind=kind,
        hidden_variant=hidden_variant,
        metadata=metadata or {},
    )


def fit_c0_stat(
    recursive_hidden: np.ndarray,
    target_hidden: np.ndarray,
    depths: np.ndarray,
    *,
    depth_count: int,
    eps: float = 1e-6,
    hidden_variant: str = "pre_norm",
    metadata: dict[str, Any] | None = None,
) -> DiagonalAffineCorrector:
    return _fit_depthwise_affine(
        recursive_hidden,
        target_hidden,
        depths,
        depth_count=depth_count,
        eps=eps,
        diagonal=False,
        kind="c0_stat",
        hidden_variant=hidden_variant,
        metadata=metadata,
    )


def fit_c1_diagonal(
    recursive_hidden: np.ndarray,
    target_hidden: np.ndarray,
    depths: np.ndarray,
    *,
    depth_count: int,
    eps: float = 1e-6,
    hidden_variant: str = "pre_norm",
    metadata: dict[str, Any] | None = None,
) -> DiagonalAffineCorrector:
    return _fit_depthwise_affine(
        recursive_hidden,
        target_hidden,
        depths,
        depth_count=depth_count,
        eps=eps,
        diagonal=True,
        kind="c1_diagonal_affine",
        hidden_variant=hidden_variant,
        metadata=metadata,
    )


def blend_with_identity(
    corrector: DiagonalAffineCorrector,
    strength: float,
    *,
    kind: str | None = None,
) -> DiagonalAffineCorrector:
    """Dampen an affine corrector toward no-op.

    ``strength=0`` is exact no-op and ``strength=1`` is the original
    corrector. This is useful when the full diagonal fit improves hidden MSE
    but is too aggressive for the LM-head logit surface.
    """

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    scale = 1.0 + strength * (corrector.scale - 1.0)
    bias = strength * corrector.bias
    metadata = dict(corrector.metadata)
    metadata["blend_source_kind"] = corrector.kind
    metadata["blend_strength"] = float(strength)
    return DiagonalAffineCorrector(
        scale=scale.astype(np.float32),
        bias=bias.astype(np.float32),
        kind=kind or f"{corrector.kind}_blend",
        hidden_variant=corrector.hidden_variant,
        metadata=metadata,
    )
