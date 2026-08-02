"""Small runtime proposal selectors for native-MTP experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


def _candidate_target_probs(
    candidates: np.ndarray,
    target_indices: np.ndarray,
    target_probs: np.ndarray,
) -> np.ndarray:
    out = np.zeros(candidates.shape, dtype=np.float64)
    for row in range(candidates.shape[0]):
        p_map = {
            int(token): float(prob)
            for token, prob in zip(target_indices[row], target_probs[row], strict=True)
            if float(prob) > 0.0
        }
        for col, token in enumerate(candidates[row]):
            out[row, col] = p_map.get(int(token), 0.0)
    return out


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - np.max(values.astype(np.float64))
    exp_values = np.exp(shifted)
    denom = float(np.sum(exp_values))
    if denom <= 0.0:
        return np.full(values.shape, 1.0 / max(1, values.size), dtype=np.float64)
    return exp_values / denom


@dataclass(frozen=True)
class _DepthPrior:
    global_mean: float
    rank: tuple[float, ...]
    token: dict[int, float]


@dataclass(frozen=True)
class TopKProposalReranker:
    """Depth-gated MTP top-k selector with a known one-hot proposal q.

    This intentionally stays tiny: it is a diagnostic acceptance-side probe,
    not a learned adapter.  It chooses from the current recursive MTP top-k
    using local q probability plus calibration priors, then the runtime treats
    the chosen token as a one-hot proposal distribution so speculative sampling
    remains exact.
    """

    source: str
    topk: int
    q_weight: float
    token_weight: float
    rank_weight: float
    depth_priors: dict[int, _DepthPrior]

    @classmethod
    def from_calibration(
        cls,
        calib_npz: Path | str,
        *,
        depths: set[int],
        topk: int,
        q_weight: float,
        token_weight: float,
        rank_weight: float,
        prefix_active_only: bool = True,
        smooth: float = 4.0,
    ) -> "TopKProposalReranker":
        path = Path(calib_npz)
        data = np.load(path, allow_pickle=True)
        all_depths = np.asarray(data["depths"], dtype=np.int16)
        row_filter = np.isin(all_depths, np.asarray(sorted(depths), dtype=np.int16))
        if prefix_active_only and "recursive_prefix_active" in data.files:
            row_filter &= np.asarray(data["recursive_prefix_active"], dtype=bool)
        if not np.any(row_filter):
            raise ValueError("top-k proposal reranker calibration has no matching rows")

        candidates = np.asarray(data["recursive_top_indices"], dtype=np.int64)[row_filter, :topk]
        target_indices = np.asarray(data["target_ar_top_indices"], dtype=np.int64)[row_filter]
        target_probs = np.asarray(data["target_ar_top_probs"], dtype=np.float64)[row_filter]
        depths_arr = all_depths[row_filter].astype(np.int16)
        p_values = _candidate_target_probs(candidates, target_indices, target_probs)
        if "target_ar_p_recursive_draft" in data.files:
            p_values[:, 0] = np.asarray(
                data["target_ar_p_recursive_draft"], dtype=np.float64
            )[row_filter]

        depth_priors: dict[int, _DepthPrior] = {}
        for depth in sorted(depths):
            local = depths_arr == int(depth)
            if not np.any(local):
                continue
            local_candidates = candidates[local]
            local_p = p_values[local]
            global_mean = max(float(np.mean(local_p[:, 0])), 1e-9)
            rank_values: list[float] = []
            for rank in range(local_candidates.shape[1]):
                rank_p = local_p[:, rank]
                rank_values.append(
                    float((np.sum(rank_p) + smooth * global_mean) / (rank_p.size + smooth))
                )
            token_sum: dict[int, float] = {}
            token_count: dict[int, float] = {}
            for row in range(local_candidates.shape[0]):
                for rank, token in enumerate(local_candidates[row]):
                    token_id = int(token)
                    token_sum[token_id] = token_sum.get(token_id, 0.0) + float(local_p[row, rank])
                    token_count[token_id] = token_count.get(token_id, 0.0) + 1.0
            token_prior = {
                token: float((value + smooth * global_mean) / (token_count[token] + smooth))
                for token, value in token_sum.items()
            }
            depth_priors[int(depth)] = _DepthPrior(
                global_mean=global_mean,
                rank=tuple(rank_values),
                token=token_prior,
            )
        if not depth_priors:
            raise ValueError("top-k proposal reranker calibration produced no priors")
        return cls(
            source=str(path),
            topk=int(topk),
            q_weight=float(q_weight),
            token_weight=float(token_weight),
            rank_weight=float(rank_weight),
            depth_priors=depth_priors,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "topk": self.topk,
            "q_weight": self.q_weight,
            "token_weight": self.token_weight,
            "rank_weight": self.rank_weight,
            "depths": sorted(self.depth_priors),
        }

    def select(self, logits: mx.array, *, depth: int) -> tuple[int, dict[str, Any]] | None:
        prior = self.depth_priors.get(int(depth))
        if prior is None:
            return None
        flat = logits.reshape(-1).astype(mx.float32)
        vocab_size = int(flat.shape[-1])
        k = min(int(self.topk), vocab_size)
        if k <= 1:
            return None
        top_idx = mx.argpartition(-flat, kth=k - 1, axis=-1)[:k]
        top_vals = flat[top_idx]
        order = mx.argsort(-top_vals, axis=-1)
        top_idx = top_idx[order]
        top_vals = top_vals[order]
        mx.eval(top_idx, top_vals)

        token_ids = np.asarray(top_idx, dtype=np.int64).reshape(-1)
        values = np.asarray(top_vals, dtype=np.float64).reshape(-1)
        q_probs = _softmax(values)
        scores = self.q_weight * np.log(np.maximum(q_probs, 1e-12))
        for rank, token_id in enumerate(token_ids):
            rank_prior = prior.rank[min(rank, len(prior.rank) - 1)]
            token_prior = prior.token.get(int(token_id), prior.global_mean)
            scores[rank] += self.rank_weight * math.log(max(rank_prior, 1e-12))
            scores[rank] += self.token_weight * math.log(max(token_prior, 1e-12))

        selected_rank = int(np.argmax(scores))
        token = int(token_ids[selected_rank])
        base_token = int(token_ids[0])
        return token, {
            "depth": int(depth),
            "topk": int(k),
            "base_token": base_token,
            "selected_token": token,
            "selected_rank": selected_rank,
            "changed": bool(token != base_token),
            "score_delta_vs_base": float(scores[selected_rank] - scores[0]),
            "selected_local_q": float(q_probs[selected_rank]),
        }
