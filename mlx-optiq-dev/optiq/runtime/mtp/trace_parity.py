"""Trace-parity helpers for comparing Linux/vLLM and MLX MTP contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TRACE_SCHEMA_VERSION = 1

DIVERGENCE_ORDER = (
    "input_ids",
    "positions",
    "slot_or_cache_offset",
    "target_hidden",
    "hidden_in",
    "embedding_raw",
    "embedding_normed",
    "hidden_normed",
    "concat",
    "fc_out",
    "input_layernorm_out",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_norm",
    "k_norm",
    "rope_q",
    "rope_k",
    "attention_kv_cache_before",
    "attention_kv_cache_after",
    "attention_out",
    "post_attention_norm",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "residual_after_attn",
    "residual_after_mlp",
    "final_norm_hidden",
    "lm_head_top50_values",
    "lm_head_top50_ids",
)


@dataclass(frozen=True)
class ArrayComparison:
    boundary: str
    left_key: str
    right_key: str
    shape_match: bool
    dtype_left: str
    dtype_right: str
    max_abs_error: float | None
    relative_l2: float | None
    cosine: float | None
    exact_match: bool
    topk_overlap: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "left_key": self.left_key,
            "right_key": self.right_key,
            "shape_match": self.shape_match,
            "dtype_left": self.dtype_left,
            "dtype_right": self.dtype_right,
            "max_abs_error": self.max_abs_error,
            "relative_l2": self.relative_l2,
            "cosine": self.cosine,
            "exact_match": self.exact_match,
            "topk_overlap": self.topk_overlap,
        }


def safe_array_key(*parts: object) -> str:
    raw = ".".join(str(part) for part in parts if str(part) != "")
    return (
        raw.replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "_")
    )


def to_numpy(value: Any, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert MLX/Torch/NumPy-like tensors into a detached NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    elif hasattr(value, "item") and not hasattr(value, "shape"):
        value = np.asarray(value)
    else:
        try:
            import mlx.core as mx

            if isinstance(value, mx.array):
                mx.eval(value)
                value = np.asarray(value.astype(mx.float32))
        except Exception:
            pass
    arr = np.asarray(value)
    if arr.dtype.kind in {"f", "c"}:
        return arr.astype(dtype, copy=False)
    return arr


def array_stats(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value)
    flat = arr.astype(np.float64, copy=False).reshape(-1) if arr.size else arr
    if arr.size == 0:
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "mean": None,
            "std": None,
            "rms": None,
            "min": None,
            "max": None,
        }
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "rms": float(np.sqrt(np.mean(flat * flat))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def compare_arrays(
    boundary: str,
    left_key: str,
    left: np.ndarray,
    right_key: str,
    right: np.ndarray,
    *,
    atol: float = 1e-4,
) -> ArrayComparison:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    shape_match = left_arr.shape == right_arr.shape
    if not shape_match:
        return ArrayComparison(
            boundary=boundary,
            left_key=left_key,
            right_key=right_key,
            shape_match=False,
            dtype_left=str(left_arr.dtype),
            dtype_right=str(right_arr.dtype),
            max_abs_error=None,
            relative_l2=None,
            cosine=None,
            exact_match=False,
        )
    if left_arr.dtype.kind in {"i", "u", "b"} or right_arr.dtype.kind in {"i", "u", "b"}:
        exact = bool(np.array_equal(left_arr, right_arr))
        overlap = None
        if boundary.endswith("top50_ids") or boundary == "lm_head_top50_ids":
            left_set = set(int(x) for x in left_arr.reshape(-1))
            right_set = set(int(x) for x in right_arr.reshape(-1))
            overlap = len(left_set & right_set) / max(1, len(left_set | right_set))
        return ArrayComparison(
            boundary=boundary,
            left_key=left_key,
            right_key=right_key,
            shape_match=True,
            dtype_left=str(left_arr.dtype),
            dtype_right=str(right_arr.dtype),
            max_abs_error=0.0 if exact else 1.0,
            relative_l2=0.0 if exact else None,
            cosine=1.0 if exact else None,
            exact_match=exact,
            topk_overlap=overlap,
        )

    left_f = left_arr.astype(np.float64, copy=False).reshape(-1)
    right_f = right_arr.astype(np.float64, copy=False).reshape(-1)
    diff = left_f - right_f
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    left_norm = float(np.linalg.norm(left_f))
    diff_norm = float(np.linalg.norm(diff))
    right_norm = float(np.linalg.norm(right_f))
    denom = max(left_norm, 1e-12)
    relative_l2 = diff_norm / denom
    cosine = None
    if left_norm > 0 and right_norm > 0:
        cosine = float(np.dot(left_f, right_f) / (left_norm * right_norm))
    return ArrayComparison(
        boundary=boundary,
        left_key=left_key,
        right_key=right_key,
        shape_match=True,
        dtype_left=str(left_arr.dtype),
        dtype_right=str(right_arr.dtype),
        max_abs_error=max_abs,
        relative_l2=float(relative_l2),
        cosine=cosine,
        exact_match=bool(max_abs <= atol),
    )


def record_id(record: dict[str, Any]) -> str:
    return "|".join(
        str(record.get(key, ""))
        for key in ("prompt_id", "window_index", "mode", "depth")
    )


def trace_records(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return records that participate in cross-runtime boundary comparison."""
    records = list(metadata.get("records", []))
    records.extend(metadata.get("worker_records", []))
    return records


def load_trace(metadata_path: Path | str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    meta_path = Path(metadata_path)
    metadata = json.loads(meta_path.read_text())
    arrays_path = metadata.get("arrays_file")
    if not arrays_path:
        arrays_path = meta_path.with_suffix(".npz").name
    arrays_full_path = meta_path.parent / arrays_path
    arrays = dict(np.load(arrays_full_path)) if arrays_full_path.exists() else {}
    return metadata, arrays


def compare_traces(
    left_metadata: dict[str, Any],
    left_arrays: dict[str, np.ndarray],
    right_metadata: dict[str, Any],
    right_arrays: dict[str, np.ndarray],
    *,
    atol: float = 1e-4,
) -> dict[str, Any]:
    left_tokens = left_metadata.get("prompt_token_ids")
    right_tokens = right_metadata.get("prompt_token_ids")
    token_match = left_tokens == right_tokens

    left_records = {record_id(row): row for row in trace_records(left_metadata)}
    right_records = {record_id(row): row for row in trace_records(right_metadata)}
    common_ids = sorted(set(left_records) & set(right_records))
    missing_left = sorted(set(right_records) - set(left_records))
    missing_right = sorted(set(left_records) - set(right_records))

    comparisons: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    for rid in common_ids:
        left_row = left_records[rid]
        right_row = right_records[rid]
        left_boundary_map = left_row.get("arrays", {})
        right_boundary_map = right_row.get("arrays", {})
        for boundary in DIVERGENCE_ORDER:
            left_key = left_boundary_map.get(boundary)
            right_key = right_boundary_map.get(boundary)
            if left_key is None or right_key is None:
                continue
            if left_key not in left_arrays or right_key not in right_arrays:
                continue
            comparison = compare_arrays(
                boundary,
                left_key,
                left_arrays[left_key],
                right_key,
                right_arrays[right_key],
                atol=atol,
            )
            payload = {"record_id": rid, **comparison.to_dict()}
            comparisons.append(payload)
            if first_divergence is None and not comparison.exact_match:
                first_divergence = payload
                break
        if first_divergence is not None:
            break

    return {
        "left_source": left_metadata.get("source"),
        "right_source": right_metadata.get("source"),
        "token_ids_match": token_match,
        "left_prompt_tokens": len(left_tokens or []),
        "right_prompt_tokens": len(right_tokens or []),
        "common_records": len(common_ids),
        "missing_left_records": missing_left[:20],
        "missing_right_records": missing_right[:20],
        "first_divergence": first_divergence,
        "comparisons_until_first_divergence": comparisons,
    }


def render_comparison_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# MTP Trace-Parity Comparison",
        "",
        f"- left: `{result.get('left_source')}`",
        f"- right: `{result.get('right_source')}`",
        f"- token IDs match: `{result.get('token_ids_match')}`",
        f"- common records: `{result.get('common_records')}`",
        "",
    ]
    divergence = result.get("first_divergence")
    if divergence is None:
        lines.append("## First Divergence")
        lines.append("")
        lines.append("No divergence found across compared boundaries.")
    else:
        lines.append("## First Divergence")
        lines.append("")
        for key in (
            "record_id",
            "boundary",
            "shape_match",
            "dtype_left",
            "dtype_right",
            "max_abs_error",
            "relative_l2",
            "cosine",
            "topk_overlap",
        ):
            lines.append(f"- {key}: `{divergence.get(key)}`")
    lines.append("")
    lines.append("## Compared Boundaries")
    lines.append("")
    for row in result.get("comparisons_until_first_divergence", []):
        lines.append(
            "- "
            f"{row['record_id']} / {row['boundary']}: "
            f"exact={row['exact_match']} max_abs={row['max_abs_error']} "
            f"rel_l2={row['relative_l2']} cosine={row['cosine']}"
        )
    lines.append("")
    return "\n".join(lines)
