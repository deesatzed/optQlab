"""Dense-logit top-k/logsumexp kernels for speculative sampling probes.

The verifier already produces dense target logits.  The default sampler path
then asks MLX to argpartition the full vocabulary and separately compute a
full-vocab logsumexp.  This module keeps the same exact sampling contract at
the Python boundary, but replaces the large-vocab top-k stage with tile-local
Metal work and a much smaller global merge.

This is intentionally opt-in.  It changes numerical reduction order for the
logsumexp denominator, so callers must gate it with distribution/sampled-output
QA before promoting it.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def fused_logits_topk_enabled() -> bool:
    return _env_truthy("MTPLX_FUSED_LOGITS_TOPK")


def _tile_size_from_env(default: int = 256) -> int:
    try:
        tile_size = int(os.environ.get("MTPLX_FUSED_LOGITS_TOPK_TILE") or default)
    except ValueError:
        tile_size = default
    if tile_size < 64:
        return 64
    if tile_size > 1024:
        return 1024
    # The kernels use one thread per tile element and are cached per tile size.
    # Keep this to common Metal-friendly powers of two.
    for candidate in (64, 128, 256, 512, 1024):
        if tile_size <= candidate:
            return candidate
    return 256


@lru_cache(maxsize=16)
def _tile_topk_kernel(top_k: int, tile_size: int, dtype: mx.Dtype):
    if not mx.metal.is_available():
        return None
    if top_k <= 0 or top_k > 64:
        return None
    if tile_size not in {64, 128, 256, 512, 1024}:
        return None

    header = f"""
        using namespace metal;

        constant constexpr int TOPK = {int(top_k)};
        constant constexpr int TILE = {int(tile_size)};
    """
    source = """
        const int tile = int(threadgroup_position_in_grid.x);
        const int row = int(threadgroup_position_in_grid.y);
        const int tid = int(thread_position_in_threadgroup.x);
        const int rows = int(row_count);
        const int vocab = int(vocab_size);
        const int base = tile * TILE;

        threadgroup float values[TILE];
        threadgroup int indices[TILE];

        float value = -INFINITY;
        int index = -1;
        if (row < rows && tid < TILE) {
            int vocab_index = base + tid;
            if (vocab_index < vocab) {
                value = float(logits[row * vocab + vocab_index]) / float(temperature);
                index = vocab_index;
            }
        }
        values[tid] = value;
        indices[tid] = index;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float top_values[TOPK];
            int top_indices[TOPK];
            for (int k = 0; k < TOPK; ++k) {
                top_values[k] = -INFINITY;
                top_indices[k] = -1;
            }
            for (int item = 0; item < TILE; ++item) {
                float candidate = values[item];
                int candidate_index = indices[item];
                if (candidate_index < 0) {
                    continue;
                }
                for (int pos = 0; pos < TOPK; ++pos) {
                    if (
                        candidate > top_values[pos]
                        || (candidate == top_values[pos] && candidate_index < top_indices[pos])
                    ) {
                        for (int shift = TOPK - 1; shift > pos; --shift) {
                            top_values[shift] = top_values[shift - 1];
                            top_indices[shift] = top_indices[shift - 1];
                        }
                        top_values[pos] = candidate;
                        top_indices[pos] = candidate_index;
                        break;
                    }
                }
            }
            const int out_base = (row * int(threadgroups_per_grid.x) + tile) * TOPK;
            for (int k = 0; k < TOPK; ++k) {
                tile_values[out_base + k] = top_values[k];
                tile_indices[out_base + k] = top_indices[k];
            }
            tile_max[row * int(threadgroups_per_grid.x) + tile] = top_values[0];
        }
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_logits_tile_topk_k{int(top_k)}_t{int(tile_size)}",
        input_names=["logits", "row_count", "vocab_size", "temperature"],
        output_names=["tile_values", "tile_indices", "tile_max"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=8)
def _tile_exp_sum_kernel(tile_size: int, dtype: mx.Dtype):
    if not mx.metal.is_available():
        return None
    if tile_size not in {64, 128, 256, 512, 1024}:
        return None

    header = f"""
        using namespace metal;

        constant constexpr int TILE = {int(tile_size)};
    """
    source = """
        const int tile = int(threadgroup_position_in_grid.x);
        const int row = int(threadgroup_position_in_grid.y);
        const int tid = int(thread_position_in_threadgroup.x);
        const int rows = int(row_count);
        const int vocab = int(vocab_size);
        const int base = tile * TILE;

        threadgroup float partials[TILE];
        float value = 0.0f;
        if (row < rows && tid < TILE) {
            int vocab_index = base + tid;
            if (vocab_index < vocab) {
                float scaled = float(logits[row * vocab + vocab_index]) / float(temperature);
                value = fast::exp(scaled - float(global_max[row]));
            }
        }
        partials[tid] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int stride = TILE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partials[tid] += partials[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            tile_sums[row * int(threadgroups_per_grid.x) + tile] = partials[0];
        }
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_logits_tile_exp_sum_t{int(tile_size)}",
        input_names=["logits", "global_max", "row_count", "vocab_size", "temperature"],
        output_names=["tile_sums"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def dense_logits_topk_logsumexp(
    logits: mx.array,
    *,
    top_k: int,
    temperature: float,
    tile_size: int | None = None,
) -> tuple[mx.array, mx.array] | None:
    """Return sorted top-k token ids and full-distribution probabilities.

    The returned probabilities are ``exp(top_value - logsumexp(full_vocab))``;
    top-p filtering/renormalization remains in Python so this can slot behind
    the existing sparse sampler without changing the public semantics.
    """

    if not mx.metal.is_available():
        return None
    if temperature <= 0 or top_k <= 0 or top_k > 64:
        return None
    if logits.ndim < 1:
        return None
    if logits.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return None

    rows = mx.contiguous(logits.reshape(-1, logits.shape[-1]))
    row_count = int(rows.shape[0])
    vocab_size = int(rows.shape[1])
    if row_count <= 0 or vocab_size <= 0:
        return None
    k = min(int(top_k), vocab_size)
    resolved_tile = _tile_size_from_env() if tile_size is None else int(tile_size)
    if resolved_tile not in {64, 128, 256, 512, 1024}:
        return None
    tile_count = (vocab_size + resolved_tile - 1) // resolved_tile

    topk_kernel = _tile_topk_kernel(k, resolved_tile, rows.dtype)
    sum_kernel = _tile_exp_sum_kernel(resolved_tile, rows.dtype)
    if topk_kernel is None or sum_kernel is None:
        return None

    tile_values, tile_indices, tile_max = topk_kernel(
        inputs=[
            rows,
            mx.array(row_count, dtype=mx.int32),
            mx.array(vocab_size, dtype=mx.int32),
            mx.array(float(temperature), dtype=mx.float32),
        ],
        template=[("T", rows.dtype)],
        grid=(tile_count * resolved_tile, row_count, 1),
        threadgroup=(resolved_tile, 1, 1),
        output_shapes=[
            (row_count, tile_count, k),
            (row_count, tile_count, k),
            (row_count, tile_count),
        ],
        output_dtypes=[mx.float32, mx.int32, mx.float32],
    )
    global_max = mx.max(tile_max, axis=-1)
    (tile_sums,) = sum_kernel(
        inputs=[
            rows,
            global_max,
            mx.array(row_count, dtype=mx.int32),
            mx.array(vocab_size, dtype=mx.int32),
            mx.array(float(temperature), dtype=mx.float32),
        ],
        template=[("T", rows.dtype)],
        grid=(tile_count * resolved_tile, row_count, 1),
        threadgroup=(resolved_tile, 1, 1),
        output_shapes=[(row_count, tile_count)],
        output_dtypes=[mx.float32],
    )
    log_total = mx.log(mx.sum(tile_sums, axis=-1)) + global_max

    candidate_values = tile_values.reshape(row_count, tile_count * k)
    candidate_indices = tile_indices.reshape(row_count, tile_count * k)
    selected = mx.argpartition(-candidate_values, kth=k - 1, axis=-1)[:, :k]
    top_values = mx.take_along_axis(candidate_values, selected, axis=-1)
    top_indices = mx.take_along_axis(candidate_indices, selected, axis=-1)
    order = mx.argsort(-top_values, axis=-1)
    top_values = mx.take_along_axis(top_values, order, axis=-1)
    top_indices = mx.take_along_axis(top_indices, order, axis=-1)
    top_probs_full = mx.exp(top_values - log_total[:, None])
    return top_indices, top_probs_full
