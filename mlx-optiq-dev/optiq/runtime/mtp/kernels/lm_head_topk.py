"""Draft-side quantized LM-head top-k kernel probes."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def is_qmv4_topk_eligible(x: mx.array, head: Any, *, top_k: int) -> bool:
    if not mx.metal.is_available():
        return False
    if not isinstance(head, nn.QuantizedLinear):
        return False
    if int(getattr(head, "bits", 0) or 0) != 4:
        return False
    if str(getattr(head, "mode", "affine")) != "affine":
        return False
    if int(getattr(head, "group_size", 0) or 0) not in {32, 64, 128}:
        return False
    if "bias" in head:
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if len(x.shape) == 1:
        k = int(x.shape[0])
    elif len(x.shape) == 2 and int(x.shape[0]) == 1:
        k = int(x.shape[1])
    else:
        return False
    if int(head.weight.shape[1]) * 8 != k:
        return False
    return 1 <= int(top_k) <= 64


def is_qmv8_topk_eligible(x: mx.array, head: Any, *, top_k: int) -> bool:
    if not mx.metal.is_available():
        return False
    if not isinstance(head, nn.QuantizedLinear):
        return False
    if int(getattr(head, "bits", 0) or 0) != 8:
        return False
    if str(getattr(head, "mode", "affine")) != "affine":
        return False
    if int(getattr(head, "group_size", 0) or 0) not in {32, 64, 128}:
        return False
    if "bias" in head:
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if len(x.shape) == 1:
        k = int(x.shape[0])
    elif len(x.shape) == 2 and int(x.shape[0]) == 1:
        k = int(x.shape[1])
    else:
        return False
    if int(head.weight.shape[1]) * 4 != k:
        return False
    return 1 <= int(top_k) <= 64


@lru_cache(maxsize=None)
def _qmv4_topk_kernel(group_size: int, top_k: int, num_simdgroups: int, subtiles: int, dtype: mx.Dtype):
    if top_k > 64:
        raise ValueError("top_k must be <= 64")
    if num_simdgroups not in {2, 4, 8}:
        raise ValueError("num_simdgroups must be 2, 4, or 8")
    if subtiles < 1:
        raise ValueError("subtiles must be >= 1")
    header = f"""
        using namespace metal;

        constant constexpr int PACK_FACTOR = 8;
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BYTES_PER_PACK = 4;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
        constant constexpr int RESULTS_PER_SIMDGROUP = 4;
        constant constexpr int NUM_SIMDGROUPS = {int(num_simdgroups)};
        constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;
        constant constexpr int SUBTILES = {int(subtiles)};
        constant constexpr int OUTS_PER_TILE = BN * SUBTILES;
        constant constexpr int TOPK = {int(top_k)};
        constant constexpr int GS = {int(group_size)};
        constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;

        template <typename T>
        inline float load_vector4_exact_topk(const device T* x, thread float* x_thread) {{
          float sum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; i += 4) {{
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }}
          return sum;
        }}

        inline float qdot4_exact_topk(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {{
          const device uint16_t* ws = (const device uint16_t*)w;
          float accum = 0.0f;
          for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {{
            uint16_t packed = ws[i];
            accum +=
              x_thread[4 * i] * float(packed & 0x000f) +
              x_thread[4 * i + 1] * float(packed & 0x00f0) +
              x_thread[4 * i + 2] * float(packed & 0x0f00) +
              x_thread[4 * i + 3] * float(packed & 0xf000);
          }}
          return scale * accum + sum * bias;
        }}
    """
    source = f"""
        uint tile = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;
        int K = int(K_size);
        int N = int(N_size);
        int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K / GS;
        int tile_base = int(tile) * OUTS_PER_TILE;

        threadgroup float top_values[{int(top_k)}];
        threadgroup int top_indices[{int(top_k)}];
        threadgroup float cand_values[{int(num_simdgroups) * 4}];
        threadgroup int cand_indices[{int(num_simdgroups) * 4}];

        if (simd_gid == 0 && simd_lid == 0) {{
          for (int i = 0; i < TOPK; ++i) {{
            top_values[i] = -INFINITY;
            top_indices[i] = -1;
          }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float x_thread[VALUES_PER_THREAD];
        for (int subtile = 0; subtile < SUBTILES; ++subtile) {{
          int out_row = tile_base + subtile * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
          float result[RESULTS_PER_SIMDGROUP] = {{0.0f}};

          const device uint8_t* w_base =
            (const device uint8_t*)w + out_row * in_vec_size_w
            + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
          const device T* scales_base =
            scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
          const device T* biases_base =
            biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

          for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE) {{
            const device T* x_ptr = x + k_block + int(simd_lid) * VALUES_PER_THREAD;
            float x_sum = load_vector4_exact_topk<T>(x_ptr, x_thread);
            const device uint8_t* w_block =
              w_base + k_block * BYTES_PER_PACK / PACK_FACTOR;
            const device T* scales_block = scales_base + k_block / GS;
            const device T* biases_block = biases_base + k_block / GS;

            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {{
              int n = out_row + row;
              if (n < N) {{
                const device uint8_t* wl = w_block + row * in_vec_size_w;
                const device T* sl = scales_block + row * in_vec_size_g;
                const device T* bl = biases_block + row * in_vec_size_g;
                result[row] += qdot4_exact_topk(
                  wl, x_thread, float(sl[0]), float(bl[0]), x_sum);
              }}
            }}
          }}

          float summed[RESULTS_PER_SIMDGROUP];
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {{
            summed[row] = simd_sum(result[row]);
          }}
          if (simd_lid == 0) {{
            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {{
              int cand = int(simd_gid) * RESULTS_PER_SIMDGROUP + row;
              int n = out_row + row;
              cand_values[cand] = (n < N) ? summed[row] : -INFINITY;
              cand_indices[cand] = (n < N) ? n : -1;
            }}
          }}
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (simd_gid == 0 && simd_lid == 0) {{
            for (int cand = 0; cand < BN; ++cand) {{
              float value = cand_values[cand];
              int index = cand_indices[cand];
              if (index < 0) {{
                continue;
              }}
              for (int pos = 0; pos < TOPK; ++pos) {{
                if (value > top_values[pos]) {{
                  for (int shift = TOPK - 1; shift > pos; --shift) {{
                    top_values[shift] = top_values[shift - 1];
                    top_indices[shift] = top_indices[shift - 1];
                  }}
                  top_values[pos] = value;
                  top_indices[pos] = index;
                  break;
                }}
              }}
            }}
          }}
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (simd_gid == 0 && simd_lid == 0) {{
          for (int i = 0; i < TOPK; ++i) {{
            values[int(tile) * TOPK + i] = top_values[i];
            indices[int(tile) * TOPK + i] = top_indices[i];
          }}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"qmv4_lm_head_topk_gs{group_size}_k{top_k}_sg{num_simdgroups}_st{subtiles}",
        input_names=["x", "w", "scales", "biases", "K_size", "N_size"],
        output_names=["values", "indices"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def qmv4_lm_head_topk_tiles(
    x: mx.array,
    head: nn.QuantizedLinear,
    *,
    top_k: int = 20,
    num_simdgroups: int = 4,
    subtiles: int = 8,
) -> tuple[mx.array, mx.array]:
    """Return per-tile top-k values/indices for a 4-bit quantized LM head.

    This is a prototype reducer. The second-stage global top-k merge remains in
    Python/NumPy so we can measure whether the first-stage kernel is worth
    productionizing before adding another Metal reduction pass.
    """
    if not is_qmv4_topk_eligible(x, head, top_k=top_k):
        raise ValueError("qmv4_lm_head_topk_tiles received an ineligible shape/head")
    flat_x = x.reshape(-1)
    k = int(flat_x.shape[0])
    n = int(head.weight.shape[0])
    bn = 4 * int(num_simdgroups)
    outs_per_tile = bn * int(subtiles)
    tile_count = (n + outs_per_tile - 1) // outs_per_tile
    kernel = _qmv4_topk_kernel(
        int(head.group_size),
        int(top_k),
        int(num_simdgroups),
        int(subtiles),
        flat_x.dtype,
    )
    values, indices = kernel(
        inputs=[
            flat_x,
            head.weight,
            head.scales,
            head.biases,
            mx.array(k, dtype=mx.int32),
            mx.array(n, dtype=mx.int32),
        ],
        template=[("T", flat_x.dtype)],
        output_shapes=[(tile_count, int(top_k)), (tile_count, int(top_k))],
        output_dtypes=[mx.float32, mx.int32],
        grid=(32 * tile_count, int(num_simdgroups), 1),
        threadgroup=(32, int(num_simdgroups), 1),
    )
    return values, indices


def _merge_tile_topk(
    values: mx.array,
    indices: mx.array,
    *,
    top_k: int,
) -> tuple[mx.array, mx.array]:
    flat_values = values.reshape(-1)
    flat_indices = indices.reshape(-1)
    k = min(int(top_k), int(flat_values.shape[0]))
    selected = mx.argpartition(-flat_values, kth=k - 1, axis=-1)[:k]
    top_values = flat_values[selected]
    top_indices = flat_indices[selected]
    order = mx.argsort(-top_values, axis=-1)
    return top_values[order], top_indices[order]


def qmv4_lm_head_topk(
    x: mx.array,
    head: nn.QuantizedLinear,
    *,
    top_k: int = 20,
    num_simdgroups: int = 4,
    subtiles: int = 8,
) -> tuple[mx.array, mx.array]:
    """Return global top-k values/indices for a 4-bit quantized LM head."""

    values, indices = qmv4_lm_head_topk_tiles(
        x,
        head,
        top_k=top_k,
        num_simdgroups=num_simdgroups,
        subtiles=subtiles,
    )
    return _merge_tile_topk(values, indices, top_k=top_k)


@lru_cache(maxsize=None)
def _qmv8_topk_kernel(group_size: int, top_k: int, num_simdgroups: int, subtiles: int, dtype: mx.Dtype):
    if top_k > 64:
        raise ValueError("top_k must be <= 64")
    if num_simdgroups not in {2, 4, 8}:
        raise ValueError("num_simdgroups must be 2, 4, or 8")
    if subtiles < 1:
        raise ValueError("subtiles must be >= 1")
    header = f"""
        using namespace metal;

        constant constexpr int PACK_FACTOR = 4;
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BYTES_PER_PACK = 4;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
        constant constexpr int RESULTS_PER_SIMDGROUP = 4;
        constant constexpr int NUM_SIMDGROUPS = {int(num_simdgroups)};
        constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;
        constant constexpr int SUBTILES = {int(subtiles)};
        constant constexpr int OUTS_PER_TILE = BN * SUBTILES;
        constant constexpr int TOPK = {int(top_k)};
        constant constexpr int GS = {int(group_size)};
        constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;

        template <typename T>
        inline float load_vector8_exact_topk(const device T* x, thread float* x_thread) {{
          float sum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; ++i) {{
            float value = static_cast<float>(x[i]);
            sum += value;
            x_thread[i] = value;
          }}
          return sum;
        }}

        inline float qdot8_exact_topk(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {{
          float accum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; ++i) {{
            accum += x_thread[i] * float(w[i]);
          }}
          return scale * accum + sum * bias;
        }}
    """
    source = f"""
        uint tile = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;
        int K = int(K_size);
        int N = int(N_size);
        int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K / GS;
        int tile_base = int(tile) * OUTS_PER_TILE;

        threadgroup float top_values[{int(top_k)}];
        threadgroup int top_indices[{int(top_k)}];
        threadgroup float cand_values[{int(num_simdgroups) * 4}];
        threadgroup int cand_indices[{int(num_simdgroups) * 4}];

        if (simd_gid == 0 && simd_lid == 0) {{
          for (int i = 0; i < TOPK; ++i) {{
            top_values[i] = -INFINITY;
            top_indices[i] = -1;
          }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float x_thread[VALUES_PER_THREAD];
        for (int subtile = 0; subtile < SUBTILES; ++subtile) {{
          int out_row = tile_base + subtile * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
          float result[RESULTS_PER_SIMDGROUP] = {{0.0f}};

          const device uint8_t* w_base =
            (const device uint8_t*)w + out_row * in_vec_size_w
            + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
          const device T* scales_base =
            scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
          const device T* biases_base =
            biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

          for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE) {{
            const device T* x_ptr = x + k_block + int(simd_lid) * VALUES_PER_THREAD;
            float x_sum = load_vector8_exact_topk<T>(x_ptr, x_thread);
            const device uint8_t* w_block =
              w_base + k_block * BYTES_PER_PACK / PACK_FACTOR;
            const device T* scales_block = scales_base + k_block / GS;
            const device T* biases_block = biases_base + k_block / GS;

            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {{
              int n = out_row + row;
              if (n < N) {{
                const device uint8_t* wl = w_block + row * in_vec_size_w;
                const device T* sl = scales_block + row * in_vec_size_g;
                const device T* bl = biases_block + row * in_vec_size_g;
                result[row] += qdot8_exact_topk(
                  wl, x_thread, float(sl[0]), float(bl[0]), x_sum);
              }}
            }}
          }}

          float summed[RESULTS_PER_SIMDGROUP];
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {{
            summed[row] = simd_sum(result[row]);
          }}
          if (simd_lid == 0) {{
            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {{
              int cand = int(simd_gid) * RESULTS_PER_SIMDGROUP + row;
              int n = out_row + row;
              cand_values[cand] = (n < N) ? float(T(summed[row])) : -INFINITY;
              cand_indices[cand] = (n < N) ? n : -1;
            }}
          }}
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (simd_gid == 0 && simd_lid == 0) {{
            for (int cand = 0; cand < BN; ++cand) {{
              float value = cand_values[cand];
              int index = cand_indices[cand];
              if (index < 0) {{
                continue;
              }}
              for (int pos = 0; pos < TOPK; ++pos) {{
                if (value > top_values[pos]) {{
                  for (int shift = TOPK - 1; shift > pos; --shift) {{
                    top_values[shift] = top_values[shift - 1];
                    top_indices[shift] = top_indices[shift - 1];
                  }}
                  top_values[pos] = value;
                  top_indices[pos] = index;
                  break;
                }}
              }}
            }}
          }}
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (simd_gid == 0 && simd_lid == 0) {{
          for (int i = 0; i < TOPK; ++i) {{
            values[int(tile) * TOPK + i] = top_values[i];
            indices[int(tile) * TOPK + i] = top_indices[i];
          }}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"qmv8_lm_head_topk_gs{group_size}_k{top_k}_sg{num_simdgroups}_st{subtiles}",
        input_names=["x", "w", "scales", "biases", "K_size", "N_size"],
        output_names=["values", "indices"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def qmv8_lm_head_topk_tiles(
    x: mx.array,
    head: nn.QuantizedLinear,
    *,
    top_k: int = 20,
    num_simdgroups: int = 4,
    subtiles: int = 8,
) -> tuple[mx.array, mx.array]:
    """Return per-tile top-k values/indices for an 8-bit quantized LM head."""
    if not is_qmv8_topk_eligible(x, head, top_k=top_k):
        raise ValueError("qmv8_lm_head_topk_tiles received an ineligible shape/head")
    flat_x = x.reshape(-1)
    k = int(flat_x.shape[0])
    n = int(head.weight.shape[0])
    bn = 4 * int(num_simdgroups)
    outs_per_tile = bn * int(subtiles)
    tile_count = (n + outs_per_tile - 1) // outs_per_tile
    kernel = _qmv8_topk_kernel(
        int(head.group_size),
        int(top_k),
        int(num_simdgroups),
        int(subtiles),
        flat_x.dtype,
    )
    values, indices = kernel(
        inputs=[
            flat_x,
            head.weight,
            head.scales,
            head.biases,
            mx.array(k, dtype=mx.int32),
            mx.array(n, dtype=mx.int32),
        ],
        template=[("T", flat_x.dtype)],
        output_shapes=[(tile_count, int(top_k)), (tile_count, int(top_k))],
        output_dtypes=[mx.float32, mx.int32],
        grid=(32 * tile_count, int(num_simdgroups), 1),
        threadgroup=(32, int(num_simdgroups), 1),
    )
    return values, indices


def qmv8_lm_head_topk(
    x: mx.array,
    head: nn.QuantizedLinear,
    *,
    top_k: int = 20,
    num_simdgroups: int = 4,
    subtiles: int = 8,
) -> tuple[mx.array, mx.array]:
    """Return global top-k values/indices for an 8-bit quantized LM head."""

    values, indices = qmv8_lm_head_topk_tiles(
        x,
        head,
        top_k=top_k,
        num_simdgroups=num_simdgroups,
        subtiles=subtiles,
    )
    return _merge_tile_topk(values, indices, top_k=top_k)
