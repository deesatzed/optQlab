"""Phase 7 VerifyCore MLP kernel experiments.

The first diagnostic kernel fuses the gate/up projections with SwiGLU
activation for M<=6 and leaves down_proj on the stock MLX path. This preserves
activation reuse across down-projection output tiles, unlike a naive full-MLP
single dispatch that would recompute gate/up for every down tile.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def is_gate_up_swiglu_qmv4_eligible(
    x: mx.array,
    gate_module: Any,
    up_module: Any,
) -> bool:
    """Return whether the diagnostic gate+up+SwiGLU qmv4 kernel can run."""
    if not mx.metal.is_available():
        return False
    if not isinstance(gate_module, nn.QuantizedLinear):
        return False
    if not isinstance(up_module, nn.QuantizedLinear):
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if len(x.shape) < 2:
        return False
    m = int(x.shape[-2])
    if m <= 0 or m > 6:
        return False
    if int(getattr(gate_module, "bits", 0) or 0) != 4:
        return False
    if int(getattr(up_module, "bits", 0) or 0) != 4:
        return False
    if int(getattr(gate_module, "group_size", 0) or 0) != int(
        getattr(up_module, "group_size", 0) or 0
    ):
        return False
    if int(gate_module.group_size) not in {32, 64, 128}:
        return False
    if str(getattr(gate_module, "mode", "affine")) != "affine":
        return False
    if str(getattr(up_module, "mode", "affine")) != "affine":
        return False
    if "bias" in gate_module or "bias" in up_module:
        return False
    if gate_module.scales.dtype != x.dtype or gate_module.biases.dtype != x.dtype:
        return False
    if up_module.scales.dtype != x.dtype or up_module.biases.dtype != x.dtype:
        return False
    if tuple(gate_module.weight.shape) != tuple(up_module.weight.shape):
        return False
    if tuple(gate_module.scales.shape) != tuple(up_module.scales.shape):
        return False
    if tuple(gate_module.biases.shape) != tuple(up_module.biases.shape):
        return False

    k = int(x.shape[-1])
    n = int(gate_module.weight.shape[0])
    packed_k = int(gate_module.weight.shape[1]) * 8
    if k != packed_k or k % 512 != 0 or n % 8 != 0:
        return False

    batch_count = 1
    for dim in x.shape[:-2]:
        batch_count *= int(dim)
    return batch_count == 1


@lru_cache(maxsize=None)
def _gate_up_swiglu_qmv4_kernel(group_size: int, dtype: mx.Dtype):
    header = """
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int PACK_FACTOR = 8;
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BYTES_PER_PACK = 4;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
        constant constexpr int RESULTS_PER_SIMDGROUP = 4;
        constant constexpr int NUM_SIMDGROUPS = 2;
        constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;
        constant constexpr int MAX_M = 6;

        template <typename T>
        inline T sigmoid_mlx_exact(T x) {
          auto y = 1 / (1 + metal::exp(metal::abs(x)));
          return (x < T(0)) ? y : 1 - y;
        }

        template <typename T>
        inline T swiglu_mlx_exact(T gate, T up) {
          T silu = gate * sigmoid_mlx_exact<T>(gate);
          return T(silu * up);
        }

        template <typename T>
        inline float load_vector4_exact(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          return sum;
        }

        inline float qdot4_exact(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          const device uint16_t* ws = (const device uint16_t*)w;
          float accum = 0.0f;
          for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
            uint16_t packed = ws[i];
            accum +=
              x_thread[4 * i] * float(packed & 0x000f) +
              x_thread[4 * i + 1] * float(packed & 0x00f0) +
              x_thread[4 * i + 2] * float(packed & 0x0f00) +
              x_thread[4 * i + 3] * float(packed & 0xf000);
          }
          return scale * accum + sum * bias;
        }
    """

    source = """
        uint n_tile = threadgroup_position_in_grid.y;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        int M = int(M_size);
        int K = int(K_size);
        int N = int(N_size);
        constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
        int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K / GS;

        const device uint8_t* gate_w_base =
          (const device uint8_t*)gate_w + out_row * in_vec_size_w
          + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device uint8_t* up_w_base =
          (const device uint8_t*)up_w + out_row * in_vec_size_w
          + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* gate_scales_base =
          gate_scales + out_row * in_vec_size_g
          + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* gate_biases_base =
          gate_biases + out_row * in_vec_size_g
          + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* up_scales_base =
          up_scales + out_row * in_vec_size_g
          + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* up_biases_base =
          up_biases + out_row * in_vec_size_g
          + int(simd_lid) / SCALE_STEP_PER_THREAD;

        float gate_result[MAX_M][RESULTS_PER_SIMDGROUP];
        float up_result[MAX_M][RESULTS_PER_SIMDGROUP];
        float x_thread[MAX_M][VALUES_PER_THREAD];
        float x_sum[MAX_M];

        for (int m = 0; m < MAX_M; ++m) {
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            gate_result[m][row] = 0.0f;
            up_result[m][row] = 0.0f;
          }
        }

        for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE) {
          for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
              const device T* x_m =
                x + m * K + k_block + int(simd_lid) * VALUES_PER_THREAD;
              x_sum[m] = load_vector4_exact<T>(x_m, x_thread[m]);
            }
          }

          const device uint8_t* gate_w_block =
            gate_w_base + k_block * BYTES_PER_PACK / PACK_FACTOR;
          const device uint8_t* up_w_block =
            up_w_base + k_block * BYTES_PER_PACK / PACK_FACTOR;
          const device T* gate_scales_block = gate_scales_base + k_block / GS;
          const device T* gate_biases_block = gate_biases_base + k_block / GS;
          const device T* up_scales_block = up_scales_base + k_block / GS;
          const device T* up_biases_block = up_biases_base + k_block / GS;

          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = out_row + row;
            if (n < N) {
              const device uint8_t* gate_w_row =
                gate_w_block + row * in_vec_size_w;
              const device uint8_t* up_w_row =
                up_w_block + row * in_vec_size_w;
              const device T* gate_sc_row =
                gate_scales_block + row * in_vec_size_g;
              const device T* gate_bs_row =
                gate_biases_block + row * in_vec_size_g;
              const device T* up_sc_row =
                up_scales_block + row * in_vec_size_g;
              const device T* up_bs_row =
                up_biases_block + row * in_vec_size_g;
              float gate_scale = float(gate_sc_row[0]);
              float gate_bias = float(gate_bs_row[0]);
              float up_scale = float(up_sc_row[0]);
              float up_bias = float(up_bs_row[0]);

              for (int m = 0; m < MAX_M; ++m) {
                if (m < M) {
                  gate_result[m][row] += qdot4_exact(
                    gate_w_row, x_thread[m], gate_scale, gate_bias, x_sum[m]
                  );
                  up_result[m][row] += qdot4_exact(
                    up_w_row, x_thread[m], up_scale, up_bias, x_sum[m]
                  );
                }
              }
            }
          }
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int n = out_row + row;
          if (n < N) {
            for (int m = 0; m < MAX_M; ++m) {
              if (m < M) {
                float gate_sum = simd_sum(gate_result[m][row]);
                float up_sum = simd_sum(up_result[m][row]);
                if (simd_lid == 0) {
                  T gate_value = T(gate_sum);
                  T up_value = T(up_sum);
                  y[m * N + n] = swiglu_mlx_exact<T>(gate_value, up_value);
                }
              }
            }
          }
        }
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_gate_up_swiglu_qmv4_gs{group_size}_{dtype_tag}",
        input_names=[
            "x",
            "gate_w",
            "gate_scales",
            "gate_biases",
            "up_w",
            "up_scales",
            "up_biases",
            "M_size",
            "K_size",
            "N_size",
        ],
        output_names=["y"],
        source=source,
        header=header,
    )


def gate_up_swiglu_qmv4_activation(
    x: mx.array,
    gate_module: nn.QuantizedLinear,
    up_module: nn.QuantizedLinear,
) -> mx.array:
    """Compute ``swiglu(gate_module(x), up_module(x))`` for M<=6.

    Unsupported shapes fall back to the stock MLX path so diagnostic callers can
    use this without changing correctness behavior.
    """
    from mlx_lm.models.qwen3_next import swiglu

    def fallback() -> mx.array:
        return swiglu(gate_module(x), up_module(x))
    if not is_gate_up_swiglu_qmv4_eligible(x, gate_module, up_module):
        return fallback()

    leading = x.shape[:-2]
    m = int(x.shape[-2])
    k = int(x.shape[-1])
    n = int(gate_module.weight.shape[0])
    x2 = x.reshape(m, k)
    kernel = _gate_up_swiglu_qmv4_kernel(int(gate_module.group_size), x.dtype)
    grid_y = 2 * ((n + 7) // 8)
    (y,) = kernel(
        inputs=[
            x2,
            gate_module.weight,
            gate_module.scales,
            gate_module.biases,
            up_module.weight,
            up_module.scales,
            up_module.biases,
            m,
            k,
            n,
        ],
        template=[("T", x.dtype), ("GS", int(gate_module.group_size))],
        grid=(32, grid_y, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(*leading, m, n)


@lru_cache(maxsize=None)
def _gate_up_swiglu_qmv4_rowwise_kernel(
    group_size: int,
    dtype: mx.Dtype,
    results_per_simdgroup: int,
    num_simdgroups: int,
):
    header = """
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int PACK_FACTOR = 8;
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BYTES_PER_PACK = 4;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
        constant constexpr int RESULTS_PER_SIMDGROUP = __RESULTS_PER_SIMDGROUP__;
        constant constexpr int NUM_SIMDGROUPS = __NUM_SIMDGROUPS__;
        constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

        template <typename T>
        inline T sigmoid_mlx_exact(T x) {
          auto y = 1 / (1 + metal::exp(metal::abs(x)));
          return (x < T(0)) ? y : 1 - y;
        }

        template <typename T>
        inline T swiglu_mlx_exact(T gate, T up) {
          T silu = gate * sigmoid_mlx_exact<T>(gate);
          return T(silu * up);
        }

        template <typename T>
        inline float load_vector4_exact(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          return sum;
        }

        inline float qdot4_exact(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          const device uint16_t* ws = (const device uint16_t*)w;
          float accum = 0.0f;
          for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
            uint16_t packed = ws[i];
            accum +=
              x_thread[4 * i] * float(packed & 0x000f) +
              x_thread[4 * i + 1] * float(packed & 0x00f0) +
              x_thread[4 * i + 2] * float(packed & 0x0f00) +
              x_thread[4 * i + 3] * float(packed & 0xf000);
          }
          return scale * accum + sum * bias;
        }
    """.replace("__RESULTS_PER_SIMDGROUP__", str(int(results_per_simdgroup))).replace(
        "__NUM_SIMDGROUPS__", str(int(num_simdgroups))
    )

    source = """
        uint m_idx = threadgroup_position_in_grid.x;
        uint n_tile = threadgroup_position_in_grid.y;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        int M = int(M_size);
        int K = int(K_size);
        int N = int(N_size);
        if (int(m_idx) >= M) {
          return;
        }

        constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
        int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K / GS;

        const device uint8_t* gate_w_base =
          (const device uint8_t*)gate_w + out_row * in_vec_size_w
          + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device uint8_t* up_w_base =
          (const device uint8_t*)up_w + out_row * in_vec_size_w
          + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* gate_scales_base =
          gate_scales + out_row * in_vec_size_g
          + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* gate_biases_base =
          gate_biases + out_row * in_vec_size_g
          + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* up_scales_base =
          up_scales + out_row * in_vec_size_g
          + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* up_biases_base =
          up_biases + out_row * in_vec_size_g
          + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* x_base =
          x + int(m_idx) * K + int(simd_lid) * VALUES_PER_THREAD;

        float gate_result[RESULTS_PER_SIMDGROUP] = {0.0f};
        float up_result[RESULTS_PER_SIMDGROUP] = {0.0f};
        float x_thread[VALUES_PER_THREAD];

        const device uint8_t* gate_ws = gate_w_base;
        const device uint8_t* up_ws = up_w_base;
        const device T* gate_sc = gate_scales_base;
        const device T* gate_bs = gate_biases_base;
        const device T* up_sc = up_scales_base;
        const device T* up_bs = up_biases_base;
        const device T* x_ptr = x_base;

        for (int k = 0; k < K; k += BLOCK_SIZE) {
          float x_sum = load_vector4_exact<T>(x_ptr, x_thread);

          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = out_row + row;
            if (n < N) {
              const device uint8_t* gate_wl = gate_ws + row * in_vec_size_w;
              const device uint8_t* up_wl = up_ws + row * in_vec_size_w;
              const device T* gate_sl = gate_sc + row * in_vec_size_g;
              const device T* gate_bl = gate_bs + row * in_vec_size_g;
              const device T* up_sl = up_sc + row * in_vec_size_g;
              const device T* up_bl = up_bs + row * in_vec_size_g;
              float gate_scale = float(gate_sl[0]);
              float gate_bias = float(gate_bl[0]);
              float up_scale = float(up_sl[0]);
              float up_bias = float(up_bl[0]);
              gate_result[row] += qdot4_exact(
                gate_wl, x_thread, gate_scale, gate_bias, x_sum
              );
              up_result[row] += qdot4_exact(
                up_wl, x_thread, up_scale, up_bias, x_sum
              );
            }
          }

          gate_ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          up_ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          gate_sc += BLOCK_SIZE / GS;
          gate_bs += BLOCK_SIZE / GS;
          up_sc += BLOCK_SIZE / GS;
          up_bs += BLOCK_SIZE / GS;
          x_ptr += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int n = out_row + row;
          if (n < N) {
            float gate_sum = simd_sum(gate_result[row]);
            float up_sum = simd_sum(up_result[row]);
            if (simd_lid == 0) {
              T gate_value = T(gate_sum);
              T up_value = T(up_sum);
              y[int(m_idx) * N + n] = swiglu_mlx_exact<T>(gate_value, up_value);
            }
          }
        }
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_gate_up_swiglu_qmv4_rowwise_bn{num_simdgroups * results_per_simdgroup}"
            f"_sg{num_simdgroups}_gs{group_size}_{dtype_tag}"
        ),
        input_names=[
            "x",
            "gate_w",
            "gate_scales",
            "gate_biases",
            "up_w",
            "up_scales",
            "up_biases",
            "M_size",
            "K_size",
            "N_size",
        ],
        output_names=["y"],
        source=source,
        header=header,
    )


def gate_up_swiglu_qmv4_activation_rowwise(
    x: mx.array,
    gate_module: nn.QuantizedLinear,
    up_module: nn.QuantizedLinear,
    *,
    results_per_simdgroup: int = 4,
    num_simdgroups: int = 2,
) -> mx.array:
    """Row-parallel exact qmv fusion for ``swiglu(gate(x), up(x))``."""
    from mlx_lm.models.qwen3_next import swiglu

    def fallback() -> mx.array:
        return swiglu(gate_module(x), up_module(x))
    if not is_gate_up_swiglu_qmv4_eligible(x, gate_module, up_module):
        return fallback()
    if results_per_simdgroup not in {4, 8, 16}:
        return fallback()
    if num_simdgroups not in {2, 4}:
        return fallback()

    leading = x.shape[:-2]
    m = int(x.shape[-2])
    k = int(x.shape[-1])
    n = int(gate_module.weight.shape[0])
    x2 = x.reshape(m, k)
    kernel = _gate_up_swiglu_qmv4_rowwise_kernel(
        int(gate_module.group_size),
        x.dtype,
        int(results_per_simdgroup),
        int(num_simdgroups),
    )
    bn = int(num_simdgroups) * int(results_per_simdgroup)
    grid_y = int(num_simdgroups) * ((n + bn - 1) // bn)
    (y,) = kernel(
        inputs=[
            x2,
            gate_module.weight,
            gate_module.scales,
            gate_module.biases,
            up_module.weight,
            up_module.scales,
            up_module.biases,
            m,
            k,
            n,
        ],
        template=[("T", x.dtype), ("GS", int(gate_module.group_size))],
        grid=(32 * m, grid_y, 1),
        threadgroup=(32, int(num_simdgroups), 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(*leading, m, n)


@lru_cache(maxsize=None)
def _gate_up_swiglu_qmv4_split_kernel(group_size: int, dtype: mx.Dtype):
    header = """
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int PACK_FACTOR = 8;
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BYTES_PER_PACK = 4;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
        constant constexpr int RESULTS_PER_SIMDGROUP = 4;
        constant constexpr int PROJECTION_SIMDGROUPS = 2;
        constant constexpr int NUM_SIMDGROUPS = 4;
        constant constexpr int BN = RESULTS_PER_SIMDGROUP * PROJECTION_SIMDGROUPS;

        template <typename T>
        inline T sigmoid_mlx_exact(T x) {
          auto y = 1 / (1 + metal::exp(metal::abs(x)));
          return (x < T(0)) ? y : 1 - y;
        }

        template <typename T>
        inline T swiglu_mlx_exact(T gate, T up) {
          T silu = gate * sigmoid_mlx_exact<T>(gate);
          return T(silu * up);
        }

        template <typename T>
        inline float load_vector4_exact(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          return sum;
        }

        inline float qdot4_exact(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          const device uint16_t* ws = (const device uint16_t*)w;
          float accum = 0.0f;
          for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
            uint16_t packed = ws[i];
            accum +=
              x_thread[4 * i] * float(packed & 0x000f) +
              x_thread[4 * i + 1] * float(packed & 0x00f0) +
              x_thread[4 * i + 2] * float(packed & 0x0f00) +
              x_thread[4 * i + 3] * float(packed & 0xf000);
          }
          return scale * accum + sum * bias;
        }
    """

    source = """
        uint m_idx = threadgroup_position_in_grid.x;
        uint n_tile = threadgroup_position_in_grid.y;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        int M = int(M_size);
        int K = int(K_size);
        int N = int(N_size);
        if (int(m_idx) >= M) {
          return;
        }

        bool up_lane = simd_gid >= PROJECTION_SIMDGROUPS;
        uint pair_gid = simd_gid - (up_lane ? PROJECTION_SIMDGROUPS : 0);
        int out_row = int(n_tile) * BN + int(pair_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K / GS;
        constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;

        const device uint8_t* w_base =
          (const device uint8_t*)(up_lane ? up_w : gate_w)
          + out_row * in_vec_size_w
          + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* scales_base =
          (up_lane ? up_scales : gate_scales)
          + out_row * in_vec_size_g
          + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* biases_base =
          (up_lane ? up_biases : gate_biases)
          + out_row * in_vec_size_g
          + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* x_base =
          x + int(m_idx) * K + int(simd_lid) * VALUES_PER_THREAD;

        float result[RESULTS_PER_SIMDGROUP] = {0.0f};
        float x_thread[VALUES_PER_THREAD];

        const device uint8_t* ws = w_base;
        const device T* sc = scales_base;
        const device T* bs = biases_base;
        const device T* x_ptr = x_base;

        for (int k = 0; k < K; k += BLOCK_SIZE) {
          float x_sum = load_vector4_exact<T>(x_ptr, x_thread);

          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = out_row + row;
            if (n < N) {
              const device uint8_t* wl = ws + row * in_vec_size_w;
              const device T* sl = sc + row * in_vec_size_g;
              const device T* bl = bs + row * in_vec_size_g;
              result[row] += qdot4_exact(
                wl, x_thread, float(sl[0]), float(bl[0]), x_sum
              );
            }
          }

          ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          x_ptr += BLOCK_SIZE;
        }

        threadgroup T gate_values[BN];
        threadgroup T up_values[BN];

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int n = out_row + row;
          if (n < N) {
            float reduced = simd_sum(result[row]);
            if (simd_lid == 0) {
              int local_n = int(pair_gid) * RESULTS_PER_SIMDGROUP + row;
              if (up_lane) {
                up_values[local_n] = T(reduced);
              } else {
                gate_values[local_n] = T(reduced);
              }
            }
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!up_lane && simd_lid == 0) {
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = out_row + row;
            if (n < N) {
              int local_n = int(pair_gid) * RESULTS_PER_SIMDGROUP + row;
              y[int(m_idx) * N + n] = swiglu_mlx_exact<T>(
                gate_values[local_n], up_values[local_n]
              );
            }
          }
        }
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_gate_up_swiglu_qmv4_split_gs{group_size}_{dtype_tag}",
        input_names=[
            "x",
            "gate_w",
            "gate_scales",
            "gate_biases",
            "up_w",
            "up_scales",
            "up_biases",
            "M_size",
            "K_size",
            "N_size",
        ],
        output_names=["y"],
        source=source,
        header=header,
    )


def gate_up_swiglu_qmv4_activation_split(
    x: mx.array,
    gate_module: nn.QuantizedLinear,
    up_module: nn.QuantizedLinear,
) -> mx.array:
    """Split simdgroups between gate/up qmv work, then combine inside one TG.

    This keeps the stock qmv output-tile parallelism but avoids carrying both
    gate and up accumulators in every simdgroup.
    """
    from mlx_lm.models.qwen3_next import swiglu

    def fallback() -> mx.array:
        return swiglu(gate_module(x), up_module(x))
    if not is_gate_up_swiglu_qmv4_eligible(x, gate_module, up_module):
        return fallback()

    leading = x.shape[:-2]
    m = int(x.shape[-2])
    k = int(x.shape[-1])
    n = int(gate_module.weight.shape[0])
    x2 = x.reshape(m, k)
    kernel = _gate_up_swiglu_qmv4_split_kernel(int(gate_module.group_size), x.dtype)
    (y,) = kernel(
        inputs=[
            x2,
            gate_module.weight,
            gate_module.scales,
            gate_module.biases,
            up_module.weight,
            up_module.scales,
            up_module.biases,
            m,
            k,
            n,
        ],
        template=[("T", x.dtype), ("GS", int(gate_module.group_size))],
        grid=(32 * m, 4 * ((n + 7) // 8), 1),
        threadgroup=(32, 4, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(*leading, m, n)


def is_small_m_qmm4_eligible(x: mx.array, module: Any) -> bool:
    """Return whether the BM=8 simdgroup-MMA qmm prototype can run."""
    if not mx.metal.is_available():
        return False
    if not isinstance(module, nn.QuantizedLinear):
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if len(x.shape) < 2:
        return False
    m = int(x.shape[-2])
    if m <= 0 or m > 8:
        return False
    if int(getattr(module, "bits", 0) or 0) != 4:
        return False
    if int(getattr(module, "group_size", 0) or 0) not in {32, 64, 128}:
        return False
    if str(getattr(module, "mode", "affine")) != "affine":
        return False
    if module.scales.dtype != x.dtype or module.biases.dtype != x.dtype:
        return False
    k = int(x.shape[-1])
    n = int(module.weight.shape[0])
    if k != int(module.weight.shape[1]) * 8:
        return False
    if k % 32 != 0 or n % 32 != 0:
        return False
    batch_count = 1
    for dim in x.shape[:-2]:
        batch_count *= int(dim)
    return batch_count == 1


@lru_cache(maxsize=None)
def _small_m_qmm4_kernel(group_size: int, dtype: mx.Dtype):
    source = f"""
        using namespace metal;
        constexpr int BM = 8;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = {group_size};

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint tg_n  = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8  = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;

        threadgroup T B_tile[BK * BN];

        simdgroup_matrix<T, 8, 8> a, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_L =
          simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_R =
          simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;
        int sg_n_off = int(sg_id) * 16;

        for (int k0 = 0; k0 < K; k0 += BK) {{
            {{
                int n_global = n0 + dq_n_a;
                int k_base = k0 + dq_k_a * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[(dq_k_a * 8 + ki) * BN + dq_n_a] =
                      T(float(nib) * s + b);
                }}
            }}
            {{
                int n_global = n0 + dq_n_b;
                int k_base = k0 + dq_k_b * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[(dq_k_b * 8 + ki) * BN + dq_n_b] =
                      T(float(nib) * s + b);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a, x + k0 + ks * BK_SUB, K);
                simdgroup_load(
                  b_L, B_tile + ks * BK_SUB * BN + sg_n_off, BN
                );
                simdgroup_load(
                  b_R, B_tile + ks * BK_SUB * BN + sg_n_off + 8, BN
                );
                simdgroup_multiply_accumulate(c_L, a, b_L, c_L);
                simdgroup_multiply_accumulate(c_R, a, b_R, c_R);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        simdgroup_matrix<T, 8, 8> c_L_T, c_R_T;
        c_L_T.thread_elements()[0] = T(c_L.thread_elements()[0]);
        c_L_T.thread_elements()[1] = T(c_L.thread_elements()[1]);
        c_R_T.thread_elements()[0] = T(c_R.thread_elements()[0]);
        c_R_T.thread_elements()[1] = T(c_R.thread_elements()[1]);
        simdgroup_store(c_L_T, y + n0 + sg_n_off, N);
        simdgroup_store(c_R_T, y + n0 + sg_n_off + 8, N);
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_small_m_qmm4_bm8_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )


def small_m_qmm4_matmul(x: mx.array, module: nn.QuantizedLinear) -> mx.array:
    """Compute ``module(x)`` with a BM=8 dflash-style qmm prototype."""
    def fallback() -> mx.array:
        return module(x)
    if not is_small_m_qmm4_eligible(x, module):
        return fallback()

    orig_shape = x.shape
    m = int(orig_shape[-2])
    k = int(orig_shape[-1])
    n = int(module.weight.shape[0])
    x2 = x.reshape(m, k)
    if m < 8:
        x2 = mx.concatenate([x2, mx.zeros((8 - m, k), dtype=x.dtype)], axis=0)
    x2 = mx.contiguous(x2)
    kernel = _small_m_qmm4_kernel(int(module.group_size), x.dtype)
    (y8,) = kernel(
        inputs=[x2, module.weight, module.scales, module.biases, m, k, n],
        template=[("T", x.dtype)],
        grid=(64, n // 32, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(8, n)],
        output_dtypes=[x.dtype],
    )
    y = y8[:m, :]
    if "bias" in module:
        y = y + module["bias"]
    return y.reshape(*orig_shape[:-1], n)
