"""Metal owner-copy helper for cache leaves."""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


def _numel(shape) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


@lru_cache(maxsize=None)
def _copy_leaf_kernel(dtype: mx.Dtype):
    source = """
        uint index = thread_position_in_grid.x;
        if (index < uint(total_size)) {
          y[index] = x[index];
        }
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16", mx.float32: "fp32"}.get(dtype, "generic")
    return mx.fast.metal_kernel(
        name=f"mtplx_copy_leaf_{dtype_tag}",
        input_names=["x", "total_size"],
        output_names=["y"],
        source=source,
    )


def metal_copy_leaf(value: mx.array, *, threadgroup_size: int = 256) -> mx.array:
    """Return an independent Metal-kernel copy of ``value``.

    Unsupported devices fall back to ``mx.contiguous`` so callers can keep one
    detach mode in tests and non-Metal environments.
    """

    source = mx.contiguous(value)
    if not mx.metal.is_available():
        mx.eval(source)
        return source
    total = _numel(source.shape)
    kernel = _copy_leaf_kernel(source.dtype)
    (copied,) = kernel(
        inputs=[source.reshape(total), int(total)],
        template=[("T", source.dtype)],
        grid=(total, 1, 1),
        threadgroup=(int(threadgroup_size), 1, 1),
        output_shapes=[(total,)],
        output_dtypes=[source.dtype],
    )
    copied = copied.reshape(*value.shape)
    mx.eval(copied)
    return copied
