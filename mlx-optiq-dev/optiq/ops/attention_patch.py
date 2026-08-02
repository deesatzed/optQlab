"""Route mlx-lm's ``scaled_dot_product_attention`` through a memory-bounded
attention during LoRA training — but only when it pays for itself.

Stock MLX autographs its fused SDPA and materializes the ``[B, Hq, T, T]``
score tensor in the backward. Gradient checkpointing keeps one layer's copy
live at a time but cannot shrink it, so a high-head-count model at long context
runs out of memory: 32 heads at 16k is 48 GiB of scores, and Metal dies.

``flash_attention_tiled`` keeps the stock fused forward and replaces only the
backward with FlashAttention-2 recomputation over query blocks. Measured on
Hq=32, Hkv=8, D=256, bf16 causal::

      T    stock                tiled
   8192    755 ms / 12.75 GiB   1360 ms / 5.86 GiB
  16384    Metal OOM            5391 ms / 18.25 GiB

So the tiled path costs ~2x stock and buys the memory back. (Its predecessor,
the hand-written ``flash_attention_metal`` kernel, bounded memory even harder
but ran 14-137x slower than stock because it does scalar loops instead of
matmul; it is superseded and kept only for reference.)

2x is still 2x, so routing is memory-aware: estimate what stock's backward
would materialize, compare against a budget (25% of Metal's wired limit by
default), and take the tiled path only when stock would not fit. Short contexts
and small head counts keep the fastest path available.

``OPTIQ_FLASH_ATTN=always|never`` forces the decision,
``OPTIQ_FLASH_ATTN_BUDGET_GB`` overrides the budget, and ``OPTIQ_FLASH_BLOCK``
sets the query-block size (default 128).

``enable_flash_attention_training()`` is a context manager (with a nested-enter
counter) that installs the router while active and restores the original on
exit. It falls through to the original function when:

  * cache is not ``None`` (decoding, not training)
  * dtypes of q/k/v disagree, or they are not rank-4
  * Hq is not a multiple of Hkv (GQA grouping)
  * queries seq length < 32 (block overhead dominates)
  * sinks is not ``None``
  * the mask is neither ``None`` nor an effectively causal array

so generation, MoE routing and sinks-enabled models stay bit-identical.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from contextlib import contextmanager
from typing import Optional

import mlx.core as mx

from .flash_attention_tiled import flash_attention_tiled

logger = logging.getLogger(__name__)

GIB = 1024 ** 3

# The tiled backward costs ~2x stock (measured 1.91-2.03x across shapes) and
# bounds peak memory. Stock is still the fastest thing available when its
# O(seq^2) score tensor fits, so route on that: what stock materializes is
# [B, Hq, T, T] (independent of head_dim), and gradient checkpointing keeps one
# layer's copy live at a time but cannot shrink it.
_ATTN_MATRIX_FACTOR = 3          # scores + softmax + grad; calibrated, upper bound
_DEFAULT_BUDGET_FRACTION = 0.25  # of usable GPU memory


def _usable_gpu_bytes() -> float:
    """Metal's wired limit if set, else ~75% of installed RAM."""
    try:
        mb = int(subprocess.check_output(
            ["sysctl", "-n", "iogpu.wired_limit_mb"], text=True).strip())
        if mb > 0:
            return mb * 1024 * 1024
    except Exception:
        pass
    try:
        total = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True).strip())
        return total * 0.75
    except Exception:
        return 16 * GIB


def _budget_bytes() -> float:
    env = os.environ.get("OPTIQ_FLASH_ATTN_BUDGET_GB")
    if env:
        try:
            return float(env) * GIB
        except ValueError:
            pass
    return _usable_gpu_bytes() * _DEFAULT_BUDGET_FRACTION


def _stock_attention_bytes(queries) -> float:
    """Bytes stock SDPA's backward will materialize for [B, Hq, T, T]."""
    B, Hq, T, _ = queries.shape
    return _ATTN_MATRIX_FACTOR * B * Hq * T * T * queries.dtype.size


_logged: set = set()


def _use_kernel(queries) -> bool:
    """auto: only pay the kernel when stock would blow the memory budget."""
    mode = os.environ.get("OPTIQ_FLASH_ATTN", "auto").lower()
    if mode == "always":
        return True
    if mode == "never":
        return False
    need = _stock_attention_bytes(queries)
    budget = _budget_bytes()
    use = need > budget
    key = (queries.shape, use)
    if key not in _logged:
        _logged.add(key)
        logger.info(
            "attention backward: %s. Stock SDPA would materialize %.2f GiB of "
            "scores, budget %.2f GiB (shape %s). Override with "
            "OPTIQ_FLASH_ATTN=always|never.",
            "tiled (memory-bounded, ~2x stock)" if use else "stock (fits)",
            need / GIB, budget / GIB, tuple(queries.shape),
        )
    return use


# Guard so we can nest / re-enter without double-patching.
_patch_state = threading.local()


def _router(original_fn):
    """Return a patched scaled_dot_product_attention that routes to our
    Metal kernel when the input shape is compatible with the kernel.
    """

    def _patched(queries, keys, values, cache, scale, mask, sinks=None):
        # Fast fail-outs: anything we don't handle goes to the original.
        if cache is not None:
            return original_fn(queries, keys, values, cache, scale, mask, sinks=sinks)
        if sinks is not None:
            return original_fn(queries, keys, values, cache, scale, mask, sinks=sinks)
        if keys.dtype != queries.dtype or values.dtype != queries.dtype:
            return original_fn(queries, keys, values, cache, scale, mask, sinks=sinks)
        if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
            return original_fn(queries, keys, values, cache, scale, mask, sinks=sinks)
        if queries.shape[1] % keys.shape[1] != 0:      # GQA groups must divide
            return original_fn(queries, keys, values, cache, scale, mask, sinks=sinks)
        # Very short sequences: block overhead dominates, MLX fused SDPA wins.
        if queries.shape[-2] < 32:
            return original_fn(queries, keys, values, cache, scale, mask, sinks=sinks)

        # We only support causal masks or unmasked for now. mlx-lm's LM
        # training always passes a causal mask (either string "causal",
        # an additive float mask, or None when L==1). We treat array
        # masks as causal iff they look lower-triangular by shape; if
        # not we fall back.
        causal = True
        if mask is None:
            causal = False  # no mask at all, allow full attention
        elif isinstance(mask, str):
            if mask != "causal":
                return original_fn(queries, keys, values, cache, scale, mask, sinks=sinks)
            causal = True
        elif isinstance(mask, mx.array):
            # mlx-lm's ``create_attention_mask`` yields a 2D additive
            # float mask (0 and -inf) that IS lower-triangular. We
            # trust shape (S, S) or (1, 1, S, S) here. If it's not
            # square in the last two dims, we bail.
            if mask.shape[-1] != queries.shape[-2] or mask.shape[-2] != queries.shape[-2]:
                return original_fn(queries, keys, values, cache, scale, mask, sinks=sinks)
            causal = True
        else:
            return original_fn(queries, keys, values, cache, scale, mask, sinks=sinks)

        # Shape is compatible. Now: is the tiled backward worth its cost here?
        if not _use_kernel(queries):
            return original_fn(queries, keys, values, cache, scale, mask, sinks=sinks)
        return flash_attention_tiled(queries, keys, values, scale=scale, causal=causal)

    return _patched


def _get_base_module():
    import mlx_lm.models.base as _base
    return _base


def _install() -> None:
    base = _get_base_module()
    if getattr(base, "_optiq_flash_installed", False):
        base._optiq_flash_depth = getattr(base, "_optiq_flash_depth", 0) + 1
        return
    orig = base.scaled_dot_product_attention
    base._optiq_flash_original_sdpa = orig
    base.scaled_dot_product_attention = _router(orig)
    base._optiq_flash_installed = True
    base._optiq_flash_depth = 1

    # The mlx-lm submodules import ``scaled_dot_product_attention`` by name
    # at module-load time. Rebinding on the base module alone isn't enough —
    # the submodules captured the original reference. Walk a known set and
    # rebind explicitly. This is a surgical list, safe to extend.
    for modname in (
        "mlx_lm.models.qwen3_next",
        "mlx_lm.models.qwen3_5",
        "mlx_lm.models.qwen3",
        "mlx_lm.models.qwen2",
        "mlx_lm.models.gemma4_text",
        "mlx_lm.models.gemma3",
        "mlx_lm.models.gemma3_text",
        "mlx_lm.models.llama",
    ):
        try:
            import importlib
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        if hasattr(mod, "scaled_dot_product_attention"):
            # Remember originals so we can restore on uninstall.
            if not hasattr(mod, "_optiq_flash_original_sdpa"):
                mod._optiq_flash_original_sdpa = mod.scaled_dot_product_attention
            mod.scaled_dot_product_attention = base.scaled_dot_product_attention


def _uninstall() -> None:
    base = _get_base_module()
    if not getattr(base, "_optiq_flash_installed", False):
        return
    depth = getattr(base, "_optiq_flash_depth", 1) - 1
    base._optiq_flash_depth = depth
    if depth > 0:
        return
    base.scaled_dot_product_attention = base._optiq_flash_original_sdpa
    del base._optiq_flash_original_sdpa
    base._optiq_flash_installed = False

    for modname in (
        "mlx_lm.models.qwen3_next",
        "mlx_lm.models.qwen3_5",
        "mlx_lm.models.qwen3",
        "mlx_lm.models.qwen2",
        "mlx_lm.models.gemma4_text",
        "mlx_lm.models.gemma3",
        "mlx_lm.models.gemma3_text",
        "mlx_lm.models.llama",
    ):
        try:
            import importlib
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        if hasattr(mod, "_optiq_flash_original_sdpa"):
            mod.scaled_dot_product_attention = mod._optiq_flash_original_sdpa
            del mod._optiq_flash_original_sdpa


@contextmanager
def enable_flash_attention_training():
    """Context manager that routes LM training through Metal flash attention."""
    _install()
    try:
        yield
    finally:
        _uninstall()
