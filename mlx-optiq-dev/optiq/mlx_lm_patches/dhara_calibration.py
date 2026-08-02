"""Tri-mode calibration for dhara sensitivity analysis.

OptiQ's sensitivity engine (``analyze_sensitivity_exact``) measures per-layer KL of
the output logits against an unquantized reference, over a set of calibration
forwards. For an autoregressive LM those forwards are plain ``model(input_ids)``
calls — a CAUSAL forward.

dhara is not an autoregressive LM. It is **tri-mode**: the same weights serve a causal
AR path and a **block-diffusion** path (masked tokens + a block-causal mask, blocks
defined by global position ``pos // block_len``). Calibrating only on the causal
forward means the mixed-precision optimizer never observes the diffusion path — so it
happily crushes the layers only block-diffusion depends on, because crushing them costs
nothing on the metric it is watching.

That is not hypothetical: an OptiQ-quantized 4-bit dhara has a pristine AR path and a
block-diffusion path that collapses into repetition ("water, water, water..."), while a
*uniform* 4-bit build of the same weights keeps diffusion coherent. Same bit-width —
the difference is which path the bit allocation was optimized for.

DiffusionGemma already solves this with ``vlm/diffusion_gemma/calibration.py``
(masked-canvas calibration). This is dhara's equivalent, with one addition: because
dhara must serve BOTH modes from one set of weights, the calibration probes BOTH — so
sensitivity reflects the true cost of quantizing a layer for the model as a whole.

Returns a ``calibration_fn`` of exactly the shape ``analyze_sensitivity_exact`` expects:
a zero-arg callable returning ``[(args_tuple, kwargs_dict), ...]``, each consumed as
``model(*args, **kwargs)``.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from .dhara_ar import build_block_causal_mask

DEFAULT_BLOCK_LEN = 32          # dhara's training block ("block 32")

# The block-diffusion decoder does not sit at one noise level. It appends a block
# that is ENTIRELY MASK and iteratively commits tokens down to 0% (see the
# reference implementation: `seq = cat([cur, full((1, block_len), MASK)])`, then
# `for _ in range(block_len)`). Probing one fraction measures a single slice of
# that trajectory — and 1.0, the state every block starts from, was never sampled
# at all. Sweep the whole path, the same reason DiffusionGemma's calibration
# samples several denoising steps.
DEFAULT_MASK_FRACS = (1.0, 0.75, 0.5, 0.25)
DEFAULT_MASK_FRAC = 0.5         # kept for callers that pass a single fraction


def _block_causal_mask(seq_len: int, block_len: int, dt) -> mx.array:
    """The additive (1,1,S,S) block-causal bias the decoder actually attends under.

    Delegates to ``dhara_ar.build_block_causal_mask`` — the same function
    ``dhara_decode`` uses at inference — rather than re-deriving it. A second copy
    would silently drift from the real one, and then the calibration would be
    scoring a forward the model never runs.
    """
    return build_block_causal_mask(seq_len, block_len, dt)


def _mask_trailing_block(ids, mask_token_id: int, block_len: int, frac: float):
    """Replace a random ``frac`` of the final block's tokens with the MASK id.

    This reproduces what the diffusion decoder actually feeds the model mid-refinement:
    a partially-filled block (some positions committed, some still MASK).
    """
    arr = np.asarray(ids, dtype=np.int32).copy()
    n = arr.shape[-1]
    start = max(0, n - block_len)
    span = np.arange(start, n)
    if span.size == 0:
        return arr
    k = max(1, int(span.size * frac))
    rng = np.random.default_rng(0)                    # deterministic: sensitivity must be reproducible
    arr[rng.choice(span, size=k, replace=False)] = mask_token_id
    return arr


def make_dhara_calibration(base_calibration_fn, model, *,
                           block_len: int = DEFAULT_BLOCK_LEN,
                           mask_fracs: tuple[float, ...] = DEFAULT_MASK_FRACS):
    """Wrap an AR calibration_fn so half its samples probe the DIFFUSION forward.

    ``base_calibration_fn`` is the standard LLM calibration (``load_llm_calibration``),
    which yields causal samples ``((input_ids,), {})``. We keep those (the AR path still
    matters) and convert every other sample into a diffusion probe:

        ((masked_input_ids,), {"mask": block_causal_mask})

    so ``model(*args, **kwargs)`` exercises the block-diffusion path. Sensitivity is
    then measured over both modes, and the bit allocation preserves both.

    The diffusion probes cycle through ``mask_fracs``, because the decoder refines a
    block from ~100% masked down to 0% — one fixed fraction would measure a single
    slice of that trajectory.
    """
    mask_id = int(model.args.mask_token_id)
    dt = getattr(model, "dtype", None) or mx.bfloat16
    fracs = tuple(mask_fracs) if isinstance(mask_fracs, (tuple, list)) else (float(mask_fracs),)

    def calibration_fn():
        out = []
        n_diff = 0
        for i, (args, kwargs) in enumerate(base_calibration_fn()):
            ids = args[0]
            if i % 2 == 0:
                out.append(((ids,), dict(kwargs)))                 # AR / causal probe
                continue
            frac = fracs[n_diff % len(fracs)]
            n_diff += 1
            arr = np.asarray(ids)
            flat = arr[0] if arr.ndim == 2 else arr
            masked = _mask_trailing_block(flat, mask_id, block_len, frac)
            masked = mx.array(masked)[None] if arr.ndim == 2 else mx.array(masked)
            S = masked.shape[-1]
            out.append(((masked,), {"mask": _block_causal_mask(S, block_len, dt)}))
        print(f"  [dhara-calib] {len(out)} samples: {len(out) - n_diff} causal AR + "
              f"{n_diff} block-diffusion over mask fractions {fracs}")
        return out

    return calibration_fn


def is_dhara(model) -> bool:
    """True for dhara's tri-mode arch (registered by OptiQ as ``dhara_ar``)."""
    return getattr(model, "model_type", "") == "dhara_ar" or hasattr(
        getattr(model, "args", None), "mask_token_id")
