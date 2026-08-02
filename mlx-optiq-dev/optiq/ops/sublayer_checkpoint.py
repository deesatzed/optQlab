"""Sub-layer gradient checkpointing.

mlx-lm's ``grad_checkpoint(layer)`` patches ``type(layer).__call__`` so ALL
instances of that class are checkpointed. Typical usage wraps the outer
``DecoderLayer``, which means backward recomputes the entire layer
(attention + norms + MLP + residuals) as one atomic unit.

Sub-layer checkpointing targets the inner modules — ``Attention``,
``MLP``, ``GatedDeltaNet`` — so each gets its own checkpoint boundary.
The trade-off:

  + Finer backward working set (recompute just attn or just MLP, never
    both simultaneously).
  - One extra saved input per sub-layer per block (attn_in + mlp_in
    instead of just layer_in), so forward state is slightly larger.

Worth measuring before trusting. The theoretical win is that the PEAK
during backward's recompute is bounded by max(attn_fwd, mlp_fwd) rather
than the union. On Qwen3.5 where MLP intermediates (~280 MB) dominate
attention intermediates (~100 MB), this could save a few hundred MB.
"""

from __future__ import annotations

import mlx.nn as nn

from mlx_lm.tuner.trainer import grad_checkpoint as _mlx_grad_checkpoint


# Class name fragments we know about — we walk the model and patch any
# matching class once. Keep the list narrow so we don't patch accidentally.
_SUBLAYER_CLASSES = (
    "Qwen3NextAttention",
    "Qwen3NextMLP",
    "GatedDeltaNet",
    "Attention",   # Gemma-4 / Llama style
    "MLP",          # Gemma-4 / Llama style
    "FeedForward", # some models
)


def enable_sublayer_checkpoint(model: nn.Module) -> list[str]:
    """Apply ``grad_checkpoint`` to sub-layer modules (Attention, MLP, …)
    instead of the whole ``DecoderLayer``.

    Returns the list of class names we patched. Raises nothing — if a class
    isn't found, it's skipped.
    """
    patched_types: set[type] = set()
    patched_names: list[str] = []
    for name, mod in model.named_modules():
        cls = type(mod)
        if cls in patched_types:
            continue
        if cls.__name__ not in _SUBLAYER_CLASSES:
            continue
        # Avoid accidentally patching the outer LM wrappers (``Model``).
        if cls.__name__ in ("Model", "TextModel"):
            continue
        _mlx_grad_checkpoint(mod)
        patched_types.add(cls)
        patched_names.append(cls.__name__)
    return patched_names
