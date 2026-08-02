"""NEFTune — Noisy Embedding instruction Fine-Tuning (Jain et al. 2023).

Adds uniform noise, scaled by ``alpha / sqrt(seq_len * embed_dim)``, to the
token-embedding output during the *training* forward pass and nothing at
inference. A cheap regularizer that improves instruction-following on small
SFT datasets (the paper reports sizeable AlpacaEval gains).

Implementation notes:

  * The noise is gated on the embedding module's ``training`` flag, which
    mlx-lm's trainer toggles (``model.eval()`` around validation, ``model.train()``
    for the step loop). So validation loss is measured on clean embeddings —
    exactly HuggingFace/TRL's behaviour.
  * We intercept the embedding's ``__call__`` via a per-instance subclass swap.
    Python resolves ``emb(x)`` through ``type(emb).__call__``, so assigning an
    instance attribute wouldn't take; swapping ``__class__`` to a dynamically
    created subclass of the same type does, and leaves the instance dict
    (weights, tied ``as_linear`` head, LoRA params) untouched and reversible.
  * Only the lookup path (``__call__``) gets noise. The tied LM head uses the
    same weight via ``as_linear``, which is a separate method and stays clean.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn


def _find_token_embedding(model: nn.Module) -> Optional[nn.Module]:
    """Locate the input token-embedding module across mlx-lm arches.

    Tries the conventional attribute paths first (``model.model.embed_tokens``
    for Llama/Qwen/Gemma, ``model.model.embed`` for a few others, and the VLM
    ``language_model`` nesting), then falls back to scanning for the single
    Embedding whose vocab dimension matches the model's ``vocab_size``.
    """
    from mlx.nn import Embedding
    try:
        from mlx.nn import QuantizedEmbedding
        _emb_types = (Embedding, QuantizedEmbedding)
    except Exception:
        _emb_types = (Embedding,)

    for path in ("model.embed_tokens", "model.embed", "embed_tokens", "embed",
                 "language_model.model.embed_tokens",
                 "language_model.embed_tokens", "model.model.embed_tokens"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, _emb_types):
            return obj

    # Fallback: scan named modules for an embedding matching the vocab size.
    vocab = None
    for attr in ("args", "config"):
        cfg = getattr(model, attr, None)
        vocab = getattr(cfg, "vocab_size", None) if cfg is not None else None
        if vocab:
            break
    try:
        modules = model.named_modules()
    except Exception:
        modules = []
    fallback = None
    for _name, mod in modules:
        if isinstance(mod, _emb_types):
            w = getattr(mod, "weight", None)
            if w is None:
                continue
            if vocab and w.shape[0] == vocab:
                return mod
            fallback = fallback or mod
    return fallback


def enable_neftune(model: nn.Module, alpha: float) -> Optional[Callable[[], None]]:
    """Turn on NEFTune for ``model``'s token embeddings.

    Returns a zero-arg callable that restores the original embedding class, or
    ``None`` if no token embedding could be found (caller should warn + proceed).
    A non-positive ``alpha`` is a no-op returning ``None``.
    """
    alpha = float(alpha)
    if alpha <= 0:
        return None
    emb = _find_token_embedding(model)
    if emb is None:
        return None

    orig_type = type(emb)

    class _NEFTuneEmbedding(orig_type):  # type: ignore[valid-type,misc]
        def __call__(self, x):  # noqa: D401
            out = super().__call__(x)
            # Only during training (mlx-lm sets model.eval() around validation).
            if getattr(self, "training", False) and alpha > 0 and out.ndim >= 2:
                seq_len, dim = out.shape[-2], out.shape[-1]
                scale = alpha / math.sqrt(seq_len * dim)
                noise = mx.random.uniform(
                    low=-1.0, high=1.0, shape=out.shape).astype(out.dtype)
                out = out + scale * noise
            return out

    emb.__class__ = _NEFTuneEmbedding

    def _restore() -> None:
        emb.__class__ = orig_type

    return _restore
