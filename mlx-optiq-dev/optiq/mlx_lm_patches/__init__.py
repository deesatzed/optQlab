"""Runtime patches that fix mlx-lm gaps for OptiQ.

Importing this module (or just ``optiq``) registers:

  * ``qwen3_5_text`` — a pure-text Model wrapper around mlx-lm's
    ``qwen3_5.Qwen3_5TextModel``, without the VLM ``.language_model``
    wrapper. Fills the gap left by mlx-lm only shipping ``qwen3_5.py``
    (VLM-flavored) while text-only deployments need a matching shape.

  * **chunked ``mlx.nn.quantize``** — replaces stock ``mlx.nn.quantize``
    with a per-module-flushing variant so large MoE bases (Gemma-4-26B-A4B,
    Qwen3.5/3.6-MoE) don't blow Metal's GPU command-buffer timeout during
    ``mlx_lm.convert()``. See ``_chunked_quantize.py`` for the full story.

After import, ``mlx_lm.utils.load(...)`` accepts models whose
``config.json`` declares ``model_type: qwen3_5_text``, and
``mlx_lm.convert(..., quantize=True)`` survives quantizing models with
hundreds of large expert tensors.
"""

from . import qwen3_5_text as _qwen3_5_text
from . import _register
from . import _chunked_quantize

_register.register()
_chunked_quantize.register_chunked_quantize()

__all__ = ["_register", "_chunked_quantize"]
