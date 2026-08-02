"""Pure-text ``qwen3_5_text`` model type for mlx-lm.

mlx-lm's ``models/qwen3_5.py`` defines two top-level LM classes:

  * ``TextModel`` — pure text LLM wrapper (``self.model = Qwen3_5TextModel``,
    optional ``self.lm_head``, clean ``sanitize`` / ``make_cache`` /
    ``quant_predicate``).
  * ``Model`` — VLM wrapper that nests TextModel under
    ``self.language_model``.

Upstream only registers ``Model`` for ``model_type: qwen3_5``. Text-only
deployments (our OptiQ variants) are better served by ``TextModel``
directly — no VLM prefix, cleaner Hugging Face metadata, PEFT-compatible
weight paths out of the box.

This module aliases ``TextModel`` as the ``Model`` class for
``model_type: qwen3_5_text``, and reuses ``TextModelArgs`` as
``ModelArgs``. Registration lives in ``_register.py``.

This is a pure facade — no duplicate code. If upstream changes
``TextModel``, our ``qwen3_5_text`` follows automatically.

Caveat: models carrying ``model_type: qwen3_5_text`` require ``optiq``
to be installed on the loader side (or an equivalent upstream patch).
Bare ``mlx_lm`` can't resolve the type. For HF-published models intended
to work with stock mlx-lm, keep ``model_type: qwen3_5`` — the existing
VLM wrapper loads the weights correctly and generation output is
bit-identical.
"""

from __future__ import annotations

from mlx_lm.models.qwen3_5 import (  # type: ignore
    TextModel as _TextModel,
    TextModelArgs as _TextModelArgs,
)


class ModelArgs(_TextModelArgs):
    """Alias for mlx-lm's Qwen3_5 TextModelArgs, with our model_type."""
    pass


class Model(_TextModel):
    """Alias for mlx-lm's Qwen3_5 TextModel under the ``qwen3_5_text`` type."""
    pass
