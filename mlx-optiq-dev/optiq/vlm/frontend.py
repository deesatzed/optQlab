"""VisionFrontend protocol + per-architecture registry.

A ``VisionFrontend`` owns everything mlx-lm does NOT: the image/audio
preprocessing, the vision/audio encoder forward, and the merge of the resulting
features into the language model's input-embedding sequence. The language decode
itself stays in mlx-lm.

Each multimodal arch family (``gemma4``, ``qwen3_5``, …) registers a frontend.
Resolution is by the base ``config.json`` ``model_type`` (after OptiQ's
remappings, e.g. ``gemma4_unified`` → ``gemma4``).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import mlx.core as mx


@runtime_checkable
class VisionFrontend(Protocol):
    """The minimal surface OptiQ needs to add image/audio to an mlx-lm decode."""

    @classmethod
    def from_pretrained(
        cls, model_path: str, language_model: Any, *, sidecar_path: str | None = None
    ) -> "VisionFrontend":
        """Load the vision/audio towers (from the ``optiq_vision`` sidecar when
        present, else the base) and bind to an already-loaded mlx-lm
        ``language_model`` (for embed_tokens / embed_scale / per-layer inputs)."""
        ...

    def preprocess(self, messages: list[dict], **kwargs) -> dict:
        """Turn a chat ``messages`` list (with image/audio parts) into the model
        inputs: ``{input_ids, pixel_values?, audio_features?, ...}``."""
        ...

    def merged_embeddings(self, inputs: dict) -> tuple[mx.array, dict]:
        """Return ``(input_embeddings, extra_kwargs)`` ready to pass straight to
        the mlx-lm language model's ``__call__`` (``input_embeddings=...`` plus,
        e.g. for gemma-4, ``per_layer_inputs=...``). Text-only inputs return the
        plain scaled token embeddings."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., VisionFrontend]] = {}

# model_type aliases → canonical frontend key. (gemma4_unified is NOT aliased
# to gemma4: it has a distinct encoder-free vision path with its own frontend.)
_ALIASES: dict[str, str] = {}


def register_frontend(model_type: str, factory: Callable[..., VisionFrontend]) -> None:
    _REGISTRY[model_type] = factory


def get_frontend(model_type: str) -> Callable[..., VisionFrontend] | None:
    key = _ALIASES.get(model_type, model_type)
    return _REGISTRY.get(key)
