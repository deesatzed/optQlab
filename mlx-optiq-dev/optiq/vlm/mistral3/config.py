"""Pixtral / Mistral-3 multimodal config (vendored from mlx-vlm, BSD-3).

Only the fields the vision path needs. The language tower is mlx-lm's, so the
text config is read as a plain dict for the two values the projector wants
(``hidden_size``, ``rms_norm_eps``).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass


class BaseModelConfig:
    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls()
        return cls(**{k: v for k, v in params.items()
                      if k in inspect.signature(cls).parameters})

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class VisionConfig(BaseModelConfig):
    """Pixtral vision tower.

    Devstral-2 / Mistral-Small-3 publish the rope base under
    ``rope_parameters.rope_theta`` rather than a flat ``rope_theta``, so
    :meth:`from_dict` normalizes both spellings; older Pixtral checkpoints that
    only have the flat key keep working.
    """

    model_type: str = "pixtral"
    num_hidden_layers: int = 24
    hidden_size: int = 1024
    head_dim: int = 64
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    image_size: int = 1540
    patch_size: int = 14
    num_channels: int = 3
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @classmethod
    def from_dict(cls, params: dict) -> "VisionConfig":
        params = dict(params or {})
        rp = params.pop("rope_parameters", None)
        if isinstance(rp, dict) and "rope_theta" in rp and "rope_theta" not in params:
            params["rope_theta"] = rp["rope_theta"]
        return super().from_dict(params)
