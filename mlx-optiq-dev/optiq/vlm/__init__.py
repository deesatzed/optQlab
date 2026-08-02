"""OptiQ VLM support: vision/audio front-end + sidecar, mlx-lm language back-end.

Design (see also the MTP sidecar in ``optiq.runtime.mtp_convert``):

OptiQ quantizes only the *language* tower of a multimodal model; the vision and
audio towers are kept at bf16 in a **sidecar** (``optiq_vision.safetensors``)
that rides alongside the quantized language shards. Because mlx-lm picks weight
files via ``glob("model*.safetensors")`` (or the index ``weight_map``), a sidecar
named ``optiq_vision.safetensors`` is invisible to mlx-lm — so the *same
artifact* loads as a text-only model under stock mlx-lm, while OptiQ loads the
sidecar too for full vision+text inference.

The image path reuses mlx-lm's optimized language decode: the vision front-end
encodes pixels into the language hidden space and scatters those features into
the text-token embedding sequence, then feeds the merged embeddings to mlx-lm's
language model via its existing ``input_embeddings`` hook. Text-only requests
never touch the vision code.
"""

from .frontend import VisionFrontend, get_frontend, register_frontend
from .sidecar import build_vision_sidecar, VISION_SIDECAR_NAME, has_vision_sidecar

# Importing the per-arch front-ends registers them in the frontend registry.
from .gemma4 import frontend as _gemma4_frontend  # noqa: F401,E402
from .gemma4_unified import frontend as _gemma4_unified_frontend  # noqa: F401,E402
from .qwen3_5 import frontend as _qwen3_5_frontend  # noqa: F401,E402
from .mistral3 import frontend as _mistral3_frontend  # noqa: F401,E402

__all__ = [
    "VisionFrontend",
    "get_frontend",
    "register_frontend",
    "build_vision_sidecar",
    "has_vision_sidecar",
    "VISION_SIDECAR_NAME",
]
