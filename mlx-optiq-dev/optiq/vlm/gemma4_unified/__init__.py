"""Gemma-4 *unified* (gemma4_unified) vision front-end.

Unlike the e2b/e4b/26B/31B ``gemma4`` models (which carry a full SigLIP
``vision_tower``), the unified 12B model is **encoder-free**: image patches go
through a light ``vision_embedder`` (patch projection + 2D position embedding)
and are then processed by the *shared* language backbone. So there is no
separate vision encoder to run; OptiQ embeds the patches and splices them into
mlx-lm's ``gemma4_text`` decode, exactly like the SigLIP path but with the
patch-embedder standing in for the tower.
"""

from .frontend import Gemma4UnifiedVisionFrontend  # noqa: F401
