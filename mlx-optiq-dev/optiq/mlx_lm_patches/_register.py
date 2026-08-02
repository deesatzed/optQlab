"""Register OptiQ-shipped model types into mlx-lm's loader.

mlx-lm resolves ``model_type`` via ``importlib.import_module(
f"mlx_lm.models.{model_type}")``. To make our ``qwen3_5_text`` module
discoverable without patching the mlx-lm install, we inject ourselves
into ``sys.modules`` under the ``mlx_lm.models.qwen3_5_text`` path.

This is idempotent — safe to call multiple times.
"""

from __future__ import annotations

import sys


_REGISTERED = False


# Model types whose architecture is identical to an existing mlx-lm class but
# which ship under a different ``model_type`` string in config.json. mlx-lm
# resolves the architecture by importing ``mlx_lm.models.{model_type}`` after
# consulting ``MODEL_REMAPPING``; adding entries here makes those configs load
# without an mlx-lm-side change.
#
# gemma4_unified: Google's unified text+vision+audio Gemma-4 (e.g.
# mlx-community/gemma-4-12B-it). The full multimodal model lives in mlx-vlm,
# but its language tower IS mlx-lm's ``gemma4_text`` (MoE, double-wide MLP,
# global_head_dim, attention_k_eq_v are all implemented there on main), and
# ``gemma4.py``'s ``sanitize()`` already drops the vision/audio towers. Routing
# the unified model_types at the existing gemma4 classes lets OptiQ load,
# convert, and serve the text path. Requires an mlx-lm build with the unified
# gemma4_text fields (the main build OptiQ pins).
_EXTRA_REMAPPINGS = {
    "gemma4_unified": "gemma4",
    "gemma4_unified_text": "gemma4_text",
}


def register() -> None:
    """Alias OptiQ-shipped model modules into ``mlx_lm.models.*`` and add the
    extra ``MODEL_REMAPPING`` entries OptiQ depends on. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return

    # Import our modules (triggers the files to load)
    from optiq.mlx_lm_patches import dhara_ar as _dhara_ar
    from optiq.mlx_lm_patches import qwen3_5_text as _qwen3_5_text
    from optiq.mlx_lm_patches import nanbeige as _nanbeige
    from optiq.mlx_lm_patches import laguna as _laguna

    # Alias into mlx-lm's namespace so its importlib-based loader finds us
    sys.modules["mlx_lm.models.qwen3_5_text"] = _qwen3_5_text
    # dhara_ar (codelion/dhara-250m) has no upstream mlx-lm class — OptiQ ships it.
    sys.modules["mlx_lm.models.dhara_ar"] = _dhara_ar
    # nanbeige (Nanbeige4.x looped transformer) has no upstream mlx-lm class and
    # the pending PRs never landed (mlx-lm is unmaintained) — OptiQ ships it.
    sys.modules["mlx_lm.models.nanbeige"] = _nanbeige
    # laguna (poolside Laguna-XS/S/M) has no upstream mlx-lm class; the public
    # "mlx" quants ship the PyTorch modeling file which stock mlx-lm can't run.
    sys.modules["mlx_lm.models.laguna"] = _laguna

    # Inject extra model_type remappings (e.g. gemma4_unified -> gemma4).
    try:
        from mlx_lm import utils as _mlx_utils

        for src, dst in _EXTRA_REMAPPINGS.items():
            _mlx_utils.MODEL_REMAPPING.setdefault(src, dst)
    except Exception:
        # mlx-lm layout changed; the remap is best-effort. Conversion of
        # gemma4_unified models will fail loudly at load if this didn't take.
        pass

    # A qwen3_5_moe checkpoint may ship its experts UNFUSED (one tensor per
    # expert per projection). mlx-lm's sanitize only fuses an already-fused
    # checkpoint and silently does nothing otherwise, leaving every expert at
    # random init -- a model that loads, serves, and talks gibberish. Teach it
    # the unfused layout. Best-effort: a future mlx-lm may handle it natively.
    try:
        from optiq.mlx_lm_patches import qwen3_5_moe_unfused as _moe_unfused

        _moe_unfused.install()
    except Exception:
        pass

    _REGISTERED = True
