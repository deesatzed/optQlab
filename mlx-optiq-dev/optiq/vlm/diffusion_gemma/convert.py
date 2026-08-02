"""Quantize + save a DiffusionGemma checkpoint through OptiQ's pipeline.

The heavy lifting (RAM-bounded streaming quantize + sharded save) is reused
straight from ``mlx_lm.utils`` — ``quantize_model`` and ``save_model`` — which
are core OptiQ dependencies. Only the model construction goes through the
vendored ``optiq.vlm._mlxvlm`` code, so there is still no ``mlx-vlm`` runtime
dependency. This mirrors what ``mlx_vlm.convert`` does internally (lazy
``load_model`` → ``mlx_lm.quantize_model`` → ``save_weights``).

Two uses:
  * ``convert_diffusion_gemma(bf16, out, bits=4)`` — uniform-4 baseline, the
    consistent reference the streaming sensitivity sweep probes against.
  * ``convert_diffusion_gemma(bf16, out, quant_predicate=fn)`` — the final OptiQ
    mixed-precision quant, ``fn`` returning the knapsack's per-layer bits.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx

from .._mlxvlm.models.diffusion_gemma.config import ModelConfig
from .._mlxvlm.models.diffusion_gemma.diffusion_gemma import Model
from .loader import _resolve_dir

# Non-weight files worth carrying into the quant dir (tokenizer / processor).
_AUX_GLOBS = (
    "tokenizer*.json", "tokenizer.model", "*.txt", "special_tokens_map.json",
    "preprocessor_config.json", "processor_config.json", "chat_template.*",
    "generation_config.json",
)


def _load_bf16_lazy(model_dir: str):
    """Build the vendored Model and attach the bf16 weights lazily (mmap)."""
    config = json.load(open(os.path.join(model_dir, "config.json")))
    model = Model(ModelConfig.from_dict(config))
    weights: dict = {}
    for shard in sorted(glob.glob(os.path.join(model_dir, "model*.safetensors"))):
        weights.update(mx.load(shard))  # lazy — not materialized until eval
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))
    return model, config


def is_vision_layer(path: str) -> bool:
    """True for the multimodal towers (and their projectors)."""
    return "vision_tower" in path or "embed_vision" in path or "audio_tower" in path


def convert_diffusion_gemma(
    bf16_path: str,
    output_path: str,
    *,
    bits: int = 4,
    group_size: int = 64,
    mode: str = "affine",
    quant_predicate: Optional[Callable[[str, object], object]] = None,
    quantize_vision: bool = False,
) -> str:
    """Quantize ``bf16_path`` → ``output_path``.

    With ``quant_predicate=None`` every quantizable layer goes to ``bits``
    (uniform baseline). Pass a predicate ``(path, module) -> {bits, group_size}``
    for OptiQ's per-layer mixed precision.

    ``quantize_vision=False`` (the default) keeps the vision/audio towers at
    **bf16**, which is OptiQ's policy for every other VLM family — their towers
    ride along at bf16 in the ``optiq_vision`` sidecar rather than being
    quantized. DiffusionGemma used to quantize its 164 tower layers inline, and
    because the calibration is text-only they scored KL == 0 and were handed the
    *floor* bit-width by default — a precision the sweep never actually measured.
    Pass ``quantize_vision=True`` to restore the old behavior.
    """
    from mlx_lm.utils import quantize_model, save_config, save_model

    model_dir = _resolve_dir(bf16_path)
    out = Path(output_path)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"  [diffusion convert] loading bf16 (lazy) from {model_dir}")
    model, config = _load_bf16_lazy(model_dir)

    # mlx-lm's quantize_model falls back to ``model.quant_predicate`` when the
    # arg is None — and DiffusionGemma's Model exposes the *heuristic* predicate
    # (routers/MLP @ 8-bit). For a genuine UNIFORM baseline we must override it
    # with an explicit "quantize everything at the default bits" predicate;
    # quantize_model already guards non-quantizable / non-divisible modules.
    inner_predicate = quant_predicate or (lambda path, module: True)

    if quantize_vision:
        effective_predicate = inner_predicate
    else:
        def effective_predicate(path, module, _inner=inner_predicate):
            # False => leave the module unquantized (bf16).
            return False if is_vision_layer(path) else _inner(path, module)
        print("  [diffusion convert] vision/audio towers stay bf16 "
              "(--quantize-vision to override)")
    # quantize_model round-trips the config through the typed ModelConfig, which
    # drops nested multimodal keys (notably ``vision_config``) — without them the
    # loaded model builds vision_tower=None and the (present!) vision weights are
    # never loaded, silently shipping a text-only quant. Snapshot + restore them.
    _MM_KEYS = ("vision_config", "vision_soft_tokens_per_image",
                "boi_token_id", "eoi_token_id", "image_token_id",
                "audio_config", "audio_soft_tokens_per_image")
    mm_snapshot = {k: config[k] for k in _MM_KEYS if k in config}
    print(f"  [diffusion convert] quantizing (bits={bits}, group_size={group_size}, "
          f"{'per-layer predicate' if quant_predicate else 'uniform'})")
    model, config = quantize_model(
        model, config, group_size, bits, mode=mode, quant_predicate=effective_predicate
    )
    for k, v in mm_snapshot.items():
        config.setdefault(k, v)

    # Lift the towers into the bf16 sidecar, exactly as every other OptiQ VLM does,
    # instead of leaving them inline. `has_vision_sidecar` is how `optiq serve` and
    # the Lab decide a model can take images, and it keeps the main shards a clean
    # language-only artifact. The sidecar lands under `optiq/`, out of reach of a
    # `*.safetensors` glob.
    lifted = False
    if not quantize_vision:
        from ...sidecars import attach_sidecars

        # Shared with the LLM pipeline: one policy, one place to fix. It raises
        # rather than shipping a model that advertises images it cannot see --
        # this used to swallow the failure and leave the towers inline.
        # DiffusionGemma has no MTP head, so only the vision sidecar is built.
        report = attach_sidecars(model_dir, out, mtp=False)

        # Drop the towers from the model only once they are safely in the sidecar,
        # so a failure can never leave an artifact with them in neither place. A
        # base with no vision_config gets no sidecar and keeps whatever it has --
        # there is nothing to lift, and nothing to drop.
        if report["vision"] is not None:
            enc = model.model.encoder
            for attr in ("vision_tower", "embed_vision", "audio_tower"):
                if getattr(enc, attr, None) is not None:
                    setattr(enc, attr, None)
            lifted = True

    print(f"  [diffusion convert] saving sharded weights → {out}")
    save_model(out, model, donate_model=True)
    save_config(config, config_path=out / "config.json")

    # save_config *deliberately* strips vision_config — `config.pop("vision_config")`
    # is right there in mlx-lm, because mlx-lm's convert produces text-only artifacts.
    # For a VLM it is fatal: without it the model rebuilds vision_tower=None and the
    # towers, sidecar or not, are never loaded. Snapshotting the keys before
    # quantize_model (above) is not enough — save_config runs after. Put them back.
    q_cfg_path = out / "config.json"
    q_cfg = json.load(open(q_cfg_path))
    restored = [k for k, v in mm_snapshot.items() if q_cfg.setdefault(k, v) is v]
    if restored:
        json.dump(q_cfg, open(q_cfg_path, "w"), indent=2)
        print(f"  [diffusion convert] restored multimodal config keys: {restored}")
    if lifted and "vision_config" not in q_cfg:
        raise RuntimeError(
            "vision_config missing from the saved config — the towers are in the "
            "sidecar but the model would rebuild vision_tower=None and never load "
            "them. Refusing to ship a silently text-only VLM."
        )

    # Carry tokenizer / processor files so the quant is self-contained.
    for pattern in _AUX_GLOBS:
        for src in glob.glob(os.path.join(model_dir, pattern)):
            shutil.copy2(src, out / os.path.basename(src))

    total = sum(
        os.path.getsize(out / f) for f in os.listdir(out)
        if f.endswith(".safetensors")
    )
    print(f"  [diffusion convert] done — {total / 1e9:.2f} GB")
    return str(out)
