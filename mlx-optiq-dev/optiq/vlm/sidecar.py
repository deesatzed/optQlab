"""Build and detect the bf16 vision/audio sidecar.

The sidecar (``optiq_vision.safetensors``) holds the multimodal towers a VLM
needs but mlx-lm drops. It rides alongside the quantized language shards in the
same repo: mlx-lm's ``glob("model*.safetensors")`` never matches it, so the
artifact still loads text-only under stock mlx-lm, while OptiQ loads the sidecar
for full image+text inference.

``build_vision_sidecar`` extracts the multimodal weights from the bf16 base,
keeps them at bf16, writes the sidecar into an existing quant directory, and
restores ``vision_config`` / ``audio_config`` (+ the multimodal token ids) into
the quant's ``config.json`` so the towers can be reconstructed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx

VISION_SIDECAR_NAME = "optiq_vision.safetensors"

# Weight-name prefixes that belong to the vision/audio towers (and their
# projectors), matched with or without a leading ``model.``. These mirror the
# keys mlx-lm's Gemma-4 ``sanitize()`` drops.
_MM_PREFIXES = (
    "vision_tower.",       # e2b/e4b/26B/31B: full SigLIP encoder
    "audio_tower.",
    "multi_modal_projector.",
    "embed_vision.",
    "embed_audio.",
    "vision_embedder.",    # gemma4_unified (12B): encoder-free patch embedder
    "audio_embedder.",
    "visual.",             # qwen3_5 / qwen3_6: the tower is `model.visual.…`
)

# Base checkpoints that name a tower differently from what the front-ends load.
# The qwen3_5 front-end reads `vision_tower.…` out of the sidecar (as does the
# convert path, see backends/mlx_backend._pt_to_mlx_vlm_path), but the Qwen bases
# store it as `model.visual.…`. Rename on store so both agree.
_MM_STORE_RENAMES = (
    ("visual.", "vision_tower."),
)

# Config sub-keys to restore onto the quant so the towers are reconstructable.
_MM_CONFIG_KEYS = (
    "vision_config",
    "audio_config",
    "image_token_id",
    "audio_token_id",
    "video_token_id",
    "boi_token_id",
    "eoi_token_id",
    "boa_token_id",
    "eoa_token_index",
    "eoa_token_id",
    "image_token_index",
    # mistral3 / pixtral: the projector cannot be rebuilt without these.
    "spatial_merge_size",
    "vision_feature_layer",
    "multimodal_projector_bias",
)


def _is_mm_key(key: str) -> bool:
    # Gemma-4 / Qwen nest the towers directly (``model.vision_tower.…``);
    # DiffusionGemma puts them one level deeper, under its encoder
    # (``model.encoder.vision_tower.…``), while its language tower lives under
    # ``model.decoder.…``. Strip both containers before matching, or the
    # DiffusionGemma towers look like ordinary weights and the sidecar comes out
    # empty.
    k = key[len("model."):] if key.startswith("model.") else key
    if k.startswith("encoder."):
        k = k[len("encoder."):]
    return any(k.startswith(p) for p in _MM_PREFIXES)


def _store_key(key: str) -> str:
    """The name a multimodal weight takes inside the sidecar.

    Strips the base's containers (``model.``, and DiffusionGemma's ``encoder.``)
    because the front-ends load the un-prefixed form, then applies the
    per-family renames in ``_MM_STORE_RENAMES``.
    """
    k = key[len("model."):] if key.startswith("model.") else key
    if k.startswith("encoder."):
        k = k[len("encoder."):]
    for src, dst in _MM_STORE_RENAMES:
        if k.startswith(src):
            return dst + k[len(src):]
    return k


def _sanitize_conv(key: str, v: "mx.array") -> "mx.array":
    """Transpose conv weights from PyTorch channel-first to MLX channel-last.

    The bf16 base (an HF/PyTorch checkpoint) stores conv weights channel-first;
    ``mlx_vlm.convert`` applies this transpose before saving, so every stock
    mlx-community conversion ships channel-last. OptiQ historically copied the
    towers raw, so the sidecar shipped channel-first convs and any loader that
    strict-checks shapes rejects the whole multimodal graph.

    Doing it here, at build time, is what makes the *published artifact* correct
    for every loader rather than only for the one front-end that remembers to
    call ``sanitize()`` on the way in.

    Covers:
      * gemma-4 audio: conv2d ``subsample_conv_projection`` + depthwise conv1d.
      * qwen3_5 / qwen3_6 vision: a Conv3d patch embed. Gemma-4's SigLIP patches
        are Linear, which is why this was long assumed not to apply to vision --
        it does, for Qwen, and a raw copy makes the tower fail to load with
        "Expected shape (768, 2, 16, 16, 3) but received (768, 3, 2, 16, 16)".
      * mistral3 / pixtral vision: a Conv2d patch embed named
        ``vision_tower.vision_model.patch_conv.weight``. It matched none of the
        cases above -- ``subsample_conv_projection`` is a gemma-4 audio name and
        the Conv3d rule needs ndim 5 -- so a Mistral base stored channel-first
        would have shipped channel-first. mlx-community's already-converted bf16
        repos store it channel-last, so both layouts reach this function and the
        guard has to tell them apart rather than transposing unconditionally.
    """
    # Guards keep it idempotent (safe to re-run on an already-fixed sidecar):
    # PyTorch conv2d has the square kernel in the last two dims (shape[2]==shape[3]);
    # MLX channel-last does not. PyTorch depthwise conv1d has in==1 in the middle.
    if "subsample_conv_projection" in key and "conv.weight" in key and v.ndim == 4:
        if v.shape[2] == v.shape[3]:
            return v.transpose(0, 2, 3, 1)  # [out, in, kH, kW] -> [out, kH, kW, in]
    if "depthwise_conv1d.weight" in key and v.ndim == 3:
        if v.shape[1] == 1:
            return v.transpose(0, 2, 1)     # [out, in=1, kW] -> [out, kW, in=1]
    if "patch_embed.proj.weight" in key and v.ndim == 5:
        # PyTorch Conv3d: [out, in, kT, kH, kW]; MLX: [out, kT, kH, kW, in].
        # `in` is the image channel count (3), and it is the last axis once
        # converted -- so a trailing 3 means this is already MLX-layout.
        if v.shape[-1] != 3:
            return v.transpose(0, 2, 3, 4, 1)
    if "patch_conv.weight" in key and v.ndim == 4:
        # PyTorch Conv2d: [out, in, kH, kW] (kH == kW, in == num_channels);
        # MLX: [out, kH, kW, in]. Only the PyTorch layout has a square *trailing*
        # pair, so `shape[2] == shape[3]` is the discriminator and re-running on
        # an already-converted sidecar is a no-op. (Guarding on `shape[1] == 3`
        # instead would misfire on a 3x3 kernel.)
        if v.shape[2] == v.shape[3] and v.shape[1] != v.shape[2]:
            return v.transpose(0, 2, 3, 1)  # [out, in, kH, kW] -> [out, kH, kW, in]
    return v


def _register_vision_in_index(quant_dir: Path, sidecar_rel: str,
                              keys: list[str]) -> bool:
    """List the vision tensors in ``model.safetensors.index.json``.

    This is what makes a stock VLM loader able to find the tower.

    mlx-vlm reads the index's ``weight_map`` and loads exactly the shards named
    there; its ``*.safetensors`` glob is only a fallback for un-indexed models.
    So hiding the sidecar in a subfolder was never the thing that broke it --
    the tensors simply were not in the index, and mlx-vlm therefore built a
    ``vision_tower`` it had no weights for::

        Missing 153 parameters: vision_tower.blocks.0.attn.proj.bias, ...

    (Reported against Qwen3.6-35B-A3B-OptiQ-4bit; reproduced exactly here with
    mlx-vlm 0.6.4. oMLX issue #72 is the same failure on a *non-OptiQ* quant.)

    Pointing the index at the sidecar fixes every loader at once:

      * mlx-lm globs ``model*.safetensors`` and ignores the index, so it never
        reads the sidecar and text-only loading is untouched. (It also drops
        ``vision_tower.*`` in ``sanitize()`` regardless.)
      * mlx-vlm / oMLX / LM Studio read the index, find the tower, and load it.
        The tower is bf16 with no ``.scales``, and both loaders quantize a
        module only when ``f"{p}.scales" in weights``, so it stays a plain
        ``Linear`` whose ``weight``/``bias`` match what they expect.
      * ``mtp.safetensors`` stays out of the index and out of both globs. That
        is the file that caused the original "Received N parameters not in
        model" -- ``mtp.*`` belongs to no model, so it must never be visible.

    Returns True if the index was updated (a single-file model has none).
    """
    idx_path = quant_dir / "model.safetensors.index.json"
    if not idx_path.is_file():
        return False
    idx = json.load(open(idx_path))
    wm = idx.setdefault("weight_map", {})
    for k in keys:
        wm[k] = sidecar_rel
    # total_size is advisory; leave it alone rather than half-maintain it.
    json.dump(idx, open(idx_path, "w"), indent=2)
    return True


def resolve_vision_sidecar(model_dir: str | Path) -> Path | None:
    """Existing vision sidecar path — ``optiq/`` subfolder first, then legacy root."""
    from ..sidecar_layout import resolve
    return resolve(model_dir, VISION_SIDECAR_NAME)


def has_vision_sidecar(model_dir: str | Path) -> bool:
    return resolve_vision_sidecar(model_dir) is not None


def _resolve_dir(base: str) -> str:
    if os.path.isdir(base):
        return base
    from huggingface_hub import snapshot_download
    return snapshot_download(base)


def _resolve_base_selective(base: str) -> str:
    """Resolve a base to a local dir, downloading only what the sidecar needs.

    For a local dir, returns it. For an HF repo, downloads config.json and the
    safetensors index, figures out which shards actually hold vision/audio
    weights, and downloads ONLY those shards (plus config.json). This avoids
    pulling a 50-60 GB bf16 base just to extract a ~1-2 GB vision tower. Falls
    back to a full ``snapshot_download`` for single-file (un-indexed) bases,
    where the multimodal weights share the one shard with everything else."""
    if os.path.isdir(base):
        return base
    from huggingface_hub import hf_hub_download, snapshot_download

    try:
        idx_local = hf_hub_download(base, "model.safetensors.index.json")
    except Exception:
        # No index -> single-file model; must pull the whole shard.
        return snapshot_download(base)

    weight_map = json.load(open(idx_local)).get("weight_map", {})
    mm_shards = sorted({weight_map[k] for k in weight_map if _is_mm_key(k)})
    if not mm_shards:
        # Vision keys not in the index (unexpected) -> fall back to full.
        return snapshot_download(base)

    # Pull config.json + only the shards holding multimodal weights. They all
    # land in the same snapshot dir alongside the already-downloaded index.
    hf_hub_download(base, "config.json")
    local_shard = None
    for shard in mm_shards:
        local_shard = hf_hub_download(base, shard)
    # The snapshot dir is the parent of the downloaded files.
    return os.path.dirname(local_shard)


def _shards_with_mm(base_dir: str) -> tuple[list[str], list[str]]:
    """Return (shard_files, mm_keys). Uses the index when present so we only
    open shards that actually contain multimodal weights."""
    import glob

    idx_path = os.path.join(base_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        weight_map = json.load(open(idx_path)).get("weight_map", {})
        mm_keys = [k for k in weight_map if _is_mm_key(k)]
        shards = sorted({weight_map[k] for k in mm_keys})
        return [os.path.join(base_dir, s) for s in shards], mm_keys

    # Single-file model: one shard, scan its header for mm keys.
    single = sorted(glob.glob(os.path.join(base_dir, "model*.safetensors")))
    if not single:
        return [], []
    from safetensors import safe_open
    with safe_open(single[0], framework="numpy") as f:
        mm_keys = [k for k in f.keys() if _is_mm_key(k)]
    return single, mm_keys


def build_vision_sidecar(
    base: str,
    quant_dir: str | Path,
    *,
    dtype: str = "bfloat16",
    force: bool = False,
) -> dict:
    """Extract the bf16 vision/audio towers from ``base`` into a sidecar in
    ``quant_dir`` and restore the multimodal config keys.

    Args:
        base: bf16 base model (HF repo id or local dir) that still has the
            vision/audio towers.
        quant_dir: an existing OptiQ quant directory (the language quant). The
            sidecar and config edits land here.
        dtype: sidecar weight dtype. Default ``bfloat16`` (no quantization of
            vision/audio — matches the ecosystem norm).

    Returns:
        Summary dict (n_tensors, bytes, base, dtype).
    """
    from ..sidecar_layout import canonical_rel, write_path

    quant_dir = Path(quant_dir)
    # Hidden from a `*.safetensors` glob: write under the optiq/ subfolder.
    existing = resolve_vision_sidecar(quant_dir)
    if existing is not None and not force:
        raise FileExistsError(f"{existing} exists; pass force=True to overwrite.")
    out = write_path(quant_dir, VISION_SIDECAR_NAME)
    sidecar_rel = canonical_rel(VISION_SIDECAR_NAME)

    base_dir = _resolve_base_selective(base)
    shards, mm_keys = _shards_with_mm(base_dir)
    if not mm_keys:
        raise RuntimeError(
            f"No vision/audio tower weights found in {base}. Is it a multimodal base?"
        )

    target = getattr(mx, dtype)
    sidecar: dict[str, mx.array] = {}
    for shard in shards:
        arrs = mx.load(shard)  # mlx reads bf16 natively
        for k, v in arrs.items():
            if _is_mm_key(k):
                store_key = _store_key(k)
                sidecar[store_key] = _sanitize_conv(store_key, v).astype(target)
        del arrs
        mx.eval(*sidecar.values()) if sidecar else None

    mx.save_safetensors(str(out), sidecar, metadata={"format": "mlx"})
    _register_vision_in_index(quant_dir, sidecar_rel, list(sidecar))

    # Restore the multimodal config keys onto the quant's config.json.
    base_cfg = json.load(open(os.path.join(base_dir, "config.json")))
    q_cfg_path = quant_dir / "config.json"
    q_cfg = json.load(open(q_cfg_path)) if q_cfg_path.exists() else {}
    for key in _MM_CONFIG_KEYS:
        if key in base_cfg:
            q_cfg[key] = base_cfg[key]
    n_bytes = sum(v.nbytes for v in sidecar.values())
    q_cfg["optiq_vision"] = {
        "sidecar": sidecar_rel,
        "dtype": dtype,
        "n_tensors": len(sidecar),
        "base_model": base,
    }
    json.dump(q_cfg, open(q_cfg_path, "w"), indent=2)

    return {
        "sidecar": str(out),
        "n_tensors": len(sidecar),
        "bytes": n_bytes,
        "dtype": dtype,
        "base": base,
    }
