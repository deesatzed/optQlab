"""Gemma-4 unified vision front-end: patches -> merged input embeddings.

Encoder-free path for the 12B ``gemma4_unified`` model: image patches go through
the vendored ``VisionEmbedder`` + ``MultimodalEmbedder`` and are scattered into
mlx-lm's ``gemma4_text`` embedding sequence. The shared backbone does the rest.

Scaling note (same as the SigLIP path): mlx-lm's ``gemma4_text`` always
re-multiplies ``input_embeddings`` by ``embed_scale``, so the vision features are
pre-divided by it. The unified text config has ``hidden_size_per_layer_input=0``,
so there are no per-layer inputs to build.
"""

from __future__ import annotations

import json
import os
from typing import Any

import mlx.core as mx
from PIL import Image

from .config import UnifiedVisionConfig
from .image_processing import preprocess_images
from .vision_embedder import VisionEmbedder
from ..gemma4.frontend import _load_image
from ..gemma4.merge import MultimodalEmbedder, masked_scatter
from ..frontend import register_frontend
from ..sidecar import VISION_SIDECAR_NAME, resolve_vision_sidecar


def _resolve_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path)


def _install_vision_bidirectional_mask(inner) -> None:
    """Patch a gemma4_text inner model so image tokens attend bidirectionally.

    The unified model is trained with ``use_bidirectional_attention == "vision"``
    (text stays causal). mlx-lm's gemma4_text builds causal masks; this wraps
    ``_make_masks`` to OR-in full attention among the image-token positions. It
    is one-shot: the frontend sets ``inner._optiq_vision_bidir`` (a bool [L]
    image mask) before the prefill, and the wrapper consumes+clears it, so
    decode steps and text requests stay purely causal."""
    if getattr(inner, "_optiq_bidir_installed", False):
        return
    import types

    import mlx.core as mx

    orig = inner._make_masks

    def _patched(self, h, cache):
        masks = orig(h, cache)
        im = getattr(self, "_optiq_vision_bidir", None)
        if im is None or h.shape[1] != im.shape[0]:
            return masks
        self._optiq_vision_bidir = None  # one-shot
        L = h.shape[1]
        bidir = im[:, None] & im[None, :]  # image x image
        out = []
        for mk in masks:
            if mk is None or isinstance(mk, str):
                idx = mx.arange(L)
                causal = idx[None, :] <= idx[:, None]
                allow = causal | bidir
                out.append(mx.where(allow, mx.array(0.0, dtype=h.dtype),
                                    mx.array(-1e9, dtype=h.dtype)))
            else:
                out.append(mx.where(bidir, mx.array(0.0, dtype=mk.dtype), mk))
        return out

    inner._make_masks = types.MethodType(_patched, inner)
    inner._optiq_bidir_installed = True


class Gemma4UnifiedVisionFrontend:
    """Vision front-end for the gemma4_unified (12B) family."""

    def __init__(self, config, vision_embedder, embed_vision, language_model):
        self.config = config
        self.vision_embedder = vision_embedder
        self.embed_vision = embed_vision
        self.language_model = language_model
        self.image_token_id = config.get("image_token_id", 258880)
        self.boi_token_id = config.get("boi_token_id", 255999)
        self.eoi_token_id = config.get("eoi_token_id", 258882)

    @classmethod
    def from_pretrained(cls, model_path, language_model, *, sidecar_path=None):
        model_dir = _resolve_dir(model_path)
        config = json.load(open(os.path.join(model_dir, "config.json")))
        vcfg = UnifiedVisionConfig.from_dict(config["vision_config"])
        text_hidden = config["text_config"]["hidden_size"]

        vision_embedder = VisionEmbedder(vcfg)
        embed_vision = MultimodalEmbedder(vcfg.output_proj_dims, text_hidden,
                                          eps=vcfg.rms_norm_eps)

        sc_path = sidecar_path or str(resolve_vision_sidecar(model_dir)
                                or os.path.join(model_dir, VISION_SIDECAR_NAME))
        if not os.path.exists(sc_path):
            raise FileNotFoundError(f"vision sidecar not found at {sc_path}")
        sc = mx.load(sc_path)
        ve_w = {k[len("vision_embedder."):]: v for k, v in sc.items()
                if k.startswith("vision_embedder.")}
        ev_w = {k[len("embed_vision."):]: v for k, v in sc.items()
                if k.startswith("embed_vision.")}
        vision_embedder.load_weights(list(ve_w.items()))
        embed_vision.load_weights(list(ev_w.items()))
        vision_embedder.eval()
        embed_vision.eval()
        return cls(config, vision_embedder, embed_vision, language_model)

    # preprocessing reuses the SigLIP frontend's message handling
    @staticmethod
    def _extract_images(messages):
        from ..gemma4.frontend import Gemma4VisionFrontend
        return Gemma4VisionFrontend._extract_images(messages)

    def preprocess(self, messages, *, tokenizer, enable_thinking: bool = False):
        images, rendered = self._extract_images(messages)
        tmpl_kw = {}
        try:
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "x"}], add_generation_prompt=True,
                tokenize=False, enable_thinking=enable_thinking)
            tmpl_kw["enable_thinking"] = enable_thinking
        except Exception:
            pass

        if not images:
            ids = tokenizer.apply_chat_template(
                rendered, add_generation_prompt=True, **tmpl_kw)
            return {"input_ids": mx.array(ids)[None], "pixel_values": None}

        pixel_values, image_position_ids, soft_tokens = preprocess_images(images)

        text = tokenizer.apply_chat_template(
            rendered, add_generation_prompt=True, tokenize=False, **tmpl_kw)
        segments = text.split("\x00IMG\x00")
        assert len(segments) == len(images) + 1, "image sentinel/count mismatch"

        ids: list[int] = []
        first = True
        for i, seg in enumerate(segments):
            ids.extend(tokenizer.encode(seg, add_special_tokens=first))
            first = False
            if i < len(images):
                ids.append(self.boi_token_id)
                ids.extend([self.image_token_id] * soft_tokens[i])
                ids.append(self.eoi_token_id)

        return {
            "input_ids": mx.array(ids, dtype=mx.int32)[None],
            "pixel_values": mx.array(pixel_values),
            "image_position_ids": mx.array(image_position_ids),
            "soft_tokens": soft_tokens,
        }

    def merged_embeddings(self, inputs):
        input_ids = inputs["input_ids"]
        lm = self.language_model.model
        embeds = lm.embed_tokens(input_ids)  # unscaled; mlx-lm scales later

        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            return embeds, {}

        # Encode each image's real (unpadded) patches, in order.
        pos_ids = inputs["image_position_ids"]
        soft = inputs["soft_tokens"]
        feats = []
        for i in range(pixel_values.shape[0]):
            n = soft[i]
            emb = self.vision_embedder(pixel_values[i:i + 1], pos_ids[i:i + 1])
            emb = self.embed_vision(emb)[0, :n]  # drop padding
            feats.append(emb)
        features = mx.concatenate(feats, axis=0).astype(embeds.dtype)
        features = features / lm.embed_scale  # pre-divide; mlx-lm re-scales

        mask = mx.broadcast_to(
            mx.expand_dims(input_ids == self.image_token_id, -1), embeds.shape)
        merged = masked_scatter(embeds, mask, features)

        # The unified model attends bidirectionally over vision tokens (text
        # stays causal). Arm the one-shot mask wrapper for this prefill.
        _install_vision_bidirectional_mask(lm)
        lm._optiq_vision_bidir = input_ids[0] == self.image_token_id

        extra = {}
        if getattr(lm, "hidden_size_per_layer_input", 0):
            text_mask = input_ids != self.image_token_id
            zeroed = mx.where(text_mask, input_ids, mx.zeros_like(input_ids))
            extra["per_layer_inputs"] = lm._get_per_layer_inputs(zeroed)
        return merged, extra


register_frontend("gemma4_unified", Gemma4UnifiedVisionFrontend.from_pretrained)
