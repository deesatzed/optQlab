"""Qwen3.5 vision front-end: pixels -> merged input embeddings for mlx-lm.

Runs the vendored Qwen3-VL vision tower (deepstack disabled), scatters its
visual tokens into mlx-lm's qwen3_5 text-embedding sequence at the image-pad
positions, and hands the merged embeddings to mlx-lm's decode. The ViT's merger
already projects to the language hidden size, and qwen3_5 has no embed_scale, so
the features go in unscaled.

mRoPE note: Qwen gives image tokens 2D spatial positions (mrope_section
[11,11,10]). mlx-lm's qwen3_5 currently applies sequential positions; the
``image_position_ids`` are computed here and stashed for an (optional) mRoPE
patch on the attention.
"""

from __future__ import annotations

import json
import os
from typing import Any

import mlx.core as mx
from PIL import Image

from .config import QwenVisionConfig
from .image_processing import preprocess_images
from .vision import VisionModel
from ..frontend import register_frontend
from ..gemma4.frontend import _load_image
from ..gemma4.merge import masked_scatter
from ..sidecar import VISION_SIDECAR_NAME, resolve_vision_sidecar


def _resolve_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path)


class Qwen35VisionFrontend:
    """Vision front-end for the Qwen3.5 / Qwen3.6 family."""

    def __init__(self, config, vision_tower, language_model):
        self.config = config
        self.vision_tower = vision_tower
        self.language_model = language_model
        self.image_token_id = config.get("image_token_id", 248056)
        self.vision_start_id = config.get("vision_start_token_id", 248053)
        self.vision_end_id = config.get("vision_end_token_id", 248054)
        self.merge_size = config["vision_config"].get("spatial_merge_size", 2)

    @classmethod
    def from_pretrained(cls, model_path, language_model, *, sidecar_path=None):
        model_dir = _resolve_dir(model_path)
        config = json.load(open(os.path.join(model_dir, "config.json")))
        vcfg = QwenVisionConfig.from_dict(config["vision_config"])
        vision_tower = VisionModel(vcfg)

        sc_path = sidecar_path or str(resolve_vision_sidecar(model_dir)
                                or os.path.join(model_dir, VISION_SIDECAR_NAME))
        if not os.path.exists(sc_path):
            raise FileNotFoundError(f"vision sidecar not found at {sc_path}")
        sc = mx.load(sc_path)
        vt_w = {k[len("vision_tower."):]: v for k, v in sc.items()
                if k.startswith("vision_tower.")}
        vision_tower.load_weights(list(vt_w.items()))
        vision_tower.eval()
        return cls(config, vision_tower, language_model)

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

        pixel_values, grid_thw = preprocess_images(images)
        # merged visual tokens per image = (t*h*w) / merge_size**2
        m2 = self.merge_size ** 2
        n_tokens = [int(grid_thw[i, 0] * grid_thw[i, 1] * grid_thw[i, 2]) // m2
                    for i in range(grid_thw.shape[0])]

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
                ids.append(self.vision_start_id)
                ids.extend([self.image_token_id] * n_tokens[i])
                ids.append(self.vision_end_id)

        return {
            "input_ids": mx.array(ids, dtype=mx.int32)[None],
            "pixel_values": mx.array(pixel_values),
            "grid_thw": mx.array(grid_thw),
        }

    def merged_embeddings(self, inputs):
        input_ids = inputs["input_ids"]
        inner = self.language_model.model
        embeds = inner.embed_tokens(input_ids)  # qwen3_5 has no embed_scale

        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            return embeds, {}

        pv = pixel_values.astype(self.vision_tower.patch_embed.proj.weight.dtype)
        feats, _ = self.vision_tower(pv, inputs["grid_thw"])
        features = feats.astype(embeds.dtype)

        mask = mx.broadcast_to(
            mx.expand_dims(input_ids == self.image_token_id, -1), embeds.shape)
        merged = masked_scatter(embeds, mask, features)
        return merged, {}


# Register under both the dense (``qwen3_5``) and MoE (``qwen3_5_moe``) text
# model_types: the published Qwen3.5/3.6 35B-A3B quants are ``qwen3_5_moe``,
# and the Qwen3-VL vision tower is identical regardless of the language
# backbone, so one frontend serves both.
register_frontend("qwen3_5", Qwen35VisionFrontend.from_pretrained)
register_frontend("qwen3_5_moe", Qwen35VisionFrontend.from_pretrained)
