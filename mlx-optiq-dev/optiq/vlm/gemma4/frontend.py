"""Gemma-4 vision front-end: pixels -> merged input embeddings for mlx-lm.

Owns everything mlx-lm's language model does not: image preprocessing, the
vendored SigLIP vision tower, the multimodal projection, and the scatter of the
resulting soft-token features into the text-embedding sequence. The merged
embeddings are then handed to mlx-lm's ``gemma4_text`` model via its
``input_embeddings`` / ``per_layer_inputs`` hooks, so the language decode stays
fully in mlx-lm (KV-share, sliding window, quantized weights, MTP, …).

Scaling note: mlx-lm's ``gemma4_text`` ``__call__`` always multiplies the
incoming ``input_embeddings`` by ``embed_scale`` (= hidden_size**0.5). mlx-vlm,
by contrast, scales the *text* embeddings before scatter and leaves the vision
features unscaled. To reproduce mlx-vlm exactly through mlx-lm we therefore pass
*unscaled* token embeddings with the vision features pre-divided by
``embed_scale``; mlx-lm's later ``* embed_scale`` restores both to the right
magnitude.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
from PIL import Image

from .config import VisionConfig
from .image_processing import preprocess_images
from .merge import MultimodalEmbedder, masked_scatter
from .vision import VisionModel
from ..frontend import register_frontend
from ..sidecar import VISION_SIDECAR_NAME, resolve_vision_sidecar


def _resolve_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path)


class Gemma4VisionFrontend:
    """Vision front-end for the Gemma-4 (gemma4_unified) family."""

    def __init__(
        self,
        config: dict,
        vision_tower: VisionModel,
        embed_vision: MultimodalEmbedder,
        language_model: Any,
    ):
        self.config = config
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision
        self.language_model = language_model
        self.image_token_id = config.get("image_token_id", 258880)
        self.boi_token_id = config.get("boi_token_id", 255999)
        self.eoi_token_id = config.get("eoi_token_id", 258882)

    # ------------------------------------------------------------------ load
    @classmethod
    def from_pretrained(
        cls, model_path: str, language_model: Any, *, sidecar_path: str | None = None
    ) -> "Gemma4VisionFrontend":
        model_dir = _resolve_dir(model_path)
        config = json.load(open(os.path.join(model_dir, "config.json")))

        vcfg = VisionConfig.from_dict(config["vision_config"])
        vision_tower = VisionModel(vcfg)
        text_hidden = config["text_config"]["hidden_size"]
        embed_vision = MultimodalEmbedder(vcfg.hidden_size, text_hidden)

        sc_path = sidecar_path or str(resolve_vision_sidecar(model_dir)
                                or os.path.join(model_dir, VISION_SIDECAR_NAME))
        if not os.path.exists(sc_path):
            raise FileNotFoundError(
                f"vision sidecar not found at {sc_path}; build it with "
                f"optiq.vlm.build_vision_sidecar(...)"
            )
        sc = mx.load(sc_path)
        vt_w = {k[len("vision_tower."):]: v for k, v in sc.items()
                if k.startswith("vision_tower.")}
        ev_w = {k[len("embed_vision."):]: v for k, v in sc.items()
                if k.startswith("embed_vision.")}
        vision_tower.load_weights(list(vt_w.items()))
        embed_vision.load_weights(list(ev_w.items()))
        vision_tower.eval()
        embed_vision.eval()

        return cls(config, vision_tower, embed_vision, language_model)

    # --------------------------------------------------------------- preprocess
    @staticmethod
    def _extract_images(messages: list[dict]) -> tuple[list[Image.Image], list[dict]]:
        """Pull PIL images out of chat ``messages`` and return them plus a copy of
        the messages with each image part replaced by a ``{IMG}`` sentinel in a
        text part (so the chat template renders text only; we splice image tokens
        in afterwards)."""
        images: list[Image.Image] = []
        rendered: list[dict] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                rendered.append({"role": msg["role"], "content": content})
                continue
            parts = []
            for part in content:
                ptype = part.get("type")
                if ptype in ("image", "image_url"):
                    src = part.get("image") or part.get("image_url") or part.get("url")
                    if isinstance(src, dict):
                        src = src.get("url")
                    images.append(_load_image(src))
                    parts.append("\x00IMG\x00")
                elif ptype == "text":
                    parts.append(part.get("text", ""))
            rendered.append({"role": msg["role"], "content": "".join(parts)})
        return images, rendered

    def preprocess(
        self, messages: list[dict], *, tokenizer, enable_thinking: bool = False
    ) -> dict:
        images, rendered = self._extract_images(messages)
        tmpl_kw = {}
        try:  # not every tokenizer template accepts the flag
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "x"}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
            tmpl_kw["enable_thinking"] = enable_thinking
        except Exception:
            pass

        if not images:
            ids = tokenizer.apply_chat_template(
                rendered, add_generation_prompt=True, **tmpl_kw
            )
            return {"input_ids": mx.array(ids)[None], "pixel_values": None}

        pixel_values, soft_tokens = preprocess_images(images)

        text = tokenizer.apply_chat_template(
            rendered, add_generation_prompt=True, tokenize=False, **tmpl_kw
        )
        # Splice each \x00IMG\x00 sentinel into boi + image_token*n + eoi.
        segments = text.split("\x00IMG\x00")
        assert len(segments) == len(images) + 1, "image sentinel/count mismatch"

        ids: list[int] = []
        first = True
        for i, seg in enumerate(segments):
            seg_ids = tokenizer.encode(seg, add_special_tokens=first)
            first = False
            ids.extend(seg_ids)
            if i < len(images):
                n = soft_tokens[i]
                ids.append(self.boi_token_id)
                ids.extend([self.image_token_id] * n)
                ids.append(self.eoi_token_id)

        return {
            "input_ids": mx.array(ids, dtype=mx.int32)[None],
            "pixel_values": [mx.array(pv)[None] for pv in pixel_values],
        }

    # --------------------------------------------------------------- merge
    def merged_embeddings(self, inputs: dict) -> tuple[mx.array, dict]:
        input_ids = inputs["input_ids"]
        lm = self.language_model.model
        embeds = lm.embed_tokens(input_ids)  # UNSCALED (mlx-lm scales later)

        pixel_values = inputs.get("pixel_values")
        if not pixel_values:
            return embeds, {}

        embed_scale = lm.embed_scale
        # Encode every image and concatenate soft tokens in order.
        feats = [self.embed_vision(self.vision_tower(pv)) for pv in pixel_values]
        features = mx.concatenate([f.reshape(-1, f.shape[-1]) for f in feats], axis=0)
        features = features.astype(embeds.dtype) / embed_scale  # pre-divide

        mask = mx.broadcast_to(
            mx.expand_dims(input_ids == self.image_token_id, -1), embeds.shape
        )
        merged = masked_scatter(embeds, mask, features)

        extra: dict = {}
        if getattr(lm, "hidden_size_per_layer_input", 0):
            text_mask = input_ids != self.image_token_id
            zeroed = mx.where(text_mask, input_ids, mx.zeros_like(input_ids))
            extra["per_layer_inputs"] = lm._get_per_layer_inputs(zeroed)
        return merged, extra


def _load_image(src) -> Image.Image:
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, (str, os.PathLike)):
        s = str(src)
        if s.startswith(("http://", "https://")):
            import io
            import urllib.request

            with urllib.request.urlopen(s) as r:  # noqa: S310
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        if s.startswith("data:"):
            import base64
            import io

            b64 = s.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        return Image.open(s).convert("RGB")
    raise TypeError(f"Unsupported image source: {type(src)}")


register_frontend("gemma4", Gemma4VisionFrontend.from_pretrained)
