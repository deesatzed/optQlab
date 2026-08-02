"""Mistral-3 / Pixtral vision front-end: pixels -> merged input embeddings.

Serves the ``mistral3`` family (Devstral-Small-2-24B, Mistral-Small-3.x) the same
way ``optiq/vlm/gemma4`` serves Gemma-4: OptiQ quantizes only the language tower,
the Pixtral vision tower + multimodal projector ride along at bf16 in the
``optiq_vision`` sidecar, and this module turns an image into features that get
scattered into mlx-lm's text-embedding sequence.

Three things that differ from Gemma-4 and are easy to copy across by mistake:

  * **No ``embed_scale``.** mlx-lm's ``ministral3`` takes ``input_embeddings``
    verbatim, so nothing is pre-divided here. Gemma-4's divide-then-remultiply
    dance would silently shrink every image feature by ``hidden**0.5``.
  * **No ``per_layer_inputs``.** That hook is Gemma-4's alone.
  * **Native resolution.** Pixtral packs differently-sized images into one
    sequence behind a block-diagonal mask, so ``image_sizes`` is load-bearing and
    must reach the tower. Nothing may assume a fixed square input.

The prompt layout is Pixtral's, not a flat run of image tokens: each patch row is
``[IMG] * n_width`` followed by ``[IMG_BREAK]``, and the very last break becomes
``[IMG_END]``. Only the ``[IMG]`` positions receive vision features; the breaks
are ordinary text tokens that tell the model where the rows end.
"""

from __future__ import annotations

import json
import os
from typing import Any

import mlx.core as mx

from .config import VisionConfig
from .image_processing import pad_to_batch, preprocess_images
from .merge import Mistral3MultiModalProjector
from .vision import VisionModel
from ..frontend import register_frontend
from ..gemma4.frontend import Gemma4VisionFrontend, _load_image  # noqa: F401
from ..gemma4.merge import masked_scatter
from ..sidecar import VISION_SIDECAR_NAME, resolve_vision_sidecar

# params.json / tekken defaults for the Mistral tokenizers.
DEFAULT_IMAGE_TOKEN_ID = 10
DEFAULT_IMAGE_BREAK_TOKEN_ID = 12
DEFAULT_IMAGE_END_TOKEN_ID = 13


def _resolve_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path)


def _inner_text_model(language_model: Any):
    """The module that owns ``embed_tokens``.

    mlx-lm exposes the Gemma-4 / Qwen text towers directly (``lm.model``) but
    wraps Mistral-3 one level deeper (``lm.language_model.model``), because its
    ``mistral3.Model`` is a VLM shell around ``ministral3.Model``. Probe rather
    than hardcode, so an OptiQ quant published with either config still loads.
    """
    for path in (("model",), ("language_model", "model")):
        node = language_model
        try:
            for attr in path:
                node = getattr(node, attr)
        except AttributeError:
            continue
        if hasattr(node, "embed_tokens"):
            return node
    raise AttributeError(
        "could not find embed_tokens on the language model "
        f"({type(language_model).__name__}); mistral3 vision needs it to build "
        "the text embedding sequence"
    )


class Mistral3VisionFrontend:
    """Vision front-end for the Mistral-3 (Pixtral tower) family."""

    def __init__(self, config: dict, vision_tower, projector, language_model):
        self.config = config
        self.vision_tower = vision_tower
        self.projector = projector
        self.language_model = language_model

        vcfg = config.get("vision_config", {})
        self.patch_size = vcfg.get("patch_size", 14)
        self.spatial_merge_size = config.get("spatial_merge_size", 2)
        self.longest_edge = vcfg.get("image_size", 1540)
        self.vision_feature_layer = config.get("vision_feature_layer", -1)

        self.image_token_id = config.get(
            "image_token_index", config.get("image_token_id", DEFAULT_IMAGE_TOKEN_ID)
        )
        self.image_break_token_id = config.get(
            "image_break_token_id", DEFAULT_IMAGE_BREAK_TOKEN_ID
        )
        self.image_end_token_id = config.get(
            "image_end_token_id", DEFAULT_IMAGE_END_TOKEN_ID
        )

    # ------------------------------------------------------------------ load
    @classmethod
    def from_pretrained(
        cls, model_path: str, language_model: Any, *, sidecar_path: str | None = None
    ) -> "Mistral3VisionFrontend":
        model_dir = _resolve_dir(model_path)
        config = json.load(open(os.path.join(model_dir, "config.json")))

        vcfg = VisionConfig.from_dict(config.get("vision_config", {}))
        text_cfg = config.get("text_config", {})
        vision_tower = VisionModel(vcfg)
        projector = Mistral3MultiModalProjector(
            vcfg.hidden_size,
            text_cfg.get("hidden_size", 5120),
            spatial_merge_size=config.get("spatial_merge_size", 2),
            patch_size=vcfg.patch_size,
            rms_norm_eps=text_cfg.get("rms_norm_eps", 1e-5),
            bias=config.get("multimodal_projector_bias", False),
        )

        sc_path = sidecar_path or str(
            resolve_vision_sidecar(model_dir)
            or os.path.join(model_dir, VISION_SIDECAR_NAME)
        )
        if not os.path.exists(sc_path):
            raise FileNotFoundError(
                f"vision sidecar not found at {sc_path}; build it with "
                f"optiq.vlm.build_vision_sidecar(...)"
            )
        sc = mx.load(sc_path)
        cls._load_towers(sc, vision_tower, projector)
        return cls(config, vision_tower, projector, language_model)

    @staticmethod
    def _load_towers(weights: dict, vision_tower, projector) -> None:
        """Split a sidecar's flat weight dict across the tower and the projector.

        Both prefixes are stored exactly as the base checkpoint names them
        (``vision_tower.vision_model.…`` / ``multi_modal_projector.…``), so the
        published artifact stays loadable by mlx-vlm as well.
        """
        vt = {k[len("vision_tower."):]: v for k, v in weights.items()
              if k.startswith("vision_tower.")}
        pj = {k[len("multi_modal_projector."):]: v for k, v in weights.items()
              if k.startswith("multi_modal_projector.")}
        if not vt:
            raise ValueError("sidecar has no vision_tower.* weights")
        if not pj:
            raise ValueError("sidecar has no multi_modal_projector.* weights")
        vision_tower.load_weights(list(vt.items()))
        projector.load_weights(list(pj.items()))
        vision_tower.eval()
        projector.eval()

    # --------------------------------------------------------------- preprocess
    @staticmethod
    def _extract_images(messages: list[dict]):
        return Gemma4VisionFrontend._extract_images(messages)

    def _image_token_ids(self, height: int, width: int) -> list[int]:
        """Pixtral's row-major image block: ``[IMG]*w  [IMG_BREAK]`` per row, with
        the final break swapped for ``[IMG_END]``."""
        step = self.patch_size * self.spatial_merge_size
        n_h, n_w = height // step, width // step
        ids: list[int] = []
        for _ in range(n_h):
            ids.extend([self.image_token_id] * n_w)
            ids.append(self.image_break_token_id)
        ids[-1] = self.image_end_token_id
        return ids

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

        pixel_values, image_sizes = preprocess_images(
            images,
            patch_size=self.patch_size,
            spatial_merge_size=self.spatial_merge_size,
            longest_edge=self.longest_edge,
        )

        text = tokenizer.apply_chat_template(
            rendered, add_generation_prompt=True, tokenize=False, **tmpl_kw
        )
        segments = text.split("\x00IMG\x00")
        assert len(segments) == len(images) + 1, "image sentinel/count mismatch"

        ids: list[int] = []
        first = True
        for i, seg in enumerate(segments):
            ids.extend(tokenizer.encode(seg, add_special_tokens=first))
            first = False
            if i < len(images):
                ids.extend(self._image_token_ids(*image_sizes[i]))

        return {
            "input_ids": mx.array(ids, dtype=mx.int32)[None],
            "pixel_values": mx.array(pad_to_batch(pixel_values)),
            "image_sizes": image_sizes,
        }

    # --------------------------------------------------------------- merge
    def image_features(self, pixel_values: mx.array, image_sizes) -> mx.array:
        """Pixels -> projected soft tokens, in reading order across images.

        ``pixel_values`` is ``[N, 3, H, W]`` (channel-first, as the processor
        emits); the tower wants NHWC, hence the transpose. ``vision_feature_layer``
        indexes the tuple of hidden states -- ``-1`` (the shipped value) is the
        last encoder block's output, and there is no post-layernorm in Pixtral.
        """
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None, ...]
        _, hidden_states = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1),
            output_hidden_states=True,
            image_sizes=image_sizes,
        )
        feature = hidden_states[self.vision_feature_layer]
        if feature.ndim == 3 and feature.shape[0] == 1:
            feature = feature.squeeze(0)
        return self.projector(feature, image_sizes)

    def merged_embeddings(self, inputs: dict) -> tuple[mx.array, dict]:
        input_ids = inputs["input_ids"]
        embeds = _inner_text_model(self.language_model).embed_tokens(input_ids)

        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            return embeds, {}

        features = self.image_features(
            pixel_values, inputs["image_sizes"]
        ).astype(embeds.dtype)

        mask = mx.broadcast_to(
            mx.expand_dims(input_ids == self.image_token_id, -1), embeds.shape
        )
        return masked_scatter(embeds, mask, features), {}


register_frontend("mistral3", Mistral3VisionFrontend.from_pretrained)
