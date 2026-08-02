"""Project constants for MTPLX gates and defaults."""

from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "MTPLX"
PRIMARY_MODEL_REPO = "trevon/Qwen3.6-27B-mtp"
OFFICIAL_MODEL_REPO = "Qwen/Qwen3.6-27B"
DFLASH_MODEL_REPO = "z-lab/Qwen3.6-27B-DFlash"

PRIMARY_MODEL_DIR = Path("models/Qwen3.6-27B-mtp")
DEFAULT_RUNTIME_MODEL_DIR = Path("models/Qwen3.6-27B-MTPLX-Optimized-Speed")
LEGACY_SPEED_BASELINE_MODEL_DIR = Path("models/Qwen3.6-27B-MLXCommunity-4bit-mtp-graft")

DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20

EXPECTED_MTP_TENSOR_COUNT = 15
EXPECTED_MTP_KEYS = (
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
)

MTP_QUANTIZED_LINEAR_WEIGHT_KEYS = (
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
)

MTP_ALL_QUANTIZED_LINEAR_WEIGHT_KEYS = (
    "mtp.fc.weight",
    *MTP_QUANTIZED_LINEAR_WEIGHT_KEYS,
)

EXPECTED_PREQUANTIZED_MTP_KEYS = tuple(
    sorted(
        EXPECTED_MTP_KEYS
        + tuple(key.rsplit(".", 1)[0] + ".scales" for key in MTP_QUANTIZED_LINEAR_WEIGHT_KEYS)
        + tuple(key.rsplit(".", 1)[0] + ".biases" for key in MTP_QUANTIZED_LINEAR_WEIGHT_KEYS)
    )
)
EXPECTED_PREQUANTIZED_MTP_TENSOR_COUNT = len(EXPECTED_PREQUANTIZED_MTP_KEYS)

EXPECTED_ALL_PREQUANTIZED_MTP_KEYS = tuple(
    sorted(
        EXPECTED_MTP_KEYS
        + tuple(
            key.rsplit(".", 1)[0] + ".scales"
            for key in MTP_ALL_QUANTIZED_LINEAR_WEIGHT_KEYS
        )
        + tuple(
            key.rsplit(".", 1)[0] + ".biases"
            for key in MTP_ALL_QUANTIZED_LINEAR_WEIGHT_KEYS
        )
    )
)
EXPECTED_ALL_PREQUANTIZED_MTP_TENSOR_COUNT = len(EXPECTED_ALL_PREQUANTIZED_MTP_KEYS)

MULTIMODAL_SIDECARS = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)
