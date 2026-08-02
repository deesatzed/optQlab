"""Small MTPLX-side TurboQuant configuration helpers.

The vLLM-Metal reference implementation keeps its public constants behind a
``vllm`` import.  MTPLX only needs the stable cache-layout metadata to drive the
already-loaded Metal ops, so keep this dependency-free and deliberately narrow.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


KEY_QUANTS: dict[str, dict[str, int | bool | str]] = {
    "q8_0": {"bits": 8, "signed": True, "dtype": "int8"},
    "int8": {"bits": 8, "signed": True, "dtype": "int8"},
    "uint8": {"bits": 8, "signed": False, "dtype": "uint8"},
    "q5_0": {"bits": 5, "signed": False, "dtype": "uint8"},
    "q4_0": {"bits": 4, "signed": False, "dtype": "uint8"},
    "int4": {"bits": 4, "signed": False, "dtype": "uint8"},
    "uint4": {"bits": 4, "signed": False, "dtype": "uint8"},
    "int2": {"bits": 2, "signed": False, "dtype": "uint8"},
    "uint2": {"bits": 2, "signed": False, "dtype": "uint8"},
}

VALUE_QUANTS: dict[str, int] = {
    "q2_0": 2,
    "q3_0": 3,
    "q4_0": 4,
    "q5_0": 5,
    "q8_0": 8,
}

FWHT_SUPPORTED_HEAD_DIMS = {64, 128, 256, 512}
SCALE_GROUP_SIZE = 32

# vLLM-Metal's built-in 3-bit Lloyd-Max table.  This is the default and the
# only value quant we need for the first project-level diagnostic.
CENTROIDS_3BIT = (
    -2.15195,
    -1.34391,
    -0.75601,
    -0.24509,
    0.24509,
    0.75601,
    1.34391,
    2.15195,
)


@dataclass(frozen=True)
class TurboQuantConfig:
    key_quant: str = "q8_0"
    value_quant: str = "q3_0"

    @property
    def key_bits(self) -> int:
        return int(KEY_QUANTS[self.key_quant]["bits"])

    @property
    def value_bits(self) -> int:
        return int(VALUE_QUANTS[self.value_quant])

    @property
    def key_signed(self) -> bool:
        return bool(KEY_QUANTS[self.key_quant]["signed"])

    @property
    def key_dtype_name(self) -> str:
        return str(KEY_QUANTS[self.key_quant]["dtype"])


def env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def config_from_env() -> TurboQuantConfig | None:
    if not env_enabled("MTPLX_VLLM_METAL_PAGED_TURBOQUANT"):
        return None
    key_quant = (os.environ.get("MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT") or "q8_0").strip().lower()
    value_quant = (os.environ.get("MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT") or "q3_0").strip().lower()
    if key_quant not in KEY_QUANTS:
        raise ValueError(
            f"Unsupported TurboQuant key quant {key_quant!r}; "
            f"available={sorted(KEY_QUANTS)}"
        )
    if value_quant not in VALUE_QUANTS:
        raise ValueError(
            f"Unsupported TurboQuant value quant {value_quant!r}; "
            f"available={sorted(VALUE_QUANTS)}"
        )
    return TurboQuantConfig(key_quant=key_quant, value_quant=value_quant)


def packed_dim(head_dim: int, bits: int) -> int:
    if (int(head_dim) * int(bits)) % 8 != 0:
        raise ValueError(
            f"TurboQuant packed dim is not byte-aligned: head_dim={head_dim}, bits={bits}"
        )
    return int(head_dim) * int(bits) // 8


def validate_head_dim(head_dim: int) -> None:
    if int(head_dim) % SCALE_GROUP_SIZE != 0:
        raise ValueError(
            f"TurboQuant requires head_dim divisible by {SCALE_GROUP_SIZE}, got {head_dim}"
        )
    if int(head_dim) not in FWHT_SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"TurboQuant FWHT supports head_dim in {sorted(FWHT_SUPPORTED_HEAD_DIMS)}, got {head_dim}"
        )


def compression_ratio(*, head_dim: int, key_bits: int, value_bits: int) -> float:
    fp16_bytes = 2 * int(head_dim) * 2
    quant_bytes = (
        packed_dim(head_dim, key_bits)
        + packed_dim(head_dim, value_bits)
        + 3 * (int(head_dim) // SCALE_GROUP_SIZE) * 2
    )
    return float(fp16_bytes) / float(quant_bytes)
