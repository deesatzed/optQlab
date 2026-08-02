"""Draft-only LM-head helpers for MTPLX speculative proposals."""

from __future__ import annotations

import time
from typing import Any


def normalize_draft_lm_head_spec(
    value: Any,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a validated draft LM-head spec from profile/contract metadata."""
    if value is None:
        return fallback
    if not isinstance(value, dict):
        raise ValueError("draft LM-head spec must be an object")
    if "bits" not in value:
        raise ValueError("draft LM-head spec missing bits")
    bits = int(value["bits"])
    group_size = int(value.get("group_size", 64))
    mode = str(value.get("mode", "affine"))
    if bits <= 0:
        raise ValueError("draft LM-head bits must be positive")
    if group_size <= 0:
        raise ValueError("draft LM-head group_size must be positive")
    if mode not in {"affine", "symmetric"}:
        raise ValueError("draft LM-head mode must be 'affine' or 'symmetric'")
    return {"bits": bits, "group_size": group_size, "mode": mode}


def draft_lm_head_spec_from_runtime_contract(
    contract_data: Any,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve a model-specific draft-head recommendation from contract data."""
    if not isinstance(contract_data, dict):
        return fallback
    return normalize_draft_lm_head_spec(
        contract_data.get("recommended_draft_lm_head"),
        fallback=fallback,
    )


def _text_model(model: Any) -> Any:
    return getattr(model, "language_model", model)


def _make_requantized_head(module: Any, *, bits: int, group_size: int, mode: str) -> tuple[Any, dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn

    started = time.perf_counter()
    dense = mx.dequantize(
        module.weight,
        module.scales,
        module.biases,
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
    ).astype(mx.bfloat16)
    mx.eval(dense)
    linear = nn.Linear(int(dense.shape[1]), int(dense.shape[0]), bias=("bias" in module))
    linear.weight = dense
    if "bias" in module:
        linear.bias = module.bias
    quantized = nn.QuantizedLinear.from_linear(
        linear,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    mx.eval(quantized.weight, quantized.scales, quantized.biases)
    report = {
        "original": {
            "bits": int(module.bits),
            "group_size": int(module.group_size),
            "mode": str(module.mode),
            "weight_shape": list(module.weight.shape),
            "scales_shape": list(module.scales.shape),
        },
        "draft_only": {
            "bits": int(quantized.bits),
            "group_size": int(quantized.group_size),
            "mode": str(quantized.mode),
            "weight_shape": list(quantized.weight.shape),
            "scales_shape": list(quantized.scales.shape),
        },
        "elapsed_s": time.perf_counter() - started,
    }
    return quantized, report


def _embedding_report(module: Any) -> dict[str, Any]:
    return {
        "bits": int(getattr(module, "bits")),
        "group_size": int(getattr(module, "group_size")),
        "mode": str(getattr(module, "mode")),
        "weight_shape": list(module.weight.shape),
        "scales_shape": list(module.scales.shape),
    }


def _make_embedding_as_linear_head(
    module: Any,
    *,
    bits: int,
    group_size: int,
    mode: str,
) -> tuple[Any, dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn

    class _EmbeddingAsLinear(nn.Module):
        def __init__(self, embedding: Any):
            super().__init__()
            self.embedding = embedding

        def __call__(self, x):
            return self.embedding.as_linear(x)

    started = time.perf_counter()
    if isinstance(module, nn.QuantizedEmbedding):
        original = _embedding_report(module)
        if (
            int(module.bits) == int(bits)
            and int(module.group_size) == int(group_size)
            and str(module.mode) == str(mode)
        ):
            return _EmbeddingAsLinear(module), {
                "source": "tied_embedding",
                "original": original,
                "draft_only": original,
                "reused_existing_quantization": True,
                "elapsed_s": time.perf_counter() - started,
            }
        dense = mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
        ).astype(mx.bfloat16)
        mx.eval(dense)
    elif isinstance(module, nn.Embedding):
        dense = module.weight.astype(mx.bfloat16)
        original = {
            "bits": "bf16",
            "group_size": None,
            "mode": "none",
            "weight_shape": list(module.weight.shape),
            "scales_shape": None,
        }
    else:
        raise TypeError(f"embed_tokens is not Embedding/QuantizedEmbedding: {type(module)!r}")

    embedding = nn.Embedding(int(dense.shape[0]), int(dense.shape[1]))
    embedding.weight = dense
    quantized = nn.QuantizedEmbedding.from_embedding(
        embedding,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    mx.eval(quantized.weight, quantized.scales, quantized.biases)
    return _EmbeddingAsLinear(quantized), {
        "source": "tied_embedding",
        "original": original,
        "draft_only": _embedding_report(quantized),
        "reused_existing_quantization": False,
        "elapsed_s": time.perf_counter() - started,
    }


def _install_draft_lm_head(rt: Any, *, bits: int, group_size: int, mode: str) -> dict[str, Any]:
    import mlx.nn as nn

    text = _text_model(rt.model)
    module = getattr(text, "lm_head", None)
    if module is not None:
        if not isinstance(module, nn.QuantizedLinear):
            raise TypeError(f"lm_head is not QuantizedLinear: {type(module)!r}")
        draft_head, report = _make_requantized_head(
            module,
            bits=bits,
            group_size=group_size,
            mode=mode,
        )
    elif bool(getattr(getattr(text, "args", None), "tie_word_embeddings", False)):
        embed_tokens = getattr(getattr(text, "model", None), "embed_tokens", None)
        draft_head, report = _make_embedding_as_linear_head(
            embed_tokens,
            bits=bits,
            group_size=group_size,
            mode=mode,
        )
    else:
        raise AttributeError("model has no lm_head and does not tie output projection to embeddings")
    text._mtplx_draft_lm_head = draft_head
    return report
