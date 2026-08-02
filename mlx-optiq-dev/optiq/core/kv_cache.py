"""Mixed-precision KV cache quantization for MLX.

The KV cache stores key/value projections for all past tokens during
autoregressive generation. For long contexts, this dominates memory:

  KV memory = 2 * n_layers * seq_len * n_kv_heads * head_dim * dtype_bytes

For Qwen3-0.6B-base at 4K context:
  = 2 * 28 * 4096 * 8 * 64 * 2 bytes = 234 MB (float16)

Quantizing to 4-bit: 234 * 4/16 = 58 MB (75% reduction)

But uniform quantization hurts quality — some layers' KV caches are much
more sensitive than others. OptiQ measures per-layer KV sensitivity and
assigns different bit-widths, preserving quality where it matters.

This module provides:
  1. Per-layer KV cache sensitivity analysis
  2. Mixed-precision KV cache bit assignment
  3. Integration with mlx-lm's generate pipeline
"""

import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class KVSensitivityResult:
    """Per-layer KV cache sensitivity measurement."""
    layer_idx: int
    layer_name: str
    sensitivities: dict[int, float]  # bits -> KL divergence
    n_kv_heads: int
    head_dim: int
    cache_bytes_per_token: float  # bytes per token at full precision


def measure_kv_sensitivity(
    model_path: str,
    candidate_bits: list[int] = [4, 8],
    n_samples: int = 5,
    seq_len: int = 512,
    group_size: int = 64,
) -> list[KVSensitivityResult]:
    """Measure per-layer KV cache sensitivity to quantization.

    For each layer, quantizes only that layer's KV cache and measures
    the KL divergence of the output logits vs reference (full-precision
    KV cache).

    This is analogous to weight sensitivity analysis but for the KV cache:
    some layers' KV projections are robust to quantization (attention
    patterns are simple), while others are highly sensitive (complex
    multi-head interactions).

    Args:
        model_path: Path to MLX model directory
        candidate_bits: Bit-widths to test (e.g., [4, 8])
        n_samples: Number of calibration sequences
        seq_len: Length of each calibration sequence
        group_size: Quantization group size
    """
    from mlx_lm import load

    print(f"  Loading model from {model_path}...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())

    n_layers = len(model.layers)

    # Load calibration data
    calibration_tokens = _get_calibration_tokens(tokenizer, n_samples, seq_len)
    print(f"  Calibration: {len(calibration_tokens)} sequences x {seq_len} tokens")

    # Step 1: Get reference logits with full-precision KV cache
    print(f"  Computing reference logits (full-precision KV cache)...")
    ref_logits_list = []
    for tokens in calibration_tokens:
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        ref_logits_list.append(logits)
        mx.eval(logits)

    # Step 2: For each full-attention layer, quantize only that layer's KV cache.
    # Hybrid models (e.g. Qwen3.5) mix linear-attention layers — those don't
    # have a KV cache to quantize and are skipped. For sliding-window models
    # (Gemma 3/4, Cohere R2, OLMo 3, etc.) install our RotatingQuantizedKVCache
    # patch so the rotating-cache layers can be substituted with a quantized
    # rotating variant; without it mlx-lm raises ``NotImplementedError``.
    from ..runtime.kv.rotating import empty_quantized_like
    from mlx_lm.models.cache import (
        KVCache,
        QuantizedKVCache,
        RotatingKVCache,
        make_prompt_cache,
    )

    from optiq.runtime.kv import (
        RotatingQuantizedKVCache,
        patch_rotating_to_quantized,
    )
    patch_rotating_to_quantized()

    results = []

    def _build_reference_cache():
        """Fresh prompt cache honoring the model's own make_cache if present."""
        return make_prompt_cache(model)

    # Precompute which layer indices have a quantizable cache.
    # `make_prompt_cache` returns one cache slot per *layer* (including
    # Mamba/SSM and MLP-only slots on hybrids like NemotronH), and some of
    # those non-attention slots expose `to_quantized` for their state
    # tensors. We only want to quantize *real* attention KV caches, so we
    # also require the layer's mixer module to look like attention
    # (q_proj / qkv_proj / wq present).
    def _attention_module(layer):
        for name in ("self_attn", "attention", "mixer"):
            sub = getattr(layer, name, None)
            if sub is None:
                continue
            if hasattr(sub, "q_proj") or hasattr(sub, "qkv_proj") or hasattr(sub, "wq"):
                return sub
        return None

    _probe = _build_reference_cache()
    # Build cache-index → model-layer-index map. On hybrid models like
    # NemotronH the prompt cache is shorter than the layer list because
    # MLP-only layers are skipped — caches are returned only for
    # stateful blocks (Mamba/SSM + attention). Map by walking the layers
    # in order and assigning cache slots to layers that have any cache-
    # bearing module (mixer / self_attn / attention). For flat
    # transformer models this collapses to the identity mapping.
    def _is_cache_bearing(layer):
        # On hybrid models like NemotronH every layer carries some kind of
        # ``mixer`` attribute, but MLP-only mixers (no q_proj / in_proj /
        # conv1d) are stateless and don't get a slot in the prompt cache.
        # We treat a layer as cache-bearing iff it looks like attention
        # (q_proj / qkv_proj / wq) or a state-space block (in_proj +
        # conv1d / x_proj / dt_proj). Pure MLP blocks have only
        # up_proj/down_proj and return False.
        for name in ("self_attn", "attention"):
            if getattr(layer, name, None) is not None:
                return True
        mixer = getattr(layer, "mixer", None)
        if mixer is None:
            return False
        attn_like = any(hasattr(mixer, a) for a in ("q_proj", "qkv_proj", "wq"))
        ssm_like = hasattr(mixer, "in_proj") and any(
            hasattr(mixer, a) for a in ("conv1d", "x_proj", "dt_proj", "A_log", "D")
        )
        return attn_like or ssm_like

    if len(_probe) == n_layers:
        cache_to_layer = {i: i for i in range(n_layers)}
    else:
        cache_to_layer = {}
        cache_pos = 0
        for layer_idx, layer in enumerate(model.layers):
            if cache_pos >= len(_probe):
                break
            if _is_cache_bearing(layer):
                cache_to_layer[cache_pos] = layer_idx
                cache_pos += 1

    quantizable_idxs = []
    for cache_idx, cache in enumerate(_probe):
        if not hasattr(cache, "to_quantized"):
            continue
        layer_idx = cache_to_layer.get(cache_idx)
        if layer_idx is None:
            continue
        if _attention_module(model.layers[layer_idx]) is None:
            continue
        quantizable_idxs.append((cache_idx, layer_idx))

    print(f"  {len(quantizable_idxs)}/{n_layers} layers have a KV cache (others are linear-attn / skipped)")

    for cache_idx, layer_idx in quantizable_idxs:
        layer = model.layers[layer_idx]
        sa = _attention_module(layer)
        if sa is None:
            print(f"  Layer {layer_idx}: no attention module, skipping")
            continue
        n_kv_heads = getattr(sa, "n_kv_heads", None) or getattr(sa, "num_key_value_heads", 0)
        head_dim = getattr(sa, "head_dim", None)
        if head_dim is None:
            try:
                head_dim = round(1.0 / (sa.scale ** 2))
            except Exception:
                head_dim = 128
        bytes_per_token = 2 * n_kv_heads * head_dim * 2  # K+V, float16

        sensitivities = {}

        for bits in candidate_bits:
            kl_divs = []

            for sample_idx, tokens in enumerate(calibration_tokens):
                input_ids = mx.array([tokens])

                # Fresh reference-layout cache, with only the target layer quantized
                cache_q = _build_reference_cache()
                # Replace only the target layer with a quantized variant.
                # Match the existing cache type: rotating → rotating quantized,
                # plain → plain quantized. This keeps sliding-window semantics
                # intact for Gemma 3/4 / Cohere R2 / etc. Index into the
                # cache list by ``cache_idx`` (not ``layer_idx``) because on
                # hybrid models the cache list is shorter than the layer
                # list (NemotronH skips MLP slots).
                cache_q[cache_idx] = empty_quantized_like(
                    cache_q[cache_idx], bits=bits, group_size=group_size,
                )

                logits_q = model(input_ids, cache=cache_q)
                mx.eval(logits_q)

                ref_logits = ref_logits_list[sample_idx]
                kl = _compute_kl_divergence(ref_logits, logits_q)
                kl_divs.append(kl)

            mean_kl = float(np.mean(kl_divs))
            # Clamp to 0 — negative values are numerical noise
            sensitivities[bits] = max(mean_kl, 0.0)

        layer_name = f"model.layers.{layer_idx}"
        results.append(KVSensitivityResult(
            layer_idx=layer_idx,
            layer_name=layer_name,
            sensitivities=sensitivities,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            cache_bytes_per_token=bytes_per_token,
        ))

        sens_str = ", ".join(f"{b}b:{sensitivities[b]:.4f}" for b in candidate_bits)
        print(f"  Layer {layer_idx:>2}/{n_layers}: {sens_str}")

    del model
    import gc
    gc.collect()

    return results


def _compute_kl_divergence(ref_logits: mx.array, test_logits: mx.array) -> float:
    """Compute KL divergence between two sets of logits."""
    # Use last position's logits for efficiency
    ref = ref_logits[0, -1, :]  # (vocab_size,)
    test = test_logits[0, -1, :]

    # Log-softmax
    ref_log_probs = ref - mx.logsumexp(ref, keepdims=True)
    test_log_probs = test - mx.logsumexp(test, keepdims=True)

    # KL(ref || test) = sum(ref_probs * (log_ref - log_test))
    ref_probs = mx.softmax(ref)
    kl = mx.sum(ref_probs * (ref_log_probs - test_log_probs))
    return float(kl.item())


def _get_calibration_tokens(tokenizer, n_samples, seq_len):
    """Load calibration tokens from WikiText-2."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [t for t in ds["text"] if len(t.strip()) > 100]
    except Exception:
        texts = [
            "The transformer architecture has revolutionized natural language processing. "
            "Self-attention mechanisms allow models to capture long-range dependencies "
            "in text sequences, enabling breakthrough performance on many tasks. "
        ] * n_samples * 5

    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= seq_len:
            all_tokens.append(tokens[:seq_len])
        if len(all_tokens) >= n_samples:
            break

    # Pad if needed
    while len(all_tokens) < n_samples:
        # Combine texts to reach seq_len
        combined = []
        for text in texts:
            combined.extend(tokenizer.encode(text, add_special_tokens=False))
            if len(combined) >= seq_len:
                all_tokens.append(combined[:seq_len])
                break
        if len(all_tokens) >= n_samples:
            break

    return all_tokens[:n_samples]


@dataclass
class KVCacheConfig:
    """Per-layer KV cache quantization configuration."""
    layer_idx: int
    bits: int  # 0 = full precision (no quantization)
    group_size: int = 64


def optimize_kv_cache(
    sensitivity_results: list[KVSensitivityResult],
    target_kv_bits: float = 6.0,
    candidate_bits: list[int] = [4, 8],
    group_size: int = 64,
) -> list[KVCacheConfig]:
    """Assign per-layer KV cache bit-widths via greedy knapsack.

    Same algorithm as weight optimization: start all layers at min bits,
    greedily upgrade the most sensitive layers to higher bits until the
    average KV bits target is reached.

    Args:
        sensitivity_results: Per-layer KV sensitivity measurements
        target_kv_bits: Target average bits for KV cache
        candidate_bits: Available bit-widths
        group_size: Quantization group size

    Returns:
        List of per-layer KV cache configurations
    """
    import heapq

    candidate_bits = sorted(candidate_bits)
    min_bits = candidate_bits[0]
    max_bits = candidate_bits[-1]
    n_layers = len(sensitivity_results)

    # All layers have the same KV cache size per token (same heads/dim)
    # So we optimize purely on sensitivity. Hybrid models (Qwen3.5 etc)
    # only expose KV at every Nth layer, so layer_idx is sparse — index
    # results by layer_idx, not by list position.
    by_layer = {r.layer_idx: r for r in sensitivity_results}
    allocation = {idx: min_bits for idx in by_layer}

    # Build upgrade heap
    upgrade_heap = []
    for r in sensitivity_results:
        _push_kv_upgrade(upgrade_heap, r, min_bits, candidate_bits)

    # Policy: target_kv_bits is a FLOOR — we keep upgrading the most-
    # sensitive layers until the running average meets or exceeds the
    # target. Slight overshoot (driven by the discrete candidate grid) is
    # accepted because users care about a quality floor; undershooting
    # silently would betray the contract.
    def _current_avg() -> float:
        return sum(allocation.values()) / len(allocation)

    while upgrade_heap and _current_avg() < target_kv_bits:
        neg_eff, layer_idx, old_bits, new_bits, kl_reduction = heapq.heappop(upgrade_heap)

        if allocation[layer_idx] != old_bits:
            continue

        allocation[layer_idx] = new_bits

        r = by_layer[layer_idx]
        _push_kv_upgrade(upgrade_heap, r, new_bits, candidate_bits)

    configs = [
        KVCacheConfig(layer_idx=r.layer_idx, bits=allocation[r.layer_idx], group_size=group_size)
        for r in sensitivity_results
    ]

    achieved_bits = sum(c.bits for c in configs) / len(configs)
    n_high = sum(1 for c in configs if c.bits == max_bits)
    n_low = sum(1 for c in configs if c.bits == min_bits)

    print(f"  KV cache optimization: {n_low} layers @ {min_bits}-bit, "
          f"{n_high} layers @ {max_bits}-bit")
    print(f"  Target KV bits: {target_kv_bits:.1f}, achieved: {achieved_bits:.2f}")

    return configs


def _push_kv_upgrade(heap, r, current_bits, candidate_bits):
    """Push next KV cache upgrade onto the heap."""
    import heapq

    idx = candidate_bits.index(current_bits) if current_bits in candidate_bits else -1
    if idx < 0 or idx >= len(candidate_bits) - 1:
        return

    next_bits = candidate_bits[idx + 1]
    kl_current = r.sensitivities.get(current_bits, 0)
    kl_next = r.sensitivities.get(next_bits, 0)
    kl_reduction = kl_current - kl_next

    if kl_reduction <= 0:
        return

    # All layers have the same cache size, so efficiency = kl_reduction / bit_cost
    bit_cost = next_bits - current_bits
    efficiency = kl_reduction / max(bit_cost, 1)

    heapq.heappush(heap, (-efficiency, r.layer_idx, current_bits, next_bits, kl_reduction))


def make_mixed_kv_cache(kv_configs: list[KVCacheConfig]) -> list:
    """Create a per-layer mixed-precision KV cache. DEMO USE ONLY.

    This cannot be correct for a sliding-window model, and it is not used by
    `optiq serve` or `optiq eval`, which is why it is still here. It builds caches
    from a config list alone, so it has no way to know which layers the model
    wants as `RotatingKVCache` — on Gemma-4 that is four of every five. The
    serving path does the only thing that works: let the model construct its own
    caches, then convert each in place with
    `runtime.kv.rotating.quantize_cache_layer`, which reads the layer's actual
    type.

    Callers: demo/demo_combined.py, demo/demo_kv_cache.py. Do not use it for
    anything a user runs.
    """
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    caches = []
    for config in kv_configs:
        cache = QuantizedKVCache(group_size=config.group_size, bits=config.bits)
        caches.append(cache)

    return caches


def maybe_quantize_kv_mixed(prompt_cache, kv_configs, quantized_kv_start=0):
    """Per-layer KV cache quantization (replaces mlx-lm's uniform version).

    Args:
        prompt_cache: List of KVCache objects (one per layer)
        kv_configs: Per-layer bit-width assignments
        quantized_kv_start: Minimum offset before quantization triggers
    """
    from mlx_lm.models.cache import KVCache

    for config in kv_configs:
        idx = config.layer_idx
        if idx >= len(prompt_cache):
            continue
        c = prompt_cache[idx]
        # Only convert KVCache (not already quantized)
        if isinstance(c, KVCache) and c.offset >= quantized_kv_start:
            prompt_cache[idx] = c.to_quantized(
                group_size=config.group_size,
                bits=config.bits,
            )


def generate_with_mixed_kv(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    kv_configs: list[KVCacheConfig] | None = None,
    temp: float = 0.0,
    verbose: bool = False,
) -> dict:
    """Generate text with per-layer mixed-precision KV cache.

    Uses mlx-lm's generate_step but with a custom KV cache that has
    different bit-widths per layer.

    Returns dict with generated text, tokens, timing info.
    """
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    tokens = tokenizer.encode(prompt)
    prompt_tokens = mx.array(tokens)

    sampler = make_sampler(temp=temp)

    # Create mixed-precision cache from the start
    if kv_configs is not None:
        prompt_cache = make_mixed_kv_cache(kv_configs)
    else:
        prompt_cache = None

    generated_tokens = []
    token_times = []

    tic = time.perf_counter()
    for token, logprobs in generate_step(
        prompt_tokens,
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=prompt_cache,
    ):
        token_times.append(time.perf_counter() - tic)
        generated_tokens.append(token.item() if hasattr(token, 'item') else int(token))
        tic = time.perf_counter()

    text = tokenizer.decode(generated_tokens)

    # Memory estimate for KV cache
    n_total_tokens = len(tokens) + len(generated_tokens)
    kv_memory = 0
    if kv_configs:
        for config in kv_configs:
            # Each layer: 2 (K+V) * n_kv_heads * head_dim * n_tokens * bits/8
            # We don't know head_dim here, estimate from model
            layer = model.layers[config.layer_idx]
            sa = (
                getattr(layer, "self_attn", None)
                or getattr(layer, "attention", None)
                or getattr(layer, "mixer", None)
            )
            n_kv_heads = getattr(sa, "n_kv_heads", None) or getattr(sa, "num_key_value_heads", 0)
            head_dim = getattr(sa, "head_dim", None)
            if head_dim is None:
                head_dim = round(1.0 / (sa.scale ** 2))
            kv_memory += 2 * n_kv_heads * head_dim * n_total_tokens * config.bits / 8

    return {
        "text": text,
        "n_prompt_tokens": len(tokens),
        "n_generated_tokens": len(generated_tokens),
        "total_time": sum(token_times),
        "tokens_per_sec": len(generated_tokens) / sum(token_times) if token_times else 0,
        "kv_memory_bytes": kv_memory,
        "kv_memory_mb": kv_memory / (1024 * 1024),
    }


def estimate_kv_memory(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    bits: float = 16.0,
) -> float:
    """Estimate KV cache memory in bytes.

    Args:
        n_layers: Number of transformer layers
        n_kv_heads: Number of KV heads per layer
        head_dim: Dimension per head
        seq_len: Sequence length
        bits: Average bits per element (16 for float16, 8/4 for quantized)

    Returns: Memory in bytes
    """
    # K + V = 2 tensors per layer
    elements = 2 * n_layers * seq_len * n_kv_heads * head_dim
    return elements * bits / 8
