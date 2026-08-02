"""Mixed-precision quantization optimizer: select per-layer bit-widths.

Supports multi-tier allocation (e.g., 2/3/4/8-bit) via greedy knapsack,
not just binary high/low. This is the key to achieving same quality at
fewer bits — robust layers get 2-3 bits, sensitive layers get 8.

Also supports hardware-aware optimization that considers Apple Silicon
Metal kernel latency, not just model quality and size.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union
import heapq

import numpy as np

from .sensitivity import SensitivityResult


@dataclass
class LayerQuantConfig:
    layer_name: str
    bits: int
    group_size: int = 64
    param_count: int = 0


@dataclass
class OptimizationResult:
    configs: list[LayerQuantConfig]
    target_bpw: float
    achieved_bpw: float
    threshold: float  # kept for backward compat; 0 for multi-tier
    n_high_bits: int
    n_low_bits: int
    total_params: int
    estimated_size_mb: float
    # Hardware-aware fields (populated by optimize_latency_aware)
    estimated_decode_us: float = 0.0
    estimated_prefill_us: float = 0.0
    latency_weight: float = 0.0  # alpha used for latency-aware optimization


from .sensitivity import KEEP_BF16_BITS  # 16 = leave the layer at bf16


def compute_bpw(configs: list[LayerQuantConfig]) -> float:
    """Compute average bits-per-weight from layer configs."""
    total_bits = sum(c.bits * c.param_count for c in configs)
    total_params = sum(c.param_count for c in configs)
    if total_params == 0:
        return 0.0
    return total_bits / total_params


def optimize_mixed_precision(
    sensitivity_results: list[SensitivityResult],
    target_bpw: float = 4.0,
    candidate_bits: list[int] | None = None,
    group_size: int = 64,
    protect_first_last: bool = True,
    n_protect: int = 1,
    n_floor_per_block: int = 2,
    max_low_bit_run: int = 3,
    # Pre-N-tier shims; ignored when candidate_bits is set explicitly
    low_bits: int = 4,
    high_bits: int = 8,
) -> OptimizationResult:
    """Select per-layer bit-widths via greedy knapsack optimization.

    For each layer, picks from candidate bit-widths to minimize total
    KL divergence subject to the BPW budget constraint.

    Algorithm:
    1. Start all layers at minimum bits.
    2. **Block-aware floor**: pre-upgrade the top-N most-sensitive components
       in EACH transformer block to the 2nd-lowest bit-width. Prevents the
       "many consecutive blocks fully at the lowest bit" failure mode of
       pure per-layer sensitivity (compounding error across blocks).
    3. Compute remaining bit budget.
    4. Greedily upgrade layers with highest KL-reduction per bit spent.
    5. **Block-run guard**: post-greedy, scan for any run of `max_low_bit_run`+
       consecutive blocks where >50% of components sit at the lowest bit;
       upgrade the worst offenders to 2nd-lowest.

    Args:
        sensitivity_results: Per-layer KL scores at each candidate bit-width
        target_bpw: Target average bits per weight
        candidate_bits: Available bit-widths (default: inferred from sensitivity data)
        group_size: Quantization group size
        protect_first_last: Protect first/last transformer blocks
        n_protect: Number of first/last blocks to protect
        n_floor_per_block: Top-N most-sensitive components per block kept
            at >= 2nd-lowest bit-width. Set to 0 to disable. Critical when
            using aggressive bit-widths (e.g. ``candidate_bits=[2, 4, 8]``)
            on dense models.
        max_low_bit_run: Maximum consecutive transformer blocks where >50%
            of components sit at the lowest bit-width. Set to 0 to disable.
            Backstop for when ``n_floor_per_block`` isn't enough.
    """
    if not sensitivity_results:
        return OptimizationResult(
            configs=[], target_bpw=target_bpw, achieved_bpw=0,
            threshold=0, n_high_bits=0, n_low_bits=0,
            total_params=0, estimated_size_mb=0,
        )

    # Infer candidate bits from sensitivity data if not provided
    if candidate_bits is None:
        all_bits = set()
        for r in sensitivity_results:
            all_bits.update(r.sensitivities.keys())
        candidate_bits = sorted(all_bits)
    if not candidate_bits:
        candidate_bits = [low_bits, high_bits]

    candidate_bits = sorted(candidate_bits)
    min_bits = candidate_bits[0]
    max_bits = candidate_bits[-1]

    # Identify layers to always protect
    protected_layers = set()
    protected_layers.update(_identify_output_layers(
        [r.layer_name for r in sensitivity_results]
    ))
    protected_layers.update(_identify_moe_protected_layers(
        [r.layer_name for r in sensitivity_results]
    ))
    if protect_first_last:
        protected_layers.update(_identify_boundary_layers(
            [r.layer_name for r in sensitivity_results], n_protect
        ))

    total_params = sum(r.param_count for r in sensitivity_results)

    # Initialize all layers at minimum bits — protected layers participate
    # in the greedy allocation like everyone else, but get a KL bonus
    # to ensure they are upgraded first
    allocation = {}
    for r in sensitivity_results:
        allocation[r.layer_name] = min_bits

    # Block-aware floor: pre-upgrade the most-sensitive components per
    # transformer block. Prevents "many consecutive blocks fully at the
    # lowest bit" — the failure mode of pure per-layer sensitivity when
    # candidate_bits includes very aggressive widths (e.g. 2-bit).
    floor_upgrades = 0
    if n_floor_per_block > 0 and len(candidate_bits) >= 2:
        second_lowest = candidate_bits[1]
        floor_upgrades = _apply_block_aware_floor(
            allocation, sensitivity_results, candidate_bits[0], second_lowest,
            n_floor_per_block,
        )

    # Handle NaN sensitivities — treat as maximally sensitive (assign max bits)
    # Replace NaN with large finite values so the greedy allocator upgrades them
    import math
    nan_layers = set()
    cleaned_results = []
    for r in sensitivity_results:
        has_nan = any(isinstance(v, float) and math.isnan(v) for v in r.sensitivities.values())
        if has_nan:
            nan_layers.add(r.layer_name)
            # Replace NaN: high KL at low bits, low KL at high bits → strong upgrade signal
            max_finite = max((v for v in r.sensitivities.values()
                             if isinstance(v, (int, float)) and not math.isnan(v)), default=1e6)
            fixed_sens = {}
            for b, v in r.sensitivities.items():
                if isinstance(v, float) and math.isnan(v):
                    # Lower bits get higher sensitivity, higher bits get lower
                    fixed_sens[b] = max_finite * 10 * (max_bits / max(b, 1))
                else:
                    fixed_sens[b] = v
            cleaned_results.append(SensitivityResult(
                layer_name=r.layer_name, sensitivities=fixed_sens, param_count=r.param_count,
            ))
        else:
            cleaned_results.append(r)
    if nan_layers:
        print(f"  Note: {len(nan_layers)} layers had NaN sensitivity — treating as maximally sensitive")
        sensitivity_results = cleaned_results

    # Build effective sensitivity results — protected layers get a KL bonus
    # so the greedy allocator upgrades them first (they're most important)
    effective_results = {}
    for r in sensitivity_results:
        if r.layer_name in protected_layers:
            effective_results[r.layer_name] = SensitivityResult(
                layer_name=r.layer_name,
                sensitivities={b: v * 100.0 for b, v in r.sensitivities.items()},
                param_count=r.param_count,
            )
        else:
            effective_results[r.layer_name] = r

    # Build upgrade candidates
    upgrade_heap = []  # min-heap on -efficiency
    for r in sensitivity_results:
        current = allocation[r.layer_name]
        _push_next_upgrade(upgrade_heap, effective_results[r.layer_name], current, candidate_bits)

    # Policy: target_bpw is a FLOOR — keep upgrading the most-sensitive
    # layers until the running BPW meets or exceeds the target. Slight
    # overshoot from the discrete candidate grid is accepted because
    # users care about a quality floor; silently undershooting would
    # betray the contract.
    while upgrade_heap and _compute_allocation_bpw(allocation, sensitivity_results) < target_bpw:
        neg_eff, layer_name, old_bits, new_bits, kl_reduction = heapq.heappop(upgrade_heap)

        # Check this upgrade is still valid (layer may have been upgraded already)
        if allocation[layer_name] != old_bits:
            continue

        # Apply upgrade
        allocation[layer_name] = new_bits

        # Push next upgrade for this layer (using effective/boosted results)
        _push_next_upgrade(upgrade_heap, effective_results[layer_name], new_bits, candidate_bits)

    # Post-greedy block-run guard: scan for runs of consecutive blocks
    # dominated by the lowest bit-width, upgrade their worst components.
    runfix_upgrades = 0
    if max_low_bit_run > 0 and len(candidate_bits) >= 2:
        runfix_upgrades = _enforce_block_run_limit(
            allocation, sensitivity_results, candidate_bits[0], candidate_bits[1],
            max_low_bit_run,
        )

    # Build configs
    configs = []
    for r in sensitivity_results:
        configs.append(LayerQuantConfig(
            layer_name=r.layer_name,
            bits=allocation[r.layer_name],
            group_size=group_size,
            param_count=r.param_count,
        ))

    achieved_bpw = compute_bpw(configs)
    n_max = sum(1 for c in configs if c.bits == max_bits)
    n_min = sum(1 for c in configs if c.bits == min_bits)
    est_size = sum(c.bits * c.param_count for c in configs) / 8
    est_size_mb = est_size / (1024 * 1024)

    # Print allocation summary
    bit_counts = {}
    for c in configs:
        bit_counts[c.bits] = bit_counts.get(c.bits, 0) + 1
    summary = ", ".join(f"{count} @ {bits}-bit" for bits, count in sorted(bit_counts.items()))
    print(f"  Optimization result: {summary}")
    print(f"  Target BPW: {target_bpw:.2f}, Achieved BPW: {achieved_bpw:.2f}")
    print(f"  Estimated model size: {est_size_mb:.1f} MB")
    if floor_upgrades > 0 or runfix_upgrades > 0:
        print(f"  Block-aware constraints: floor={floor_upgrades} pre-upgrades, "
              f"run-guard={runfix_upgrades} post-fixes")

    return OptimizationResult(
        configs=configs,
        target_bpw=target_bpw,
        achieved_bpw=achieved_bpw,
        threshold=0.0,
        n_high_bits=n_max,
        n_low_bits=n_min,
        total_params=total_params,
        estimated_size_mb=est_size_mb,
    )


def _block_index(layer_name: str) -> int | None:
    """Extract transformer block index from layer name (e.g. ``...layers.7.mlp.up_proj`` → 7)."""
    import re
    m = re.search(r"layers\.(\d+)", layer_name)
    return int(m.group(1)) if m else None


_ATTENTION_TOKENS = (
    # Full-attention (Qwen, Llama, Gemma): q/k/v/o
    "q_proj", "k_proj", "v_proj", "o_proj",
    # Linear-attention / GatedDeltaNet (Qwen3.5/3.6 hybrid)
    "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
    # MoE attention variants and generic fall-throughs
    "qkv_proj", "wqkv", "Wqkv", "attn_qkv",
)


def _is_attention_component(layer_name: str) -> bool:
    """Heuristic: does this linear sit inside a transformer block's attention
    sub-module (vs MLP / experts / router)? We check the trailing component
    against a known set of attention-projection suffixes that covers all
    architectures shipped in v0.1.0 (Qwen3.5/3.6 hybrid, Gemma-4 dense and MoE).
    """
    last = layer_name.rsplit(".", 1)[-1]
    return last in _ATTENTION_TOKENS


def _apply_block_aware_floor(
    allocation: dict, sensitivity_results: list[SensitivityResult],
    lowest_bit: int, second_lowest_bit: int, n_floor: int,
) -> int:
    """Pre-upgrade per-block components from ``lowest_bit`` to ``second_lowest_bit``.

    For each transformer block we promote ``n_floor`` components, but the
    first slot is reserved for the most-sensitive **attention** component
    (q/k/v/o or the linear-attn projection equivalents) when the block has
    any. Without this carve-out, MLP up/down projections systematically win
    the global per-block sensitivity ranking and the entire q_proj/k_proj/
    v_proj/o_proj stack stays at ``lowest_bit`` — which collapses
    instruction-following on late layers (observed on Qwen3.5-4B v0.1.0:
    layers 23 and 31 had ALL four attention components at 4-bit, producing
    a -12.9 pp regression on IFEval despite wins on every other metric).

    The remaining ``n_floor - 1`` slots fall back to the global per-block
    sensitivity ranking. Sensitivity ranking uses the KL of the LOWEST
    candidate bit-width: large KL = "this layer suffers most when squeezed".
    Returns the number of upgrades applied.
    """
    block_layers: dict[int, list[tuple[float, str]]] = {}
    for r in sensitivity_results:
        idx = _block_index(r.layer_name)
        if idx is None:
            continue
        kl_low = r.sensitivities.get(lowest_bit, 0)
        block_layers.setdefault(idx, []).append((kl_low, r.layer_name))

    upgrades = 0
    for idx, layers in block_layers.items():
        if not layers:
            continue
        layers.sort(key=lambda x: -x[0])  # most-sensitive first

        attn = [(kl, n) for kl, n in layers if _is_attention_component(n)]

        # Reserve attention slots first. Full-attention blocks (q/k/v/o all
        # present) get 2 reserved slots; linear-attn / single-projection
        # blocks get 1. This is the load-bearing bit that prevents late
        # full-attention layers from collapsing to all-4-bit.
        n_attn_reserve = 2 if len(attn) >= 4 else (1 if attn else 0)
        n_attn_reserve = min(n_attn_reserve, n_floor)

        # Per-block floor budget. Full-attention blocks honor the global
        # n_floor (default 2). Mamba/SSM/MLP blocks typically have only 2
        # linears per block (e.g. NemotronH mixer in_proj/out_proj, MLP
        # up_proj/down_proj), so flooring 2 of 2 promotes the entire block
        # and makes the bpw target unreachable. Cap their floor at 1.
        is_full_attn_block = len(attn) >= 4
        block_floor = n_floor if is_full_attn_block else min(n_floor, 1)

        picks: list[str] = []
        for kl, name in attn[:n_attn_reserve]:
            picks.append(name)

        # Remaining slots come from the overall most-sensitive list,
        # skipping anything we already picked.
        picked_set = set(picks)
        for _, name in layers:
            if len(picks) >= block_floor:
                break
            if name in picked_set:
                continue
            picks.append(name)
            picked_set.add(name)

        for name in picks:
            if allocation.get(name, lowest_bit) < second_lowest_bit:
                allocation[name] = second_lowest_bit
                upgrades += 1
    return upgrades


def _enforce_block_run_limit(
    allocation: dict, sensitivity_results: list[SensitivityResult],
    lowest_bit: int, second_lowest_bit: int, max_run: int,
) -> int:
    """Scan for runs of consecutive transformer blocks where the majority
    (>50%) of components sit at ``lowest_bit``. If a run exceeds
    ``max_run`` blocks, upgrade the highest-sensitivity ``lowest_bit``
    components in those blocks to ``second_lowest_bit``.

    Returns the number of upgrades applied. This is a backstop to the
    block-aware floor — kicks in only when an entire stretch is still
    misallocated.
    """
    # Group layers by block index
    by_block: dict[int, list[str]] = {}
    sens_map: dict[str, SensitivityResult] = {r.layer_name: r for r in sensitivity_results}
    for name in allocation:
        idx = _block_index(name)
        if idx is None:
            continue
        by_block.setdefault(idx, []).append(name)

    # Compute per-block "low-bit dominance" flag
    block_indices = sorted(by_block)
    low_dom = {}
    for idx in block_indices:
        layers = by_block[idx]
        low = sum(1 for n in layers if allocation[n] == lowest_bit)
        low_dom[idx] = (low / max(len(layers), 1)) > 0.5

    # Find runs of consecutive low-dominated blocks
    upgrades = 0
    run_start = None
    for i, idx in enumerate(block_indices):
        if low_dom[idx]:
            if run_start is None:
                run_start = i
        if not low_dom[idx] or i == len(block_indices) - 1:
            run_end = i if not low_dom[idx] else i + 1
            if run_start is not None and (run_end - run_start) > max_run:
                # Upgrade most-sensitive lowest_bit components in the
                # MIDDLE of the run (early/late edges already protected
                # by boundary protection).
                middle = block_indices[run_start:run_end]
                # Take every Nth block in the run to upgrade
                stride = max(1, max_run)
                for blk in middle[::stride]:
                    candidates = [
                        (sens_map[n].sensitivities.get(lowest_bit, 0), n)
                        for n in by_block[blk]
                        if allocation[n] == lowest_bit
                    ]
                    candidates.sort(key=lambda x: -x[0])
                    # Upgrade top-1 in this block
                    for _, name in candidates[:1]:
                        allocation[name] = second_lowest_bit
                        upgrades += 1
            run_start = None
    return upgrades


def optimize_latency_aware(
    sensitivity_results: list[SensitivityResult],
    target_bpw: float = 4.0,
    candidate_bits: list[int] | None = None,
    group_size: int = 64,
    protect_first_last: bool = True,
    n_protect: int = 1,
    model_config: dict | None = None,
    latency_weight: float = 0.5,
    hw_profile: "HardwareProfile | None" = None,
) -> OptimizationResult:
    """Hardware-aware mixed-precision optimizer for Apple Silicon.

    Uses the same greedy knapsack as the quality-only optimizer, but
    changes the upgrade ordering to account for latency cost:

        efficiency = kl_reduction / (bit_cost + alpha * latency_increase)

    This REORDERS which layers get upgraded (not how many), achieving
    the same BPW with better speed by preferring to upgrade small-but-
    sensitive layers over large-but-sensitive ones.

    On Apple Silicon at decode time, latency is memory-bandwidth-bound:
      latency ∝ bytes read ∝ bits * param_count / 8
    So upgrading a large layer (e.g. gate_proj with 3M params) from 4→8
    bit adds ~4.5 us latency, while upgrading k_proj (1M params) adds
    only ~1.5 us. If both have similar KL sensitivity, the HW-aware
    optimizer picks k_proj first.

    Args:
        sensitivity_results: Per-layer KL scores at each candidate bit-width
        target_bpw: Target average bits per weight
        candidate_bits: Available bit-widths
        model_config: Model architecture config (for layer dimensions)
        latency_weight: Weight for latency penalty in efficiency score.
            0 = pure quality (same as optimize_mixed_precision)
            Higher values = more aggressive latency optimization
        hw_profile: Hardware profile for latency estimation
    """
    from .latency import (
        estimate_layer_latency, get_layer_dimensions, detect_hardware,
    )

    if not sensitivity_results:
        return OptimizationResult(
            configs=[], target_bpw=target_bpw, achieved_bpw=0,
            threshold=0, n_high_bits=0, n_low_bits=0,
            total_params=0, estimated_size_mb=0,
        )

    if hw_profile is None:
        hw_profile = detect_hardware()

    if model_config is None:
        model_config = {
            "hidden_size": 1024, "num_attention_heads": 16,
            "num_key_value_heads": 8, "intermediate_size": 3072,
            "vocab_size": 151936,
        }

    # Infer candidate bits
    if candidate_bits is None:
        all_bits = set()
        for r in sensitivity_results:
            all_bits.update(r.sensitivities.keys())
        candidate_bits = sorted(all_bits)
    if not candidate_bits:
        candidate_bits = [4, 8]

    candidate_bits = sorted(candidate_bits)
    min_bits = candidate_bits[0]
    max_bits = candidate_bits[-1]

    # Identify protected layers
    protected_layers = set()
    protected_layers.update(_identify_output_layers(
        [r.layer_name for r in sensitivity_results]
    ))
    protected_layers.update(_identify_moe_protected_layers(
        [r.layer_name for r in sensitivity_results]
    ))
    if protect_first_last:
        protected_layers.update(_identify_boundary_layers(
            [r.layer_name for r in sensitivity_results], n_protect
        ))

    total_params = sum(r.param_count for r in sensitivity_results)

    # Precompute latency for each layer at each bit-width
    layer_latency = {}  # (layer_name, bits) -> decode_us
    for r in sensitivity_results:
        try:
            in_f, out_f = get_layer_dimensions(r.layer_name, model_config)
        except Exception:
            import math
            side = int(math.sqrt(r.param_count))
            in_f, out_f = side, side
        for bits in candidate_bits:
            lat = estimate_layer_latency(
                r.layer_name, r.param_count, in_f, out_f, bits,
                group_size=group_size, hw=hw_profile,
            )
            layer_latency[(r.layer_name, bits)] = lat.decode_us

    # Handle NaN sensitivities
    import math
    cleaned_results = []
    for r in sensitivity_results:
        has_nan = any(isinstance(v, float) and math.isnan(v) for v in r.sensitivities.values())
        if has_nan:
            max_finite = max((v for v in r.sensitivities.values()
                             if isinstance(v, (int, float)) and not math.isnan(v)), default=1e6)
            fixed_sens = {}
            for b, v in r.sensitivities.items():
                if isinstance(v, float) and math.isnan(v):
                    fixed_sens[b] = max_finite * 10 * (max_bits / max(b, 1))
                else:
                    fixed_sens[b] = v
            cleaned_results.append(SensitivityResult(
                layer_name=r.layer_name, sensitivities=fixed_sens, param_count=r.param_count,
            ))
        else:
            cleaned_results.append(r)
    sensitivity_results = cleaned_results

    # Build effective results with protection bonus
    effective_results = {}
    for r in sensitivity_results:
        if r.layer_name in protected_layers:
            effective_results[r.layer_name] = SensitivityResult(
                layer_name=r.layer_name,
                sensitivities={b: v * 100.0 for b, v in r.sensitivities.items()},
                param_count=r.param_count,
            )
        else:
            effective_results[r.layer_name] = r

    # Initialize all layers at minimum bits
    allocation = {r.layer_name: min_bits for r in sensitivity_results}

    alpha = latency_weight

    # Build upgrade heap with latency-aware efficiency
    upgrade_heap = []
    for r in sensitivity_results:
        current = allocation[r.layer_name]
        _push_latency_aware_upgrade(
            upgrade_heap, effective_results[r.layer_name], current,
            candidate_bits, layer_latency, alpha,
        )

    # Policy: target_bpw is a FLOOR — keep upgrading the highest-efficiency
    # layers (KL reduction per latency cost) until the running BPW meets
    # or exceeds the target. Mirrors the standard knapsack policy above.
    while upgrade_heap and _compute_allocation_bpw(allocation, sensitivity_results) < target_bpw:
        neg_eff, layer_name, old_bits, new_bits, kl_red, lat_inc = heapq.heappop(upgrade_heap)

        if allocation[layer_name] != old_bits:
            continue

        allocation[layer_name] = new_bits

        _push_latency_aware_upgrade(
            upgrade_heap, effective_results[layer_name], new_bits,
            candidate_bits, layer_latency, alpha,
        )

    # Build configs
    configs = []
    for r in sensitivity_results:
        configs.append(LayerQuantConfig(
            layer_name=r.layer_name,
            bits=allocation[r.layer_name],
            group_size=group_size,
            param_count=r.param_count,
        ))

    achieved_bpw = compute_bpw(configs)
    n_max = sum(1 for c in configs if c.bits == max_bits)
    n_min = sum(1 for c in configs if c.bits == min_bits)
    est_size = sum(c.bits * c.param_count for c in configs) / 8
    est_size_mb = est_size / (1024 * 1024)

    # Compute estimated latency
    total_decode_us = sum(
        layer_latency.get((c.layer_name, c.bits), 0) for c in configs
    )

    # Print allocation summary
    bit_counts = {}
    for c in configs:
        bit_counts[c.bits] = bit_counts.get(c.bits, 0) + 1
    summary = ", ".join(f"{count} @ {bits}-bit" for bits, count in sorted(bit_counts.items()))
    print(f"  Latency-aware optimization (alpha={alpha:.2f}): {summary}")
    print(f"  Target BPW: {target_bpw:.2f}, Achieved BPW: {achieved_bpw:.2f}")
    print(f"  Estimated size: {est_size_mb:.1f} MB, decode latency: {total_decode_us:.0f} us")
    print(f"  Hardware: {hw_profile.name} ({hw_profile.memory_bandwidth_gbps:.0f} GB/s)")

    return OptimizationResult(
        configs=configs,
        target_bpw=target_bpw,
        achieved_bpw=achieved_bpw,
        threshold=0.0,
        n_high_bits=n_max,
        n_low_bits=n_min,
        total_params=total_params,
        estimated_size_mb=est_size_mb,
        estimated_decode_us=total_decode_us,
        latency_weight=alpha,
    )


def optimize_for_latency_budget(
    sensitivity_results: list[SensitivityResult],
    target_decode_tok_per_sec: float,
    candidate_bits: list[int] | None = None,
    group_size: int = 64,
    protect_first_last: bool = True,
    n_protect: int = 1,
    model_config: dict | None = None,
    hw_profile: "HardwareProfile | None" = None,
) -> OptimizationResult:
    """Find optimal bit allocation that meets a throughput target.

    Given a desired decode throughput (tok/s), finds the maximum-quality
    allocation that runs at least that fast. This answers the practical
    question: "What's the best quality I can get at 500 tok/s?"

    Uses binary search over BPW targets with the standard optimizer,
    then computes estimated latency for each allocation.

    Args:
        target_decode_tok_per_sec: Minimum required decode throughput
        Other args: same as optimize_mixed_precision
    """
    from .latency import (
        estimate_layer_latency, get_layer_dimensions, detect_hardware,
    )

    if hw_profile is None:
        hw_profile = detect_hardware()

    if model_config is None:
        model_config = {
            "hidden_size": 1024, "num_attention_heads": 16,
            "num_key_value_heads": 8, "intermediate_size": 3072,
            "vocab_size": 151936,
        }

    if candidate_bits is None:
        all_bits = set()
        for r in sensitivity_results:
            all_bits.update(r.sensitivities.keys())
        candidate_bits = sorted(all_bits)
    if not candidate_bits:
        candidate_bits = [4, 8]

    min_bits = min(candidate_bits)
    max_bits = max(candidate_bits)

    # Target latency in microseconds
    target_us = 1e6 / target_decode_tok_per_sec

    def _estimate_latency(opt_result):
        total_us = 0.0
        for c in opt_result.configs:
            try:
                in_f, out_f = get_layer_dimensions(c.layer_name, model_config)
            except Exception:
                import math
                side = int(math.sqrt(c.param_count))
                in_f, out_f = side, side
            lat = estimate_layer_latency(
                c.layer_name, c.param_count, in_f, out_f,
                c.bits, group_size=group_size, hw=hw_profile,
            )
            total_us += lat.decode_us
        return total_us

    # Binary search: find highest BPW that meets the latency target
    bpw_lo = float(min_bits)
    bpw_hi = float(max_bits)

    print(f"  Finding optimal BPW for {target_decode_tok_per_sec:.0f} tok/s "
          f"({target_us:.0f} us per token)...")
    print(f"  Hardware: {hw_profile.name} ({hw_profile.memory_bandwidth_gbps:.0f} GB/s)")

    best_result = None
    best_latency = float("inf")

    # Search with ~0.1 BPW resolution
    for _ in range(15):
        bpw_mid = (bpw_lo + bpw_hi) / 2

        opt = optimize_mixed_precision(
            sensitivity_results,
            target_bpw=bpw_mid,
            candidate_bits=candidate_bits,
            group_size=group_size,
            protect_first_last=protect_first_last,
            n_protect=n_protect,
        )

        latency_us = _estimate_latency(opt)
        opt.estimated_decode_us = latency_us
        tok_per_sec = 1e6 / latency_us if latency_us > 0 else float("inf")

        if latency_us <= target_us:
            # This BPW is fast enough; try higher BPW for better quality
            bpw_lo = bpw_mid
            if best_result is None or opt.achieved_bpw > best_result.achieved_bpw:
                best_result = opt
                best_latency = latency_us
        else:
            # Too slow; try lower BPW
            bpw_hi = bpw_mid

        if bpw_hi - bpw_lo < 0.05:
            break

    if best_result is None:
        # Even minimum BPW is too slow; return minimum
        best_result = optimize_mixed_precision(
            sensitivity_results,
            target_bpw=float(min_bits),
            candidate_bits=candidate_bits,
            group_size=group_size,
            protect_first_last=protect_first_last,
            n_protect=n_protect,
        )
        best_latency = _estimate_latency(best_result)
        best_result.estimated_decode_us = best_latency

    tok_per_sec = 1e6 / best_latency if best_latency > 0 else 0
    print(f"  Result: BPW {best_result.achieved_bpw:.2f}, "
          f"{tok_per_sec:.0f} tok/s, {best_result.estimated_size_mb:.1f} MB")

    return best_result


def _push_latency_aware_upgrade(
    heap, r: SensitivityResult, current_bits: int,
    candidate_bits: list[int], layer_latency: dict,
    alpha: float,
):
    """Push upgrade with latency-penalized efficiency.

    Uses quality-per-total-cost where total cost blends bit budget and
    latency budget:

        efficiency = kl_reduction / ((1-alpha) * bit_cost + alpha * lat_cost_normalized)

    The normalization ensures bit_cost and lat_cost are on comparable scales
    by converting latency to "equivalent bits" using the layer's own ratio.

    At alpha=0: same as quality-only (kl_reduction / bit_cost)
    At alpha>0: penalizes upgrades with high dequant overhead (e.g. 3-bit)
                and layers with disproportionate latency vs their param count
    """
    idx = candidate_bits.index(current_bits) if current_bits in candidate_bits else -1
    if idx < 0 or idx >= len(candidate_bits) - 1:
        return

    next_bits = candidate_bits[idx + 1]
    kl_reduction = _kl_reduction(r, current_bits, next_bits)

    if kl_reduction <= 0:
        return

    # Latency increase from this upgrade (microseconds)
    lat_current = layer_latency.get((r.layer_name, current_bits), 0)
    lat_next = layer_latency.get((r.layer_name, next_bits), 0)
    lat_increase = max(lat_next - lat_current, 0)

    bit_cost = (next_bits - current_bits) * r.param_count

    # Normalize latency to same scale as bit_cost
    # For pure bandwidth-bound: lat_increase ∝ bit_cost (no effect)
    # For dequant-overhead-sensitive: lat_increase has non-proportional component
    # Normalization: convert us to "bit-equivalents" using ratio at the layer level
    if lat_increase > 0 and bit_cost > 0:
        lat_normalized = lat_increase * (bit_cost / max(lat_increase, 1e-10))
        # This equals bit_cost when lat ∝ bits (no change in ranking)
        # but differs when dequant overhead creates non-proportionality
        effective_cost = (1 - alpha) * bit_cost + alpha * lat_normalized
    else:
        effective_cost = bit_cost

    efficiency = kl_reduction / max(effective_cost, 1)

    heapq.heappush(heap, (-efficiency, r.layer_name, current_bits, next_bits, kl_reduction, lat_increase))


def _kl_reduction(r: SensitivityResult, current_bits: int, next_bits: int) -> float:
    """Quality benefit of upgrading from ``current_bits`` to ``next_bits``.

    Two reference-mode semantics, autodetected:

    * **bf16 reference** — ``sensitivities[b]`` is KL(bf16 ∥ b-bit-quant).
      Upgrading reduces KL: benefit = ``kl_current - kl_next``.

    * **uniform-N reference** (e.g. uniform_4bit baseline used for big
      models that don't fit bf16 in RAM) — ``sensitivities[lowest_bit]``
      is identically 0 (probing 4-bit against a 4-bit reference is a
      no-op). ``sensitivities[higher_b]`` measures *how much* upgrading
      moves the layer's output back toward bf16, i.e. the benefit
      directly. Without this distinction the greedy heap stays empty
      and the optimizer leaves every layer at the minimum bit-width.
    """
    kl_current = r.sensitivities.get(current_bits, 0)
    kl_next = r.sensitivities.get(next_bits, 0)
    # Detect uniform-reference mode: under bf16 reference, kl_next < kl_current
    # because higher bits = closer to bf16 = smaller perturbation. Under
    # uniform-N reference (where the running model itself is N-bit), kl at
    # the matching bit-width is ~0 (often a tiny float-roundoff like 5e-18,
    # not exactly zero) and kl_next > 0 because going to higher bits *moves*
    # the layer toward bf16. The sign of (kl_next - kl_current) flips
    # between the two modes; treat ``kl_next > kl_current`` as the signal.
    if next_bits > current_bits and kl_next > kl_current:
        return kl_next - kl_current
    return kl_current - kl_next


def _push_next_upgrade(heap, r: SensitivityResult, current_bits: int, candidate_bits: list[int]):
    """Push the next upgrade step for a layer onto the heap."""
    idx = candidate_bits.index(current_bits) if current_bits in candidate_bits else -1
    if idx < 0 or idx >= len(candidate_bits) - 1:
        return  # Already at max

    next_bits = candidate_bits[idx + 1]
    kl_reduction = _kl_reduction(r, current_bits, next_bits)

    if kl_reduction <= 0:
        return  # No benefit from upgrading

    bit_cost = (next_bits - current_bits) * r.param_count
    efficiency = kl_reduction / max(bit_cost, 1)

    # Max-heap via negative efficiency
    heapq.heappush(heap, (-efficiency, r.layer_name, current_bits, next_bits, kl_reduction))


def _compute_allocation_bpw(allocation: dict, results: list[SensitivityResult]) -> float:
    """Compute BPW from a layer->bits allocation dict."""
    total_bits = sum(allocation[r.layer_name] * r.param_count for r in results)
    total_params = sum(r.param_count for r in results)
    return total_bits / max(total_params, 1)


def _get_param_count(results: list[SensitivityResult], name: str) -> int:
    for r in results:
        if r.layer_name == name:
            return r.param_count
    return 0


def _get_result(results: list[SensitivityResult], name: str) -> SensitivityResult:
    for r in results:
        if r.layer_name == name:
            return r
    raise KeyError(f"Layer {name} not found")


def make_quant_predicate(
    opt_result: OptimizationResult,
    default_bits: int | None = None,
) -> Callable[[str, object], Union[bool, dict]]:
    """Create an mlx-lm compatible quant_predicate from optimization result.

    The predicate returns a dict with {"bits": b, "group_size": g} for each
    layer. For unmatched layers (e.g. embed_tokens which isn't in the
    sensitivity analysis), uses the max bit-width from the allocation to
    preserve quality of critical layers not covered by sensitivity analysis.

    Args:
        opt_result: Optimization result with per-layer configs.
        default_bits: Default bits for unmatched layers. If None, uses the
            max bit-width from the allocation (safest for quality).
    """
    config_map = {c.layer_name: c for c in opt_result.configs}

    # Default to max bits for unmatched layers (e.g. embed_tokens).
    # These layers are typically critical (embeddings, tied weights) and
    # should get the best quality available.
    if default_bits is None:
        default_bits = max((c.bits for c in opt_result.configs), default=4)

    # Get the most common group_size
    from collections import Counter
    gs_counts = Counter(c.group_size for c in opt_result.configs)
    default_gs = gs_counts.most_common(1)[0][0] if gs_counts else 64

    def _emit(c) -> Union[bool, dict]:
        # 16 bits is not a quantized format — MLX has none. It is the LOSSLESS
        # tier: returning False tells mlx-lm to skip the module, so the weight
        # stays at the source dtype (bf16).
        if c.bits >= KEEP_BF16_BITS:
            return False
        return {"bits": c.bits, "group_size": c.group_size}

    def predicate(path: str, module: object) -> Union[bool, dict]:
        # Try exact match first
        if path in config_map:
            return _emit(config_map[path])

        # Try suffix match (mlx-lm paths may differ from PyTorch paths)
        for name, c in config_map.items():
            # Strip common prefixes like "model." for matching
            clean_path = path.replace("model.", "")
            clean_name = name.replace("model.", "")
            if clean_path == clean_name or path.endswith(name) or name.endswith(path):
                return _emit(c)

        # Unmatched layers (e.g. embed_tokens, lm_head when tied) get
        # the most common bit-width from the allocation
        if default_bits >= KEEP_BF16_BITS:
            return False
        return {"bits": default_bits, "group_size": default_gs}

    return predicate


def _identify_output_layers(layer_names: list[str]) -> set[str]:
    """Identify output projection layers that should always get high bits.

    These layers directly produce model output and are highly sensitive
    to quantization error. Includes lm_head, embed_tokens, output projections.
    """
    protected = set()
    output_keywords = ["lm_head", "embed_tokens", "wte", "wpe", "output_proj",
                       "embed_out", "head"]
    for name in layer_names:
        name_lower = name.lower()
        for kw in output_keywords:
            if kw in name_lower.split(".")[-1] or kw in name_lower.split(".")[-2:]:
                protected.add(name)
                break
        # Also protect if the layer is NOT inside a "layers.N" block
        parts = name.split(".")
        if "layers" not in parts:
            protected.add(name)
    return protected


def _identify_moe_protected_layers(layer_names: list[str]) -> set[str]:
    """Identify MoE router projections and shared-expert MLPs to protect.

    Two name patterns end up here:

    - **Router projections** — small linears that produce per-token
      routing scores over the expert pool. Quantization noise here
      directly perturbs the routing distribution, which can change which
      experts a token visits and produces a much larger downstream
      error than the params on the projection itself would suggest.
      Patterns: ``*.mlp.gate`` (Qwen3 MoE, where the SwiGLU gate is named
      ``gate_proj``; bare ``gate`` is the router), ``*.router``,
      ``*.router.proj`` (Gemma-4 MoE).

    - **Shared experts** — MoE blocks frequently include a small dense
      sub-MLP that fires for EVERY token in addition to the top-k routed
      experts. By construction these are sensitive (one per-token forward
      pass each); pure per-layer KL already lands them at high bits on
      every shipped quant we have, but we encode the rule so it doesn't
      depend on the calibration mix or budget happening to land that way.
      Pattern: any layer containing ``.shared_experts.`` as a segment.

    Both rules are name-based; they catch the architectures we ship for
    (Qwen3.5/3.6 MoE, Gemma-4 MoE) without needing model_config injected.
    The KL knapsack would protect these layers anyway on the shipped
    artifacts — this just makes the behavior guaranteed by design instead
    of emergent.
    """
    protected = set()
    for name in layer_names:
        parts = name.split(".")
        if not parts:
            continue
        last = parts[-1]
        prev = parts[-2] if len(parts) >= 2 else ""

        # Router patterns. `mlp.gate` is Qwen3's MoE router (distinct from
        # `mlp.gate_proj`, the SwiGLU activation gate inside each expert).
        # `router` / `router.proj` covers the Gemma-style naming.
        if (last == "gate" and prev == "mlp") or \
           last == "router" or \
           (last == "proj" and prev == "router"):
            protected.add(name)
            continue

        # Shared-expert MLPs (any segment).
        if ".shared_experts." in name or ".shared_expert." in name:
            protected.add(name)
            continue

    return protected


def _identify_boundary_layers(layer_names: list[str], n_protect: int) -> set[str]:
    """Identify first and last transformer block layers to protect."""
    layer_indices = {}
    for name in layer_names:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    idx = int(parts[i + 1])
                    layer_indices.setdefault(idx, []).append(name)
                except ValueError:
                    pass

    if not layer_indices:
        return set()

    sorted_indices = sorted(layer_indices.keys())
    protected = set()

    for idx in sorted_indices[:n_protect]:
        protected.update(layer_indices[idx])
    for idx in sorted_indices[-n_protect:]:
        protected.update(layer_indices[idx])

    return protected
