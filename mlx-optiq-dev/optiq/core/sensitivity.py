"""Per-layer quantization sensitivity analysis for MLX models.

OptiQ uses a single sensitivity method — calibration-driven exact analysis.
For each (layer, bit-width) pair, simulate quantizing just that layer, run a
forward pass on calibration data, and measure KL divergence of the output
logits against an unquantized reference. Cost: ``n_layers × n_bits ×
n_samples`` forward passes.

Two physical execution paths, picked by the caller via ``bf16_source_dir``:

  * **bf16 reference** (preferred when bf16 fits in RAM): the supplied
    ``model`` is the bf16 base, fully resident. Each layer's weight is
    swapped in-place between bf16 and a simulate-quantized copy. Reference
    logits come from the unmodified bf16 model.

  * **uniform-4-bit reference + bf16 streaming** (fallback for big models):
    the supplied ``model`` is a uniform-4-bit MLX baseline of the same
    architecture. ``bf16_source_dir`` is a local snapshot of the bf16
    source. For each layer, the bf16 weight is mmap-streamed from disk
    and used to replace the corresponding 4-bit layer with a freshly
    quantized version at each candidate bit-width. Reference logits come
    from the unmodified 4-bit baseline. The signal is "marginal benefit
    of bumping this layer above 4-bit" — slightly weaker than the bf16
    reference but still calibration-driven.

Both paths return ``list[SensitivityResult]`` — drop-in for
``optimize_mixed_precision``.

Public entry point:

    from optiq.core.sensitivity import analyze_sensitivity_exact
    # bf16 in-RAM:
    results = analyze_sensitivity_exact(model, calibration_fn)
    # uniform-4-bit baseline + bf16 streamed from disk:
    results = analyze_sensitivity_exact(
        baseline_4bit_model, calibration_fn,
        bf16_source_dir="/path/to/bf16/snapshot",
    )

The bf16 source is also expected for MoE bases (Gemma-4-MoE, Qwen3.5/3.6-MoE)
so the streamer can apply the same expert-name sanitization mlx-lm's loader
applies (``*.experts.gate_up_proj`` → ``*.{switch_glu|switch_mlp}.{gate,up}_proj``).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as mnn


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


@dataclass
class SensitivityResult:
    """Per-layer quantization sensitivity, one entry per Linear-like leaf."""

    layer_name: str
    sensitivities: dict[int, float]
    param_count: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "layer_name": self.layer_name,
            "sensitivities": {str(k): v for k, v in self.sensitivities.items()},
            "param_count": self.param_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SensitivityResult":
        return cls(
            layer_name=d["layer_name"],
            sensitivities={int(k): float(v) for k, v in d["sensitivities"].items()},
            param_count=int(d["param_count"]),
            metadata=d.get("metadata", {}),
        )


# --------------------------------------------------------------------------
# Quantization-error / KL primitives
# --------------------------------------------------------------------------


#: A candidate bit-width of 16 means "do not quantize this layer" — keep it at the
#: source dtype (bf16). MLX has no 16-bit quantized format (``mx.quantize`` accepts
#: 2-8), so this is not a finer grid, it is the *lossless* tier: the layer is left
#: alone and ``make_quant_predicate`` emits ``False`` for it so mlx-lm skips the
#: module entirely. Useful in the opposite regime from the 2-bit MoE quants — a
#: small model has no redundancy to spend, so the knapsack should be able to buy
#: back exactness on the layers that need it.
KEEP_BF16_BITS = 16


def _simulate_quantize(
    w: mx.array, bits: int, group_size: int = 64
) -> mx.array:
    """Quantize then dequantize ``w`` under MLX's group-quantization scheme.
    Result is in ``w``'s original dtype."""
    if bits >= KEEP_BF16_BITS:
        return w  # the lossless tier: no round-trip, so no error to measure
    w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    return mx.dequantize(
        w_q, scales, biases, group_size=group_size, bits=bits
    ).astype(w.dtype)


def _quantizable_linears(
    model, group_size: int
) -> list[tuple[str, mnn.Module]]:
    """List ``(name, module)`` for every BF16 Linear whose weight is compatible
    with group quantization. Skips already-quantized modules."""
    out: list[tuple[str, mnn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, mnn.Linear):
            continue
        if isinstance(module, mnn.QuantizedLinear):
            continue
        w = module.weight
        if w.ndim == 2 and w.shape[-1] % group_size == 0:
            out.append((name, module))
    return out


def _quantized_layers(model) -> list[tuple[str, mnn.Module]]:
    """List ``(path, module)`` for every QuantizedLinear / QuantizedSwitchLinear
    in a uniform-4-bit baseline model. Used by the bf16-streaming path."""
    out: list[tuple[str, mnn.Module]] = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name in ("QuantizedLinear", "QuantizedSwitchLinear"):
            out.append((name, module))
    return out


def _extract_logits(out) -> mx.array:
    """Accept either a raw logits array or an object with ``.logits``."""
    return out.logits if hasattr(out, "logits") else out


def _kl_from_ref(cur_logits: mx.array, ref_logits: mx.array) -> mx.array:
    """KL(reference || current) averaged across batch/seq."""
    cur_logits = cur_logits.astype(mx.float32)
    ref_logits = ref_logits.astype(mx.float32)
    log_cur = cur_logits - mx.logsumexp(cur_logits, axis=-1, keepdims=True)
    log_ref = ref_logits - mx.logsumexp(ref_logits, axis=-1, keepdims=True)
    ref_probs = mx.softmax(ref_logits, axis=-1)
    return mx.mean(mx.sum(ref_probs * (log_ref - log_cur), axis=-1))


# --------------------------------------------------------------------------
# Safetensors mmap streaming utilities (used by the uniform-4bit path)
# --------------------------------------------------------------------------


def _dtype_from_str(dtype_str: str):
    m = {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}
    if dtype_str not in m:
        raise ValueError(f"unsupported weight dtype from safetensors: {dtype_str}")
    return m[dtype_str]


def _safe_open_to_mx(safe_file, tensor_key: str, dtype_str: str) -> mx.array:
    """Fetch one tensor from a ``safe_open(framework='pt')`` handle, convert
    to ``mx.array`` preserving bfloat16."""
    t = safe_file.get_tensor(tensor_key)
    target_dtype = _dtype_from_str(dtype_str)
    if dtype_str == "BF16":
        return mx.array(t.float().numpy()).astype(target_dtype)
    return mx.array(t.numpy()).astype(target_dtype)


def _index_bf16_layers(model_path: str, group_size: int) -> dict[str, dict]:
    """Walk the bf16 safetensors at ``model_path`` and return a mapping
    ``mlx_lm_path -> {shard, source_key, slice_spec, shape, dtype}`` so the
    streamer can fetch a layer's weight on demand.

    Indexed under both the on-disk key (with ``.weight`` stripped) AND
    common mlx-lm-side rewrites:
      * leading ``model.`` stripped (mlx-lm sometimes drops it)
      * ``model.language_model.`` ↔ ``language_model.model.`` reordering
        (mlx-lm's VLM-text loaders)

    For MoE bases the gate_up_proj is split per the architecture's sanitize:
      * Gemma-4 MoE keeps ``experts`` (``experts.switch_glu.{gate,up,down}_proj``)
      * Qwen3.5/3.6 MoE drops ``experts`` (``mlp.switch_mlp.{gate,up,down}_proj``)
    """
    import glob
    from safetensors import safe_open

    shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors under {model_path}")

    wrapper = "switch_glu"
    drop_experts_segment = False
    fused_experts_direct = False
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as _f:
                _cfg = json.load(_f)
            arch = (_cfg.get("architectures") or [""])[0].lower()
            mt = str(_cfg.get("model_type", "")).lower()
            if "qwen3" in mt or "qwen3" in arch:
                wrapper = "switch_mlp"
                drop_experts_segment = True
            elif "diffusion_gemma" in mt or "diffusion_gemma" in arch:
                # DiffusionGemma keeps its routed experts FUSED: the running
                # QuantizedSwitchLinear modules are named ``experts.gate_up_proj``
                # / ``experts.down_proj`` (no switch_glu split, unlike mlx-lm's
                # dense Gemma-4 MoE). Index the bf16 fused 3D tensors directly
                # under those names so the streamer matches the model module.
                fused_experts_direct = True
            elif "laguna" in mt or "laguna" in arch:
                # poolside Laguna ships experts UNFUSED — 256 separate per-expert
                # 2D tensors (``mlp.experts.{e}.{gate,up,down}_proj.weight``) — while
                # the running model's SwitchGLU is a single 3D
                # ``mlp.switch_mlp.{gate,up,down}_proj``. Stacked in the post-pass below.
                wrapper = "switch_mlp"
        except Exception:
            pass

    def _moe_base(key: str, suffix: str) -> str:
        b = key.removesuffix(suffix)
        if drop_experts_segment and b.endswith(".experts"):
            b = b.removesuffix(".experts")
        return b

    def _add(idx: dict[str, dict], name: str, entry: dict) -> None:
        idx[name] = entry
        if name.startswith("model."):
            idx[name[len("model."):]] = entry
        if "model.language_model." in name:
            idx[name.replace(
                "model.language_model.", "language_model.model.", 1)] = entry

    import re as _re
    # Unfused per-expert bf16 tensors (poolside Laguna): collected here across all
    # shards, then stacked into ``switch_mlp.{proj}`` entries after the walk.
    _expert_re = _re.compile(
        r"^(.*\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    unfused: dict[tuple, list] = {}

    index: dict[str, dict] = {}
    for shard in shards:
        # framework="numpy" reads tensor METADATA (shape/dtype/keys) for every
        # dtype incl. bf16 without materializing — and needs no torch, so the
        # streaming path works in an mlx-only env (e.g. the DiffusionGemma venv).
        with safe_open(shard, framework="numpy") as f:
            for key in f.keys():
                info = f.get_slice(key)
                shape = tuple(info.get_shape())
                dtype_str = info.get_dtype()
                if dtype_str not in ("BF16", "F16", "F32"):
                    continue

                m_exp = _expert_re.match(key)
                if m_exp is not None:
                    # Collect an unfused per-expert 2D tensor; skip the normal
                    # per-expert indexing (the running model has no such module).
                    unfused.setdefault((m_exp.group(1), m_exp.group(3)), []).append(
                        (int(m_exp.group(2)), shard, key, shape, dtype_str)
                    )
                    continue

                if key.endswith(".weight") and len(shape) == 2 and shape[-1] % group_size == 0:
                    name = key[: -len(".weight")]
                    entry = {
                        "shard": shard,
                        "source_key": key,
                        "slice_spec": None,
                        "shape": shape,
                        "dtype": dtype_str,
                    }
                    _add(index, name, entry)

                if len(shape) == 3 and shape[-1] % group_size == 0:
                    if fused_experts_direct and (
                        key.endswith(".experts.gate_up_proj")
                        or key.endswith(".experts.down_proj")
                        or key.endswith(".experts.gate_up_proj.weight")
                        or key.endswith(".experts.down_proj.weight")
                    ):
                        # DiffusionGemma: fused expert tensor maps 1:1 to its
                        # QuantizedSwitchLinear module. Strip a trailing
                        # ``.weight`` if present; no split, no rename.
                        nm = key[: -len(".weight")] if key.endswith(".weight") else key
                        _add(index, nm, {
                            "shard": shard,
                            "source_key": key,
                            "slice_spec": None,
                            "shape": shape,
                            "dtype": dtype_str,
                        })
                    elif key.endswith(".experts.gate_up_proj"):
                        mid = shape[-2] // 2
                        if shape[-2] != 2 * mid:
                            continue
                        half_shape = (shape[0], mid, shape[-1])
                        base = _moe_base(key, ".gate_up_proj")
                        for nm, sl in (
                            (f"{base}.{wrapper}.gate_proj",
                             (slice(None), slice(0, mid), slice(None))),
                            (f"{base}.{wrapper}.up_proj",
                             (slice(None), slice(mid, 2 * mid), slice(None))),
                        ):
                            _add(index, nm, {
                                "shard": shard,
                                "source_key": key,
                                "slice_spec": sl,
                                "shape": half_shape,
                                "dtype": dtype_str,
                            })
                    elif key.endswith(".experts.down_proj"):
                        base = _moe_base(key, ".down_proj")
                        nm = f"{base}.{wrapper}.down_proj"
                        _add(index, nm, {
                            "shard": shard,
                            "source_key": key,
                            "slice_spec": None,
                            "shape": shape,
                            "dtype": dtype_str,
                        })
                    elif key.endswith(".switch_mlp.fc1.weight") or \
                            key.endswith(".switch_mlp.fc2.weight"):
                        # Already-fused mlx-lm-format routed experts (3D:
                        # [n_experts, out, in]). NemotronH's mlx-community
                        # bf16 ships its MoE this way — a plain fc1→fc2 MLP
                        # per expert (no gate/up split), with the running
                        # QuantizedSwitchLinear named identically. Index the
                        # tensor directly under its own name (drop the
                        # trailing ``.weight``); no slicing or rename needed,
                        # unlike the HF ``.experts.gate_up_proj`` path above.
                        nm = key[: -len(".weight")]
                        _add(index, nm, {
                            "shard": shard,
                            "source_key": key,
                            "slice_spec": None,
                            "shape": shape,
                            "dtype": dtype_str,
                        })

    # Stack unfused per-expert tensors into the ``switch_mlp.{proj}`` names the
    # running model uses. Each switch_mlp weight is [n_experts, out, in]; the
    # loader gathers the per-expert 2D sources (possibly across shards) and stacks.
    for (base, proj), items in unfused.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda t: t[0])  # by expert index
        out_d, in_d = items[0][3]
        if in_d % group_size != 0:
            continue
        _add(index, f"{base}.{wrapper}.{proj}", {
            "shard": None,
            "source_key": None,
            "slice_spec": None,
            "stack_sources": [(sh, k) for (_, sh, k, _, _) in items],
            "shape": (len(items), out_d, in_d),
            "dtype": items[0][4],
        })
    return index


def _load_bf16_weight(index_entry: dict) -> mx.array:
    """Load (and slice, if needed) a bf16 weight given an index entry.

    Uses mlx-native ``mx.load`` (lazy mmap) rather than a torch-backed
    ``safe_open``: mlx reads bfloat16 directly, so the streaming sensitivity
    path needs no torch and runs in an mlx-only environment. ``mx.load`` mmaps
    the shard and only the indexed tensor (+ optional slice) is materialized on
    ``eval``; the rest of the lazy dict is dropped.
    """
    stack_sources = index_entry.get("stack_sources")
    if stack_sources:
        # Gather unfused per-expert 2D tensors and stack to [n_experts, out, in].
        # Group by shard so each shard is mmapped once, not once per expert.
        parts: list = [None] * len(stack_sources)
        by_shard: dict[str, list] = {}
        for i, (shard, key) in enumerate(stack_sources):
            by_shard.setdefault(shard, []).append((i, key))
        for shard, items in by_shard.items():
            arrs = mx.load(shard)
            for i, key in items:
                parts[i] = arrs[key]
        w = mx.stack(parts, axis=0)
        mx.eval(w)
        return w

    arrs = mx.load(index_entry["shard"])
    w = arrs[index_entry["source_key"]]
    if index_entry["slice_spec"] is not None:
        w = w[index_entry["slice_spec"]]
    mx.eval(w)
    return w


# --------------------------------------------------------------------------
# In-place mutation of a quantized layer to a different bit-width
# --------------------------------------------------------------------------


def _capture_quantized_layer_state(layer: mnn.Module) -> dict:
    return {
        "weight": layer.weight,
        "scales": layer.scales,
        "biases": getattr(layer, "biases", None),
        "bits": getattr(layer, "bits", None),
        "group_size": getattr(layer, "group_size", None),
    }


def _restore_quantized_layer_state(layer: mnn.Module, state: dict) -> None:
    layer.weight = state["weight"]
    layer.scales = state["scales"]
    if state["biases"] is not None:
        layer.biases = state["biases"]
    if state["bits"] is not None:
        layer.bits = state["bits"]
    if state["group_size"] is not None:
        layer.group_size = state["group_size"]


def _mutate_quantized_layer_to_bits(
    layer: mnn.Module, new_bf16_weight: mx.array, new_bits: int, group_size: int
) -> None:
    """In-place mutation of a QuantizedLinear / QuantizedSwitchLinear to use
    ``new_bf16_weight`` quantized at ``new_bits``. Works because mlx-lm's
    quantized layers compute forward via ``mx.dequantize(self.weight,
    self.scales, self.biases, bits=self.bits, group_size=self.group_size)`` —
    all parameters live on the layer instance."""
    q, scales, *biases_list = mx.quantize(
        new_bf16_weight, group_size=group_size, bits=new_bits, mode="affine"
    )
    layer.weight = q
    layer.scales = scales
    if biases_list:
        layer.biases = biases_list[0]
    layer.bits = new_bits
    layer.group_size = group_size
    mx.eval(layer.weight, layer.scales)


# --------------------------------------------------------------------------
# Internal: bf16 reference path (preferred, used when bf16 fits in RAM)
# --------------------------------------------------------------------------


def _structural_priority(name: str, n_layers: int) -> float:
    """Architecture-only priority in (0, 1] for the ``static`` method: how much
    a layer 'wants' high precision, with no measurement. Encodes the standard
    quantization priors — embedding/output head and the first/last transformer
    block are the most fragile; attention and the MoE router matter more than
    the dense MLP and the (robust, large) routed experts; the network edges
    matter more than the middle."""
    import re
    lname = name.lower()
    if "embed" in lname or "lm_head" in lname:
        return 1.0

    p = 0.30  # dense-MLP / mid-network baseline
    m = re.search(r"layers?\.(\d+)\.", lname)
    if m is not None and n_layers > 1:
        d = int(m.group(1))
        if d == 0 or d == n_layers - 1:
            p = 0.90                       # first / last block: protect
        elif d <= 1 or d >= n_layers - 2:
            p = 0.65                       # near-edge blocks
        else:
            # taper across the middle: edges slightly above the centre
            edge = min(d, n_layers - 1 - d) / max(n_layers / 2, 1)
            p = 0.30 + 0.15 * (1.0 - edge)

    if "self_attn" in lname or ".attn" in lname:
        p += 0.30                          # attention is quant-sensitive
    if "router" in lname or "gate.weight" in lname or ".gate." in lname:
        p += 0.40                          # MoE router: small + decisive
    if "experts" in lname or "switch" in lname:
        p -= 0.10                          # routed experts are robust + huge
    return max(0.05, min(p, 1.0))


def analyze_sensitivity_static(
    model: mnn.Module,
    *,
    candidate_bits: list[int] = (4, 8),
    group_size: int = 64,
) -> list[SensitivityResult]:
    """Rule-based **structural** sensitivity for the ``static`` method — no
    measurement at all. Each layer is scored by architecture alone (see
    :func:`_structural_priority`): embedding/output head and the first/last
    block rank highest, attention and the MoE router above the dense MLP and
    routed experts. The scores are packed as ``sensitivities`` so the same
    allocator that the measured methods use spends the bit budget on the
    highest-priority layers — at the requested ``candidate_bits`` and target
    BPW, not a fixed recipe. Fast (no forward passes, no calibration), but the
    coarsest signal of the three methods."""
    candidate_bits = sorted(candidate_bits)
    lo, hi = candidate_bits[0], candidate_bits[-1]

    quantizable = [
        (name, m) for name, m in model.named_modules()
        if type(m).__name__ not in ("QuantizedLinear", "QuantizedSwitchLinear")
        and getattr(m, "weight", None) is not None
        and hasattr(m.weight, "ndim")
        and (
            (isinstance(m, mnn.Linear) and m.weight.ndim == 2)
            or (type(m).__name__ == "SwitchLinear" and m.weight.ndim == 3)
        )
        and m.weight.shape[-1] % group_size == 0
    ]
    n_layers = 1
    import re
    for name, _ in quantizable:
        m = re.search(r"layers?\.(\d+)\.", name)
        if m:
            n_layers = max(n_layers, int(m.group(1)) + 1)

    print(f"  [static] {len(quantizable)} layers — rule-based structural priority (no measurement)")
    results: list[SensitivityResult] = []
    for name, module in quantizable:
        prio = _structural_priority(name, n_layers)
        # Encode as a sensitivity curve the allocator reads: cost falls from
        # ``prio`` at the lowest bit-width to 0 at the highest, linearly across
        # tiers, so upgrading a high-priority layer buys the most "KL".
        span = max(hi - lo, 1)
        sensitivities = {b: prio * (hi - b) / span for b in candidate_bits}
        param_count = 1
        for s in module.weight.shape:
            param_count *= s
        results.append(SensitivityResult(name, sensitivities, param_count))
    return results


def _exact_with_bf16_reference(
    model: mnn.Module,
    cal_samples: list,
    *,
    candidate_bits: list[int],
    group_size: int,
    completed: dict[str, SensitivityResult],
    checkpoint_path: Optional[str],
) -> list[SensitivityResult]:
    quantizable = _quantizable_linears(model, group_size)
    print(
        f"  [exact/bf16] {len(quantizable)} layers × {len(candidate_bits)} bit-widths"
    )

    print("  [exact/bf16] reference logits…")
    ref_logits_list: list[mx.array] = []
    for args, kwargs in cal_samples:
        ref_logits_list.append(_extract_logits(model(*args, **kwargs)))
    mx.eval(*ref_logits_list)

    results: list[SensitivityResult] = []
    for layer_idx, (layer_name, module) in enumerate(quantizable):
        if layer_name in completed:
            results.append(completed[layer_name])
            continue

        sensitivities: dict[int, float] = {}
        orig_w = module.weight

        for bits in candidate_bits:
            if bits >= KEEP_BF16_BITS:
                # The lossless tier leaves the weight untouched, so KL is 0 by
                # construction. Skip the forwards.
                sensitivities[bits] = 0.0
                continue
            module.weight = _simulate_quantize(orig_w, bits, group_size)
            mx.eval(module.weight)

            kls: list[float] = []
            for i, (args, kwargs) in enumerate(cal_samples):
                cur = _extract_logits(model(*args, **kwargs))
                kls.append(float(_kl_from_ref(cur, ref_logits_list[i]).item()))
            sensitivities[bits] = sum(kls) / max(len(kls), 1)

        module.weight = orig_w
        mx.eval(module.weight)

        param_count = 1
        for s in orig_w.shape:
            param_count *= s

        results.append(SensitivityResult(
            layer_name=layer_name,
            sensitivities=sensitivities,
            param_count=param_count,
        ))

        print(
            f"  [exact/bf16] {layer_idx + 1:3d}/{len(quantizable)} "
            f"{layer_name}  "
            + "  ".join(f"{b}b={s:.3e}" for b, s in sensitivities.items())
        )
        if checkpoint_path:
            _save_checkpoint(results, checkpoint_path)

    return results


# --------------------------------------------------------------------------
# Internal: uniform-4-bit baseline + bf16 streaming (RAM fallback)
# --------------------------------------------------------------------------


def _exact_with_quantized_baseline(
    model: mnn.Module,
    cal_samples: list,
    *,
    bf16_source_dir: str,
    candidate_bits: list[int],
    group_size: int,
    completed: dict[str, SensitivityResult],
    checkpoint_path: Optional[str],
) -> list[SensitivityResult]:
    quantized = _quantized_layers(model)
    print(
        f"  [exact/uniform_4bit] {len(quantized)} layers × "
        f"{len(candidate_bits)} bit-widths (bf16 stream from {bf16_source_dir})"
    )

    bf16_index = _index_bf16_layers(bf16_source_dir, group_size)
    print(f"  [exact/uniform_4bit] bf16 index has {len(bf16_index)} entries")

    print("  [exact/uniform_4bit] reference logits (from uniform-4-bit baseline)…")
    # Store reference logits as fp16 — halves RAM (~16 GB → 8 GB on a
    # 256k-vocab model with 2 calibration samples). KL math casts to fp32
    # internally for numerical stability, so fp16 storage is safe.
    ref_logits_list: list[mx.array] = []
    for args, kwargs in cal_samples:
        ref_logits_list.append(
            _extract_logits(model(*args, **kwargs)).astype(mx.float16)
        )
    mx.eval(*ref_logits_list)
    import gc
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass

    results: list[SensitivityResult] = []
    skipped = 0
    for layer_idx, (layer_name, layer) in enumerate(quantized):
        if layer_name in completed:
            results.append(completed[layer_name])
            continue

        bf16_key = layer_name
        if bf16_key not in bf16_index:
            alt = layer_name[len("model."):] if layer_name.startswith("model.") else layer_name
            if alt in bf16_index:
                bf16_key = alt
            else:
                skipped += 1
                if skipped <= 5:
                    print(
                        f"  [exact/uniform_4bit] [skip] {layer_name}: "
                        f"no bf16 source"
                    )
                continue

        bf16_w = _load_bf16_weight(bf16_index[bf16_key])

        sensitivities: dict[int, float] = {}
        orig_state = _capture_quantized_layer_state(layer)

        # The running model is the uniform-4-bit baseline, and the reference
        # logits were taken from it. Probing the candidate bit that equals the
        # baseline's bit-width re-quantizes the layer to exactly what it already
        # is, so KL(reference || current) is 0 by construction (observed as
        # ~1e-18 float noise). Skip the forward passes for that bit and record
        # 0.0 directly — this roughly halves the sweep for the common
        # candidate_bits=[4, 8] case with no effect on the optimizer, which only
        # needs the *improvement* from upgrading 4 -> 8.
        baseline_bits = getattr(layer, "bits", None)

        for bits in candidate_bits:
            if bits == baseline_bits:
                sensitivities[bits] = 0.0
                continue
            if bits >= KEEP_BF16_BITS:
                # Reaching the lossless tier here would mean swapping a
                # QuantizedLinear for a plain Linear mid-sweep — module surgery,
                # not a parameter mutation. This reference mode exists for models
                # too big to hold in bf16, which is the opposite regime from the
                # one that wants a bf16 tier, so refuse rather than pretend.
                raise ValueError(
                    f"candidate bit-width {bits} (keep-bf16) is not supported with "
                    "--reference uniform_4bit; use --reference bf16 (small models "
                    "fit, and that is where an 8/16 mix makes sense)"
                )
            _mutate_quantized_layer_to_bits(layer, bf16_w, bits, group_size)

            kls: list[float] = []
            for i, (args, kwargs) in enumerate(cal_samples):
                cur = _extract_logits(model(*args, **kwargs))
                kl_val = float(_kl_from_ref(cur, ref_logits_list[i]).item())
                kls.append(kl_val)
                del cur  # release activations before next sample
            sensitivities[bits] = sum(kls) / max(len(kls), 1)

        _restore_quantized_layer_state(layer, orig_state)
        mx.eval(layer.weight, layer.scales)

        param_count = 1
        for s in bf16_w.shape:
            param_count *= int(s)
        del bf16_w
        del orig_state

        results.append(SensitivityResult(
            layer_name=layer_name,
            sensitivities=sensitivities,
            param_count=param_count,
        ))

        # Aggressive memory hygiene per probe — large MoE switch_mlp
        # tensors (~0.5 GB bf16) accumulate in MLX's page cache and
        # macOS Jetsam-kills the process within ~10 layers without this.
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

        print(
            f"  [exact/uniform_4bit] {layer_idx + 1:3d}/{len(quantized)} "
            f"{layer_name}  "
            + "  ".join(f"{b}b={s:.3e}" for b, s in sensitivities.items())
        )
        if checkpoint_path:
            _save_checkpoint(results, checkpoint_path)

    if skipped:
        print(f"  [exact/uniform_4bit] {skipped} layers skipped (no bf16 source match)")
    return results


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def analyze_sensitivity_exact(
    model: mnn.Module,
    calibration_fn: Callable[[], list[tuple[tuple, dict]]],
    *,
    candidate_bits: list[int] = (4, 8),
    group_size: int = 64,
    n_calibration: int = 2,
    checkpoint_path: Optional[str] = None,
    bf16_source_dir: Optional[str] = None,
) -> list[SensitivityResult]:
    """Per-layer KL sensitivity of an MLX model.

    Path is selected by ``bf16_source_dir``:
      * ``None`` (default): ``model`` is bf16, do in-place weight swaps.
      * ``str``: ``model`` is a uniform-4-bit MLX baseline; bf16 weights
        are streamed off disk from this snapshot for layer-by-layer probes.
    """
    t_start = time.time()
    candidate_bits = list(candidate_bits)

    completed: dict[str, SensitivityResult] = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                for d in json.load(f):
                    entry = SensitivityResult.from_dict(d)
                    if set(candidate_bits).issubset(entry.sensitivities):
                        completed[entry.layer_name] = entry
            if completed:
                print(
                    f"  [exact] resuming from {checkpoint_path}: "
                    f"{len(completed)} layers already done"
                )
        except Exception as e:
            print(f"  [exact] checkpoint read failed ({e!r}); ignoring")

    cal_samples = calibration_fn()[:n_calibration]
    print(f"  [exact] calibration: {len(cal_samples)} samples")

    if bf16_source_dir is None:
        results = _exact_with_bf16_reference(
            model, cal_samples,
            candidate_bits=candidate_bits, group_size=group_size,
            completed=completed, checkpoint_path=checkpoint_path,
        )
    else:
        results = _exact_with_quantized_baseline(
            model, cal_samples,
            bf16_source_dir=bf16_source_dir,
            candidate_bits=candidate_bits, group_size=group_size,
            completed=completed, checkpoint_path=checkpoint_path,
        )

    if checkpoint_path:
        _save_checkpoint(results, checkpoint_path)

    print(f"  [exact] done in {time.time() - t_start:.1f}s")
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _save_checkpoint(results: list[SensitivityResult], path: str) -> None:
    """Persist sensitivity results, **merging** with any existing checkpoint so
    it only ever grows.

    A resumed run re-appends already-``completed`` layers to ``results`` only as
    its loop passes them, so a periodic save (or a kill) mid-loop could otherwise
    write fewer entries than the file already held and drop scored layers — the
    next resume would then re-score them. Merging by ``layer_name`` (existing
    file ∪ current results) makes the checkpoint monotonic, and the atomic
    tmp-then-replace write means a crash mid-save can't corrupt it.
    """
    merged: dict[str, dict] = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                for d in json.load(f):
                    merged[d["layer_name"]] = d
        except Exception:
            merged = {}
    for r in results:
        merged[r.layer_name] = r.to_dict()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(list(merged.values()), f, indent=2)
    os.replace(tmp, path)


def print_sensitivity_report(
    results: list[SensitivityResult], top_n: int = 20
) -> None:
    if not results:
        print("  (no layers to report)")
        return

    bits_seen: set[int] = set()
    for r in results:
        bits_seen.update(r.sensitivities.keys())

    for bits in sorted(bits_seen):
        ranked = sorted(
            (r for r in results if bits in r.sensitivities),
            key=lambda r: r.sensitivities[bits],
            reverse=True,
        )
        print(f"\n  Top {top_n} most-sensitive layers at {bits}-bit:")
        for i, r in enumerate(ranked[:top_n]):
            print(
                f"    {i + 1:3d}. {r.layer_name:60s} "
                f" sens={r.sensitivities[bits]:.3e}  "
                f"params={r.param_count:,}"
            )
