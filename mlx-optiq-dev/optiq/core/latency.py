"""Hardware-aware latency model for Apple Silicon MLX inference.

Models per-layer inference cost based on the roofline model:
  - Decode (seq_len=1): memory-bandwidth-bound → latency ∝ bytes read
  - Prefill (seq_len>1): compute-bound → latency ∝ FLOPS / peak throughput

The key insight: on Apple Silicon unified memory, quantized inference is
almost always memory-bound at decode time. Lower bit-widths read fewer bytes
per weight, so they run faster — but the relationship is NOT linear because:
  1. Quantized matmul has per-group scale/bias overhead
  2. Dequantization compute has fixed cost per element
  3. Kernel dispatch has per-layer fixed overhead

This module provides analytical latency estimates that match measured behavior
without requiring noisy micro-benchmarks.
"""

from dataclasses import dataclass


@dataclass
class HardwareProfile:
    """Apple Silicon GPU characteristics."""
    name: str
    memory_bandwidth_gbps: float  # GB/s
    peak_flops_tflops: float      # TFLOPS (FP16/BF16)
    n_gpu_cores: int

    @property
    def memory_bandwidth_bps(self) -> float:
        """Bandwidth in bytes/sec."""
        return self.memory_bandwidth_gbps * 1e9

    @property
    def peak_flops(self) -> float:
        """Peak FLOPS."""
        return self.peak_flops_tflops * 1e12


# Known Apple Silicon profiles
HARDWARE_PROFILES = {
    "m1":      HardwareProfile("M1",          68.25, 2.6,  8),
    "m1_pro":  HardwareProfile("M1 Pro",     200.0,  5.2, 16),
    "m1_max":  HardwareProfile("M1 Max",     400.0, 10.4, 32),
    "m1_ultra":HardwareProfile("M1 Ultra",   800.0, 20.8, 64),
    "m2":      HardwareProfile("M2",         100.0,  3.6, 10),
    "m2_pro":  HardwareProfile("M2 Pro",     200.0,  6.8, 19),
    "m2_max":  HardwareProfile("M2 Max",     400.0, 13.6, 38),
    "m2_ultra":HardwareProfile("M2 Ultra",   800.0, 27.2, 76),
    "m3":      HardwareProfile("M3",         100.0,  3.6, 10),
    "m3_pro":  HardwareProfile("M3 Pro",     150.0,  5.4, 14),
    "m3_max":  HardwareProfile("M3 Max",     400.0, 14.2, 40),
    "m4":      HardwareProfile("M4",         120.0,  4.3, 10),
    "m4_pro":  HardwareProfile("M4 Pro",     273.0,  8.7, 20),
    "m4_max":  HardwareProfile("M4 Max",     546.0, 17.4, 40),
}


def detect_hardware() -> HardwareProfile:
    """Detect current Apple Silicon hardware profile.

    Falls back to conservative M3 Max estimate if detection fails.
    """
    try:
        import mlx.core as mx
        # Use new API, fall back to deprecated one
        try:
            info = mx.device_info()
        except AttributeError:
            info = mx.metal.device_info()
        device_name = info.get("device_name", "").lower()

        # Match from most specific to least specific
        # (e.g. "m3 max" before "m3")
        sorted_profiles = sorted(
            HARDWARE_PROFILES.items(),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )
        for key, profile in sorted_profiles:
            if key.replace("_", " ") in device_name:
                return profile

        # Fallback: estimate from architecture string
        return HardwareProfile(
            name=info.get("device_name", "Unknown Apple Silicon"),
            memory_bandwidth_gbps=200.0,  # conservative
            peak_flops_tflops=7.0,
            n_gpu_cores=16,
        )
    except Exception:
        return HARDWARE_PROFILES["m3_max"]


@dataclass
class LayerLatency:
    """Per-layer latency estimate."""
    layer_name: str
    bits: int
    param_count: int
    group_size: int
    # Weight data sizes
    weight_bytes: float      # quantized weight bytes
    scale_bias_bytes: float  # per-group scales + biases
    total_bytes: float       # total data read for this layer
    # Latency estimates
    decode_us: float         # decode latency (memory-bound)
    prefill_us: float        # prefill latency at reference seq_len
    # Compute stats
    flops: float             # FLOPS for one forward pass
    arithmetic_intensity: float  # FLOPS / bytes


def estimate_layer_bytes(param_count: int, bits: int, group_size: int = 64) -> tuple[float, float, float]:
    """Estimate memory footprint for a quantized layer.

    MLX QuantizedLinear stores:
      - weight: packed uint32, ceil(param_count * bits / 32) * 4 bytes
      - scales: float16, param_count / group_size * 2 bytes
      - biases: float16, param_count / group_size * 2 bytes

    Returns: (weight_bytes, scale_bias_bytes, total_bytes)
    """
    weight_bytes = param_count * bits / 8
    n_groups = param_count / group_size
    scale_bias_bytes = n_groups * 2 * 2  # scales + biases, float16 each
    total_bytes = weight_bytes + scale_bias_bytes
    return weight_bytes, scale_bias_bytes, total_bytes


def estimate_layer_flops(in_features: int, out_features: int, seq_len: int = 1) -> float:
    """Estimate FLOPS for a linear layer forward pass.

    FLOPS = 2 * batch * seq_len * in_features * out_features
    (multiply-add = 2 ops per element)
    """
    return 2.0 * seq_len * in_features * out_features


def estimate_layer_latency(
    layer_name: str,
    param_count: int,
    in_features: int,
    out_features: int,
    bits: int,
    group_size: int = 64,
    seq_len_decode: int = 1,
    seq_len_prefill: int = 512,
    hw: HardwareProfile | None = None,
    dispatch_overhead_us: float = 5.0,
    dequant_overhead_factor: float = 1.15,
) -> LayerLatency:
    """Estimate latency for a single quantized layer.

    Decode (seq_len=1): memory-bandwidth-bound
      latency = total_bytes / bandwidth + dispatch_overhead
      Adjusted by dequant_overhead_factor for quantization decode cost.

    Prefill (seq_len>1): compute-bound
      latency = flops / peak_throughput + dispatch_overhead
      Lower bits have slightly higher dequant cost per element.

    Args:
        dispatch_overhead_us: Fixed per-layer kernel dispatch cost (us).
        dequant_overhead_factor: Multiplier on bandwidth-limited time to
            account for dequantization compute. ~1.15 for 4-bit, ~1.25
            for 3-bit (more complex unpacking).
    """
    if hw is None:
        hw = detect_hardware()

    weight_bytes, scale_bias_bytes, total_bytes = estimate_layer_bytes(
        param_count, bits, group_size
    )

    # Decode: memory-bandwidth-bound
    # At seq_len=1, we read all weights but do minimal compute
    # Also need to read input activation (small) and write output (small)
    input_bytes = in_features * 2  # float16 input
    output_bytes = out_features * 2  # float16 output
    total_read_bytes = total_bytes + input_bytes

    # Effective bandwidth: dequant overhead reduces effective throughput
    # 3-bit has more complex unpacking than 4/8-bit
    dequant_factor = dequant_overhead_factor
    if bits == 3:
        dequant_factor *= 1.08  # 3-bit unpacking is ~8% slower
    elif bits == 2:
        dequant_factor *= 1.12  # 2-bit even more overhead

    decode_time_s = (total_read_bytes / hw.memory_bandwidth_bps) * dequant_factor
    decode_us = decode_time_s * 1e6 + dispatch_overhead_us

    # Prefill: compute-bound
    flops_prefill = estimate_layer_flops(in_features, out_features, seq_len_prefill)
    # Quantized matmul runs at different effective throughput per bit-width
    # 4-bit is roughly on par with FP16 on modern Apple Silicon
    # 8-bit is slightly slower due to wider dequant, 3-bit slightly slower due to complex unpack
    effective_flops = hw.peak_flops
    if bits <= 3:
        effective_flops *= 0.85  # 3-bit kernels less optimized
    elif bits == 8:
        effective_flops *= 0.95  # 8-bit slightly wider dequant

    prefill_time_s = flops_prefill / effective_flops
    prefill_us = prefill_time_s * 1e6 + dispatch_overhead_us

    # Arithmetic intensity
    flops_decode = estimate_layer_flops(in_features, out_features, seq_len_decode)
    arith_intensity = flops_decode / total_read_bytes if total_read_bytes > 0 else 0

    return LayerLatency(
        layer_name=layer_name,
        bits=bits,
        param_count=param_count,
        group_size=group_size,
        weight_bytes=weight_bytes,
        scale_bias_bytes=scale_bias_bytes,
        total_bytes=total_bytes,
        decode_us=decode_us,
        prefill_us=prefill_us,
        flops=flops_decode,
        arithmetic_intensity=arith_intensity,
    )


@dataclass
class CalibratedLatencyModel:
    """Latency model calibrated against a single actual measurement.

    The raw model only accounts for quantized linear layers. The actual
    per-token latency includes attention, normalization, KV cache, and
    framework overhead. This calibrates by learning the constant offset:

        total_latency = linear_layer_latency + overhead

    The overhead is estimated from one actual throughput measurement.
    After calibration, the model predicts relative throughput differences
    between quantization configs with high accuracy.
    """
    hw: HardwareProfile
    overhead_us: float = 0.0  # Non-linear-layer overhead per token
    calibrated: bool = False

    def calibrate(self, linear_layer_us: float, actual_tok_per_sec: float):
        """Calibrate from one actual measurement.

        Args:
            linear_layer_us: Predicted linear-layer-only latency (us)
            actual_tok_per_sec: Measured actual throughput (tok/s)
        """
        actual_us = 1e6 / actual_tok_per_sec
        self.overhead_us = max(actual_us - linear_layer_us, 0)
        self.calibrated = True

    def predict_tok_per_sec(self, linear_layer_us: float) -> float:
        """Predict throughput for a given linear-layer latency."""
        total_us = linear_layer_us + self.overhead_us
        return 1e6 / total_us if total_us > 0 else 0

    def predict_total_us(self, linear_layer_us: float) -> float:
        """Predict total per-token latency."""
        return linear_layer_us + self.overhead_us


def estimate_model_latency(
    layer_configs: list[dict],
    hw: HardwareProfile | None = None,
    seq_len_prefill: int = 512,
    calibrated_model: CalibratedLatencyModel | None = None,
) -> dict:
    """Estimate total model inference latency.

    Args:
        layer_configs: List of dicts with keys:
            name, param_count, in_features, out_features, bits, group_size
        hw: Hardware profile (auto-detected if None)
        seq_len_prefill: Sequence length for prefill estimate
        calibrated_model: If provided, uses calibrated overhead

    Returns: dict with decode_us, prefill_us, per_layer breakdown, model_bytes
    """
    if hw is None:
        hw = detect_hardware()

    per_layer = []
    total_decode_us = 0.0
    total_prefill_us = 0.0
    total_bytes = 0.0

    for lc in layer_configs:
        lat = estimate_layer_latency(
            layer_name=lc["name"],
            param_count=lc["param_count"],
            in_features=lc["in_features"],
            out_features=lc["out_features"],
            bits=lc["bits"],
            group_size=lc.get("group_size", 64),
            seq_len_prefill=seq_len_prefill,
            hw=hw,
        )
        per_layer.append(lat)
        total_decode_us += lat.decode_us
        total_prefill_us += lat.prefill_us
        total_bytes += lat.total_bytes

    # Apply calibration if available
    if calibrated_model and calibrated_model.calibrated:
        calibrated_decode_us = calibrated_model.predict_total_us(total_decode_us)
        decode_tps = calibrated_model.predict_tok_per_sec(total_decode_us)
    else:
        calibrated_decode_us = total_decode_us
        decode_tps = 1e6 / total_decode_us if total_decode_us > 0 else 0

    return {
        "linear_decode_us": total_decode_us,
        "decode_us": calibrated_decode_us,
        "decode_ms": calibrated_decode_us / 1000,
        "prefill_us": total_prefill_us,
        "prefill_ms": total_prefill_us / 1000,
        "decode_tok_per_sec": decode_tps,
        "prefill_tok_per_sec": seq_len_prefill * 1e6 / total_prefill_us if total_prefill_us > 0 else 0,
        "model_bytes": total_bytes,
        "model_mb": total_bytes / (1024 * 1024),
        "per_layer": per_layer,
        "hardware": hw.name,
        "calibrated": calibrated_model.calibrated if calibrated_model else False,
    }


def get_layer_dimensions(layer_name: str, model_config: dict) -> tuple[int, int]:
    """Infer in_features, out_features from layer name and model config.

    Works for Qwen3/Llama-style transformer architectures.
    """
    d_model = model_config.get("hidden_size", 1024)
    n_heads = model_config.get("num_attention_heads", 16)
    n_kv_heads = model_config.get("num_key_value_heads", 8)
    head_dim = model_config.get("head_dim", d_model // n_heads)
    d_ffn = model_config.get("intermediate_size", 3072)

    name_lower = layer_name.lower()

    if "q_proj" in name_lower:
        return d_model, n_heads * head_dim
    elif "k_proj" in name_lower:
        return d_model, n_kv_heads * head_dim
    elif "v_proj" in name_lower:
        return d_model, n_kv_heads * head_dim
    elif "o_proj" in name_lower:
        return n_heads * head_dim, d_model
    elif "gate_proj" in name_lower or "up_proj" in name_lower:
        return d_model, d_ffn
    elif "down_proj" in name_lower:
        return d_ffn, d_model
    elif "lm_head" in name_lower:
        vocab_size = model_config.get("vocab_size", 151936)
        return d_model, vocab_size
    elif "embed" in name_lower:
        vocab_size = model_config.get("vocab_size", 151936)
        return vocab_size, d_model
    else:
        # Fallback: assume square
        import math
        side = int(math.sqrt(model_config.get("param_count", d_model * d_model)))
        return side, side
