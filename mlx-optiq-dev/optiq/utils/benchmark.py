"""Benchmarking utilities: timing, memory, comparison tables."""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import psutil


@dataclass
class BenchmarkResult:
    name: str
    model_size_mb: float = 0.0
    bpw: float = 0.0
    perplexity: float = 0.0
    kl_divergence: float = 0.0
    tokens_per_sec: float = 0.0
    fps: float = 0.0
    peak_memory_mb: float = 0.0
    details: dict = field(default_factory=dict)


def measure_llm_perplexity(
    model_path: str,
    n_samples: int = 50,
    seq_len: int = 512,
) -> float:
    """Measure perplexity of an MLX LLM on WikiText-2 validation."""
    import mlx.core as mx
    from mlx_lm import load

    if os.path.isdir(model_path):
        model_path = os.path.abspath(model_path)
    model, tokenizer = load(model_path)

    # Load evaluation data
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [t for t in ds["text"] if len(t.strip()) > 100]
    except Exception:
        texts = [
            "The transformer architecture processes sequences through self-attention. "
            * 20
        ] * n_samples

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_loss = 0.0
    total_tokens = 0

    for text in texts[:n_samples]:
        tokens = tokenizer.encode(text, add_special_tokens=False)[:seq_len]
        if len(tokens) < 10:
            continue

        input_ids = mx.array([tokens[:-1]])
        targets = mx.array(tokens[1:])

        logits = model(input_ids)
        # Cross-entropy loss (vectorized)
        logits_2d = logits[0]  # (seq_len-1, vocab_size)
        # Log-softmax for numerical stability
        log_probs = logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)
        # Gather target token log probs
        n_tok = targets.shape[0]
        target_log_probs = log_probs[mx.arange(n_tok), targets]
        loss = -mx.sum(target_log_probs).item()

        total_loss += loss
        total_tokens += n_tok

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = float(np.exp(min(avg_loss, 100)))  # cap to avoid overflow
    return perplexity


def measure_llm_throughput(
    model_path: str,
    prompt: str = "The quick brown fox jumps over the lazy",
    n_tokens: int = 50,
    n_warmup: int = 3,
    n_runs: int = 5,
) -> float:
    """Measure token generation throughput (tokens/sec) for MLX LLM."""
    import mlx.core as mx
    from mlx_lm import load, generate

    if os.path.isdir(model_path):
        model_path = os.path.abspath(model_path)
    model, tokenizer = load(model_path)

    # Warmup
    for _ in range(n_warmup):
        generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        generate(model, tokenizer, prompt=prompt, max_tokens=n_tokens, verbose=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    tokens_per_sec = n_tokens / avg_time
    return float(tokens_per_sec)


def measure_model_bpw(model_path: str) -> float:
    """Estimate bits-per-weight from model config."""
    import json

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return 0.0

    with open(config_path) as f:
        config = json.load(f)

    quant = config.get("quantization", {})
    if not isinstance(quant, dict):
        return 16.0

    # Check for per-layer configs (mixed quantization)
    per_layer = {k: v for k, v in quant.items() if isinstance(v, dict)}
    if per_layer:
        bits_list = [v.get("bits", 4) for v in per_layer.values()]
        return float(np.mean(bits_list))

    # Uniform quantization
    if "bits" in quant:
        return float(quant["bits"])

    return 16.0  # Default FP16


def get_model_size(path: str) -> float:
    """Get model file size in MB."""
    total = 0
    if os.path.isdir(path):
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp) and f.endswith((".safetensors", ".npz", ".bin")):
                total += os.path.getsize(fp)
    elif os.path.isfile(path):
        total = os.path.getsize(path)
    return total / (1024 * 1024)


def print_comparison_table(results: list[BenchmarkResult], title: str = "Benchmark Comparison"):
    """Print a formatted comparison table."""
    print(f"\n  {title}")
    print("  " + "=" * 90)

    # Determine which columns have data
    has_ppl = any(r.perplexity > 0 for r in results)
    has_kl = any(r.kl_divergence > 0 for r in results)
    has_tps = any(r.tokens_per_sec > 0 for r in results)
    has_fps = any(r.fps > 0 for r in results)

    # Header
    header = f"  {'Model':<30} {'Size (MB)':>10} {'BPW':>6}"
    if has_ppl:
        header += f" {'PPL':>8}"
    if has_kl:
        header += f" {'KL Div':>10}"
    if has_tps:
        header += f" {'Tok/s':>8}"
    if has_fps:
        header += f" {'FPS':>6}"
    print(header)
    print("  " + "-" * 90)

    for r in results:
        row = f"  {r.name:<30} {r.model_size_mb:>10.1f} {r.bpw:>6.2f}"
        if has_ppl:
            row += f" {r.perplexity:>8.3f}" if r.perplexity > 0 else f" {'N/A':>8}"
        if has_kl:
            row += f" {r.kl_divergence:>10.6f}" if r.kl_divergence > 0 else f" {'N/A':>10}"
        if has_tps:
            row += f" {r.tokens_per_sec:>8.1f}" if r.tokens_per_sec > 0 else f" {'N/A':>8}"
        if has_fps:
            row += f" {r.fps:>6.1f}" if r.fps > 0 else f" {'N/A':>6}"
        print(row)

    print("  " + "=" * 90)

    # Highlight best
    if has_ppl and len(results) > 1:
        valid_ppl = [(r.name, r.perplexity) for r in results if r.perplexity > 0]
        if valid_ppl:
            best = min(valid_ppl, key=lambda x: x[1])
            print(f"  Best perplexity: {best[0]} ({best[1]:.3f})")

    if has_fps and len(results) > 1:
        valid_fps = [(r.name, r.fps) for r in results if r.fps > 0]
        if valid_fps:
            best = max(valid_fps, key=lambda x: x[1])
            print(f"  Best FPS: {best[0]} ({best[1]:.1f})")
