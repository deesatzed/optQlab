"""KL-divergence eval for quantized MLX LLMs.

Measures how much a quantized model's next-token distribution drifts from
a reference model's. The cheapest, most informative diagnostic we have:
catches subtle distribution shifts that downstream task evals miss, runs
in ~2 min on a 27 B model, and is the metric Unsloth uses as their
primary diagnostic for the Dynamic 2.0 GGUF quants.

Two reference modes, picked automatically based on what fits in RAM:

  * ``"bf16"`` — load the original bf16 base into RAM as the reference.
    Highest fidelity. RAM ≈ 2 × params in GB. Auto when bf16 fits in
    ~70 % of available RAM (i.e., models up to ~10 B on a 36 GB Mac).

  * ``"uniform_4bit"`` — use mlx-community's published uniform-4-bit
    quant of the same base as reference. Cheaper, lets us measure on
    27 B+ models where bf16 doesn't fit, but the "gold" is itself
    quantized so the absolute KL number is a lower bound on the true
    drift from full precision. Differential rankings (this quant vs
    that quant) remain valid.

Either way, the eval pipeline is the same:

  1. Load reference, run K calibration prompts, collect per-token logits.
  2. Free reference, load candidate, run same prompts, collect logits.
  3. Compute mean KL(reference ‖ candidate) per token, averaged over
     prompts.

We ship the prompts inside the package — same sourcing as
``optiq.calibration.data.optiq.jsonl`` so KL eval reflects the same
domain mix the quantizer was tuned for.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class KLResult:
    candidate_path: str
    reference_path: str
    reference_mode: str  # "bf16" or "uniform_4bit"
    n_prompts: int
    seq_len: int
    mean_kl: float                       # primary scalar
    median_kl: float                     # robust to outliers
    p95_kl: float                        # tail (long-distribution drift)
    elapsed_sec: float

    def __str__(self) -> str:
        return (
            f"KL(ref ∥ candidate)  mean={self.mean_kl:.4f}  "
            f"median={self.median_kl:.4f}  p95={self.p95_kl:.4f}\n"
            f"  reference={self.reference_mode}: {self.reference_path}\n"
            f"  candidate:                       {self.candidate_path}\n"
            f"  prompts={self.n_prompts} × {self.seq_len} tokens, "
            f"elapsed {self.elapsed_sec:.0f}s"
        )


_BF16_RAM_FIT_FRACTION = 0.70


def _uniform_4bit_reference(
    base_repo: str, stem: str, group_size: int = 64
) -> tuple[str, str]:
    """Resolve a uniform-4-bit KL reference for ``base_repo``.

    Prefers mlx-community's published ``<stem>-4bit`` quant when it actually
    exists on the Hub. For brand-new bases (and any base without a published
    uniform-4-bit sibling) that repo 404s, so we fall back to building a local
    uniform-4-bit quant of the base once and caching it under
    ``~/.cache/optiq/kl_refs/`` for reuse. Returns ``(path_or_repo, "uniform_4bit")``.
    """
    published = f"mlx-community/{stem}-4bit"
    try:
        from huggingface_hub import repo_exists

        if repo_exists(published):
            return published, "uniform_4bit"
    except Exception:
        # Can't reach the Hub (offline, rate-limited). Fall through to a
        # local build rather than returning a repo we couldn't verify.
        pass

    from glob import glob

    cache_root = os.path.join(os.path.expanduser("~"), ".cache", "optiq", "kl_refs")
    local_ref = os.path.join(cache_root, f"{stem}-uniform4bit")
    if os.path.isdir(local_ref) and os.path.exists(
        os.path.join(local_ref, "config.json")
    ) and glob(os.path.join(local_ref, "*.safetensors")):
        print(f"  [KL] reusing cached local uniform-4-bit reference {local_ref}")
        return local_ref, "uniform_4bit"

    os.makedirs(cache_root, exist_ok=True)
    print(
        f"  [KL] published reference {published} not found on the Hub — "
        f"building a local uniform-4-bit quant of {base_repo} "
        f"(one-time, cached at {local_ref})"
    )
    from optiq.backends.mlx_backend import convert_llm_uniform

    convert_llm_uniform(base_repo, local_ref, bits=4, group_size=group_size)
    return local_ref, "uniform_4bit"


def _resolve_reference(candidate_path: str, prefer: str = "auto") -> tuple[str, str]:
    """Pick a reference model path + mode for KL evaluation.

    Heuristic when ``prefer="auto"``:
      * Try to find a sibling bf16 base in the candidate's metadata or by
        guessing from the directory name. If RAM allows, use it.
      * Otherwise fall back to mlx-community's uniform-4-bit quant of the
        same base (e.g. ``mlx-community/Qwen3.5-9B-OptiQ-4bit`` →
        ``mlx-community/Qwen3.5-9B-4bit``).
    """
    import json
    cand_abs = os.path.abspath(candidate_path) if os.path.isdir(candidate_path) else candidate_path

    # Try to read base model from optiq_metadata.json or config.json.
    # optiq_metadata.json["base_model"] is the authoritative source: written
    # by the convert pipeline as "<org>/<repo>" so we can recover it for
    # local outputs whose dir name doesn't follow the HF naming convention
    # (e.g. ".../requants/Qwen3.6-27B/optiq_mixed").
    base_repo = None
    for fname in ("optiq_metadata.json", "config.json"):
        meta_path = os.path.join(cand_abs, fname) if os.path.isdir(cand_abs) else None
        if meta_path and os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path))
                base_repo = meta.get("base_model") or meta.get("_name_or_path")
                if base_repo:
                    break
            except Exception:
                pass

    # If still no metadata, guess from candidate path. Walk the parent
    # chain too, so paths like ".../requants/Qwen3.6-27B/optiq_mixed" can
    # match "Qwen3.6-27B" up one level.
    if not base_repo:
        cand_norm = cand_abs.rstrip("/")
        candidates = [os.path.basename(cand_norm)]
        parent = os.path.dirname(cand_norm)
        if parent:
            candidates.append(os.path.basename(parent))

        for leaf in candidates:
            for suffix in ("-OptiQ-4bit", "-4bit", "-OptiQ"):
                if leaf.endswith(suffix):
                    leaf = leaf[: -len(suffix)]
                    break
            stem = leaf
            if stem.lower().startswith("qwen"):
                base_repo = f"Qwen/{stem}"
                break
            if stem.lower().startswith("gemma"):
                base_repo = f"google/{stem}"
                break

    if prefer == "bf16":
        if not base_repo:
            raise RuntimeError(
                "Could not infer bf16 reference path. Pass --reference-model "
                "explicitly or rebuild the candidate with optiq metadata."
            )
        return base_repo, "bf16"

    if prefer == "uniform_4bit":
        # Published uniform-4-bit quant of the same base if it exists,
        # else a locally-built (cached) one.
        if base_repo and "/" in base_repo:
            stem = base_repo.split("/")[-1]
            return _uniform_4bit_reference(base_repo, stem)
        raise RuntimeError(
            "Could not infer uniform_4bit reference path. Pass --reference-model."
        )

    # auto: pick based on RAM
    try:
        import psutil
        avail_bytes = psutil.virtual_memory().available
    except Exception:
        avail_bytes = 28 * 1024 ** 3  # safe fallback for 36 GB Mac

    # Get bf16 size — first try HF API for real on-disk sizes (most accurate
    # for MoE etc. where the architecture-derived estimate is way off);
    # fall back to crude formula if the API is unreachable.
    bf16_bytes = 0
    if base_repo:
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(base_repo, files_metadata=True)
            bf16_bytes = sum(
                (s.size or 0) for s in (info.siblings or [])
                if s.rfilename and s.rfilename.endswith(".safetensors")
            )
        except Exception:
            cand_cfg = os.path.join(cand_abs, "config.json") if os.path.isdir(cand_abs) else None
            if cand_cfg and os.path.exists(cand_cfg):
                try:
                    cfg = json.load(open(cand_cfg))
                    tcfg = cfg.get("text_config", cfg)
                    hidden = tcfg.get("hidden_size", 4096)
                    n_layers = tcfg.get("num_hidden_layers", 32)
                    vocab = tcfg.get("vocab_size", 150000)
                    # very rough — under-counts MoE total params
                    bf16_bytes = 2 * (12 * hidden * hidden * n_layers + hidden * vocab)
                except Exception:
                    pass

    if base_repo and bf16_bytes > 0 and bf16_bytes <= _BF16_RAM_FIT_FRACTION * avail_bytes:
        print(f"  [KL/auto] bf16 ≈ {bf16_bytes / 1e9:.1f} GB fits in "
              f"{avail_bytes / 1e9:.1f} GB RAM — using bf16 reference")
        return base_repo, "bf16"

    if base_repo and "/" in base_repo:
        stem = base_repo.split("/")[-1]
        print(f"  [KL/auto] bf16 ≈ {bf16_bytes / 1e9:.1f} GB > "
              f"{_BF16_RAM_FIT_FRACTION:.0%} of {avail_bytes / 1e9:.1f} GB — "
              f"falling back to uniform_4bit reference")
        return _uniform_4bit_reference(base_repo, stem)

    raise RuntimeError(
        "Could not pick a reference model. Pass --reference-model explicitly."
    )


def _collect_logits(
    model_path: str, prompts: list, max_seq_len: int,
) -> list:
    """Load ``model_path``, forward each prompt, return list of (1, T, V) fp16
    logits arrays. Casts to fp16 to halve memory; KL math casts back to fp32
    internally for stability."""
    import mlx.core as mx
    from .decode import load_tolerant, reclaim

    if os.path.isdir(model_path):
        model_path = os.path.abspath(model_path)

    print(f"  loading {model_path} …")
    model, tokenizer = load_tolerant(model_path)

    if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token

    out_logits = []
    for prompt_text in prompts:
        ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        ids = ids[:max_seq_len]
        ids_arr = mx.array([ids], dtype=mx.int32)
        logits = model(ids_arr)
        if hasattr(logits, "logits"):
            logits = logits.logits
        out_logits.append(logits.astype(mx.float16))
    mx.eval(*out_logits)

    # Free model + framework caches before next load
    del model
    del tokenizer
    gc.collect()
    try:
        reclaim()
    except Exception:
        pass

    return out_logits


def _kl_per_token(p_logits, q_logits):
    """KL(p ∥ q) per token, averaged over vocab. Returns (T,) fp32 array."""
    import mlx.core as mx
    p = p_logits.astype(mx.float32)
    q = q_logits.astype(mx.float32)
    log_p = p - mx.logsumexp(p, axis=-1, keepdims=True)
    log_q = q - mx.logsumexp(q, axis=-1, keepdims=True)
    p_probs = mx.softmax(p, axis=-1)
    # KL(p||q) = sum_v p(v) * (log p(v) - log q(v))
    kl = mx.sum(p_probs * (log_p - log_q), axis=-1)
    # kl shape: (1, T)
    return kl.squeeze(0)


def _load_default_prompts(seq_len: int, n_prompts: int) -> list:
    """Source prompts from the bundled optiq calibration mix so KL is
    measured on the same distribution the quantizer was tuned for."""
    import json
    from pathlib import Path

    optiq_jsonl = (
        Path(__file__).parent.parent / "calibration" / "data" / "optiq.jsonl"
    )
    if not optiq_jsonl.exists():
        raise RuntimeError(
            f"Default KL prompts not found at {optiq_jsonl}. Either rebuild "
            f"the calibration mix (`python scripts/build_calibration.py`) or "
            f"pass --prompts-file."
        )

    samples = []
    with optiq_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Convert each sample to a flat prompt string. For chat samples we
    # serialize a simple "[role]: content" form — the KL signal we care
    # about is whether the candidate's distribution drifts on this content,
    # not whether the chat template renders identically.
    prompts = []
    for s in samples:
        if "text" in s:
            prompts.append(s["text"])
        elif "messages" in s:
            prompts.append(
                "\n\n".join(
                    f"[{m['role']}]: {m['content']}" for m in s["messages"]
                )
            )
        if len(prompts) >= n_prompts:
            break

    if len(prompts) < n_prompts:
        # Repeat to fill (rare — calibration mix has 40 samples)
        prompts = (prompts * ((n_prompts // len(prompts)) + 1))[:n_prompts]

    return prompts[:n_prompts]


def evaluate_kl(
    candidate_path: str,
    reference_path: str | None = None,
    reference_mode: str = "auto",
    n_prompts: int = 64,
    seq_len: int = 256,
    prompts: list | None = None,
) -> KLResult:
    """Compute KL(reference ∥ candidate) on shared prompts.

    Args:
        candidate_path: Path or HF repo of the quant being evaluated.
        reference_path: Explicit reference model. If None, auto-resolved
            via the candidate's metadata + ``reference_mode``.
        reference_mode: ``"bf16"``, ``"uniform_4bit"``, or ``"auto"`` (pick
            bf16 if it fits in RAM, else uniform_4bit).
        n_prompts: Number of distinct calibration prompts.
        seq_len: Max tokens per prompt.
        prompts: Override for the bundled calibration prompts.
    """
    import mlx.core as mx
    t0 = time.time()

    # Resolve reference
    if reference_path is None:
        reference_path, reference_mode = _resolve_reference(
            candidate_path, prefer=reference_mode,
        )

    if prompts is None:
        prompts = _load_default_prompts(seq_len, n_prompts)

    print(f"\n[KL eval] candidate: {candidate_path}")
    print(f"[KL eval] reference ({reference_mode}): {reference_path}")
    print(f"[KL eval] {len(prompts)} prompts × {seq_len} max tokens")

    # Run reference first (use its logits as p), then unload + run candidate
    ref_logits = _collect_logits(reference_path, prompts, seq_len)
    cand_logits = _collect_logits(candidate_path, prompts, seq_len)

    # Per-prompt mean KL
    per_prompt_means = []
    all_per_token = []
    for p_log, q_log in zip(ref_logits, cand_logits):
        # Truncate to common length if shapes differ slightly
        T = min(p_log.shape[1], q_log.shape[1])
        kl_t = _kl_per_token(p_log[:, :T, :], q_log[:, :T, :])
        kl_t_np = np.array(kl_t.tolist())
        per_prompt_means.append(float(kl_t_np.mean()))
        all_per_token.extend(kl_t_np.tolist())

    per_prompt_means = np.array(per_prompt_means)
    all_per_token = np.array(all_per_token)

    res = KLResult(
        candidate_path=candidate_path,
        reference_path=reference_path,
        reference_mode=reference_mode,
        n_prompts=len(prompts),
        seq_len=seq_len,
        mean_kl=float(per_prompt_means.mean()),
        median_kl=float(np.median(all_per_token)),
        p95_kl=float(np.percentile(all_per_token, 95)),
        elapsed_sec=time.time() - t0,
    )
    return res
