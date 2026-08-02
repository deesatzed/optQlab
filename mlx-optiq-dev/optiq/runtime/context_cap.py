"""Memory-aware context cap: keep a long prompt from OOM-crashing the server.

Each token of KV cache costs ``full_attention_layers × n_kv_heads × head_dim × 2
(K+V)`` values, at whatever precision the cache is stored in. For a big model on
a tight Mac, letting a request run to the model's full native context (e.g.
128k–256k) can allocate more KV than unified memory holds — the whole server
dies with a Metal OOM mid-generation.

Note ``full_attention_layers``, not ``num_hidden_layers``. Linear-attention
layers cache nothing that grows with context, and sliding-attention layers cost
a constant. And note "whatever precision": with ``--kv-bits`` or ``--kv-config``
the cache is 4–8 bit, not bf16.

``estimate_kv_token_cap`` reads the model's config + on-disk weight size and the
machine's free RAM, and returns a token cap **only when the full native context
would not fit** — otherwise ``None`` (no cap, native behavior preserved). So on
a machine with enough RAM this is a no-op; it engages exactly in the regime that
would otherwise crash.

``install`` wraps ``mlx_lm.server.make_prompt_cache`` to pass ``max_kv_size``
when the caller doesn't. Under the hood mlx-lm then uses a ``RotatingKVCache``
(which supports ``merge`` / ``trim`` / ``to_quantized``, so batching,
prompt-cache reuse, and KV quantization all keep working). Models that manage
their own cache (sliding-window: Gemma, Qwen3-Next) ignore ``max_kv_size`` and
are untouched — they already bound their own KV memory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


def _read_config(model_dir: str) -> Optional[dict]:
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        with open(cfg_path) as f:
            return json.load(f)
    except Exception:
        return None


def _text_config(cfg: dict) -> dict:
    # Multimodal configs nest the LM geometry under text_config.
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) else cfg


def _weight_bytes(model_dir: str) -> int:
    total = 0
    for p in Path(model_dir).glob("*.safetensors"):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _bytes_per_kv_value(kv_bits: Optional[float], group_size: int) -> float:
    """Bytes one cached K/V element costs, including quantization scales.

    Affine quantization stores a scale and a bias (fp16 each) per group, so a
    4-bit cache costs 0.5 B/value plus 4 B per ``group_size`` values, not 0.5.
    """
    if not kv_bits or kv_bits >= 16:
        return 2.0  # bf16 / fp16
    return kv_bits / 8.0 + 4.0 / float(group_size)


def _cache_geometry(tc: dict) -> Optional[tuple[int, int, int]]:
    """(full_attention_layers, sliding_layers, sliding_window).

    ``num_hidden_layers`` is NOT the number of layers that hold a KV cache:

    * ``linear_attention`` (GatedDeltaNet — qwen3_5, qwen3_6, qwen3_next) keeps a
      fixed-size recurrent state and caches nothing that grows with context.
      Qwen3.6-35B-A3B has 40 layers, of which only 10 cache.
    * ``sliding_attention`` (Gemma-4, gpt-oss) bounds its cache at the sliding
      window, so it costs a constant, not a per-token amount.

    Counting all 40 layers overestimated per-token KV by 4x on the Qwen3.5/3.6
    line and 6x on gemma-4-e4b, which is how a model that comfortably holds 128k
    got capped at 2048 tokens.
    """
    try:
        n_layers = int(tc.get("num_hidden_layers"))
    except (TypeError, ValueError):
        return None

    types = tc.get("layer_types")
    if not isinstance(types, list) or not types:
        return n_layers, 0, 0

    full = sum(1 for t in types if t == "full_attention")
    sliding = sum(1 for t in types if t == "sliding_attention")
    window = tc.get("sliding_window") or tc.get("attention_window_size") or 0
    try:
        window = int(window)
    except (TypeError, ValueError):
        window = 0
    # A model that is all-linear still needs a nonzero denominator; fall back to
    # the raw layer count rather than dividing by zero.
    if full == 0 and sliding == 0:
        return n_layers, 0, 0
    return full, sliding, window


def estimate_kv_token_cap(
    model_dir: str,
    *,
    headroom: float = 0.8,
    kv_bits: Optional[float] = None,
    kv_group_size: int = 64,
) -> Optional[int]:
    """Return a safe KV token cap, or ``None`` if the full context fits.

    ``headroom`` is the fraction of the KV budget we actually hand out.
    ``kv_bits`` is the average bits per cached value once ``--kv-bits`` or
    ``--kv-config`` is in play; ``None`` means an unquantized fp16 cache.
    """
    cfg = _read_config(model_dir)
    if not cfg:
        return None
    tc = _text_config(cfg)

    geom = _cache_geometry(tc)
    if not geom:
        return None
    full_layers, sliding_layers, window = geom

    try:
        n_heads = int(tc.get("num_attention_heads"))
    except (TypeError, ValueError):
        return None
    n_kv_heads = int(tc.get("num_key_value_heads", n_heads) or n_heads)
    head_dim = tc.get("head_dim")
    if not head_dim:
        hidden = tc.get("hidden_size")
        if not hidden:
            return None
        head_dim = int(hidden) // n_heads
    head_dim = int(head_dim)

    native = tc.get("max_position_embeddings") or cfg.get("max_position_embeddings")
    try:
        native = int(native)
    except (TypeError, ValueError):
        native = None

    b = _bytes_per_kv_value(kv_bits, kv_group_size)
    vals_per_token_per_layer = n_kv_heads * head_dim * 2  # K and V

    per_token_kv = full_layers * vals_per_token_per_layer * b
    # Sliding layers cost a fixed amount once the window is full.
    sliding_fixed = sliding_layers * window * vals_per_token_per_layer * b
    if per_token_kv <= 0:
        return None

    try:
        import psutil
        avail = int(psutil.virtual_memory().available)
    except Exception:
        return None

    # Weights are mmap'd, so their pages are file-backed and evictable, and a
    # sparse MoE never faults in every expert. `avail - weights` can therefore go
    # negative for a model that runs fine (Qwen3.6-35B-A3B: 23 GB of weights,
    # 6 GB actually resident). Falling back to a slice of free RAM keeps the
    # estimate conservative without hard-capping such a model at 2048.
    kv_budget = max(avail - _weight_bytes(model_dir), avail * 0.25) * float(headroom)
    kv_budget -= sliding_fixed
    if kv_budget <= 0:
        return 2048

    max_ctx = int(kv_budget // per_token_kv)
    if native is not None and max_ctx >= native:
        return None  # full native context fits — no cap needed

    # Round down to a 1024 multiple, floor at 2048 so the server stays usable.
    capped = max(2048, (max_ctx // 1024) * 1024)
    if native is not None:
        capped = min(capped, native)
    return capped


def install(max_kv_size: int) -> bool:
    """Wrap ``mlx_lm.server.make_prompt_cache`` to inject ``max_kv_size``.

    Only fills in ``max_kv_size`` when the caller passes none, so an explicit
    size elsewhere still wins. Idempotent.
    """
    if not max_kv_size or max_kv_size <= 0:
        return False
    try:
        import mlx_lm.server as server_mod
    except Exception:
        return False
    if getattr(server_mod, "_optiq_ctx_cap_installed", False):
        return True

    orig = server_mod.make_prompt_cache

    def wrapped(model, mks=None):
        if mks is None:
            mks = max_kv_size
        return orig(model, mks)

    server_mod.make_prompt_cache = wrapped
    server_mod._optiq_ctx_cap_installed = True
    return True
