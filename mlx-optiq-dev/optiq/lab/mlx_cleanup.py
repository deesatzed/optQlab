"""Threshold-based MLX buffer-pool cleanup for long-running services.

Mirrors mlx-lm's own ``_clear_cache(threshold)`` helper from
``mlx_lm/tuner/trainer.py``::

    def _clear_cache(threshold: int):
        if mx.get_cache_memory() > threshold:
            mx.clear_cache()

The reuse pool (``mx.get_cache_memory()``) is buffers MLX has allocated
but isn't currently using — kept around so the next request doesn't pay
the cost of re-allocating from Metal. Without a cap, the pool grows
until you hit either the MTLResource cap (~499 K objects) or just plain
RAM exhaustion.

A blanket "clear after every request" kills the reuse benefit (each
cleared buffer must be re-allocated from Metal next time, ~10-100 ms).
Threshold-based clearing lets reuse work most of the time and only
fires when the pool is bloated enough to matter.

Default threshold is RAM-derived (10 % of total memory) so the same code
scales sanely across 24 / 36 / 48 / 128 GB Macs without a hardcoded
magic number.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional


log = logging.getLogger(__name__)


# Fraction of total RAM to allow the buffer reuse pool to grow to before
# we trigger a clear. 10% is the "noticeable but not constraining" zone:
# big enough that reuse pays off across requests, small enough that the
# pool doesn't crowd out the model + active KV cache.
DEFAULT_FRACTION = 0.10

# Floor: never set the threshold below 1 GB. Below that, even normal
# inference activity trips the threshold every request — clearing
# becomes effectively unconditional and loses its purpose.
MIN_THRESHOLD_BYTES = 1 * 1024**3


# Separate budget for mlx-lm's LRUPromptCache (the per-request KV caches
# the server keeps around for prefix reuse). mlx-lm's default is 10
# entries with NO byte cap — at 24 k context a single cache is ~2.75 GB,
# so four unique long-context prompts OOM a 24 GB Mac before any reuse
# eviction kicks in. Setting --prompt-cache-bytes lets mlx-lm's own
# trim_to(n_bytes) path (server.py:795-798) free old caches before the
# device limit hits. 15 % of RAM ≈ one big-context cache on a 24 GB
# Mac (model + active cache + headroom), more on larger machines.
PROMPT_CACHE_FRACTION = 0.15
MIN_PROMPT_CACHE_BYTES = 2 * 1024**3
# Share of the memory left AFTER the model's weights that the prompt cache may
# claim. The rest has to cover activations and prefill transients.
PROMPT_CACHE_FREE_FRACTION = 0.25
# Hard floor when the model nearly fills the machine. Lower than
# MIN_PROMPT_CACHE_BYTES on purpose: on a tight box a small cache that fits
# beats a large one that OOMs the third turn.
MIN_PROMPT_CACHE_FLOOR = 512 * 1024**2


def _total_ram_bytes() -> int:
    try:
        import psutil
        return psutil.virtual_memory().total
    except Exception:
        # psutil is in our [lab] extras but might be missing in odd
        # environments. Conservative fallback.
        return 16 * 1024**3


def default_threshold_bytes() -> int:
    """RAM-derived default. 10 % of total RAM, floor 1 GB."""
    return max(MIN_THRESHOLD_BYTES, int(_total_ram_bytes() * DEFAULT_FRACTION))


def default_prompt_cache_bytes(model_dir: Optional[str] = None) -> int:
    """Cap for mlx-lm's --prompt-cache-bytes. Without one, mlx-lm holds up to
    10 distinct KV caches with no byte cap and OOMs on long-context workloads.

    Derived from what is actually LEFT after the model, not from total RAM.
    A fraction of total is the wrong budget for a model that nearly fills the
    machine: gemma-4-26B is 17.6 GB of weights plus a 1.1 GB vision sidecar on
    a 24 GB Mac, leaving ~5 GB for activations, prefill transients and the
    prompt cache together. The old 15 %-of-total rule handed 2.4 GB of that to
    the cache alone, and a multi-turn session OOMed on the third turn -- with
    fp16 KV as readily as quantized, which is what showed it was this and not
    the cache quantization.

    With no model_dir (or no weights found) this falls back to the old
    behavior, so callers that cannot resolve a local path are unaffected.
    """
    total = _total_ram_bytes()
    weights = _weight_bytes(model_dir) if model_dir else 0
    if weights <= 0:
        return max(MIN_PROMPT_CACHE_BYTES, int(total * PROMPT_CACHE_FRACTION))

    # Leave room for activations and prefill transients before giving anything
    # to the cache; the cache is an optimization, the forward pass is not.
    free = total - weights
    budget = int(free * PROMPT_CACHE_FREE_FRACTION)
    capped = min(int(total * PROMPT_CACHE_FRACTION), budget)
    return max(MIN_PROMPT_CACHE_FLOOR, capped)


def _weight_bytes(model_dir: str) -> int:
    """On-disk safetensors size, which is what the weights occupy resident."""
    from pathlib import Path
    total = 0
    try:
        for p in Path(model_dir).glob("*.safetensors"):
            try:
                total += p.stat().st_size
            except OSError:
                pass
    except OSError:
        return 0
    return total


def maybe_clear(threshold_bytes: int) -> bool:
    """Clear MLX's reuse pool if it exceeds ``threshold_bytes``.

    Calls ``gc.collect()`` BEFORE ``mx.clear_cache()`` — same pattern as
    mlx-omni-server's ``clear_mlx_memory``. The GC pass forces Python
    to drop any unreferenced MLX arrays so the subsequent
    ``mx.clear_cache()`` actually returns their backing buffers to
    Metal, instead of waiting for natural GC.

    NOTE: this hook only frees buffers in the *unused* reuse pool. It
    does NOT free mlx-lm's LRUPromptCache (the KV caches the server
    keeps live for prefix reuse) — those need ``--prompt-cache-bytes``
    set on the mlx_lm.server invocation. See ``inject_prompt_cache_bytes``.

    Returns True if a clear actually happened. Safe to call from any
    thread (mlx-lm's own API is thread-safe). Cheap when no clear
    happens (~few microseconds of ``get_cache_memory()``).
    """
    import gc
    try:
        import mlx.core as mx
    except ImportError:
        return False
    if mx.get_cache_memory() <= threshold_bytes:
        return False
    gc.collect()
    mx.clear_cache()
    return True


def inject_prompt_cache_bytes(argv: list[str], default_bytes: int) -> list[str]:
    """Return ``argv`` with ``--prompt-cache-bytes <N>`` appended if the
    caller didn't already pass it. Use this when spawning mlx_lm.server
    to cap the LRUPromptCache before it bloats the GPU and crashes
    Metal."""
    for a in argv:
        if a == "--prompt-cache-bytes" or a.startswith("--prompt-cache-bytes="):
            return list(argv)
    return list(argv) + ["--prompt-cache-bytes", str(default_bytes)]


def install_server_cleanup(threshold_bytes: int) -> None:
    """Patch ``mlx_lm.server.APIHandler.handle_completion`` to call
    ``maybe_clear`` after every response completes.

    Idempotent: re-installing replaces the previous patch in place.
    Safe if called before mlx_lm.server.main() — the patch sticks
    onto the class, so all later instances inherit it.
    """
    import mlx_lm.server

    sentinel = "_optiq_cleanup_installed"
    if getattr(mlx_lm.server.APIHandler, sentinel, False):
        return  # already installed

    original = mlx_lm.server.APIHandler.handle_completion

    def patched(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        finally:
            try:
                maybe_clear(threshold_bytes)
            except Exception as e:  # noqa: BLE001 - cleanup must never raise
                log.warning("mlx cleanup hook raised: %s", e)

    mlx_lm.server.APIHandler.handle_completion = patched
    setattr(mlx_lm.server.APIHandler, sentinel, True)
    log.info(
        "[optiq.lab.mlx_cleanup] installed server cleanup hook "
        "(threshold=%.2f GB)",
        threshold_bytes / 1024**3,
    )
