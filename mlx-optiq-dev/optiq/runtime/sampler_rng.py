"""Fix per-request sampling diversity on the serving worker thread.

mlx-lm's ``sample_utils.categorical_sampling`` is decorated with
``@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)`` so the
temperature sampler threads the global RNG state through the compiled function.
That works on the main thread, but mlx-lm's server runs token generation on a
dedicated worker thread, and there the compiled function's captured RNG state is
frozen: every request samples from the *same* state, so the model returns
byte-identical output regardless of ``temperature`` or ``seed`` (effectively
greedy). Confirmed empirically: the same ``stream_generate`` call is diverse on
the main thread and identical on a worker thread; dropping the ``mx.compile``
restores diversity.

The fix replaces ``categorical_sampling`` with an un-compiled equivalent that
reads the live ``mx.random.state`` on whatever thread it runs on. ``make_sampler``
looks ``categorical_sampling`` up as a module global at call time, so swapping the
module attribute is enough — no reach into the sampler factory. The compile only
fused a single per-token categorical draw (negligible next to the model forward),
so the throughput cost is ~0.
"""
from __future__ import annotations


def install() -> bool:
    """Swap mlx-lm's compiled ``categorical_sampling`` for a thread-safe one.
    Idempotent. Returns True if the patch was applied (or already present)."""
    try:
        import mlx.core as mx
        import mlx_lm.sample_utils as su
    except Exception:
        return False

    if getattr(su, "_optiq_rng_worker_fix", False):
        return True

    def categorical_sampling(logits, temp):
        # Same math as upstream, minus the @mx.compile that freezes the RNG
        # state on non-main threads.
        return mx.random.categorical(logits * (1 / temp))

    su.categorical_sampling = categorical_sampling
    su._optiq_rng_worker_fix = True
    return True
