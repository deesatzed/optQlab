"""Resilient HuggingFace downloads: Xet high-performance transfer → HTTPS fallback.

HuggingFace's Xet content-addressed transfer (the default since ``hf-xet``
shipped) is fast when it works, but on some networks — corporate proxies, flaky
links, TLS-inspection middleboxes — it stalls partway through a large shard and
never recovers, so ``snapshot_download`` hangs or raises. The plain HTTPS range
path is slower but robust.

``robust_snapshot_download`` retries the normal (Xet) path a couple of times to
ride out transient blips, then, as a last resort, forces the HTTPS path by
setting ``HF_HUB_DISABLE_XET=1`` for one final attempt. Partial downloads are
resumed from the local cache, so a retry never re-fetches completed shards.

``install`` monkeypatches ``mlx_lm.utils.snapshot_download`` (module-level, used
by ``get_model_path`` / ``hf_repo_to_path``) so every ``mlx_lm.load`` download —
``optiq serve``, ``optiq convert``, adapters — inherits the fallback with
mlx-lm's own ``allow_patterns`` intact. We never pre-download extra files.
"""

from __future__ import annotations

import os
from typing import Any


def robust_snapshot_download(*args: Any, retries: int = 2, **kwargs: Any) -> str:
    """``snapshot_download`` with Xet→HTTPS failover.

    Positional/keyword args are forwarded verbatim, so this is a drop-in for
    ``huggingface_hub.snapshot_download`` (repo id positional or ``repo_id=``).
    """
    from huggingface_hub import snapshot_download

    # If Xet is already disabled by the environment, there is no faster path to
    # fall back from — just run once and let errors propagate.
    if os.environ.get("HF_HUB_DISABLE_XET") == "1":
        return snapshot_download(*args, **kwargs)

    last_exc: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            return snapshot_download(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — retry any transfer failure
            last_exc = exc

    # Final attempt on the plain HTTPS path (Xet off). Resumes from cache, so
    # completed shards are not re-fetched.
    prev = os.environ.get("HF_HUB_DISABLE_XET")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        result = snapshot_download(*args, **kwargs)
        _log("Xet transfer stalled/failed; completed via HTTPS fallback "
             "(HF_HUB_DISABLE_XET=1)")
        return result
    except Exception as exc:  # noqa: BLE001
        # Prefer surfacing the HTTPS-path error — it is the more actionable one.
        raise exc from last_exc
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = prev


def _log(msg: str) -> None:
    print(f"[optiq.download] {msg}", flush=True)


def install() -> bool:
    """Patch ``mlx_lm.utils.snapshot_download`` with the failover wrapper.

    Idempotent; returns True if the patch is in place.
    """
    try:
        import mlx_lm.utils as mlx_utils
    except Exception:
        return False

    if getattr(mlx_utils, "_optiq_download_fallback", False):
        return True

    orig = mlx_utils.snapshot_download

    def patched(*args: Any, **kwargs: Any) -> str:
        # Bypass entirely for local-only lookups (no network to fall back on).
        if kwargs.get("local_files_only"):
            return orig(*args, **kwargs)
        return robust_snapshot_download(*args, **kwargs)

    mlx_utils.snapshot_download = patched
    mlx_utils._optiq_download_fallback = True
    return True
