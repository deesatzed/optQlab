"""Idle auto-unload: free the served model's RAM after a period of inactivity.

On Apple Silicon the model lives in unified memory, so a served 24 GB model
holds RAM the machine can't use for anything else even when nobody's calling
it. This watchdog drops the model after ``timeout`` seconds with no requests
and reloads it lazily on the next call — the same lazy-load mlx-lm already does
on first request, just triggered again.

Safety: mlx-lm's ``ModelProvider.load`` is called on every request (it returns
the cached model when the key matches). We stamp a last-access time there. A
daemon thread unloads by nulling ``model``/``tokenizer``/``draft_model``/
``model_key`` and calling ``mx.clear_cache()``. This is safe even if a
generation is somehow still in flight: that worker holds its own reference to
the model object (from ``load``'s return value), so Python keeps it alive until
the generation finishes — we only detach the provider's cached handle, and the
next request sees the null ``model_key`` and reloads. Set the timeout longer
than your longest single generation so a normal long decode is never
interrupted (default 600 s).
"""

from __future__ import annotations

import threading
import time


def install(timeout_s: float, *, poll_s: float | None = None) -> bool:
    """Install the idle-unload watchdog on ``mlx_lm.server.ModelProvider``.

    ``timeout_s <= 0`` disables the feature (returns False). Idempotent.
    """
    if not timeout_s or timeout_s <= 0:
        return False

    try:
        import mlx_lm.server as server_mod
    except Exception:
        return False

    MP = server_mod.ModelProvider
    if getattr(MP, "_optiq_idle_installed", False):
        return True

    orig_load = MP.load
    poll = poll_s if poll_s and poll_s > 0 else max(5.0, min(timeout_s / 4.0, 30.0))

    def _watchdog(inst) -> None:
        import mlx.core as mx

        while True:
            time.sleep(poll)
            last = getattr(inst, "_optiq_last_access", None)
            if last is None or inst.model is None:
                continue
            if (time.monotonic() - last) < timeout_s:
                continue
            # Idle past the threshold — detach the cached model so RAM frees.
            inst.model_key = None
            inst.model = None
            inst.tokenizer = None
            inst.draft_model = None
            try:
                mx.clear_cache()
            except Exception:
                pass
            print(
                f"[optiq.serve] idle {timeout_s:.0f}s: unloaded model, freed RAM "
                f"(reloads on the next request)",
                flush=True,
            )

    def patched_load(self, model_path, adapter_path=None, draft_model_path=None):
        self._optiq_last_access = time.monotonic()
        if not getattr(self, "_optiq_watchdog_started", False):
            self._optiq_watchdog_started = True
            threading.Thread(
                target=_watchdog, args=(self,), daemon=True,
                name="optiq-idle-unload",
            ).start()
        return orig_load(self, model_path, adapter_path, draft_model_path)

    MP.load = patched_load
    MP._optiq_idle_installed = True
    return True
