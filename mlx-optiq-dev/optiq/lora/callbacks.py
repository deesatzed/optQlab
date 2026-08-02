"""Training callbacks for OptiQ LoRA training.

These are plugged into ``mlx_lm.tuner.trainer.train`` via the
``training_callback`` arg so optiq can hook into its eval/train loss
reports without forking the trainer itself.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.tuner.callbacks import TrainingCallback


class EarlyStop(Exception):
    """Raised from a callback to break mlx-lm's training loop early.

    mlx-lm's ``train()`` has no callback-driven stop hook (its loop never
    checks a return value), so the only non-forking way to halt it is to
    raise out of ``on_val_loss_report``. ``train_lora`` catches this; by the
    time it fires the best adapter is already on disk (BestAdapterCallback
    runs first in the chain), so nothing is lost.
    """

    def __init__(self, iteration: int, best_val: float, patience: int) -> None:
        self.iteration = iteration
        self.best_val = best_val
        self.patience = patience
        super().__init__(
            f"early stop at iter {iteration}: no val improvement over the last "
            f"{patience} evals (best val_loss={best_val:.4f})"
        )


class EarlyStoppingCallback(TrainingCallback):
    """Monitors validation loss and raises :class:`EarlyStop` once it has not
    improved by more than ``min_delta`` for ``patience`` consecutive evals.

    Wraps an ``inner`` callback (progress reporting / experiment logging) and
    delegates to it *before* deciding, so the eval that triggers the stop is
    still reported/logged. A non-positive ``patience`` makes this a pure
    pass-through (never stops).
    """

    def __init__(
        self,
        patience: int,
        *,
        min_delta: float = 0.0,
        inner: Optional[TrainingCallback] = None,
    ) -> None:
        self._patience = int(patience)
        self._min_delta = float(min_delta)
        self._inner = inner
        self._best: float = float("inf")
        self._bad: int = 0

    def on_train_loss_report(self, train_info: dict) -> None:
        if self._inner is not None:
            self._inner.on_train_loss_report(train_info)

    def on_val_loss_report(self, val_info: dict) -> None:
        # Report/log this eval first, then judge it — so the triggering eval
        # still shows up in the tracker and the CLI/Lab progress stream.
        if self._inner is not None:
            self._inner.on_val_loss_report(val_info)

        if self._patience <= 0:
            return

        val_loss = float(val_info.get("val_loss", float("inf")))
        iteration = int(val_info.get("iteration", -1))
        if val_loss < self._best - self._min_delta:
            self._best = val_loss
            self._bad = 0
        else:
            self._bad += 1
            if self._bad >= self._patience:
                raise EarlyStop(iteration, self._best, self._patience)


def promote_best_adapter(adapter_dir: Path) -> bool:
    """Copy ``<adapter_dir>/best/adapters.safetensors`` over the top-level
    ``adapters.safetensors`` so the returned adapter is the best-val snapshot
    rather than the last (possibly overfit) step. Returns True if it copied.

    Used for load-best-at-end on early stop, where the trailing evals that
    tripped the patience counter are by definition worse than the best.
    """
    import shutil

    best_w = Path(adapter_dir) / "best" / "adapters.safetensors"
    top_w = Path(adapter_dir) / "adapters.safetensors"
    if best_w.exists():
        shutil.copy2(best_w, top_w)
        return True
    return False


class BestAdapterCallback(TrainingCallback):
    """Tracks the best validation loss and preserves a copy of the adapter
    whenever a new minimum is reached.

    mlx-lm's trainer already writes the *latest* adapter every
    ``steps_per_save`` iterations. We layer this on top so a bad late
    checkpoint (overfitting / noise) doesn't clobber a good earlier one.

    Output layout (under ``adapter_path``):

        adapters.safetensors        ← last saved (mlx-lm's own cadence)
        best/
            adapters.safetensors    ← best-val copy
            META.txt                ← "val_loss=X at iteration Y"
            train_log.json          ← eval history up to the best point
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_path: Path,
        *,
        inner: Optional[TrainingCallback] = None,
        final_event=None,
        total_iters: Optional[int] = None,
    ) -> None:
        self._model = model
        self._adapter_path = Path(adapter_path)
        self._best_dir = self._adapter_path / "best"
        self._best_dir.mkdir(parents=True, exist_ok=True)
        self._inner = inner
        self._best_val: float = float("inf")
        self._eval_history: list[dict] = []
        self._train_history: list[dict] = []
        # Signalled once the FINAL validation pass completes (best/ is on disk).
        # After this, the only remaining work in mlx-lm's train() is the final
        # top-level adapter save, which can deadlock in Metal on long 8k+ runs.
        # The trainer's exit-watchdog waits on this event to know it is safe to
        # force a clean exit if that save hangs. mlx-lm reports iteration=it-1,
        # and evals at it==iters, so the final eval reports total_iters-1.
        self._final_event = final_event
        self._final_iter = (total_iters - 1) if total_iters else None

    @property
    def best_val(self) -> float:
        """The true minimum validation loss seen so far (the value backing the
        ``best/`` snapshot). Distinct from an EarlyStoppingCallback's internal
        patience anchor, which is min-delta-gated and can read higher."""
        return self._best_val

    # --- delegate print/log behaviour to any wrapped callback -------------
    def on_train_loss_report(self, train_info: dict) -> None:
        self._train_history.append(dict(train_info))
        if self._inner is not None:
            self._inner.on_train_loss_report(train_info)

    def on_val_loss_report(self, val_info: dict) -> None:
        val_loss = float(val_info.get("val_loss", float("inf")))
        iteration = int(val_info.get("iteration", -1))
        self._eval_history.append({"iteration": iteration, "val_loss": val_loss,
                                    "t": time.time()})

        if val_loss < self._best_val:
            self._best_val = val_loss
            self._save_best(iteration, val_loss)

        if self._inner is not None:
            self._inner.on_val_loss_report(val_info)

        # Final eval done -> best/ is on disk. Let the exit-watchdog arm its
        # grace timer for mlx-lm's (possibly deadlocking) final top-level save.
        if (self._final_event is not None and self._final_iter is not None
                and iteration >= self._final_iter):
            self._final_event.set()

    # --- helpers ----------------------------------------------------------
    def _save_best(self, iteration: int, val_loss: float) -> None:
        trainables = dict(tree_flatten(self._model.trainable_parameters()))
        mx.save_safetensors(
            str(self._best_dir / "adapters.safetensors"), trainables
        )
        (self._best_dir / "META.txt").write_text(
            f"val_loss={val_loss:.4f} at iteration {iteration}\n"
        )
        # Mirror the current adapter_config.json into the best/ dir so the
        # best-adapter snapshot is fully self-describing (mlx-lm needs it
        # to load via --adapter-path).
        source_cfg = self._adapter_path / "adapter_config.json"
        if source_cfg.exists():
            (self._best_dir / "adapter_config.json").write_text(
                source_cfg.read_text()
            )
        (self._best_dir / "train_log.json").write_text(
            json.dumps({
                "best_val_loss": self._best_val,
                "best_iteration": iteration,
                "eval_history": self._eval_history,
                "train_history": self._train_history[-200:],
            }, indent=2)
        )
        print(f"  [best] new val minimum {val_loss:.4f} at iter {iteration} → "
              f"saved to {self._best_dir}", flush=True)
