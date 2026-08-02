"""Background LoRA training job — wraps optiq.lora.trainer.train_lora.

mlx-lm's trainer exposes a TrainingCallback class with on_train_loss /
on_val_loss hooks. We plug into both to emit live loss + grad-norm
events that the Fine-tune wizard renders as a Chart.js line plot.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable


def run(emit: Callable[[dict], None], config: dict) -> None:
    """Job entry. ``config`` shape::

        {
          "model_dir": "/path/to/optiq-quant",
          "data_dir": "/path/to/dataset",
          "adapter_path": "/path/to/output/adapters",
          "rank": 8,
          "scale": 20.0,
          "rank_scaling": "by_bits",
          "target_modules": ["q_proj", "v_proj"],
          "num_layers": 16,
          "iters": 500,
          "batch_size": 1,
          "learning_rate": 1e-4,
          "max_seq_length": 1024,
          "grad_accumulation_steps": 1,
        }
    """
    emit({"type": "stage", "stage": "load",
          "message": "Loading model + dataset…", "progress": 0.02})

    from optiq.lora.config import OptiqLoraConfig

    # mlx-lm's trainer takes a TrainingCallback subclass — define one inline
    # that re-emits onto our job event bus. Used by SFT only; DPO has its
    # own callback signature (step, metrics_dict).
    from mlx_lm.tuner.callbacks import TrainingCallback

    iters = int(config.get("iters", 500))
    method = (config.get("method") or "sft").lower()

    class _LabCallback(TrainingCallback):
        """SFT callback. mlx-lm calls on_train_loss_report / on_val_loss_report."""

        def __init__(self):
            super().__init__()
            self.started = time.time()

        def on_train_loss_report(self, info):  # type: ignore[override]
            step = int(info.get("iteration", 0))
            emit({
                "type": "metric",
                "kind": "train",
                "step": step,
                "loss": float(info.get("train_loss", 0.0)),
                "grad_norm": float(info.get("grad_norm", 0.0)),
                "learning_rate": float(info.get("learning_rate", 0.0)),
                "tokens_per_sec": float(info.get("tokens_per_sec", 0.0)),
                "progress": min(0.9, 0.05 + (step / max(iters, 1)) * 0.85),
                "message": f"step {step}/{iters} · loss {info.get('train_loss', 0.0):.3f}",
            })

        def on_val_loss_report(self, info):  # type: ignore[override]
            emit({
                "type": "metric",
                "kind": "val",
                "step": int(info.get("iteration", 0)),
                "loss": float(info.get("val_loss", 0.0)),
            })

    def _dpo_progress_cb(step: int, metrics: dict) -> None:
        """DPO callback. optiq.lora.dpo.train_dpo invokes (step, metrics)."""
        emit({
            "type": "metric",
            "kind": "train",
            "step": step,
            "loss": float(metrics.get("loss", 0.0)),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "margin": float(metrics.get("margin", 0.0)),
            "progress": min(0.9, 0.05 + (step / max(iters, 1)) * 0.85),
            "message": (
                f"step {step}/{iters} · loss {metrics.get('loss', 0.0):.3f}"
                f" · acc {metrics.get('accuracy', 0.0):.2f}"
                f" · margin {metrics.get('margin', 0.0):+.3f}"
            ),
        })

    # Vision (image+text) LoRA uses optiq.vlm.lora, not the OptiqLoraConfig /
    # mlx-lm SFT path, so it builds its own kwargs below.
    if method != "vision":
        cfg_kwargs = {
            k: v for k, v in config.items()
            if k not in {"model_dir", "data_dir"} and v is not None
        }
        # target_modules JSON-array → tuple
        if isinstance(cfg_kwargs.get("target_modules"), list):
            cfg_kwargs["target_modules"] = tuple(cfg_kwargs["target_modules"])
        optiq_cfg = OptiqLoraConfig(**cfg_kwargs)

    # Tee stdout so a log scroll is available
    real_stdout = sys.stdout
    real_stderr = sys.stderr
    sys.stdout = _LineEmitter(emit, real_stdout)
    sys.stderr = sys.stdout
    try:
        if method == "vision":
            import os as _os

            from optiq.vlm.lora import train_vlm_lora

            data_path = config["data_dir"]
            if _os.path.isdir(data_path):
                data_path = _os.path.join(data_path, "train.jsonl")
            adapter_path = (config.get("adapter_path")
                            or _os.path.join(config.get("output_dir") or ".",
                                             "vlm_adapter"))

            def _vlm_cb(info: dict) -> None:
                step = int(info.get("step", 0))
                total = int(info.get("iters", iters))
                loss = float(info.get("loss", 0.0))
                emit({"type": "metric", "kind": "train", "step": step,
                      "loss": loss,
                      "progress": min(0.9, 0.05 + step / max(total, 1) * 0.85),
                      "message": f"step {step}/{total} · loss {loss:.3f}"})

            out = train_vlm_lora(
                config["model_dir"], data_path, adapter_path,
                rank=int(config.get("rank") or 8),
                scale=float(config.get("scale") or 8.0),
                iters=iters,
                learning_rate=float(config.get("learning_rate") or 2e-4),
                image_size=int(config.get("image_size") or 512),
                max_target=int(config.get("max_seq_length") or 256),
                report_every=int(config.get("steps_per_report") or 20),
                grad_checkpoint=bool(config.get("grad_checkpoint", True)),
                progress_callback=_vlm_cb,
            )
            result = {"adapter_path": out}
        elif method == "dpo":
            from optiq.lora.dpo import train_dpo
            result = train_dpo(
                model_dir=config["model_dir"],
                data_dir=config["data_dir"],
                config=optiq_cfg,
                progress_callback=_dpo_progress_cb,
            )
        else:
            from optiq.lora.trainer import train_lora
            result = train_lora(
                model_dir=config["model_dir"],
                data_dir=config["data_dir"],
                config=optiq_cfg,
                progress_callback=_LabCallback(),
            )
    finally:
        sys.stdout = real_stdout
        sys.stderr = real_stderr

    _stopped = bool(result.get("early_stopped"))
    emit({
        "type": "stage", "stage": "done",
        "progress": 1.0,
        "message": ("Training complete (early-stopped; best adapter kept)."
                    if _stopped else "Training complete."),
        "adapter_path": result.get("adapter_path"),
        "applied_ranks": result.get("applied_ranks", {}),
        "early_stopped": _stopped,
    })


class _LineEmitter:
    def __init__(self, emit, real):
        self.emit = emit
        self.real = real
        self._buf = ""

    def write(self, s):
        try:
            self.real.write(s)
            self.real.flush()
        except Exception:
            pass
        self._buf += s
        if "\n" in self._buf:
            lines, self._buf = self._buf.rsplit("\n", 1)
            for line in lines.splitlines():
                if line:
                    self.emit({"type": "log", "line": line})
        return len(s)

    def flush(self):
        try:
            self.real.flush()
        except Exception:
            pass
