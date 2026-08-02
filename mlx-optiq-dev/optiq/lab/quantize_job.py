"""Background quantize job — wraps optiq.models.llm.run_llm_pipeline.

Runs in a multiprocessing.Process via lab.jobs.submit. Captures stdout
so the console output streams into the job log, and emits structured
progress events at the major stage boundaries for the wizard's
progress bar.
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from typing import Callable


STAGES = [
    ("resolving", "Downloading model"),
    ("sensitivity", "Running sensitivity analysis"),
    ("optimizing", "Picking per-layer bits"),
    ("converting", "Quantizing to MLX"),
]


def run(emit: Callable[[dict], None], config: dict) -> None:
    """Job entry. ``config`` shape::

        {
          "model_name": "Qwen/Qwen3.5-9B",
          "output_dir": "/path/to/output",
          "target_bpw": 5.0,
          "candidate_bits": [4, 8],
          "reference": "auto",
          "calibration_mix": "optiq",
          "n_calibration": 8,
          "preserve_mtp": true,
        }
    """
    config = dict(config)
    output_dir = config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    emit({"type": "stage", "stage": "resolving",
          "message": "Downloading model…", "progress": 0.05})

    # Tee stdout/stderr into the job log so the user sees live console output
    real_stdout = sys.stdout
    real_stderr = sys.stderr
    log_buf = _StageEmittingStream(emit, real_stdout)
    sys.stdout = log_buf
    sys.stderr = log_buf

    try:
        from optiq.models.llm import run_llm_pipeline
        result = run_llm_pipeline(
            model_name=config["model_name"],
            output_dir=output_dir,
            target_bpw=float(config.get("target_bpw", 5.0)),
            candidate_bits=list(config.get("candidate_bits") or [4, 8]),
            reference=config.get("reference", "auto"),
            calibration_mix=config.get("calibration_mix", "optiq"),
            n_calibration=int(config.get("n_calibration", 8)),
            skip_baselines=True,
        )
    finally:
        sys.stdout = real_stdout
        sys.stderr = real_stderr

    # NB: result from run_llm_pipeline contains SensitivityResult dataclasses
    # that aren't JSON-serializable. Only pass plain primitives to emit().
    summary: dict = {}
    if isinstance(result, dict):
        summary = {
            "output_dir": str(result.get("output_dir") or output_dir),
            "achieved_bpw": float(result["achieved_bpw"]) if "achieved_bpw" in result else None,
            "n_layers": int(result["n_layers"]) if "n_layers" in result else None,
        }
        # Drop None values for cleanliness
        summary = {k: v for k, v in summary.items() if v is not None}
    emit({"type": "stage", "stage": "done",
          "message": "Quantize complete.",
          "progress": 1.0,
          "output_dir": output_dir,
          "summary": summary})


class _StageEmittingStream(io.TextIOBase):
    """Mirror writes to the job log + watch for stage markers in stdout.

    run_llm_pipeline prints headers like ``[1/4] Resolving …`` and
    ``[3/4] Computing layer sensitivity…``. We use those to drive a
    coarse progress bar without modifying the pipeline.
    """

    STAGE_HINTS = [
        ("[2/4]", "sensitivity", "Picking reference mode", 0.10),
        ("[3/4]", "sensitivity", "Computing layer sensitivity", 0.25),
        ("[4/4]", "optimizing", "Picking per-layer bits", 0.65),
        ("Converting", "converting", "Quantizing to MLX", 0.80),
    ]

    def __init__(self, emit, real):
        self.emit = emit
        self.real = real
        self._buf = ""

    def writable(self):
        return True

    def write(self, s: str) -> int:
        # Always pass through so dev mode still sees output
        try:
            self.real.write(s)
            self.real.flush()
        except Exception:
            pass
        self._buf += s
        if "\n" in self._buf:
            lines, self._buf = self._buf.rsplit("\n", 1)
            for line in lines.splitlines():
                self._handle_line(line)
        return len(s)

    def _handle_line(self, line: str) -> None:
        self.emit({"type": "log", "line": line})
        for marker, stage, message, progress in self.STAGE_HINTS:
            if marker in line:
                self.emit({
                    "type": "stage", "stage": stage,
                    "message": message, "progress": progress,
                })
                break

    def flush(self) -> None:
        try:
            self.real.flush()
        except Exception:
            pass
