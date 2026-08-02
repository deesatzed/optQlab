"""Background eval job for Lab (WP-3).

Suites:
  * ``byo`` — user prompt set; generates with mlx_lm when model loads, scores
    with exact/substring match (real answers, real expected strings).
  * ``gsm8k-50`` / ``smoketest`` / single tasks — subprocess ``optiq eval``
    with ``--output-json`` (real CLI evaluation).

Config::

    {
      "model_path": "/path/or/hf-id",
      "suite": "byo" | "gsm8k-50" | "smoketest" | "mmlu" | ...,
      "eval_set_id": "evset_..."   # required for byo
      "n_samples": 50,             # optional override
      "build_id": "bld_...",       # optional
    }
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

from . import eval_service
from .config import ensure_lab_dirs


def run(emit: Callable[[dict], None], config: dict) -> None:
    config = dict(config)
    model_path = (config.get("model_path") or "").strip()
    suite = (config.get("suite") or "gsm8k-50").strip()
    if not model_path:
        raise ValueError("model_path required")

    emit({"type": "stage", "stage": "start", "message": f"Eval {suite}", "progress": 0.05})

    if suite == "byo":
        scores = _run_byo(emit, config, model_path)
    else:
        scores = _run_cli_eval(emit, config, model_path, suite)

    build_id = config.get("build_id") or eval_service.resolve_build_id_for_path(model_path)
    # job_id is not in config by default; optional for metadata linkage
    job_id = config.get("_job_id") or config.get("job_id")
    eid = eval_service.store_eval_result(
        build_id=build_id,
        model_path=model_path,
        suite=suite,
        scores=scores,
        job_id=job_id,
        eval_set_id=config.get("eval_set_id"),
        metadata={"source": "lab_eval_job"},
    )
    emit({
        "type": "result",
        "message": f"Stored {eid}",
        "progress": 1.0,
        "eval_id": eid,
        "capability_score": scores.get("capability_score"),
    })


def _run_cli_eval(emit, config, model_path: str, suite: str) -> dict:
    paths = ensure_lab_dirs()
    out_path = paths.cache_dir / f"eval_{suite}_{Path(model_path).name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "optiq.cli", "eval", model_path,
        "--task", suite if suite != "smoketest" else "smoketest",
        "--output-json", str(out_path),
    ]
    if suite == "smoketest":
        pass
    elif config.get("n_samples") is not None:
        cmd.extend(["--n-samples", str(int(config["n_samples"]))])
    if config.get("skip_kl"):
        cmd.append("--skip-kl")
    if suite in ("all",) or config.get("show_score"):
        cmd.append("--score")

    emit({"type": "stage", "stage": "cli", "message": " ".join(cmd[-6:]), "progress": 0.1})

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    # Stream tail of stdout into log via emit
    for line in (proc.stdout or "").splitlines()[-40:]:
        emit({"type": "log", "message": line})
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "eval failed")[-2000:]
        raise RuntimeError(f"optiq eval failed ({proc.returncode}): {err}")

    if not out_path.is_file():
        # smoketest may not write JSON — synthesize minimal record from stdout parse fail
        raise RuntimeError(
            f"eval produced no JSON at {out_path}; use a task with --output-json support "
            "(e.g. gsm8k-50, mmlu) or BYO suite"
        )

    emit({"type": "stage", "stage": "parse", "message": "Parsing results", "progress": 0.9})
    return eval_service.parse_cli_eval_json(out_path)


def _run_byo(emit, config, model_path: str) -> dict:
    set_id = config.get("eval_set_id")
    if not set_id:
        raise ValueError("eval_set_id required for byo suite")
    eset = eval_service.load_eval_set(set_id)
    if not eset:
        raise ValueError(f"eval set not found: {set_id}")

    items = eset.get("items") or []
    emit({
        "type": "stage", "stage": "generate",
        "message": f"Scoring {len(items)} BYO prompts",
        "progress": 0.15,
    })

    # Generate with mlx_lm — real inference
    from mlx_lm import load, generate

    emit({"type": "log", "message": f"Loading model {model_path}"})
    model, tokenizer = load(model_path)

    pairs = []
    n = len(items)
    for i, it in enumerate(items):
        prompt = it["prompt"]
        # Simple chat-style completion
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                text = prompt
        else:
            text = prompt
        actual = generate(
            model, tokenizer, prompt=text, max_tokens=int(config.get("max_tokens") or 128),
            verbose=False,
        )
        if isinstance(actual, list):
            actual = "".join(actual)
        pairs.append({
            "id": it.get("id"),
            "prompt": prompt,
            "expected": it.get("expected") or "",
            "actual": str(actual),
        })
        emit({
            "type": "progress",
            "progress": 0.15 + 0.75 * ((i + 1) / max(n, 1)),
            "message": f"BYO {i+1}/{n}",
        })

    return eval_service.score_byo_pairs(pairs)
