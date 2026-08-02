"""Evidence-grade MTPLX truth report runner."""

from __future__ import annotations

import json
import statistics
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from optiq.runtime.mtp.benchmarks.runners.mtp1_sampler_smoke import run_mtp1_sampler_smoke
from optiq.runtime.mtp.benchmarks.runners.mtp_depth_sweep import run_mtp_depth_sweep
from optiq.runtime.mtp.benchmarks.runners.preflight import run_preflight
from optiq.runtime.mtp.benchmarks.schema import encode_prompt_case, load_prompt_suite, now_run_id
from optiq.runtime.mtp.benchmarks.validators.basic import (
    validate_json_text,
    validate_no_degenerate_loop,
)
from optiq.runtime.mtp.constants import DEFAULT_RUNTIME_MODEL_DIR
from optiq.runtime.mtp.env import collect_environment
from optiq.runtime.mtp.generation import generate_ar
from optiq.runtime.mtp.runtime import load
from optiq.runtime.mtp.sampling import SamplerConfig

DEFAULT_TRUTH_MODES = (
    "ar",
    "mtp1_batched",
    "mtp1_graphbank",
    "d2_batched",
    "d2_graphbank_capture_commit",
    "d2_graphbank_capture_commit_linear_gdn",
    "d2_graphbank_capture_commit_linear_gdn_committed",
    "d2_correction_cache_d2only",
    "d2_c3_blend015",
    "d3_c3_blend015",
)

DEFAULT_C3_CORRECTOR = Path(
    "outputs/correctors/logit-corrector-20260428-012607-c3-logit-r16.npz"
)


def _parse_modes(modes: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if modes is None:
        parsed = list(DEFAULT_TRUTH_MODES)
    elif isinstance(modes, str):
        parsed = [item.strip() for item in modes.split(",") if item.strip()]
    else:
        parsed = [str(item).strip() for item in modes if str(item).strip()]
    unknown = sorted(set(parsed) - set(DEFAULT_TRUTH_MODES))
    if unknown:
        raise ValueError(f"unknown truth modes: {unknown}")
    if not parsed:
        raise ValueError("at least one truth mode is required")
    return parsed


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _validation_counts(rows: list[dict[str, Any]]) -> tuple[int, int]:
    validations = [v for row in rows for v in row.get("validations", [])]
    return (
        sum(1 for item in validations if item.get("passed")),
        len(validations),
    )


def _graphbank_fallback_calls(rows: list[dict[str, Any]]) -> int | None:
    values = []
    for row in rows:
        graphbank = row.get("graphbank") or {}
        if "fallback_calls" in graphbank:
            values.append(int(graphbank["fallback_calls"]))
    return sum(values) if values else None


def _summarize_sampler(result: dict[str, Any]) -> dict[str, Any]:
    summary = dict(result.get("summary") or {})
    rows = list(result.get("rows") or [])
    passed, total = _validation_counts(rows)
    summary.setdefault("validations_passed", passed)
    summary.setdefault("validations_total", total)
    summary["graphbank_fallback_calls"] = _graphbank_fallback_calls(rows)
    return summary


def _summarize_depth(result: dict[str, Any], *, depth: int) -> dict[str, Any]:
    selected = None
    for item in result.get("depths", []):
        if int(item.get("depth", -1)) == int(depth):
            selected = item
            break
    if selected is None:
        raise ValueError(f"depth result {depth} missing from sweep result")
    summary = dict(selected.get("summary") or {})
    rows = list(selected.get("rows") or [])
    passed, total = _validation_counts(rows)
    summary.setdefault("validations_passed", passed)
    summary.setdefault("validations_total", total)
    summary["graphbank_fallback_calls"] = _graphbank_fallback_calls(rows)
    return summary


def _run_ar_mode(
    model_path: Path | str,
    prompt_suite: Path | str,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    seed: int,
    limit: int | None,
    enable_thinking: bool | None,
) -> dict[str, Any]:
    rt = load(model_path, mtp=True)
    prompts = load_prompt_suite(prompt_suite)
    if limit is not None:
        prompts = prompts[:limit]
    sampler = SamplerConfig(temperature=temperature, top_p=top_p, top_k=top_k)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(prompts):
        ids = encode_prompt_case(
            rt.tokenizer,
            case,
            chat_template=True,
            enable_thinking=enable_thinking,
        )
        out = generate_ar(
            rt,
            ids,
            max_tokens=min(max_tokens, case.max_tokens),
            sampler=sampler,
            seed=seed + index,
        )
        validations = [asdict(validate_no_degenerate_loop(out.text))]
        if case.category == "json_tool":
            validations.append(asdict(validate_json_text(out.text.strip())))
        rows.append(
            {
                "prompt_id": case.id,
                "category": case.category,
                "prompt_sha256": case.prompt_sha256,
                "generated_tokens": out.stats.generated_tokens,
                "tok_s": out.stats.tok_s,
                "target_forward_time_s": out.stats.target_forward_time_s,
                "peak_memory_bytes": out.stats.peak_memory_bytes,
                "validations": validations,
                "text": out.text,
                "events": out.stats.events,
            }
        )
    validations_passed, validations_total = _validation_counts(rows)
    return {
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "sampler": asdict(sampler),
        "max_tokens": max_tokens,
        "seed": seed,
        "enable_thinking": enable_thinking,
        "rows": rows,
        "summary": {
            "prompts": len(rows),
            "generated_tokens": sum(row["generated_tokens"] for row in rows),
            "mean_tok_s": _mean([row["tok_s"] for row in rows]) or 0.0,
            "target_forward_time_s": sum(row["target_forward_time_s"] for row in rows),
            "validations_passed": validations_passed,
            "validations_total": validations_total,
            "peak_memory_bytes": max([row["peak_memory_bytes"] for row in rows] or [0]),
        },
    }


def _mode_result_failed(mode_result: dict[str, Any]) -> bool:
    if mode_result.get("error"):
        return True
    summary = mode_result.get("summary") or {}
    return int(summary.get("validations_passed", 0)) != int(summary.get("validations_total", 0))


def run_truth_report(
    *,
    model_path: Path | str = DEFAULT_RUNTIME_MODEL_DIR,
    prompt_suite: Path | str = Path("mtplx/benchmarks/prompts/default.jsonl"),
    modes: str | list[str] | tuple[str, ...] | None = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    draft_temperature: float = 0.0,
    draft_top_p: float | None = None,
    draft_top_k: int = 1,
    max_tokens: int = 96,
    seed: int = 0,
    limit: int | None = 1,
    enable_thinking: bool | None = False,
    mtp_hidden_variant: str = "pre_norm",
    mtp_cache_policy: str = "persistent",
    mtp_history_policy: str = "cycle",
    c3_corrector: Path | str | None = DEFAULT_C3_CORRECTOR,
    c3_blend: float = 0.15,
    project_root: Path | str = ".",
    min_free_gib: float = 120.0,
    cpu_threshold: float = 25.0,
    keep_going: bool = True,
) -> dict[str, Any]:
    selected_modes = _parse_modes(modes)
    run_id = now_run_id("truth")
    started = time.perf_counter()
    preflight = run_preflight(
        project_root,
        cpu_threshold=cpu_threshold,
        min_free_gib=min_free_gib,
    )
    claim_label = "headline_clean" if preflight.get("clean") else "diagnostic_only"
    environment = collect_environment(project_root).to_dict()
    mode_results: list[dict[str, Any]] = []

    def record_mode(name: str, runner) -> None:
        mode_started = time.perf_counter()
        try:
            raw_result, summary = runner()
            mode_results.append(
                {
                    "name": name,
                    "status": "ok",
                    "summary": summary,
                    "raw_result": raw_result,
                    "elapsed_s": time.perf_counter() - mode_started,
                }
            )
        except Exception as exc:
            mode_results.append(
                {
                    "name": name,
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "elapsed_s": time.perf_counter() - mode_started,
                }
            )
            if not keep_going:
                raise

    for mode in selected_modes:
        if mode == "ar":
            record_mode(
                mode,
                lambda: (
                    ar := _run_ar_mode(
                        model_path,
                        prompt_suite,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        max_tokens=max_tokens,
                        seed=seed,
                        limit=limit,
                        enable_thinking=enable_thinking,
                    ),
                    ar["summary"],
                ),
            )
        elif mode == "mtp1_batched":
            record_mode(
                mode,
                lambda: (
                    res := run_mtp1_sampler_smoke(
                        model_path,
                        prompt_suite,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        draft_temperature=draft_temperature,
                        draft_top_p=draft_top_p,
                        draft_top_k=draft_top_k,
                        max_tokens=max_tokens,
                        seed=seed,
                        limit=limit,
                        enable_thinking=enable_thinking,
                        verify_strategy="batched",
                    ),
                    _summarize_sampler(res),
                ),
            )
        elif mode == "mtp1_graphbank":
            record_mode(
                mode,
                lambda: (
                    res := run_mtp1_sampler_smoke(
                        model_path,
                        prompt_suite,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        draft_temperature=draft_temperature,
                        draft_top_p=draft_top_p,
                        draft_top_k=draft_top_k,
                        max_tokens=max_tokens,
                        seed=seed,
                        limit=limit,
                        enable_thinking=enable_thinking,
                        verify_strategy="graphbank",
                    ),
                    _summarize_sampler(res),
                ),
            )
        elif mode in {
            "d2_batched",
            "d2_graphbank_capture_commit",
            "d2_graphbank_capture_commit_linear_gdn",
            "d2_graphbank_capture_commit_linear_gdn_committed",
            "d2_correction_cache_d2only",
            "d2_c3_blend015",
            "d3_c3_blend015",
        }:
            depth = 3 if mode == "d3_c3_blend015" else 2
            verify_strategy = (
                "batched" if mode == "d2_batched" else "graphbank_capture_commit"
            )
            verify_core = (
                "linear-gdn"
                if mode
                in {
                    "d2_graphbank_capture_commit_linear_gdn",
                    "d2_graphbank_capture_commit_linear_gdn_committed",
                    "d2_correction_cache_d2only",
                }
                else "stock"
            )
            history_policy = (
                "committed"
                if mode == "d2_graphbank_capture_commit_linear_gdn_committed"
                else mtp_history_policy
            )
            online_correction_cache = mode == "d2_correction_cache_d2only"
            corrector_path = None
            corrector_blend = None
            if mode in {"d2_c3_blend015", "d3_c3_blend015"}:
                corrector_path = c3_corrector
                corrector_blend = c3_blend

            def depth_runner(
                *,
                depth: int = depth,
                verify_strategy: str = verify_strategy,
                verify_core: str = verify_core,
                corrector_path: Path | str | None = corrector_path,
                corrector_blend: float | None = corrector_blend,
                online_correction_cache: bool = online_correction_cache,
                history_policy: str = history_policy,
                mode_name: str = mode,
            ):
                if mode_name in {"d2_c3_blend015", "d3_c3_blend015"} and (
                    corrector_path is None or not Path(corrector_path).exists()
                ):
                    raise FileNotFoundError(
                        f"C3 corrector artifact is required for {mode_name}: {corrector_path}"
                    )
                res = run_mtp_depth_sweep(
                    model_path,
                    prompt_suite,
                    depths=[depth],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    draft_temperature=draft_temperature,
                    draft_top_p=draft_top_p,
                    draft_top_k=draft_top_k,
                    max_tokens=max_tokens,
                    seed=seed,
                    limit=limit,
                    enable_thinking=enable_thinking,
                    mtp_hidden_variant=mtp_hidden_variant,
                    mtp_cache_policy=mtp_cache_policy,
                    mtp_history_policy=history_policy,
                    verify_strategy=verify_strategy,
                    verify_core=verify_core,
                    mtp_corrector_path=corrector_path,
                    mtp_corrector_blend=corrector_blend,
                    online_correction_cache=online_correction_cache,
                    online_correction_cache_min_depth=2,
                )
                return res, _summarize_depth(res, depth=depth)

            record_mode(
                mode,
                depth_runner,
            )
        else:  # pragma: no cover - guarded by _parse_modes
            raise AssertionError(f"unhandled truth mode: {mode}")

    failed_modes = [item["name"] for item in mode_results if _mode_result_failed(item)]
    return {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "project_root": str(Path(project_root).resolve()),
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "modes": selected_modes,
        "sampler": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
        "draft_sampler": {
            "temperature": draft_temperature,
            "top_p": draft_top_p if draft_top_p is not None else top_p,
            "top_k": draft_top_k,
        },
        "max_tokens": max_tokens,
        "seed": seed,
        "limit": limit,
        "enable_thinking": enable_thinking,
        "mtp_hidden_variant": mtp_hidden_variant,
        "mtp_cache_policy": mtp_cache_policy,
        "mtp_history_policy": mtp_history_policy,
        "c3_corrector": str(c3_corrector) if c3_corrector is not None else None,
        "c3_blend": c3_blend,
        "preflight": preflight,
        "claim_label": claim_label,
        "headline_speed_allowed": bool(preflight.get("clean")),
        "environment": environment,
        "mode_results": mode_results,
        "failed_modes": failed_modes,
        "passed": not failed_modes,
        "elapsed_s": time.perf_counter() - started,
    }


def render_truth_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# MTPLX Truth Report",
        "",
        f"- run id: `{result['run_id']}`",
        f"- created: `{result['created_at']}`",
        f"- model: `{result['model_path']}`",
        f"- prompts: `{result['prompt_suite']}`",
        f"- sampler: `temp={result['sampler']['temperature']} top_p={result['sampler']['top_p']} top_k={result['sampler']['top_k']}`",
        f"- draft sampler: `temp={result['draft_sampler']['temperature']} top_k={result['draft_sampler']['top_k']}`",
        f"- claim label: `{result['claim_label']}`",
        f"- headline speed allowed: `{result['headline_speed_allowed']}`",
        f"- preflight clean: `{result['preflight'].get('clean')}`",
        f"- preflight issues: `{result['preflight'].get('issues')}`",
        "",
        "## Mode Summary",
        "",
        "| mode | status | tok/s | accepted/depth | drafted/depth | corrections | bonus | verify calls | repair s | graphbank fallback | validations |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result.get("mode_results", []):
        summary = item.get("summary") or {}
        validations = (
            f"{summary.get('validations_passed', 0)}/{summary.get('validations_total', 0)}"
            if summary
            else "0/0"
        )
        lines.append(
            "| {name} | {status} | {tok_s} | {accepted} | {drafted} | {corrections} | "
            "{bonus} | {verify_calls} | {repair} | {fallback} | {validations} |".format(
                name=item.get("name"),
                status=item.get("status"),
                tok_s=(
                    "n/a"
                    if summary.get("mean_tok_s") is None
                    else f"{summary.get('mean_tok_s'):.4f}"
                ),
                accepted=summary.get("accepted_by_depth", ""),
                drafted=summary.get("drafted_by_depth", ""),
                corrections=summary.get("correction_tokens", ""),
                bonus=summary.get("bonus_tokens", ""),
                verify_calls=summary.get("verify_calls", ""),
                repair=(
                    ""
                    if summary.get("repair_time_s") is None
                    else f"{summary.get('repair_time_s'):.4f}"
                ),
                fallback=(
                    ""
                    if summary.get("graphbank_fallback_calls") is None
                    else summary.get("graphbank_fallback_calls")
                ),
                validations=validations,
            )
        )
    if result.get("failed_modes"):
        lines.extend(["", "## Failures", ""])
        for item in result.get("mode_results", []):
            if item.get("name") not in set(result["failed_modes"]):
                continue
            lines.append(f"- `{item.get('name')}`: `{item.get('error', 'validation failure')}`")
    lines.append("")
    return "\n".join(lines)


def write_truth_report(
    json_path: Path | str,
    markdown_path: Path | str,
    result: dict[str, Any],
) -> None:
    json_out = Path(json_path)
    markdown_out = Path(markdown_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_truth_markdown(result))
