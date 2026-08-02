"""Handlers for the product-facing MTPLX CLI surface."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import importlib
import importlib.metadata
import importlib.util
import re
import webbrowser
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from optiq.runtime.mtp.artifacts import inspect_model
from optiq.runtime.mtp.benchmarks.validators.basic import (
    validate_balanced_delimiters,
    validate_no_degenerate_loop,
    validate_python_syntax,
)
from optiq.runtime.mtp.constants import DEFAULT_RUNTIME_MODEL_DIR
from optiq.runtime.mtp.default_models import (
    OPTIMIZED_QUALITY_DESCRIPTION,
    is_verified_default_model_ref,
    optimized_quality_model_ref,
    public_model_id_for_ref,
    select_default_model,
)
from optiq.runtime.mtp.env import collect_environment
from optiq.runtime.mtp.kpi import (
    EXIT_EXACTNESS,
    EXIT_QUALITY,
    EXIT_STRICT_GATE,
    EXIT_TELEMETRY,
    EXIT_UNSUPPORTED_MODEL,
    build_benchmark_envelope,
    default_output_path,
    exact_paged_attention_env,
    prompt_suite_path,
    run_exactness_smoke,
    summarize_vllm_reference,
    write_json,
)
from optiq.runtime.mtp.kpi.runtime_kpis import (
    distribution_suite_names,
    repo_root,
)
from optiq.runtime.mtp.backends.registry import (
    TIER_ARCH_COMPATIBLE_UNVERIFIED,
    architecture_catalog,
)
from optiq.runtime.mtp.profiles import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROFILE_NAME,
    DEFAULT_PUBLIC_MODEL_ID,
    apply_profile_env,
    get_profile,
)
from optiq.runtime.mtp.server_urls import (
    bind_label,
    connect_host_for_bind,
    is_wildcard_bind,
    local_url_for_bind,
)


DEFAULT_CHAMPION = "models/Qwen3.6-27B-MTPLX-Optimized-Speed"
QUICKSTART_SPEED_MIN_TOKENS = 64
QUICKSTART_SPEED_MAX_TOKENS = 192
QUICKSTART_SPEED_PROMPT = (
    "Create a compact single-file HTML5 Canvas Flappy Bird game. "
    "Draw visuals procedurally, include physics, score, restart, and no prose."
)
QUICKSTART_TARGETS = {"terminal", "openwebui", "open-webui", "pi", "opencode", "swival"}
LONG_RESPONSE_DIRECT_PROFILE = (
    "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_"
    "partition_threshold_2048_impl_mlx_vector_paged"
)
BENCH_SUSTAINED_DEFAULT_SUITES = {
    "flappy",
    "long_code_uncapped",
    "long-code-uncapped",
    "python_modules_long",
    "python-modules-long",
}
BENCH_SUSTAINED_LENGTH_SENSITIVE_SUITES = {"long_code", "long-code"}
BENCH_SUSTAINED_MAX_TOKENS_THRESHOLD = 512
EXTERNAL_RUNTIME_ENV_KEYS = (
    "MTPLX_VERIFY_OUTPUT_DEPENDS",
    "MTPLX_VERIFY_OUTPUT_DEPENDS_AFTER_TOKENS",
    "MTPLX_VERIFY_OUTPUT_DEPENDS_EVERY",
    "MTPLX_VERIFY_OUTPUT_DEPENDS_INCLUDE_MTP",
    "MTPLX_STATE_ROOT_EVAL_MODE",
    "MTPLX_EVAL_STATE_ROOTS_INCLUDE_MTP",
    "MTPLX_EVAL_STATE_ROOTS_INCLUDE_LIVE",
    "MTPLX_TARGET_LAYER_EVAL_MODE",
    "MTPLX_FUSE_ATTN_QKV_PROJECTIONS",
    "MTPLX_SDPA_2PASS_BLOCKS",
    "MTPLX_SDPA_DYNAMIC_OFFSET_ACTIVE_BLOCKS",
    "MTPLX_EXPORT_VERIFY_DOT_DIR",
    "MTPLX_EXPORT_VERIFY_DOT_CYCLES",
    "MTPLX_EXPORT_VERIFY_DOT_INCLUDE_CACHE",
    "MTPLX_EXPORT_VERIFY_DOT_INCLUDE_CAPTURES",
)
LOCALHOST_BINDS = {"", "127.0.0.1", "::1", "localhost"}
MAX_PUBLIC_SPECULATIVE_DEPTH = 3
TUNE_DEFAULT_DEPTHS = "1,2,3"
TUNE_DEFAULT_SUITE = "cold-long-code-192"
TUNE_DEFAULT_MAX_TOKENS = 192
TUNE_DEFAULT_LIMIT = 1
TUNE_DEFAULT_SEED = 0
TUNE_STATE_PATH = Path("~/.mtplx/tuning.json").expanduser()
TUNE_TELEMETRY_SAMPLE_INTERVAL_S = 0.75
TUNE_POWERMETRICS_SAMPLE_INTERVAL_S = 1.5
TUNE_CANDIDATE_SETTLE_S = 5.0
TUNE_TIE_PREFER_DEEPER_WITHIN_PCT = 2.0
TUNE_TELEMETRY_ENV = "MTPLX_BENCH_TUNE_TELEMETRY"
GENERATION_MODE_MTP = "mtp"
GENERATION_MODE_AR = "ar"
GENERATION_MODES = {GENERATION_MODE_MTP, GENERATION_MODE_AR}


def _absolute_user_path(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return (Path.cwd() / expanded).resolve()


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _is_localhost_bind(host: str | None) -> bool:
    return str(host or "").strip().lower().strip("[]") in LOCALHOST_BINDS


def _benchmark_seed(args: Any, *, runtime_profile: str, harness: str) -> int:
    explicit = getattr(args, "seed", None)
    if explicit is not None:
        return int(explicit)
    if runtime_profile == "native_mtp_60_cold" or harness == "depth-sweep":
        return 0
    return 42


def _depths_for_bench_run(args: Any) -> str:
    explicit_depths = getattr(args, "depths", None)
    if explicit_depths:
        return str(explicit_depths)
    depth = int(getattr(args, "speculative_depth", 0) or 0)
    return str(depth) if depth > 0 else "3"

def _runtime_env_with_external_overrides(runtime_env: dict[str, str]) -> dict[str, str]:
    merged = dict(runtime_env)
    for key in EXTERNAL_RUNTIME_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None and value != "":
            merged[key] = value
    return merged


def _bench_run_console_summary(envelope: dict[str, Any]) -> dict[str, Any]:
    runtime = envelope.get("runtime") or {}
    trace = envelope.get("decode_trace") or {}
    quality = envelope.get("quality") or {}
    correctness = envelope.get("correctness") or {}
    smoke = correctness.get("exactness_smoke") or {}
    return {
        "run_id": envelope.get("run_id"),
        "harness": envelope.get("harness") or "depth-sweep",
        "runtime_profile": envelope.get("runtime_profile"),
        "tok_s": runtime.get("tok_s") or runtime.get("mean_tok_s"),
        "generated_tokens": runtime.get("generated_tokens") or trace.get("generated_tokens"),
        "first64_tok_s": trace.get("first64_tok_s"),
        "last64_tok_s": trace.get("last64_tok_s"),
        "last64_over_first64": trace.get("last64_over_first64"),
        "last10_over_first10": trace.get("last10_over_first10"),
        "late_verify_ms": trace.get("late_verify_ms") or runtime.get("late_verify_ms"),
        "quality_passed": quality.get("passed"),
        "quality_failures": quality.get("failures") or [],
        "exactness_smoke_passed": smoke.get("passed"),
        "strict_passed": envelope.get("strict_passed"),
        "artifacts": envelope.get("artifacts") or {},
    }


def _model_gate(
    model: str,
    *,
    unsafe_force_unverified: bool = False,
    yes: bool = False,
) -> tuple[dict[str, Any], int | None]:
    try:
        inspection = inspect_model(model).to_dict()
    except Exception as exc:
        return {"error": "inspect failed", "model": model, "detail": str(exc)}, 1
    compatibility = inspection.get("compatibility") or {}
    tier = compatibility.get("tier")
    exit_code = int(compatibility.get("exit_code", EXIT_UNSUPPORTED_MODEL))
    if exit_code == 0 and compatibility.get("can_run"):
        if compatibility.get("unverified_model"):
            print(
                "WARNING: running a family-compatible MTPLX model without a "
                "recorded mtplx_runtime.json exactness baseline; stats will be "
                "marked unverified until this artifact is smoke-verified.",
                file=sys.stderr,
            )
        return inspection, None
    if (
        unsafe_force_unverified
        and yes
        and tier == TIER_ARCH_COMPATIBLE_UNVERIFIED
        and compatibility.get("unsafe_force_required")
    ):
        print(
            "WARNING: running an architecture-compatible but unverified MTPLX model; "
            "exactness and performance are not guaranteed.",
            file=sys.stderr,
        )
        return inspection, None
    return inspection, exit_code


def _compact_model_summary(inspection: dict[str, Any]) -> dict[str, Any]:
    compatibility = inspection.get("compatibility") or {}
    return {
        "source": inspection.get("source"),
        "model_dir": inspection.get("model_dir"),
        "architecture": inspection.get("architecture"),
        "model_type": inspection.get("model_type"),
        "mtp_arch": inspection.get("mtp_arch"),
        "mtp_supported": inspection.get("mtp_supported"),
        "recommended_backend": inspection.get("recommended_backend"),
        "recommended_profile": (
            compatibility.get("recommended_profile")
            or inspection.get("recommended_profile")
        ),
        "runtime_compatibility": inspection.get("runtime_compatibility"),
        "runtime_contract_path": (
            compatibility.get("runtime_contract_path")
            or inspection.get("runtime_contract_path")
        ),
        "compatibility": {
            "tier": compatibility.get("tier"),
            "can_run": compatibility.get("can_run"),
            "supported": compatibility.get("supported"),
            "recognized": compatibility.get("recognized"),
            "exit_code": compatibility.get("exit_code"),
            "message": compatibility.get("message"),
            "arch_id": compatibility.get("arch_id"),
            "unsafe_force_required": compatibility.get("unsafe_force_required"),
            "unverified_model": compatibility.get("unverified_model"),
            "runtime_compatibility": compatibility.get("runtime_compatibility"),
            "support_level": compatibility.get("support_level"),
            "support_notes": compatibility.get("support_notes"),
        },
    }


def _model_gate_error_lines(inspection: dict[str, Any]) -> list[str]:
    compatibility = inspection.get("compatibility") or {}
    mtp = inspection.get("mtp") or {}
    lines = [
        "error: model cannot run with MTPLX",
        f"model: {inspection.get('model_dir') or inspection.get('model') or 'unknown'}",
        f"tier: {compatibility.get('tier') or 'unknown'}",
    ]
    runtime_compatibility = compatibility.get("runtime_compatibility") or inspection.get("runtime_compatibility")
    mtp_layers = int(inspection.get("mtp_num_hidden_layers") or 0)
    missing_mtp_weights = (
        mtp_layers > 0
        and bool(mtp)
        and not bool(mtp.get("exists"))
        and compatibility.get("tier") != "no-MTP"
    )
    if missing_mtp_weights:
        runtime_compatibility = "missing-mtp-weights"
    if runtime_compatibility:
        lines.append(f"runtime: {runtime_compatibility}")
    if missing_mtp_weights:
        message = (
            "This model's config advertises MTP, but MTPLX did not find "
            "runnable MTP weights in the folder. mtplx_runtime.json is "
            "optional metadata; the blocker is missing MTP weights."
        )
    else:
        message = compatibility.get("message") or inspection.get("detail")
    if message:
        lines.append(f"reason: {message}")
    if mtp:
        lines.append(
            "mtp: "
            f"exists={str(bool(mtp.get('exists'))).lower()}, "
            f"tensors={mtp.get('tensor_count', 0)}, "
            f"gate={str(bool(mtp.get('passes_tensor_gate'))).lower()}"
        )
    if runtime_compatibility == "missing-mtp-weights":
        lines.append(
            "fix: choose a model with real MTP weights, or graft an MTP sidecar "
            "into this base model."
        )
    elif compatibility.get("unsafe_force_required"):
        lines.append("try: add --unsafe-force-unverified --yes to run without support guarantees")
    else:
        lines.append("try: mtplx inspect MODEL")
    return lines


def _print_model_gate_error(
    inspection: dict[str, Any],
    *,
    printer=print,
    json_output: bool = False,
) -> None:
    if json_output:
        _print(
            {
                "error": "model failed MTPLX compatibility gate",
                "model": _compact_model_summary(inspection),
            }
        )
        return
    for line in _model_gate_error_lines(inspection):
        printer(line)


def _print_command_error(
    payload: dict[str, Any],
    *,
    command: str,
    json_output: bool = False,
) -> None:
    if json_output:
        _print(payload)
        return
    model = payload.get("model")
    if isinstance(model, dict):
        _print_model_gate_error(model, json_output=False)
        return
    error = str(payload.get("error") or "model is not available locally")
    print(f"error: {error}")
    if model:
        print(f"model: {model}")
    detail = payload.get("detail")
    if detail:
        print(f"detail: {detail}")
    if error == "model is not available locally":
        print(f"try: mtplx {command} --download --model {model}")
        print("try: mtplx models")


def _validate_public_depth(args: Any, *, printer=print) -> int | None:
    try:
        depth = int(getattr(args, "depth", 3))
    except (TypeError, ValueError):
        printer("error: --depth must be an integer")
        return 2
    if depth < 1 or depth > MAX_PUBLIC_SPECULATIVE_DEPTH:
        printer(
            "error: --depth must be between "
            f"1 and {MAX_PUBLIC_SPECULATIVE_DEPTH} for the current MTPLX runtime"
        )
        printer("hint: omit --depth to use the model contract default")
        return 2
    args.depth = depth
    return None


def _normalize_generation_mode(value: Any) -> str:
    text = str(value or GENERATION_MODE_MTP).strip().lower()
    if text not in GENERATION_MODES:
        raise ValueError("generation mode must be 'mtp' or 'ar'")
    return text


def _generation_mode_from_args(args: Any) -> str:
    if bool(getattr(args, "stock_ar", False)):
        return GENERATION_MODE_AR
    explicit = getattr(args, "generation_mode", None)
    if explicit is not None:
        return _normalize_generation_mode(explicit)
    return GENERATION_MODE_AR if bool(getattr(args, "no_mtp", False)) else GENERATION_MODE_MTP


def _set_generation_mode_on_args(args: Any, mode: str) -> None:
    normalized = _normalize_generation_mode(mode)
    setattr(args, "generation_mode", normalized)
    setattr(args, "no_mtp", normalized == GENERATION_MODE_AR)


def _generation_mode_label(mode: str) -> str:
    return "AR target-only" if mode == GENERATION_MODE_AR else "MTP"


def _format_bytes(size_bytes: int | float | None) -> str:
    if not isinstance(size_bytes, (int, float)):
        return "unknown size"
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1000.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1000.0
    return f"{size:.1f} TB"


def _download_progress_callback(*, printer=print):
    """Plain-text download progress callback for non-interactive paths."""

    def emit(event: dict[str, Any]) -> None:
        kind = event.get("event")
        size = _format_bytes(event.get("size_bytes"))
        if kind == "start":
            printer(f"[1/4] Download started: {event.get('repo_id')} -> {event.get('path')}")
        elif kind == "resume":
            printer(
                "[1/4] Resuming partial download: "
                f"{event.get('repo_id')} ({size} already on disk)"
            )
        elif kind == "progress":
            interval = max(1, int(round(float(event.get("interval_s") or 0))))
            delta = float(event.get("delta_bytes") or 0)
            if delta > 0:
                printer(
                    "[1/4] Still downloading: "
                    f"{size} on disk (+{_format_bytes(delta)} in {interval}s)"
                )
            else:
                printer(
                    "[1/4] Still downloading: "
                    f"{size} on disk (no byte change in {interval}s; waiting on Hugging Face)"
                )
        elif kind == "complete":
            printer(f"[1/4] Download complete: {size} on disk")

    return emit


def _rich_download_progress_callback(*, repo_id: str, total_bytes: int | None = None):
    """Single-line live progress callback for interactive downloads."""

    from optiq.runtime.mtp.ui.download_progress import RichDownloadProgress, from_progress_event_callback

    progress = RichDownloadProgress(repo_id=repo_id, total_bytes=total_bytes)
    callback = from_progress_event_callback(progress=progress)

    def finalize() -> None:
        progress.stop()

    return callback, finalize


def _profile_draft_lm_head_spec(profile: Any) -> dict[str, Any] | None:
    draft = getattr(profile, "draft_lm_head", None)
    if draft is None:
        return None
    return {
        "bits": int(draft.bits),
        "group_size": int(draft.group_size),
        "mode": str(draft.mode),
    }


def _profile_draft_sampler_spec(profile: Any) -> dict[str, Any] | None:
    draft = getattr(profile, "draft_sampler", None)
    if draft is None:
        return None
    return {
        "temperature": float(draft.temperature),
        "top_p": float(draft.top_p),
        "top_k": int(draft.top_k),
    }


def _model_draft_lm_head_spec(
    inspection: dict[str, Any],
    profile: Any,
) -> dict[str, Any] | None:
    """Use model contract draft-head metadata when present, else profile default."""
    fallback = _profile_draft_lm_head_spec(profile)
    try:
        from optiq.runtime.mtp.draft_lm_head import draft_lm_head_spec_from_runtime_contract

        compatibility = inspection.get("compatibility") or {}
        contract = compatibility.get("runtime_contract")
        return draft_lm_head_spec_from_runtime_contract(contract, fallback=fallback)
    except ImportError:
        return fallback


def _model_draft_sampler_spec(
    inspection: dict[str, Any],
    profile: Any,
) -> dict[str, Any] | None:
    """Use model contract draft-sampler metadata when present, else profile default."""
    fallback = _profile_draft_sampler_spec(profile)
    try:
        from optiq.runtime.mtp.draft_sampling import draft_sampler_spec_from_runtime_contract

        compatibility = inspection.get("compatibility") or {}
        contract = compatibility.get("runtime_contract")
        return draft_sampler_spec_from_runtime_contract(contract, fallback=fallback)
    except ImportError:
        return fallback


def _model_contract_depth(inspection: dict[str, Any], *, fallback: int = 3) -> int:
    compatibility = inspection.get("compatibility") or {}
    contract = (
        compatibility.get("runtime_contract")
        if isinstance(compatibility, dict)
        else inspection.get("runtime_contract")
    )
    if not isinstance(contract, dict):
        return int(fallback)
    try:
        depth = int(contract.get("mtp_depth_max", fallback))
    except (TypeError, ValueError):
        return int(fallback)
    return max(1, min(MAX_PUBLIC_SPECULATIVE_DEPTH, depth))


def _apply_model_contract_depth_default(args: Any, inspection: dict[str, Any]) -> None:
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if "depth" in cli_flags:
        return
    args.depth = _model_contract_depth(
        inspection,
        fallback=int(getattr(args, "depth", 3)),
    )


def _draft_sampler_from_spec(spec: dict[str, Any] | None) -> Any | None:
    if spec is None:
        return None
    from optiq.runtime.mtp.sampling import SamplerConfig

    return SamplerConfig(
        temperature=float(spec["temperature"]),
        top_p=float(spec["top_p"]),
        top_k=int(spec["top_k"]),
    )


def _resolve_model_context_window(tokenizer: Any, model_path: str | Path) -> int:
    candidates: list[int] = []
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int):
        candidates.append(tokenizer_max)

    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = {}
        for section in (config, config.get("text_config", {})):
            if not isinstance(section, dict):
                continue
            for key in (
                "max_position_embeddings",
                "model_max_length",
                "max_sequence_length",
                "seq_length",
                "context_length",
            ):
                value = section.get(key)
                if isinstance(value, int):
                    candidates.append(value)

    sane = [value for value in candidates if 0 < value <= 1_000_000]
    return max(sane) if sane else 262_144


def _cli_generation_budget(
    *,
    tokenizer: Any,
    model_path: str | Path,
    prompt_token_count: int,
    explicit_max_tokens: int | None,
) -> dict[str, int | bool | None]:
    context_window = _resolve_model_context_window(tokenizer, model_path)
    remaining_context = max(1, int(context_window) - max(0, int(prompt_token_count)))
    requested_max = remaining_context
    request_max_tokens = None
    if explicit_max_tokens is not None:
        request_max_tokens = int(explicit_max_tokens)
        if request_max_tokens < 1:
            raise ValueError("--max-tokens must be >= 1; omit it to use the model context")
        requested_max = request_max_tokens
    effective_max = max(1, min(requested_max, remaining_context))
    return {
        "request_max_tokens": request_max_tokens,
        "effective_max_tokens": int(effective_max),
        "context_window": int(context_window),
        "remaining_context_tokens": int(remaining_context),
        "context_cap_applied": bool(effective_max < requested_max),
    }


def _reasoning_mode(args: Any, *, default: str = "on") -> str:
    raw = getattr(args, "reasoning", None)
    mode = str(raw or default).strip().lower()
    if mode not in {"auto", "on", "off"}:
        return default
    return mode


def _preserve_thinking_policy(args: Any) -> str:
    if bool(getattr(args, "strip_assistant_reasoning_history", False)):
        return "off"
    raw = getattr(args, "preserve_thinking", None)
    mode = str(raw or "auto").strip().lower()
    return mode if mode in {"auto", "on", "off"} else "auto"


def _enable_thinking_for_reasoning(mode: str) -> bool | None:
    if mode == "auto":
        return None
    return mode == "on"


def _redact_secret_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in ("token", "api_key", "apikey", "auth", "secret", "password")):
                redacted[str(key)] = "[redacted]"
            else:
                redacted[str(key)] = _redact_secret_value(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secret_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_secret_value(item) for item in value]
    if isinstance(value, str) and len(value) > 12:
        lowered = value.lower()
        if any(marker in lowered for marker in ("hf_", "bearer ", "api-key", "password", "secret")):
            return "[redacted]"
    return value


def _write_json_redacted(path: Path, value: Any) -> None:
    write_json(path, _redact_secret_value(value))


def _deep_doctor_report(args: Any, base: dict[str, Any]) -> dict[str, Any]:
    from optiq.runtime.mtp.config import load_user_config

    config = load_user_config()
    default_model = config.model or DEFAULT_CHAMPION
    model_report: dict[str, Any]
    try:
        model_report = _compact_model_summary(inspect_model(default_model).to_dict())
    except Exception as exc:
        model_report = {
            "model": default_model,
            "ok": False,
            "error": str(exc),
        }
    server_deps = {
        name: importlib.util.find_spec(name) is not None
        for name in ("fastapi", "uvicorn")
    }
    launchers = {
        "path_lookup": shutil.which("mtplx"),
        "homebrew_bin": {
            "path": "/opt/homebrew/bin/mtplx",
            "exists": Path("/opt/homebrew/bin/mtplx").exists(),
        },
        "local_bin": {
            "path": str(Path.home() / ".local/bin/mtplx"),
            "exists": (Path.home() / ".local/bin/mtplx").exists(),
        },
    }
    base["deep"] = {
        "config": config.to_dict(),
        "default_model": model_report,
        "server_dependencies": server_deps,
        "launchers": launchers,
        "integrations": {
            "openwebui": "mtplx integrate openwebui --port 8000",
            "claude_code": "mtplx integrate claude-code --port 8000",
            "opencode": "mtplx start opencode --port 18083",
        },
        "product_policy": {
            "fanmax_counts_for_product_gate": False,
            "safe_mode_drops_per_cycle_events": os.environ.get("MTPLX_DROP_EVENTS") == "1",
        },
    }
    return base


def _opencode_doctor_report(args: Any) -> dict[str, Any]:
    from optiq.runtime.mtp.opencode import (
        detect_opencode_desktop,
        opencode_config_path,
    )

    config_path = opencode_config_path()
    parsed: dict[str, Any] | None = None
    error: str | None = None
    if config_path.exists():
        try:
            value = json.loads(config_path.read_text(encoding="utf-8"))
            parsed = value if isinstance(value, dict) else {}
        except Exception as exc:
            error = str(exc)
    provider = (
        ((parsed or {}).get("provider") or {}).get("mtplx")
        if isinstance((parsed or {}).get("provider"), dict)
        else None
    )
    model_ref = (parsed or {}).get("model") if isinstance(parsed, dict) else None
    model_id = str(model_ref or "").split("/", 1)[-1] if model_ref else DEFAULT_PUBLIC_MODEL_ID
    model_config = None
    if isinstance(provider, dict):
        models = provider.get("models")
        if isinstance(models, dict):
            model_config = models.get(model_id) or next(iter(models.values()), None)
    options = provider.get("options") if isinstance(provider, dict) else {}
    base_url = ""
    if isinstance(options, dict):
        base_url = str(options.get("baseURL") or "")
    server_url = base_url.rstrip("/")
    if server_url.endswith("/v1"):
        server_url = server_url[:-3]
    health = _http_json(server_url + "/health", timeout=1.5) if server_url else {}
    return {
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "config_error": error,
        "detected": detect_opencode_desktop(),
        "provider_present": isinstance(provider, dict),
        "model_ref": model_ref,
        "base_url": base_url,
        "server_url": server_url,
        "server_health": health,
        "reasoning_field": (
            (model_config or {}).get("interleaved", {}).get("field")
            if isinstance(model_config, dict)
            else None
        ),
        "reasoning_enabled": (
            bool((model_config or {}).get("reasoning"))
            if isinstance(model_config, dict)
            else False
        ),
        "tool_call_enabled": (
            bool((model_config or {}).get("tool_call"))
            if isinstance(model_config, dict)
            else False
        ),
        "has_hidden_max_tokens": "maxTokens" in json.dumps(model_config or {}),
        "expected_start_command": "mtplx start opencode --port 18083 --profile sustained --max",
    }


def _android_studio_doctor_report(args: Any) -> dict[str, Any]:
    host = str(getattr(args, "host", None) or "127.0.0.1")
    port = int(getattr(args, "port", None) or 8008)
    base_url = str(getattr(args, "base_url", None) or f"http://{host}:{port}/v1").rstrip("/")
    models = _http_json(f"{base_url}/models", timeout=3.0)
    model_id = DEFAULT_PUBLIC_MODEL_ID
    data = models.get("data") if isinstance(models, dict) else None
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and first.get("id"):
            model_id = str(first["id"])
    chat_payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 16,
    }
    stream_payload = {
        **chat_payload,
        "max_completion_tokens": 8,
        "enable_thinking": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    tool_payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Say OK. Do not call tools."}],
        "max_completion_tokens": 8,
        "enable_thinking": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "android_studio_check",
                    "description": "No-op compatibility check for Android Studio.",
                    "parameters": {
                        "type": "object",
                        "properties": {"ok": {"type": "string"}},
                        "required": ["ok"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "response_format": {"type": "text"},
    }
    return {
        "base_url": base_url,
        "paste_url": base_url,
        "url_schema": "OpenAI-compatible",
        "api_key": "blank for localhost unless MTPLX was started with --api-key",
        "model": model_id,
        "models": models,
        "chat_nonstream": _http_post_json(
            f"{base_url}/chat/completions", chat_payload, timeout=20.0
        ),
        "chat_stream": _http_post_text(
            f"{base_url}/chat/completions", stream_payload, timeout=20.0
        ),
        "tool_request": _http_post_json(
            f"{base_url}/chat/completions", tool_payload, timeout=30.0
        ),
        "expected_start_command": "mtplx start --port 8008",
    }


def _git_value(args: list[str], *, cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _resolve_runtime_model_path(model: str, *, cache_dir: str | None = None) -> tuple[str, dict[str, Any] | None]:
    from optiq.runtime.mtp.hf_loader import resolve_model_path

    try:
        return str(resolve_model_path(model, cache_dir=cache_dir)), None
    except Exception as exc:
        return model, {
            "error": "model is not available locally",
            "model": model,
            "detail": str(exc),
        }


def _exactness_profile_kwargs(args: Any) -> dict[str, Any]:
    return {
        "attention_impl": getattr(args, "exactness_attention_impl", "mlx_vector_paged"),
        "block_size": int(getattr(args, "exactness_block_size", 16)),
        "num_blocks": int(getattr(args, "exactness_num_blocks", 1024)),
        "partitioned": not bool(getattr(args, "exactness_no_partitioned", False)),
        "partition_threshold": int(getattr(args, "exactness_partition_threshold", 2048)),
        "partition_size": int(getattr(args, "exactness_partition_size", 512)),
    }


def _exact_paged_env_from_args(args: Any) -> dict[str, str]:
    return exact_paged_attention_env(**_exactness_profile_kwargs(args))


def _depth_sweep_native60(
    *,
    model: str,
    prompt_suite: str,
    depths: str = "3",
    max_tokens: int,
    limit: int | None,
    seed: int,
    draft_lm_head: dict[str, Any] | None = None,
    draft_sampler: dict[str, Any] | None = None,
    compare_ar: bool = False,
    ar_only: bool = False,
) -> dict[str, Any]:
    from optiq.runtime.mtp.benchmarks.runners.mtp_depth_sweep import run_mtp_depth_sweep

    apply_profile_env("performance-cold")
    draft_lm_head = draft_lm_head or {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    return run_mtp_depth_sweep(
        model,
        prompt_suite,
        depths=depths,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=max_tokens,
        seed=seed,
        limit=limit,
        enable_thinking=False,
        compare_ar=compare_ar,
        ar_only=ar_only,
        mtp_hidden_variant="post_norm",
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        min_speculative_depth=1,
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        draft_lm_head_bits=int(draft_lm_head["bits"]),
        draft_lm_head_group_size=int(draft_lm_head["group_size"]),
        draft_lm_head_mode=str(draft_lm_head["mode"]),
        draft_temperature=(
            None if draft_sampler is None else float(draft_sampler["temperature"])
        ),
        draft_top_p=None if draft_sampler is None else float(draft_sampler["top_p"]),
        draft_top_k=None if draft_sampler is None else int(draft_sampler["top_k"]),
    )

class _temporary_env:
    def __init__(self, updates: dict[str, str]) -> None:
        self.updates = updates
        self.previous: dict[str, str | None] = {}

    def __enter__(self) -> None:
        self.previous = {key: os.environ.get(key) for key in self.updates}
        os.environ.update(self.updates)

    def __exit__(self, *_exc: object) -> None:
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def cmd_doctor(args: Any) -> int:
    env = collect_environment(args.project_root).to_dict()
    from optiq.runtime.mtp.hf_loader import hf_cache_report
    from optiq.runtime.mtp.thermal import detect_thermal_control
    from optiq.runtime.mtp.diagnostics import build_diagnostics_payload, write_doctor_bundle

    smc_path_raw = getattr(args, "smc_path", None) or shutil.which("smc") or ""
    sovereign_path_raw = getattr(args, "sovereign_path", None) or shutil.which("sovereign") or ""
    smc_path = Path(smc_path_raw) if smc_path_raw else None
    sovereign_path = Path(sovereign_path_raw) if sovereign_path_raw else None
    thermal_control = detect_thermal_control()
    server_deps = {
        name: importlib.util.find_spec(name) is not None
        for name in ("fastapi", "uvicorn")
    }

    report = {
        "environment": env,
        "huggingface": hf_cache_report(cache_dir=getattr(args, "model_cache", None)),
        "thermal_control": thermal_control,
        "tools": {
            "python": sys.executable,
            "powermetrics": shutil.which("powermetrics"),
            "sudo": shutil.which("sudo"),
            "smc_atlas": str(smc_path) if smc_path else None,
            "smc_atlas_exists": bool(smc_path and smc_path.exists()),
            "sovereign": str(sovereign_path) if sovereign_path else None,
            "sovereign_exists": bool(sovereign_path and sovereign_path.exists()),
        },
        "policy": {
            "fanmax_counts_for_product_gate": False,
            "benchmark_exactness_smoke_context": 2048,
        },
    }
    report["diagnostics"] = build_diagnostics_payload(
        model_cache=getattr(args, "model_cache", None),
        deep=bool(getattr(args, "deep", False)),
        mlx_info=env.get("mlx") if isinstance(env.get("mlx"), dict) else None,
        thermal_control=thermal_control,
        server_dependencies=server_deps if getattr(args, "deep", False) else None,
    )
    if getattr(args, "topic", None) == "opencode":
        report["opencode"] = _opencode_doctor_report(args)
    if getattr(args, "topic", None) in {"android-studio", "android_studio"}:
        report["android_studio"] = _android_studio_doctor_report(args)
    if getattr(args, "deep", False):
        report = _deep_doctor_report(args, report)
    if getattr(args, "bundle", False):
        report["bundle"] = write_doctor_bundle(
            report=report,
            output_dir=getattr(args, "output_dir", None),
            include_paths=bool(getattr(args, "include_paths", False)),
        )
    if getattr(args, "json", False):
        _print(report)
    elif getattr(args, "summary", False):
        diagnostics = report["diagnostics"]
        print(f"MTPLX doctor: {diagnostics['overall']}")
        for check in diagnostics["checks"]:
            marker = check["status"].upper()
            print(f"{marker:4} {check['id']}: {check['observed']}")
            if check.get("fix") and check["status"] != "pass":
                print(f"     fix: {check['fix']}")
        if report.get("bundle"):
            print(f"bundle: {report['bundle']['bundle_dir']}")
            print(f"zip: {report['bundle']['bundle_zip']}")
    else:
        env_info = report.get("environment") or {}
        hf = report.get("huggingface") or {}
        thermal = report.get("thermal_control") or {}
        selected = thermal.get("selected") or {}
        tools = report.get("tools") or {}
        print("MTPLX status")
        print(f"python: {tools.get('python') or sys.executable}")
        print(f"platform: {env_info.get('platform') or env_info.get('system') or 'unknown'}")
        print(f"project: {env_info.get('project_root') or os.getcwd()}")
        print(f"model cache: {hf.get('cache_dir') or 'default'}")
        print(f"cached models: {hf.get('cached_models', 'unknown')}")
        print(
            "thermal: "
            f"{'available' if thermal.get('available') else 'not configured'}"
            f" ({selected.get('kind') or 'none'})"
        )
        if getattr(args, "deep", False):
            launchers = report.get("launchers") or {}
            config = report.get("config") or {}
            print(f"launcher: {launchers.get('global_launcher') or launchers.get('global') or 'unknown'}")
            print(f"config: {config.get('path') or 'default'}")
        if report.get("opencode"):
            opencode = report["opencode"]
            print("OpenCode:")
            print(f"  config: {opencode.get('config_path')}")
            print(f"  provider: {'present' if opencode.get('provider_present') else 'missing'}")
            print(f"  model: {opencode.get('model_ref') or 'missing'}")
            print(f"  base URL: {opencode.get('base_url') or 'missing'}")
            print(f"  reasoning field: {opencode.get('reasoning_field') or 'missing'}")
            print(f"  hidden maxTokens: {str(bool(opencode.get('has_hidden_max_tokens'))).lower()}")
        if report.get("android_studio"):
            android = report["android_studio"]
            print("Android Studio:")
            print(f"  URL: {android.get('paste_url')}")
            print(f"  URL schema: {android.get('url_schema')}")
            print(f"  model: {android.get('model')}")
            print(f"  /v1/models: {'ok' if android.get('models', {}).get('data') else 'check failed'}")
            print(
                "  /v1/chat/completions: "
                f"{'ok' if android.get('chat_nonstream', {}).get('ok') else 'check failed'}"
            )
    return 0


def cmd_inspect_model_public(args: Any) -> int:
    model_args = list(getattr(args, "model_args", []) or [])
    if model_args and model_args[0] == "model":
        model_args = model_args[1:]
    model = args.model or getattr(args, "model_arg", None) or (model_args[0] if model_args else None)
    if len(model_args) > 1:
        raise SystemExit("inspect accepts exactly one model path/repo id")
    if not model:
        raise SystemExit("inspect requires MODEL or --model MODEL")
    try:
        inspection = inspect_model(model).to_dict()
    except Exception as exc:
        _print({"error": "inspect failed", "model": model, "detail": str(exc)})
        return 1
    if getattr(args, "json", False):
        _print(inspection)
    else:
        _print_inspect_human(inspection)
    compatibility = inspection.get("compatibility") or {}
    exit_code = int(compatibility.get("exit_code", 0))
    if args.require_mtp or getattr(args, "strict_exit_code", True):
        return exit_code
    return 0


def _print_inspect_human(inspection: dict[str, Any]) -> None:
    compatibility = inspection.get("compatibility") or {}
    mtp = inspection.get("mtp") or {}
    architecture = inspection.get("architecture") or inspection.get("model_type") or "unknown"
    tensor_count = mtp.get("tensor_count")
    mtp_layers = inspection.get("mtp_num_hidden_layers")
    runtime_contract_present = bool(
        inspection.get("runtime_contract_path")
        or compatibility.get("runtime_contract_path")
    )
    print("MTPLX inspect")
    print(f"model: {inspection.get('model_dir')}")
    print(f"source: {inspection.get('source') or 'local'}")
    print(f"architecture: {architecture}")
    print(f"tier: {compatibility.get('tier', 'unknown')}")
    print(f"recognized: {str(bool(compatibility.get('recognized'))).lower()}")
    print(f"can_run: {str(bool(compatibility.get('can_run'))).lower()}")
    runtime_status = compatibility.get("runtime_compatibility")
    if runtime_status:
        print(f"runtime_compatibility: {runtime_status}")
    support_level = compatibility.get("support_level")
    if support_level:
        print(f"support_level: {support_level}")
    if mtp_layers is not None:
        print(f"mtp_layers: {mtp_layers}")
    if tensor_count is not None:
        print(f"mtp_tensors_present: {tensor_count}")
    print(f"runtime_contract: {str(runtime_contract_present).lower()}")
    recommended = compatibility.get("recommended_profile") or inspection.get("recommended_profile")
    if recommended:
        print(f"recommended_profile: {recommended}")
    message = compatibility.get("message")
    if message:
        print(f"message: {message}")


def cmd_bench_public(args: Any) -> int:
    action = args.bench_action
    if action == "prefill-ladder":
        from optiq.runtime.mtp.prefill_bench import (
            UnsafePrefillDiagnosticError,
            emit_prefill_ladder,
            run_prefill_ladder,
            write_prefill_ladder,
        )

        try:
            payload = run_prefill_ladder(args)
        except UnsafePrefillDiagnosticError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if getattr(args, "output", None):
            write_prefill_ladder(args.output, payload)
        emit_prefill_ladder(payload, json_output=bool(getattr(args, "json", False)))
        return 0
    if action in {"run", "context"}:
        return _cmd_bench_run(args)
    if action == "tune":
        return _cmd_bench_tune(args)
    if action == "nightly":
        return _cmd_bench_nightly(args)
    if action == "compare":
        return _cmd_bench_compare(args)
    if action == "serve":
        return _cmd_bench_serve(args)
    if action == "reference":
        return _cmd_bench_reference(args)
    if action == "reference-vllm":
        return _cmd_bench_reference_vllm(args)
    raise SystemExit(f"unknown bench action: {action}")


def cmd_tune_public(args: Any) -> int:
    if getattr(args, "_tune_candidate", None):
        return _cmd_tune_candidate(args)
    return _cmd_tune(
        args,
        action="tune",
        save_default=not bool(getattr(args, "no_save", False)),
        verbose_default=bool(getattr(args, "verbose", False)),
    )


def _cmd_bench_tune(args: Any) -> int:
    child = SimpleNamespace(**vars(args))
    cli_flags = getattr(child, "_cli_flags", set()) or set()
    if "max-tokens" not in cli_flags:
        child.max_tokens = TUNE_DEFAULT_MAX_TOKENS
    return _cmd_tune(
        child,
        action="bench tune",
        save_default=False,
        verbose_default=True,
    )


def _cmd_tune(
    args: Any,
    *,
    action: str,
    save_default: bool,
    verbose_default: bool,
) -> int:
    try:
        depths = _parse_tune_depths(getattr(args, "depths", None) or TUNE_DEFAULT_DEPTHS)
    except ValueError as exc:
        return _tune_error(str(exc), json_output=bool(getattr(args, "json", False)))

    model = getattr(args, "model", None) or DEFAULT_CHAMPION
    settings = _tune_settings(args, depths=depths)
    run_id = getattr(args, "run_id", None) or f"tune-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir = _absolute_user_path(getattr(args, "output_dir", None) or "outputs/cli/tune")
    output_root = output_dir / run_id
    output_path = _absolute_user_path(getattr(args, "output")) if getattr(args, "output", None) else output_root / "tune.json"
    json_output = bool(getattr(args, "json", False))
    verbose = bool(getattr(args, "verbose", verbose_default) or verbose_default)
    collect_telemetry = _tune_collect_telemetry(args, action=action)

    if bool(getattr(args, "dry_run", False)):
        model_source_notes = _tune_model_source_notes(args, runtime_model=model)
        payload = _tune_dry_run_payload(
            args,
            action=action,
            model=model,
            run_id=run_id,
            output_root=output_root,
            output_path=output_path,
            settings=settings,
            depths=depths,
            save_default=save_default,
            collect_telemetry=collect_telemetry,
            model_source_notes=model_source_notes,
        )
        if json_output or action == "bench tune":
            _print(payload)
        else:
            _print_tune_dry_run_human(payload)
        return 0

    runtime_model, resolve_error = _resolve_runtime_model_path(
        model,
        cache_dir=getattr(args, "cache_dir", None),
    )
    if resolve_error is not None:
        return _tune_error(
            resolve_error.get("error") or "model is not available locally",
            detail=resolve_error.get("detail"),
            json_output=json_output,
        )
    model_source_notes = _tune_model_source_notes(args, runtime_model=runtime_model)

    profile = get_profile("performance-cold")
    hardware = _apple_hardware_context()
    software = _software_context()
    backend = _mlx_backend_context(profile)
    state_key, key_material = _tune_state_key(
        runtime_model,
        settings=settings,
        hardware=hardware,
        software=software,
        backend=backend,
    )
    cached = None if bool(getattr(args, "retune", False)) else _load_tune_record(state_key)
    if save_default and cached is not None:
        payload = dict(cached.get("payload") or {})
        payload["from_cache"] = True
        payload["state_path"] = str(_tune_state_path())
        if json_output:
            _print(payload)
        else:
            _print_tune_human(payload, verbose=verbose)
        return 0

    from optiq.runtime.mtp.thermal import MaxSession

    def _emit(line: str) -> None:
        print(line, file=sys.stderr, flush=True)

    if not json_output:
        _emit(f"[tune] model: {runtime_model}")
        for note in model_source_notes:
            _emit(note)
        if action == "bench tune":
            if collect_telemetry:
                _emit("[tune] diagnostic telemetry active; use --no-telemetry for clean speed comparison")
            else:
                _emit("[tune] diagnostic telemetry disabled; speed comparison is cleaner")
        _emit("[tune] close heavy apps now for cleaner results before measurements start")
        _emit("[tune] fans may get loud during tuning and will be restored afterward")
    max_session = MaxSession(log=_emit)
    if not max_session.start():
        verified = max_session.thermal.get("verified") or {}
        return _tune_error(
            "verified max-fan mode did not start; refusing to run a model tune",
            detail=verified.get("message"),
            actionable=verified.get("actionable"),
            thermal=max_session.thermal,
            json_output=json_output,
        )

    try:
        if not json_output:
            _emit(f"[tune] artifacts: {output_root}")
            _emit("[tune] running isolated candidates: AR, " + ", ".join(f"D{depth}" for depth in depths))
        candidate_rows = _run_tune_candidates(
            args,
            runtime_model=runtime_model,
            run_id=run_id,
            output_root=output_root,
            depths=depths,
            settings=settings,
            progress=_emit if not json_output else None,
            collect_telemetry=collect_telemetry,
        )
    finally:
        max_session.stop()

    payload = _tune_payload(
        action=action,
        run_id=run_id,
        model=runtime_model,
        settings=settings,
        rows=candidate_rows,
        output_root=output_root,
        output_path=output_path,
        hardware=hardware,
        software=software,
        backend=backend,
        thermal=max_session.thermal,
        state_key=state_key,
        diagnostics={
            "telemetry_enabled": collect_telemetry,
            "telemetry_env": TUNE_TELEMETRY_ENV,
            "model_source_notes": model_source_notes,
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)

    if payload.get("best") and save_default and not bool(getattr(args, "no_save", False)):
        payload["saved"] = True
        payload["state_path"] = str(_tune_state_path())
        _save_tune_record(state_key, key_material=key_material, payload=payload)
        write_json(output_path, payload)
    else:
        payload["saved"] = False
        payload["state_path"] = str(_tune_state_path())
        if not any(isinstance(row.get("tok_s"), (int, float)) for row in candidate_rows):
            payload["save_skipped_reason"] = "tune failed; no candidate produced usable tokens"
        elif not payload.get("best"):
            payload["save_skipped_reason"] = "no MTP depth beat AR"
        elif bool(getattr(args, "no_save", False)) or not save_default:
            payload["save_skipped_reason"] = "save disabled"
        write_json(output_path, payload)

    if json_output:
        _print(payload)
    else:
        _print_tune_human(payload, verbose=verbose)
    if not candidate_rows or not any(isinstance(row.get("tok_s"), (int, float)) for row in candidate_rows):
        return 1
    return 0


def _cmd_tune_candidate(args: Any) -> int:
    candidate = str(getattr(args, "_tune_candidate", "")).lower()
    output = Path(getattr(args, "_tune_candidate_output", None) or "")
    if candidate not in {"ar", "1", "2", "3"}:
        return _tune_error("invalid tune candidate", json_output=True)
    if not output:
        return _tune_error("missing tune candidate output path", json_output=True)
    model = getattr(args, "model", None) or DEFAULT_CHAMPION
    runtime_model, resolve_error = _resolve_runtime_model_path(
        model,
        cache_dir=getattr(args, "cache_dir", None),
    )
    if resolve_error is not None:
        _print(resolve_error)
        return 1
    inspection, gate_exit = _model_gate(
        runtime_model,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=True,
    )
    if gate_exit is not None:
        _print({"error": "model failed MTP primary gate", "model": inspection})
        return gate_exit
    profile = get_profile("performance-cold")
    draft_lm_head = _model_draft_lm_head_spec(inspection, profile)
    draft_sampler = _model_draft_sampler_spec(inspection, profile)
    result = _depth_sweep_native60(
        model=runtime_model,
        prompt_suite=prompt_suite_path(TUNE_DEFAULT_SUITE),
        depths="1" if candidate == "ar" else candidate,
        max_tokens=int(getattr(args, "max_tokens", TUNE_DEFAULT_MAX_TOKENS) or TUNE_DEFAULT_MAX_TOKENS),
        limit=int(getattr(args, "limit", TUNE_DEFAULT_LIMIT) or TUNE_DEFAULT_LIMIT),
        seed=int(getattr(args, "seed", TUNE_DEFAULT_SEED) or TUNE_DEFAULT_SEED),
        draft_lm_head=draft_lm_head,
        draft_sampler=draft_sampler,
        compare_ar=candidate == "ar",
        ar_only=candidate == "ar",
    )
    from optiq.runtime.mtp.benchmarks.runners.mtp_depth_sweep import write_depth_sweep

    write_depth_sweep(output, result)
    _print({"candidate": candidate, "output": str(output)})
    return 0


def _parse_tune_depths(raw: Any) -> list[int]:
    if raw is None:
        raw = TUNE_DEFAULT_DEPTHS
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError("tune depths must include at least one of 1,2,3")
    depths: list[int] = []
    for part in parts:
        try:
            depth = int(part)
        except ValueError as exc:
            raise ValueError("tune depths must be integers from 1 to 3") from exc
        if depth < 1 or depth > MAX_PUBLIC_SPECULATIVE_DEPTH:
            raise ValueError("tune depths must be between 1 and 3")
        if depth not in depths:
            depths.append(depth)
    return depths


def _tune_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    text = str(raw).strip().lower()
    if text in {"0", "off", "false", "none", "no"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return default


def _tune_env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", "none"}:
        return False
    return default


def _tune_collect_telemetry(args: Any, *, action: str) -> bool:
    return (
        action == "bench tune"
        and not bool(getattr(args, "no_telemetry", False))
        and _tune_env_bool(TUNE_TELEMETRY_ENV, default=True)
    )


def _normalize_model_ref_for_compare(value: str | Path | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith(("~", "/", "./", "../")):
        path = Path(text).expanduser()
        try:
            return str(path.resolve())
        except OSError:
            return str(path)
    return text


def _same_model_ref(left: str | Path | None, right: str | Path | None) -> bool:
    return _normalize_model_ref_for_compare(left) == _normalize_model_ref_for_compare(right)


def _tune_model_source_notes(args: Any, *, runtime_model: str) -> list[str]:
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if "model" in cli_flags:
        return []
    config = getattr(args, "mtplx_config", None)
    config_model = config.get("model") if isinstance(config, dict) else None
    if not config_model or not _same_model_ref(runtime_model, config_model):
        return []
    try:
        default_selection = select_default_model()
    except Exception:
        return []
    default_model = default_selection.model
    if _same_model_ref(runtime_model, default_model):
        return []
    config_path = config.get("path") if isinstance(config, dict) else "~/.mtplx/config.toml"
    return [
        f"[tune] using configured model from {config_path}: {runtime_model}",
        f"[tune] verified default for this Mac is: {default_model}",
        "[tune] pass --model <path> to benchmark a different artifact explicitly",
    ]


def _tune_settings(args: Any, *, depths: list[int]) -> dict[str, Any]:
    return {
        "profile": "performance-cold",
        "suite": TUNE_DEFAULT_SUITE,
        "depths": ",".join(str(depth) for depth in depths),
        "max_tokens": int(getattr(args, "max_tokens", TUNE_DEFAULT_MAX_TOKENS) or TUNE_DEFAULT_MAX_TOKENS),
        "limit": int(getattr(args, "limit", TUNE_DEFAULT_LIMIT) or TUNE_DEFAULT_LIMIT),
        "seed": int(getattr(args, "seed", TUNE_DEFAULT_SEED) or TUNE_DEFAULT_SEED),
        "thinking": "disabled",
        "candidate_settle_s": max(
            0.0,
            _tune_env_float("MTPLX_TUNE_CANDIDATE_SETTLE_S", TUNE_CANDIDATE_SETTLE_S),
        ),
        "tie_prefer_deeper_within_pct": max(
            0.0,
            _tune_env_float(
                "MTPLX_TUNE_TIE_PREFER_DEEPER_WITHIN_PCT",
                TUNE_TIE_PREFER_DEEPER_WITHIN_PCT,
            ),
        ),
    }


def _tune_state_path() -> Path:
    env = os.environ.get("MTPLX_TUNE_STATE")
    return Path(env).expanduser() if env else TUNE_STATE_PATH


def _load_tune_state() -> dict[str, Any]:
    path = _tune_state_path()
    if not path.exists():
        return {"schema_version": 1, "records": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "records": {}}
    if not isinstance(data, dict):
        return {"schema_version": 1, "records": {}}
    records = data.get("records")
    if not isinstance(records, dict):
        data["records"] = {}
    return data


def _load_tune_record(state_key: str) -> dict[str, Any] | None:
    record = (_load_tune_state().get("records") or {}).get(state_key)
    if not isinstance(record, dict):
        return None
    payload = record.get("payload")
    best = payload.get("best") if isinstance(payload, dict) else None
    if not isinstance(best, dict):
        return None
    if not isinstance(best.get("depth"), int):
        return None
    return record


def _save_tune_record(state_key: str, *, key_material: dict[str, Any], payload: dict[str, Any]) -> None:
    state = _load_tune_state()
    records = state.setdefault("records", {})
    records[state_key] = {
        "schema_version": 1,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "key": state_key,
        "key_material": key_material,
        "payload": payload,
    }
    path = _tune_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, state)


def _tune_state_key(
    model: str,
    *,
    settings: dict[str, Any],
    hardware: dict[str, Any],
    software: dict[str, Any],
    backend: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    model_identity = str(Path(model).expanduser().resolve()) if Path(model).expanduser().exists() else str(model)
    key_material = {
        "model": model_identity,
        "hardware": {
            "chip": hardware.get("chip"),
            "chip_family": hardware.get("chip_family"),
            "hw_model": hardware.get("hw_model"),
            "machine": hardware.get("machine"),
        },
        "software": {
            "mtplx_version": software.get("mtplx_version"),
            "mlx_version": software.get("mlx_version"),
            "mlx_lm_version": software.get("mlx_lm_version"),
        },
        "backend": {
            "mlx_core_path": backend.get("mlx_core_path"),
            "optional_fast_mlx_fork_active": backend.get("optional_fast_mlx_fork_active"),
            "stock_mlx_likely": backend.get("stock_mlx_likely"),
        },
        "settings": settings,
    }
    encoded = json.dumps(key_material, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), key_material


def _tune_dry_run_payload(
    args: Any,
    *,
    action: str,
    model: str,
    run_id: str,
    output_root: Path,
    output_path: Path,
    settings: dict[str, Any],
    depths: list[int],
    save_default: bool,
    collect_telemetry: bool,
    model_source_notes: list[str] | None = None,
) -> dict[str, Any]:
    commands = []
    for candidate in ["ar", *[str(depth) for depth in depths]]:
        candidate_output = output_root / f"{_tune_candidate_label(candidate).lower()}.json"
        commands.append(
            {
                "candidate": _tune_candidate_label(candidate),
                "command": _tune_candidate_command(
                    args,
                    candidate=candidate,
                    model=model,
                    output=candidate_output,
                    settings=settings,
                ),
                "output": str(candidate_output),
            }
        )
    return {
        "dry_run": True,
        "action": action,
        "run_id": run_id,
        "model": model,
        "settings": settings,
        "candidates": commands,
        "output": str(output_path),
        "state_path": str(_tune_state_path()),
        "save_default": save_default,
        "fan_control": "verified max-fan required before model load",
        "diagnostics": {
            "telemetry_enabled": collect_telemetry,
            "telemetry_env": TUNE_TELEMETRY_ENV,
            "model_source_notes": model_source_notes or [],
            "description": (
                "bench tune records per-candidate power, frequency, temperature, "
                "utilization, and fan telemetry when not in dry-run"
                if action == "bench tune" and collect_telemetry
                else (
                    "bench tune telemetry disabled; candidate commands match clean tune timing more closely"
                    if action == "bench tune"
                    else None
                )
            ),
        },
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _series_stats(values: list[Any]) -> dict[str, Any] | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    return {
        "avg": sum(numeric) / len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "last": numeric[-1],
        "samples": len(numeric),
    }


def _generation_windows_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    windows: list[dict[str, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start = _float_or_none(row.get("generation_started_at"))
        end = _float_or_none(row.get("generation_ended_at"))
        if start is None or end is None or end <= start:
            continue
        windows.append({"start": start, "end": end, "duration_s": end - start})
    return windows


def _convert_power_to_watts(value: str, unit: str | None) -> float:
    watts = float(value)
    if (unit or "").lower() == "mw":
        watts /= 1000.0
    return watts


def _convert_frequency_to_ghz(value: str, unit: str | None) -> float:
    ghz = float(value)
    if (unit or "").lower() == "mhz":
        ghz /= 1000.0
    return ghz


def _parse_powermetrics_text(text: str) -> dict[str, Any]:
    """Extract the MX Power Gadget-style rails from macOS powermetrics text."""

    power: dict[str, float] = {}
    for match in re.finditer(
        r"(?m)^(CPU|GPU|ANE) Power:\s*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)\b",
        text,
    ):
        rail = match.group(1).lower()
        power[rail] = _convert_power_to_watts(match.group(2), match.group(3))
    package_match = re.search(
        r"(?m)^Combined Power \(CPU \+ GPU \+ ANE\):\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*(mW|W)\b",
        text,
    )
    if package_match:
        power["package"] = _convert_power_to_watts(
            package_match.group(1),
            package_match.group(2),
        )

    p_frequencies: list[float] = []
    m_frequencies: list[float] = []
    p_utilization: list[float] = []
    m_utilization: list[float] = []
    for match in re.finditer(
        r"(?m)^([A-Za-z0-9]+)-Cluster HW active frequency:\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*(MHz|GHz)\b",
        text,
    ):
        name = match.group(1).upper()
        ghz = _convert_frequency_to_ghz(match.group(2), match.group(3))
        if name == "P":
            p_frequencies.append(ghz)
        else:
            m_frequencies.append(ghz)
    for match in re.finditer(
        r"(?m)^([A-Za-z0-9]+)-Cluster HW active residency:\s*"
        r"([0-9]+(?:\.[0-9]+)?)%",
        text,
    ):
        name = match.group(1).upper()
        residency = float(match.group(2))
        if name == "P":
            p_utilization.append(residency)
        else:
            m_utilization.append(residency)

    frequency: dict[str, float] = {}
    utilization: dict[str, float] = {}
    p_cluster = _avg(p_frequencies)
    m_cluster = _avg(m_frequencies)
    p_core = _avg(p_utilization)
    m_core = _avg(m_utilization)
    if p_cluster is not None:
        frequency["p_cluster"] = p_cluster
    if m_cluster is not None:
        frequency["m_cluster"] = m_cluster
    if p_core is not None:
        utilization["p_core"] = p_core
    if m_core is not None:
        utilization["m_core"] = m_core

    gpu_freq_match = re.search(
        r"(?m)^GPU HW active frequency:\s*([0-9]+(?:\.[0-9]+)?)\s*(MHz|GHz)\b",
        text,
    )
    if gpu_freq_match:
        frequency["gpu"] = _convert_frequency_to_ghz(
            gpu_freq_match.group(1),
            gpu_freq_match.group(2),
        )
    gpu_residency_match = re.search(
        r"(?m)^GPU HW active residency:\s*([0-9]+(?:\.[0-9]+)?)%",
        text,
    )
    if gpu_residency_match:
        utilization["gpu"] = float(gpu_residency_match.group(1))

    payload: dict[str, Any] = {
        "source": "powermetrics",
        "power_w": power,
        "frequency_ghz": frequency,
        "utilization_pct": utilization,
    }
    pressure_match = re.search(r"(?m)^Current pressure level:\s*(.+)$", text)
    if pressure_match:
        payload["thermal_pressure"] = pressure_match.group(1).strip()
    return payload


def _thermalforge_binary() -> str | None:
    owned = Path("~/.mtplx/bin/thermalforge").expanduser()
    if owned.exists():
        return str(owned)
    return shutil.which("thermalforge")


def _temperature_groups_from_thermalforge(
    temperatures: dict[str, Any],
) -> tuple[list[float], list[float]]:
    core_values: list[float] = []
    fallback_core_values: list[float] = []
    gpu_values: list[float] = []
    for key, raw_value in temperatures.items():
        value = _float_or_none(raw_value)
        if value is None:
            continue
        name = str(key)
        upper = name.upper()
        if upper.startswith("TG") or name.startswith("Tg"):
            gpu_values.append(value)
        elif upper.startswith("TC") or name.startswith(("Tp", "Tm")):
            core_values.append(value)
        else:
            fallback_core_values.append(value)
    return core_values or fallback_core_values, gpu_values


def _sample_thermalforge_status() -> dict[str, Any]:
    binary = _thermalforge_binary()
    if not binary:
        return {"source": "thermalforge", "error": "thermalforge not found"}
    try:
        proc = subprocess.run(
            [binary, "status"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=1.0,
        )
    except Exception as exc:
        return {"source": "thermalforge", "error": str(exc)}
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return {"source": "thermalforge", "error": detail or f"exit {proc.returncode}"}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"source": "thermalforge", "error": f"invalid JSON: {exc}"}

    fans = data.get("fans") if isinstance(data, dict) else []
    fan_rpms: list[float] = []
    if isinstance(fans, dict):
        fans = list(fans.values())
    for fan in fans if isinstance(fans, list) else []:
        if not isinstance(fan, dict):
            continue
        rpm = _float_or_none(fan.get("actual_rpm"))
        if rpm is None:
            rpm = _float_or_none(fan.get("target_rpm"))
        if rpm is not None:
            fan_rpms.append(rpm)

    temperatures = data.get("temperatures") if isinstance(data, dict) else {}
    core_values, gpu_values = (
        _temperature_groups_from_thermalforge(temperatures)
        if isinstance(temperatures, dict)
        else ([], [])
    )
    temperature: dict[str, float] = {}
    if core_values:
        temperature["core_avg"] = sum(core_values) / len(core_values)
        temperature["core_max"] = max(core_values)
        temperature["core_min"] = min(core_values)
    if gpu_values:
        temperature["gpu_avg"] = sum(gpu_values) / len(gpu_values)

    payload: dict[str, Any] = {
        "source": "thermalforge",
        "fans_rpm": {
            "min": min(fan_rpms),
            "max": max(fan_rpms),
            "avg": sum(fan_rpms) / len(fan_rpms),
        }
        if fan_rpms
        else {},
        "temperature_c": temperature,
    }
    if isinstance(temperatures, dict):
        payload["temperature_sensors_c"] = {
            str(key): value
            for key, value in temperatures.items()
            if isinstance(value, (int, float))
        }
    return payload


def _sample_powermetrics_once() -> dict[str, Any]:
    if not shutil.which("powermetrics"):
        return {"source": "powermetrics", "error": "powermetrics not found"}
    if not shutil.which("sudo"):
        return {"source": "powermetrics", "error": "sudo not found"}
    command = [
        "sudo",
        "-n",
        "powermetrics",
        "-n",
        "1",
        "-i",
        "250",
        "--samplers",
        "cpu_power,gpu_power,ane_power,thermal",
    ]
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=2.0,
        )
    except Exception as exc:
        return {"source": "powermetrics", "error": str(exc)}
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return {"source": "powermetrics", "error": detail or f"exit {proc.returncode}"}
    return _parse_powermetrics_text(proc.stdout)


def _summarize_tune_telemetry_samples(
    samples: list[dict[str, Any]],
    *,
    errors: list[str] | None = None,
    started_at: float | None = None,
    ended_at: float | None = None,
) -> dict[str, Any]:
    power_keys = ("package", "cpu", "ane", "gpu")
    frequency_keys = ("p_cluster", "m_cluster", "gpu")
    temperature_keys = ("core_avg", "core_max", "core_min", "gpu_avg")
    utilization_keys = ("p_core", "m_core", "gpu")

    summary: dict[str, Any] = {
        "enabled": True,
        "scope": "candidate_process",
        "sample_count": len(samples),
        "duration_s": (
            float(ended_at - started_at)
            if started_at is not None and ended_at is not None
            else None
        ),
        "samples": samples,
        "sources": {
            "thermalforge": any(sample.get("thermalforge_ok") for sample in samples),
            "powermetrics": any(sample.get("powermetrics_ok") for sample in samples),
        },
    }
    errors = list(errors or [])
    if errors:
        summary["errors"] = sorted(set(errors))

    power = {
        key: _series_stats(
            [
                ((sample.get("power_w") or {}).get(key))
                for sample in samples
            ]
        )
        for key in power_keys
    }
    frequency = {
        key: _series_stats(
            [
                ((sample.get("frequency_ghz") or {}).get(key))
                for sample in samples
            ]
        )
        for key in frequency_keys
    }
    temperature = {
        key: _series_stats(
            [
                ((sample.get("temperature_c") or {}).get(key))
                for sample in samples
            ]
        )
        for key in temperature_keys
    }
    utilization = {
        key: _series_stats(
            [
                ((sample.get("utilization_pct") or {}).get(key))
                for sample in samples
            ]
        )
        for key in utilization_keys
    }
    fans = {
        key: _series_stats(
            [
                ((sample.get("fans_rpm") or {}).get(key))
                for sample in samples
            ]
        )
        for key in ("avg", "min", "max")
    }
    summary["power_w"] = {key: value for key, value in power.items() if value}
    summary["frequency_ghz"] = {key: value for key, value in frequency.items() if value}
    summary["temperature_c"] = {key: value for key, value in temperature.items() if value}
    summary["utilization_pct"] = {
        key: value for key, value in utilization.items() if value
    }
    summary["fans_rpm"] = {key: value for key, value in fans.items() if value}

    pressures = [
        str(sample.get("thermal_pressure"))
        for sample in samples
        if sample.get("thermal_pressure")
    ]
    if pressures:
        summary["thermal_pressure"] = {
            "last": pressures[-1],
            "observed": sorted(set(pressures)),
        }

    latest_sensors = next(
        (
            sample.get("temperature_sensors_c")
            for sample in reversed(samples)
            if sample.get("temperature_sensors_c")
        ),
        None,
    )
    if isinstance(latest_sensors, dict):
        summary["latest_temperature_sensors_c"] = latest_sensors
    return summary


def _sample_in_any_window(sample: dict[str, Any], windows: list[dict[str, float]]) -> bool:
    timestamp = _float_or_none(sample.get("timestamp"))
    if timestamp is None:
        return False
    return any(window["start"] <= timestamp <= window["end"] for window in windows)


def _attach_generation_telemetry(
    telemetry: dict[str, Any] | None,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(telemetry, dict):
        return telemetry
    windows = [
        window
        for window in row.get("generation_windows", [])
        if isinstance(window, dict)
        and isinstance(window.get("start"), (int, float))
        and isinstance(window.get("end"), (int, float))
        and float(window["end"]) > float(window["start"])
    ]
    if not windows:
        telemetry["generation"] = {
            "enabled": True,
            "scope": "generation_window",
            "sample_count": 0,
            "note": "candidate artifact did not include generation timing windows",
        }
        return telemetry
    samples = [
        sample
        for sample in telemetry.get("samples", [])
        if isinstance(sample, dict) and _sample_in_any_window(sample, windows)
    ]
    started_at = min(float(window["start"]) for window in windows)
    ended_at = max(float(window["end"]) for window in windows)
    generation = _summarize_tune_telemetry_samples(
        samples,
        errors=telemetry.get("errors") or [],
        started_at=started_at,
        ended_at=ended_at,
    )
    generation["scope"] = "generation_window"
    generation["windows"] = windows
    generation["window_count"] = len(windows)
    generation["window_duration_s"] = sum(
        float(window["end"]) - float(window["start"])
        for window in windows
    )
    if not samples:
        generation["note"] = "no powermetrics/thermalforge sample landed inside the generation window"
    telemetry["generation"] = generation
    return telemetry


class _TuneTelemetrySampler:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._powermetrics_disabled = False
        self._started_at: float | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        ended_at = time.monotonic()
        with self._lock:
            samples = list(self._samples)
            errors = list(self._errors)
        return _summarize_tune_telemetry_samples(
            samples,
            errors=errors,
            started_at=self._started_at,
            ended_at=ended_at,
        )

    def _loop(self) -> None:
        next_powermetrics = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            sample_started_at = time.time()
            sample: dict[str, Any] = {}
            thermal = _sample_thermalforge_status()
            if thermal.get("error"):
                self._add_error(f"thermalforge: {thermal.get('error')}")
            else:
                sample["thermalforge_ok"] = True
                sample.update({k: v for k, v in thermal.items() if k != "source"})
            if not self._powermetrics_disabled and now >= next_powermetrics:
                power = _sample_powermetrics_once()
                next_powermetrics = now + TUNE_POWERMETRICS_SAMPLE_INTERVAL_S
                if power.get("error"):
                    self._powermetrics_disabled = True
                    self._add_error(f"powermetrics: {power.get('error')}")
                else:
                    sample["powermetrics_ok"] = True
                    for key in (
                        "power_w",
                        "frequency_ghz",
                        "utilization_pct",
                        "thermal_pressure",
                    ):
                        if key in power:
                            sample[key] = power[key]
            sample_ended_at = time.time()
            sample["sample_started_at"] = sample_started_at
            sample["sample_ended_at"] = sample_ended_at
            sample["timestamp"] = (sample_started_at + sample_ended_at) / 2.0
            with self._lock:
                self._samples.append(sample)
            self._stop.wait(TUNE_TELEMETRY_SAMPLE_INTERVAL_S)

    def _add_error(self, message: str) -> None:
        with self._lock:
            if message not in self._errors:
                self._errors.append(message)


def _run_tune_candidates(
    args: Any,
    *,
    runtime_model: str,
    run_id: str,
    output_root: Path,
    depths: list[int],
    settings: dict[str, Any],
    progress: Callable[[str], None] | None = None,
    collect_telemetry: bool = False,
) -> list[dict[str, Any]]:
    output_root = _absolute_user_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total = 1 + len(depths)
    settle_s = float(settings.get("candidate_settle_s") or 0.0)
    for candidate in ["ar", *[str(depth) for depth in depths]]:
        label = _tune_candidate_label(candidate)
        candidate_output = output_root / f"{label.lower()}.json"
        stdout_path = output_root / f"{label.lower()}.log"
        command = _tune_candidate_command(
            args,
            candidate=candidate,
            model=runtime_model,
            output=candidate_output,
            settings=settings,
        )
        if rows and settle_s > 0.0:
            if progress is not None:
                progress(f"[tune] settling {settle_s:.1f}s before {label}")
            time.sleep(settle_s)
        if progress is not None:
            progress(f"[tune] {label} ({len(rows) + 1}/{total}) starting; log: {stdout_path}")
        started = time.monotonic()
        telemetry_sampler = _TuneTelemetrySampler(enabled=collect_telemetry)
        telemetry_sampler.start()
        try:
            proc = subprocess.run(
                command,
                cwd=repo_root(),
                env={**os.environ, "MTPLX_TUNE_CHILD": "1"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        finally:
            telemetry = telemetry_sampler.stop()
        elapsed_s = time.monotonic() - started
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        row = _tune_candidate_summary(
            candidate,
            candidate_output,
            returncode=proc.returncode,
            stdout_path=stdout_path,
            command=command,
        )
        if telemetry is not None:
            row["telemetry"] = _attach_generation_telemetry(telemetry, row)
        rows.append(row)
        if progress is not None:
            if isinstance(row.get("tok_s"), (int, float)):
                progress(f"[tune] {label} finished in {elapsed_s:.1f}s: {float(row['tok_s']):.2f} tok/s")
            else:
                progress(f"[tune] {label} failed in {elapsed_s:.1f}s: {row.get('error') or 'no token rate'}")
            telemetry_line = _format_tune_telemetry_inline(row.get("telemetry"))
            if telemetry_line:
                progress(f"[tune] {label} telemetry: {telemetry_line}")
    return rows


def _tune_candidate_command(
    args: Any,
    *,
    candidate: str,
    model: str,
    output: Path,
    settings: dict[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "mtplx.cli",
        "tune",
        "--_candidate",
        candidate,
        "--_candidate-output",
        str(output),
        "--model",
        str(model),
        "--max-tokens",
        str(int(settings["max_tokens"])),
        "--limit",
        str(int(settings["limit"])),
        "--seed",
        str(int(settings["seed"])),
        "--depths",
        str(settings["depths"]),
        "--yes",
    ]
    cache_dir = getattr(args, "cache_dir", None)
    if cache_dir:
        command.extend(["--cache-dir", str(cache_dir)])
    if bool(getattr(args, "unsafe_force_unverified", False)):
        command.append("--unsafe-force-unverified")
    return command


def _tune_candidate_label(candidate: str) -> str:
    return "AR" if candidate == "ar" else f"D{candidate}"


def _tune_candidate_summary(
    candidate: str,
    path: Path,
    *,
    returncode: int,
    stdout_path: Path,
    command: list[str],
) -> dict[str, Any]:
    label = _tune_candidate_label(candidate)
    base = {
        "mode": label,
        "depth": None if candidate == "ar" else int(candidate),
        "candidate": candidate,
        "returncode": returncode,
        "artifact": str(path),
        "stdout": str(stdout_path),
        "command": command,
    }
    if not path.exists():
        return {**base, "error": "candidate did not write an artifact"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {**base, "error": f"candidate artifact is not valid JSON: {exc}"}
    if candidate == "ar":
        ar_rows = data.get("ar_rows") or []
        tok_values = [
            float(row["tok_s"])
            for row in ar_rows
            if isinstance(row.get("tok_s"), (int, float))
        ]
        generation_windows = _generation_windows_from_rows(ar_rows)
        return {
            **base,
            "tok_s": (sum(tok_values) / len(tok_values)) if tok_values else None,
            "generated_tokens": sum(int(row.get("generated_tokens") or 0) for row in ar_rows),
            "generation_windows": generation_windows,
            "generation_window_s": sum(window["duration_s"] for window in generation_windows),
            "verify_ms_per_call": None,
            "quality_passed": True if ar_rows else None,
        }
    depth_rows = data.get("depths") or data.get("depth_results") or []
    row = depth_rows[0] if depth_rows else {}
    summary = row.get("summary") or {}
    generation_rows = [
        result_row
        for depth_row in depth_rows
        for result_row in depth_row.get("rows", [])
        if isinstance(result_row, dict)
    ]
    generation_windows = _generation_windows_from_rows(generation_rows)
    verify_calls = int(summary.get("verify_calls") or 0)
    validations = [
        validation
        for depth_row in depth_rows
        for result_row in depth_row.get("rows", [])
        for validation in result_row.get("validations", [])
    ]
    acceptance = summary.get("acceptance_by_depth")
    if acceptance is None:
        acceptance = _rate_lists(
            summary.get("accepted_by_depth") or [],
            summary.get("drafted_by_depth") or [],
        )
    return {
        **base,
        "tok_s": summary.get("mean_tok_s"),
        "generated_tokens": summary.get("generated_tokens"),
        "elapsed_s": summary.get("elapsed_s"),
        "generation_windows": generation_windows,
        "generation_window_s": sum(window["duration_s"] for window in generation_windows),
        "verify_time_s": summary.get("verify_time_s"),
        "verify_calls": verify_calls,
        "verify_ms_per_call": _ms_per_call(summary.get("verify_time_s"), verify_calls),
        "verify_forward_ms_per_call": _ms_per_call(summary.get("verify_forward_time_s"), verify_calls),
        "verify_eval_ms_per_call": _ms_per_call(summary.get("verify_eval_time_s"), verify_calls),
        "verify_hidden_eval_ms_per_call": _ms_per_call(summary.get("verify_hidden_eval_time_s"), verify_calls),
        "accepted_by_depth": summary.get("accepted_by_depth"),
        "drafted_by_depth": summary.get("drafted_by_depth"),
        "acceptance_by_depth": acceptance,
        "acceptance_percent_by_depth": [
            (None if value is None else 100.0 * float(value)) for value in acceptance
        ],
        "mean_accept_probability_by_depth": summary.get("mean_accept_probability_by_depth"),
        "quality_passed": all(bool(v.get("passed")) for v in validations) if validations else None,
        "validations_total": len(validations),
        "validations_passed": sum(1 for v in validations if v.get("passed")),
    }


def _tune_payload(
    *,
    action: str,
    run_id: str,
    model: str,
    settings: dict[str, Any],
    rows: list[dict[str, Any]],
    output_root: Path,
    output_path: Path,
    hardware: dict[str, Any],
    software: dict[str, Any],
    backend: dict[str, Any],
    thermal: dict[str, Any],
    state_key: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = _annotate_multipliers(rows)
    best = _best_multiplier_summary(rows)
    return {
        "action": action,
        "run_id": run_id,
        "model": model,
        "profile": "performance-cold",
        "suite": settings["suite"],
        "settings": settings,
        "results": rows,
        "best": best.get("winner"),
        "best_multiplier": best,
        "hardware": hardware,
        "software": software,
        "backend": backend,
        "thermal": thermal,
        "diagnostics": diagnostics or {},
        "state_key": state_key,
        "artifacts": {
            "root": str(output_root),
            "summary": str(output_path),
        },
    }


def _annotate_multipliers(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ar_tok_s = None
    for row in results:
        if row.get("mode") == "AR":
            ar_tok_s = row.get("tok_s")
            break
    annotated = []
    for row in results:
        item = dict(row)
        if item.get("mode") == "AR":
            item["multiplier_vs_ar"] = 1.0 if isinstance(item.get("tok_s"), (int, float)) else None
        else:
            item["multiplier_vs_ar"] = item.get("speedup_vs_ar") or _safe_ratio(item.get("tok_s"), ar_tok_s)
        annotated.append(item)
    return annotated


def _best_multiplier_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    annotated = _annotate_multipliers(results)
    ar = next((row for row in annotated if row.get("mode") == "AR"), None)
    if not ar or not isinstance(ar.get("tok_s"), (int, float)):
        return {
            "available": False,
            "ar_tok_s": None,
            "winner": None,
            "verdict": "tune_failed_no_ar_result",
        }
    candidates = [
        row
        for row in annotated
        if row.get("depth") is not None
        and isinstance(row.get("multiplier_vs_ar"), (int, float))
        and float(row["multiplier_vs_ar"]) > 1.0
    ]
    raw_winner = max(candidates, key=lambda row: float(row["multiplier_vs_ar"]), default=None)
    winner = raw_winner
    tie_margin_pct = max(
        0.0,
        _tune_env_float(
            "MTPLX_TUNE_TIE_PREFER_DEEPER_WITHIN_PCT",
            TUNE_TIE_PREFER_DEEPER_WITHIN_PCT,
        ),
    )
    if raw_winner is not None and tie_margin_pct > 0.0:
        raw_tok_s = raw_winner.get("tok_s")
        if isinstance(raw_tok_s, (int, float)) and float(raw_tok_s) > 0.0:
            floor = float(raw_tok_s) * (1.0 - tie_margin_pct / 100.0)
            tied = [
                row
                for row in candidates
                if isinstance(row.get("tok_s"), (int, float))
                and float(row["tok_s"]) >= floor
            ]
            winner = max(
                tied,
                key=lambda row: (
                    int(row.get("depth") or 0),
                    float(row.get("tok_s") or 0.0),
                ),
                default=raw_winner,
            )
    if winner is None:
        return {
            "available": bool(ar and isinstance(ar.get("tok_s"), (int, float))),
            "ar_tok_s": ar.get("tok_s") if ar else None,
            "winner": None,
            "verdict": "no_mtp_depth_beat_ar",
        }
    raw_winner_changed = (
        raw_winner is not None
        and raw_winner.get("mode") != winner.get("mode")
    )
    return {
        "available": True,
        "ar_tok_s": ar.get("tok_s") if ar else None,
        "winner": {
            "mode": winner.get("mode"),
            "depth": winner.get("depth"),
            "tok_s": winner.get("tok_s"),
            "multiplier_vs_ar": winner.get("multiplier_vs_ar"),
        },
        "raw_winner": {
            "mode": raw_winner.get("mode"),
            "depth": raw_winner.get("depth"),
            "tok_s": raw_winner.get("tok_s"),
            "multiplier_vs_ar": raw_winner.get("multiplier_vs_ar"),
        }
        if raw_winner is not None
        else None,
        "tie_breaker": {
            "applied": raw_winner_changed,
            "prefer_deeper_within_pct": tie_margin_pct,
        },
        "verdict": "mtp_depth_wins",
    }


def _print_tune_dry_run_human(payload: dict[str, Any]) -> None:
    print("MTPLX Tune")
    print("dry-run: no model will be loaded")
    print(f"model: {payload.get('model')}")
    print(f"output: {payload.get('output')}")
    print("candidate commands:")
    for row in payload.get("candidates", []):
        print(f"  {row.get('candidate')}: {shlex.join(row.get('command') or [])}")


def _stat_avg(stats: Any) -> float | None:
    if isinstance(stats, dict) and isinstance(stats.get("avg"), (int, float)):
        return float(stats["avg"])
    return None


def _format_tune_telemetry_inline(telemetry: Any) -> str | None:
    if not isinstance(telemetry, dict) or not telemetry.get("enabled"):
        return None
    display = telemetry
    scope = "candidate"
    generation = telemetry.get("generation")
    if isinstance(generation, dict) and generation.get("sample_count"):
        display = generation
        scope = "generation"
    parts: list[str] = []
    power = display.get("power_w") or {}
    power_parts = []
    for label, key in (("pkg", "package"), ("cpu", "cpu"), ("ane", "ane"), ("gpu", "gpu")):
        value = _stat_avg(power.get(key))
        if value is not None:
            power_parts.append(f"{label}={value:.1f}W")
    if power_parts:
        parts.append("power " + " ".join(power_parts))

    frequency = display.get("frequency_ghz") or {}
    frequency_parts = []
    for label, key in (("P", "p_cluster"), ("M", "m_cluster"), ("GPU", "gpu")):
        value = _stat_avg(frequency.get(key))
        if value is not None:
            frequency_parts.append(f"{label}={value:.2f}GHz")
    if frequency_parts:
        parts.append("freq " + " ".join(frequency_parts))

    temperature = display.get("temperature_c") or {}
    temp_parts = []
    core_avg = _stat_avg(temperature.get("core_avg"))
    core_max = _stat_avg(temperature.get("core_max"))
    gpu_avg = _stat_avg(temperature.get("gpu_avg"))
    if core_avg is not None:
        temp_parts.append(f"core_avg={core_avg:.1f}C")
    if core_max is not None:
        temp_parts.append(f"core_max={core_max:.1f}C")
    if gpu_avg is not None:
        temp_parts.append(f"gpu_avg={gpu_avg:.1f}C")
    if temp_parts:
        parts.append("temp " + " ".join(temp_parts))

    utilization = display.get("utilization_pct") or {}
    utilization_parts = []
    for label, key in (("P", "p_core"), ("M", "m_core"), ("GPU", "gpu")):
        value = _stat_avg(utilization.get(key))
        if value is not None:
            utilization_parts.append(f"{label}={value:.1f}%")
    if utilization_parts:
        parts.append("util " + " ".join(utilization_parts))

    fans = display.get("fans_rpm") or {}
    fan_avg = _stat_avg(fans.get("avg"))
    if fan_avg is not None:
        parts.append(f"fans={fan_avg:.0f}rpm")

    sample_count = display.get("sample_count")
    if isinstance(sample_count, int):
        parts.append(f"samples={sample_count}")
    window_duration = display.get("window_duration_s")
    if scope == "generation" and isinstance(window_duration, (int, float)):
        parts.append(f"window={float(window_duration):.1f}s")
    errors = display.get("errors") or telemetry.get("errors") or []
    if parts and errors:
        parts.append("notes=" + "; ".join(str(error) for error in errors[:2]))
    if not parts:
        if errors:
            return f"unavailable ({'; '.join(str(error) for error in errors[:2])})"
        return None
    parts.insert(0, f"scope={scope}")
    return " | ".join(parts)


def _print_tune_human(payload: dict[str, Any], *, verbose: bool = False) -> None:
    print("MTPLX Tune")
    if payload.get("from_cache"):
        print("Using saved tuning. Run `mtplx tune --retune` to measure again.")
    else:
        artifacts = payload.get("artifacts") or {}
        if artifacts.get("root"):
            print(f"Results written to {artifacts.get('root')}")
    print()
    best = payload.get("best") or {}
    for row in payload.get("results", []):
        mode = str(row.get("mode") or "?")
        tok_s = _fmt_metric(row.get("tok_s"), digits=1)
        multiplier = _fmt_metric(row.get("multiplier_vs_ar"), digits=2)
        marker = "  BEST" if best.get("mode") == mode else ""
        print(f"{mode:<4} {tok_s:>6} tok/s   {multiplier}x{marker}")
    print()
    errors = [row for row in payload.get("results", []) if row.get("error")]
    if errors:
        print("Tune failed for one or more candidates:")
        for row in errors:
            print(f"  {row.get('mode')}: {row.get('error')} (log: {row.get('stdout')})")
    elif best:
        print(
            f"Best for this Mac: {best.get('mode')}, "
            f"{_fmt_metric(best.get('multiplier_vs_ar'), digits=2)}x AR"
        )
        best_multiplier = payload.get("best_multiplier") or {}
        tie_breaker = best_multiplier.get("tie_breaker") or {}
        raw_winner = best_multiplier.get("raw_winner") or {}
        if tie_breaker.get("applied") and raw_winner.get("mode"):
            print(
                f"Tie-break: {best.get('mode')} was within "
                f"{_fmt_metric(tie_breaker.get('prefer_deeper_within_pct'), digits=1)}% "
                f"of raw fastest {raw_winner.get('mode')}; preferred the deeper depth."
            )
    else:
        print("No MTP depth beat AR on this run")
    if payload.get("saved") and best:
        print(f"Saved: Web UI starts will use depth {best.get('depth')} for this model.")
    elif payload.get("save_skipped_reason"):
        print(f"Not saved: {payload.get('save_skipped_reason')}.")
    if verbose:
        print()
        for row in payload.get("results", []):
            if row.get("depth") is not None:
                print(
                    f"{row.get('mode')}: verify_ms={_fmt_metric(row.get('verify_ms_per_call'), digits=2)} "
                    f"acceptance={_format_depth_acceptance(row)}"
                )
            telemetry_line = _format_tune_telemetry_inline(row.get("telemetry"))
            if telemetry_line:
                print(f"{row.get('mode')}: telemetry={telemetry_line}")
        artifacts = payload.get("artifacts") or {}
        if artifacts:
            print(f"artifacts: {artifacts.get('root')}")


def _tune_error(
    message: str,
    *,
    detail: str | None = None,
    actionable: str | None = None,
    thermal: dict[str, Any] | None = None,
    json_output: bool = False,
) -> int:
    payload: dict[str, Any] = {"error": message}
    if detail:
        payload["detail"] = detail
    if actionable:
        payload["actionable"] = actionable
    if thermal is not None:
        payload["thermal"] = thermal
    if json_output:
        _print(payload)
    else:
        print(f"error: {message}", file=sys.stderr)
        if detail:
            print(f"detail: {detail}", file=sys.stderr)
        if actionable:
            print(f"action: {actionable}", file=sys.stderr)
    return 1

def cmd_pull_public(args: Any) -> int:
    from optiq.runtime.mtp.hf_loader import pull_model, repo_id_from_model_ref

    json_mode = bool(getattr(args, "json", False))
    callback = None

    def finalize() -> None:
        return None

    progress_interval_s = 10.0
    if not json_mode:
        callback, finalize = _rich_download_progress_callback(
            repo_id=repo_id_from_model_ref(args.model) or args.model,
        )
        progress_interval_s = 0.4
    try:
        result = pull_model(
            args.model,
            cache_dir=args.cache_dir,
            revision=args.revision,
            progress_callback=callback,
            progress_interval_s=progress_interval_s,
        )
    except KeyboardInterrupt:
        finalize()
        print("download cancelled")
        return 130
    except Exception as exc:
        finalize()
        if json_mode:
            _print({"error": "pull failed", "model": args.model, "detail": str(exc)})
        else:
            print("error: pull failed")
            print(f"model: {args.model}")
            print(f"detail: {exc}")
        return 1
    finalize()
    if json_mode:
        _print(result)
    else:
        print("MTPLX pull")
        print(f"model: {result.get('repo_id')}")
        print(f"path: {result.get('path')}")
        print(f"size: {_format_bytes(result.get('size_bytes'))}")
        print(f"runtime contract: {str(bool(result.get('has_runtime_contract'))).lower()}")
    return 0


def cmd_list_public(args: Any) -> int:
    from optiq.runtime.mtp.hf_loader import list_cached_models, model_cache_dir

    models = [row.to_dict() for row in list_cached_models(cache_dir=args.cache_dir)]
    payload = {"cache_dir": str(model_cache_dir(args.cache_dir)), "models": models}
    if getattr(args, "json", False):
        _print(payload)
    else:
        print("MTPLX models")
        print(f"cache: {payload['cache_dir']}")
        if not models:
            print("no cached models")
        for row in models:
            print(
                f"- {row.get('repo_id')}  "
                f"{_format_bytes(row.get('size_bytes'))}  "
                f"contract={str(bool(row.get('has_runtime_contract'))).lower()}"
            )
            print(f"  {row.get('path')}")
    return 0


def cmd_remove_public(args: Any) -> int:
    from optiq.runtime.mtp.hf_loader import remove_cached_model

    result = remove_cached_model(args.model, cache_dir=args.cache_dir)
    if getattr(args, "json", False):
        _print(result)
    else:
        print("MTPLX remove")
        print(f"model: {result.get('repo_id')}")
        print(f"path: {result.get('path')}")
        if result.get("removed"):
            print(f"removed: {_format_bytes(result.get('size_bytes_removed'))}")
        else:
            print("removed: false")
    return 0 if result["removed"] or args.missing_ok else 1


def _cmd_bench_run(args: Any) -> int:
    model = args.model or DEFAULT_CHAMPION
    suite = args.suite or "default"
    selected_profile = get_profile(_bench_run_profile_name(args, suite=suite))
    args.profile = selected_profile.name
    prompt_suite = prompt_suite_path(suite)
    run_id = args.run_id or f"cli-bench-{suite}-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir = Path(args.output_dir or "outputs/cli/bench") / run_id
    output = Path(args.output) if args.output else output_dir / "depth-sweep.json"
    envelope_output = output_dir / "envelope.json"
    decode_trace = output_dir / "decode-trace.jsonl"
    exact_paged_env = _exact_paged_env_from_args(args)
    runtime_profile = selected_profile.runtime_profile
    runtime_env = selected_profile.env_dict()
    if selected_profile.name in {"safe", "exact", "max-diagnostic"}:
        runtime_env.update(exact_paged_env)
    runtime_env = _runtime_env_with_external_overrides(runtime_env)
    harness = getattr(args, "harness", "auto")
    if harness == "auto":
        harness = "depth-sweep" if selected_profile.name == "performance-cold" else "direct-http"
    benchmark_seed = _benchmark_seed(args, runtime_profile=runtime_profile, harness=harness)
    depths = _depths_for_bench_run(args)

    if args.dry_run:
        direct_command = _direct_http_bench_command(
            args,
            model=model,
            suite=suite,
            run_id=run_id,
            output_dir=output_dir,
            seed=benchmark_seed,
        )
        _print(
            {
                "dry_run": True,
                "action": f"bench {getattr(args, 'bench_action', None) or 'run'}",
                "model": model,
                "suite": suite,
                "prompt_suite": prompt_suite,
                "exactness_smoke": {
                    "context": 2048,
                    "automatic": True,
                    "profile": _exactness_profile_kwargs(args),
                },
                "harness": harness,
                "depths": depths if harness == "depth-sweep" else None,
                "compare_ar": bool(getattr(args, "compare_ar", False))
                if harness == "depth-sweep"
                else None,
                "seed": benchmark_seed,
                "profile": selected_profile.to_dict(),
                "runtime_profile": runtime_profile,
                "runtime_env": runtime_env,
                "direct_http_command": direct_command if harness == "direct-http" else None,
                "exact_paged_env": exact_paged_env,
                "output": str(output),
                "envelope": str(envelope_output),
                "decode_trace": str(decode_trace),
            }
        )
        return 0
    runtime_model, resolve_error = _resolve_runtime_model_path(
        model,
        cache_dir=getattr(args, "cache_dir", None),
    )
    if resolve_error is not None:
        _print(resolve_error)
        return 1
    inspection, gate_exit = _model_gate(
        runtime_model,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=bool(getattr(args, "yes", False)),
    )
    if gate_exit is not None:
        _print({"error": "model failed MTP primary gate", "model": inspection})
        return gate_exit
    draft_lm_head = _model_draft_lm_head_spec(inspection, selected_profile)
    draft_sampler = _model_draft_sampler_spec(inspection, selected_profile)

    from optiq.runtime.mtp.benchmarks.runners.preflight import run_preflight

    preflight = run_preflight(".")
    smoke = run_exactness_smoke(
        runtime_model,
        context=2048,
        prompt_suite=prompt_suite_path("flappy"),
        output=output_dir / "exactness-smoke.json",
        **_exactness_profile_kwargs(args),
    )
    if not smoke["passed"]:
        write_json(envelope_output, {"run_id": run_id, "correctness": {"exactness_smoke": smoke}})
        _print({"error": "Phase 0H exactness smoke failed", "exactness_smoke": smoke})
        return EXIT_EXACTNESS

    if harness == "direct-http":
        return _cmd_bench_run_direct_http(
            args,
            model=runtime_model,
            suite=suite,
            run_id=run_id,
            output_dir=output_dir,
            envelope_output=envelope_output,
            exactness_smoke=smoke,
            runtime_profile=runtime_profile,
            runtime_env=runtime_env,
            preflight=preflight,
            model_inspection=inspection,
            seed=benchmark_seed,
        )

    from optiq.runtime.mtp.benchmarks.runners.mtp_depth_sweep import write_depth_sweep

    decode_trace.parent.mkdir(parents=True, exist_ok=True)
    with _temporary_env(
        {
            "MTPLX_DECODE_TRACE_JSONL": str(decode_trace),
            "MTPLX_DECODE_TRACE_INTERVAL_S": str(args.trace_interval_s),
            "MTPLX_DECODE_TRACE_LABEL": run_id,
            **runtime_env,
        }
    ):
        result = _depth_sweep_native60(
            model=runtime_model,
            prompt_suite=prompt_suite,
            depths=depths,
            max_tokens=args.max_tokens,
            limit=args.limit,
            seed=benchmark_seed,
            draft_lm_head=draft_lm_head,
            draft_sampler=draft_sampler,
            compare_ar=bool(getattr(args, "compare_ar", False)),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_depth_sweep(output, result)
    envelope = build_benchmark_envelope(
        result=result,
        model_inspection=inspection,
        run_id=run_id,
        suite=suite,
        runtime_profile=runtime_profile,
        runtime_env=runtime_env,
        exactness_smoke=smoke,
        fan_controlled=bool(args.fanmax),
        strict=bool(args.strict),
        strict_cold=bool(args.strict_cold),
        telemetry={
            "telemetry_unavailable": False,
            "source": "bench-preflight",
            "power": preflight.get("power"),
            "issues": preflight.get("issues"),
            "deep_smc_trace_attached": False,
        },
        decode_trace_path=decode_trace,
    )
    envelope["artifacts"] = {
        "depth_sweep": str(output),
        "envelope": str(envelope_output),
        "decode_trace": str(decode_trace),
    }
    write_json(envelope_output, envelope)
    _print(_bench_run_console_summary(envelope))

    if not envelope["quality"]["passed"]:
        return EXIT_QUALITY
    if args.strict or args.strict_cold:
        if envelope.get("strict_passed") is False:
            return EXIT_STRICT_GATE
    return 0


def _bench_run_profile_name(args: Any, *, suite: str) -> str:
    requested = getattr(args, "profile", None)
    if requested:
        return str(requested)
    if suite in BENCH_SUSTAINED_DEFAULT_SUITES:
        return "sustained"
    try:
        max_tokens = int(getattr(args, "max_tokens", 0) or 0)
    except (TypeError, ValueError):
        max_tokens = 0
    if (
        suite in BENCH_SUSTAINED_LENGTH_SENSITIVE_SUITES
        and max_tokens > BENCH_SUSTAINED_MAX_TOKENS_THRESHOLD
    ):
        return "sustained"
    return DEFAULT_PROFILE_NAME


def _direct_http_bench_command(
    args: Any,
    *,
    model: str,
    suite: str,
    run_id: str,
    output_dir: Path,
    seed: int,
) -> list[str]:
    if suite in {"python_modules_long", "python-modules-long"}:
        test_name = "python_modules_long"
    elif suite in {"long_code_uncapped", "long-code-uncapped"}:
        test_name = "long_code_uncapped"
    elif suite in {"long_code", "long-code", "cold-long-code-192"}:
        test_name = "long_code"
    else:
        test_name = suite
    if test_name not in {"flappy", "python_modules_long", "long_code_uncapped", "long_code"}:
        test_name = "flappy"
    generation_mode = _generation_mode_from_args(args)
    load_mtp_flag = "--no-load-mtp" if bool(getattr(args, "stock_ar", False)) else "--load-mtp"
    ablation_profile = (
        "sustained"
        if getattr(args, "profile", None) == "sustained"
        else LONG_RESPONSE_DIRECT_PROFILE
    )
    command = [
        sys.executable,
        str(repo_root() / "scripts" / "run_context_degradation_diagnostics.py"),
        "local-ablation",
        "--label",
        run_id,
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir / "direct-http"),
        "--port",
        str(getattr(args, "port", 8041) or 8041),
        "--model",
        model,
        "--model-id",
        Path(model).name,
        "--generation-mode",
        generation_mode,
        load_mtp_flag,
        "--depth",
        "3",
        "--verify-strategy",
        "capture_commit",
        "--verify-core",
        "linear-gdn-from-conv-tape",
        "--profiles",
        ablation_profile,
        "--tests",
        test_name,
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--seed",
        str(seed),
        "--trace-interval-s",
        str(getattr(args, "trace_interval_s", 1.0)),
        "--python-bin",
        sys.executable,
        "--request-timeout-s",
        "2400",
        "--startup-timeout-s",
        "600",
    ]
    max_tokens = getattr(args, "max_tokens", None)
    if max_tokens is not None and test_name != "long_code_uncapped":
        command.extend(["--max-tokens", str(int(max_tokens))])
    return command


def _validation_dicts_for_text(text: str, *, suite: str) -> list[dict[str, Any]]:
    validations = [validate_no_degenerate_loop(text), validate_balanced_delimiters(text)]
    if suite in {"python_modules_long", "python-modules-long", "long_code", "long-code", "cold-long-code-192"}:
        validations.append(validate_python_syntax(text))
    return [validation.__dict__ for validation in validations]


def _cmd_bench_run_direct_http(
    args: Any,
    *,
    model: str,
    suite: str,
    run_id: str,
    output_dir: Path,
    envelope_output: Path,
    exactness_smoke: dict[str, Any],
    runtime_profile: str,
    runtime_env: dict[str, str],
    preflight: dict[str, Any],
    model_inspection: dict[str, Any],
    seed: int,
) -> int:
    command = _direct_http_bench_command(
        args,
        model=model,
        suite=suite,
        run_id=run_id,
        output_dir=output_dir,
        seed=seed,
    )
    proc = subprocess.run(
        command,
        cwd=repo_root(),
        env={**os.environ, **runtime_env},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    context_summary = output_dir / "direct-http" / run_id / "ablation-summary.json"
    row: dict[str, Any] = {}
    if context_summary.exists():
        summary = json.loads(context_summary.read_text(encoding="utf-8"))
        rows = summary.get("rows") or []
        row = dict(rows[0]) if rows else {}
    row_error = row.get("error") if isinstance(row.get("error"), dict) else None
    text = ""
    content_path = row.get("content_path")
    if content_path and Path(content_path).exists():
        text = Path(content_path).read_text(encoding="utf-8", errors="replace")
    validations = _validation_dicts_for_text(text, suite=suite) if text else []
    quality_failures = [
        {
            "prompt_id": row.get("test") or row.get("suite") or suite,
            "validation": validation.get("name"),
            "detail": validation.get("detail", ""),
        }
        for validation in validations
        if not validation.get("passed")
    ]
    if row_error is not None:
        quality_failures.append(
            {
                "prompt_id": row.get("test") or row.get("suite") or suite,
                "validation": "bench_runtime_error",
                "detail": str(row_error.get("message") or row_error),
            }
        )
    trace_summary = row.get("trace_summary") or {}
    runtime = {
        "generated_tokens": row.get("completion_tokens"),
        "elapsed_s": row.get("request_elapsed_s"),
        "tok_s": row.get("decode_tok_s"),
        "mean_tok_s": row.get("decode_tok_s"),
        "verify_ms_per_call": None,
        "late_verify_ms": row.get("late_verify_ms"),
        "accepted_by_depth": None,
        "drafted_by_depth": None,
        "mean_accept_probability_by_depth": None,
        "acceptance_by_depth": None,
        "correction_tokens": None,
        "bonus_tokens": None,
    }
    trace = {
        "path": trace_summary.get("trace_path"),
        "available": bool(trace_summary),
        "buckets": trace_summary.get("trace_rows"),
        "positive_buckets": trace_summary.get("trace_usable_rows"),
        "generated_tokens": trace_summary.get("trace_generated_tokens") or row.get("completion_tokens"),
        "first64_tok_s": row.get("first64"),
        "last64_tok_s": row.get("last64"),
        "last64_over_first64": row.get("last64_over_first64"),
        "first10_tok_s": row.get("first10_tok_s"),
        "last10_tok_s": row.get("last10_tok_s"),
        "last10_over_first10": row.get("last10_over_first10"),
        "late_verify_ms": row.get("late_verify_ms"),
        "cache_gib_last": row.get("last_cache_gib"),
    }
    strict_gates: dict[str, bool] = {}
    if args.strict:
        strict_gates = {
            "flappy_tok_s_ge_50": bool(runtime["tok_s"] is not None and float(runtime["tok_s"]) >= 50.0),
            "last64_over_first64_ge_0_90": bool(trace["last64_over_first64"] is not None and float(trace["last64_over_first64"]) >= 0.90),
            "last10_over_first10_ge_0_85": bool(trace["last10_over_first10"] is not None and float(trace["last10_over_first10"]) >= 0.85),
            "late_verify_le_75ms": bool(trace["late_verify_ms"] is not None and float(trace["late_verify_ms"]) <= 75.0),
            "telemetry_available": False,
            "no_fan_control": not bool(args.fanmax),
        }
    envelope = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "suite": suite,
        "model": model_inspection,
        "fan_controlled": bool(args.fanmax),
        "harness": "direct-http",
        "runtime_profile": runtime_profile,
        "runtime_env": runtime_env,
        "fast_path_env": row.get("fast_path_env"),
        "runtime": runtime,
        "decode_trace": trace,
        "thermal": {
            "telemetry_unavailable": False,
            "source": "bench-preflight",
            "power": preflight.get("power"),
            "issues": preflight.get("issues"),
            "deep_smc_trace_attached": False,
        },
        "dispatch": {"dispatch_trace_attached": False, "command_buffers_per_token": None},
        "quality": {
            "passed": not quality_failures,
            "failures": quality_failures,
            "validations": validations,
            "acceptance_smells": [],
        },
        "correctness": {
            "exactness_smoke": exactness_smoke,
            "full_exactness": None,
            "distribution_exactness": None,
        },
        "strict_gates": strict_gates,
        "strict_passed": all(strict_gates.values()) if strict_gates else None,
        "artifacts": {
            "context_summary": str(context_summary),
            "envelope": str(envelope_output),
            "server_stdout": str(output_dir / "direct-http-command.log"),
            "content": content_path,
            "events": row.get("events_path"),
        },
        "direct_returncode": proc.returncode,
        "error": row_error,
    }
    (output_dir / "direct-http-command.log").write_text(proc.stdout, encoding="utf-8")
    write_json(envelope_output, envelope)
    _print(_bench_run_console_summary(envelope))
    if proc.returncode != 0 or row_error is not None or runtime["generated_tokens"] is None:
        return EXIT_STRICT_GATE
    if not envelope["quality"]["passed"]:
        return EXIT_QUALITY
    if args.strict and envelope.get("strict_passed") is False:
        return EXIT_STRICT_GATE
    return 0


def _nightly_tasks(args: Any) -> list[dict[str, Any]]:
    sustained_profile = get_profile(getattr(args, "profile", None) or DEFAULT_PROFILE_NAME).name
    return [
        {
            "label": "cold-long-code-192",
            "suite": "cold-long-code-192",
            "max_tokens": 192,
            "profile": "performance-cold",
            "strict": False,
            "strict_cold": True,
            "harness": "auto",
        },
        {
            "label": "flappy-6k",
            "suite": "flappy",
            "max_tokens": 6000,
            "profile": sustained_profile,
            "strict": bool(getattr(args, "strict", False)),
            "strict_cold": False,
            "harness": "direct-http",
        },
        {
            "label": "flappy-10k",
            "suite": "flappy",
            "max_tokens": 10000,
            "profile": sustained_profile,
            "strict": bool(getattr(args, "strict", False)),
            "strict_cold": False,
            "harness": "direct-http",
        },
        {
            "label": "python-modules-6k",
            "suite": "python_modules_long",
            "max_tokens": 6000,
            "profile": sustained_profile,
            "strict": False,
            "strict_cold": False,
            "harness": "direct-http",
        },
    ]


def _cmd_bench_nightly(args: Any) -> int:
    model = args.model or DEFAULT_CHAMPION
    run_id = args.run_id or f"cli-nightly-{time.strftime('%Y%m%d-%H%M%S')}"
    output = Path(args.output or Path("outputs/cli/nightly") / run_id / "summary.json")
    task_root = Path(args.output_dir or output.parent)
    tasks = _nightly_tasks(args)
    exactness_output = output.parent / "phase0h-full-exactness.json"
    exactness_cmd = [
        "qa",
        "exactness",
        "--model",
        model,
        "--contexts",
        str(getattr(args, "nightly_exactness_contexts", "64,2048,6144,10240")),
        "--output",
        str(exactness_output),
    ]
    if args.dry_run:
        dry_tasks = []
        for task in tasks:
            child = type("BenchArgs", (), vars(args).copy())()
            child.model = model
            child.suite = task["suite"]
            child.max_tokens = task["max_tokens"]
            child.profile = task["profile"]
            child.strict = task["strict"]
            child.strict_cold = task["strict_cold"]
            child.harness = task["harness"]
            child.run_id = f"{run_id}-{task['label']}"
            child.output_dir = str(task_root)
            dry_tasks.append(
                {
                    **task,
                    "run_id": child.run_id,
                    "direct_http_command": _direct_http_bench_command(
                        child,
                        model=model,
                        suite=task["suite"],
                        run_id=child.run_id,
                        output_dir=task_root / child.run_id,
                        seed=_benchmark_seed(
                            child,
                            runtime_profile=get_profile(task["profile"]).runtime_profile,
                            harness=task["harness"],
                        ),
                    )
                    if task["harness"] == "direct-http"
                    else None,
                }
            )
        _print(
            {
                "dry_run": True,
                "action": "bench nightly",
                "model": model,
                "run_id": run_id,
                "tasks": dry_tasks,
                "full_exactness_command": exactness_cmd,
                "output": str(output),
                "policy": {
                    "fanmax_counts_for_product_gate": False,
                    "cold_floor_tok_s": 59.0,
                    "sustained_target_tok_s": 50.0,
                    "sustained_first_stage_tok_s": 45.0,
                },
            }
        )
        return 0

    results: list[dict[str, Any]] = []
    worst_exit = 0
    for task in tasks:
        child = type("BenchArgs", (), vars(args).copy())()
        child.model = model
        child.suite = task["suite"]
        child.max_tokens = task["max_tokens"]
        child.profile = task["profile"]
        child.strict = task["strict"]
        child.strict_cold = task["strict_cold"]
        child.harness = task["harness"]
        child.fanmax = False
        child.run_id = f"{run_id}-{task['label']}"
        child.output_dir = str(task_root)
        code = _cmd_bench_run(child)
        envelope_path = task_root / child.run_id / "envelope.json"
        envelope = (
            json.loads(envelope_path.read_text(encoding="utf-8"))
            if envelope_path.exists()
            else {}
        )
        results.append(
            {
                "label": task["label"],
                "suite": task["suite"],
                "max_tokens": task["max_tokens"],
                "profile": task["profile"],
                "exit_code": code,
                "envelope": envelope,
                "envelope_path": str(envelope_path),
            }
        )
        worst_exit = max(worst_exit, int(code))

    exact_proc = _run_exactness_command(
        [
            "--model",
            model,
            "--contexts",
            str(getattr(args, "nightly_exactness_contexts", "64,2048,6144,10240")),
            "--attention-impl",
            str(getattr(args, "exactness_attention_impl", "mlx_vector_paged")),
            "--block-size",
            str(getattr(args, "exactness_block_size", 16)),
            "--num-blocks",
            str(getattr(args, "exactness_num_blocks", 1024)),
            "--partition-threshold",
            str(getattr(args, "exactness_partition_threshold", 2048)),
            "--partition-size",
            str(getattr(args, "exactness_partition_size", 512)),
            "--output",
            str(exactness_output),
            *(
                ["--no-partitioned"]
                if getattr(args, "exactness_no_partitioned", False)
                else ["--partitioned"]
            ),
        ]
    )
    cold = next((row for row in results if row["label"] == "cold-long-code-192"), {})
    flappy_6k = next((row for row in results if row["label"] == "flappy-6k"), {})
    flappy_10k = next((row for row in results if row["label"] == "flappy-10k"), {})
    quality_passed = all(
        bool((row.get("envelope") or {}).get("quality", {}).get("passed"))
        for row in results
    )
    cold_tok_s = ((cold.get("envelope") or {}).get("runtime") or {}).get("tok_s")
    f6_tok_s = ((flappy_6k.get("envelope") or {}).get("runtime") or {}).get("tok_s")
    f10_tok_s = ((flappy_10k.get("envelope") or {}).get("runtime") or {}).get("tok_s")
    f10_ratio = ((flappy_10k.get("envelope") or {}).get("decode_trace") or {}).get(
        "last64_over_first64"
    )
    gates = {
        "full_exactness_passed": exact_proc.returncode == 0,
        "cold_tok_s_ge_59": bool(cold_tok_s is not None and float(cold_tok_s) >= 59.0),
        "flappy_6k_tok_s_ge_45": bool(f6_tok_s is not None and float(f6_tok_s) >= 45.0),
        "flappy_10k_tok_s_ge_45": bool(f10_tok_s is not None and float(f10_tok_s) >= 45.0),
        "flappy_10k_decay_ratio_ge_0_85": bool(f10_ratio is not None and float(f10_ratio) >= 0.85),
        "quality_passed": quality_passed,
        "no_fan_product_gate": not bool(getattr(args, "fanmax", False)),
    }
    summary = {
        "action": "bench nightly",
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model,
        "tasks": results,
        "full_exactness": {
            "returncode": exact_proc.returncode,
            "passed": exact_proc.returncode == 0,
            "output": str(exactness_output),
            "stdout_tail": exact_proc.stdout[-4000:],
        },
        "gates": gates,
        "passed": all(gates.values()),
        "policy": {
            "fanmax_counts_for_product_gate": False,
            "cold_floor_tok_s": 59.0,
            "sustained_target_tok_s": 50.0,
            "sustained_first_stage_tok_s": 45.0,
        },
    }
    write_json(output, summary)
    _print(
        {
            "action": "bench nightly",
            "run_id": run_id,
            "output": str(output),
            "passed": summary["passed"],
            "gates": gates,
            "task_outputs": [
                {"label": row["label"], "exit_code": row["exit_code"], "envelope": row["envelope_path"]}
                for row in results
            ],
        }
    )
    if exact_proc.returncode != 0:
        worst_exit = max(worst_exit, EXIT_EXACTNESS)
    if getattr(args, "strict", False) and not summary["passed"]:
        worst_exit = max(worst_exit, EXIT_STRICT_GATE)
    return worst_exit


def _load_benchmark_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object in {path}")
    return data


def _envelope_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if payload.get("action") == "bench nightly" or "tasks" in payload:
        rows: dict[str, dict[str, Any]] = {}
        for task in payload.get("tasks") or []:
            label = str(task.get("label") or task.get("suite") or "unknown")
            envelope = task.get("envelope") or task
            if isinstance(envelope, dict):
                rows[label] = envelope
        return rows
    label = str(payload.get("suite") or payload.get("run_id") or "envelope")
    return {label: payload}


def _metric_float(row: dict[str, Any], *path: str) -> float | None:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cmd_bench_compare_envelopes(args: Any) -> int:
    if not args.before or not args.after:
        raise SystemExit("bench compare envelope mode requires --before and --after")
    before = _load_benchmark_json(args.before)
    after = _load_benchmark_json(args.after)
    before_rows = _envelope_rows(before)
    after_rows = _envelope_rows(after)
    labels = sorted(set(before_rows) & set(after_rows))
    if not labels:
        raise SystemExit("no matching benchmark labels between --before and --after")
    comparisons = []
    gates: dict[str, bool] = {}
    tolerance_pct = float(getattr(args, "cold_regression_tolerance_pct", 2.0))
    for label in labels:
        left = before_rows[label]
        right = after_rows[label]
        before_tok_s = _metric_float(left, "runtime", "tok_s")
        after_tok_s = _metric_float(right, "runtime", "tok_s")
        before_ratio = _metric_float(left, "decode_trace", "last64_over_first64")
        after_ratio = _metric_float(right, "decode_trace", "last64_over_first64")
        tok_s_delta = (
            after_tok_s - before_tok_s
            if before_tok_s is not None and after_tok_s is not None
            else None
        )
        tok_s_delta_pct = (
            (tok_s_delta / before_tok_s) * 100.0
            if tok_s_delta is not None and before_tok_s
            else None
        )
        quality_after = bool((right.get("quality") or {}).get("passed", True))
        exactness_after = bool(
            ((right.get("correctness") or {}).get("exactness_smoke") or {}).get(
                "passed",
                True,
            )
        )
        label_gates = {
            "quality_after": quality_after,
            "exactness_after": exactness_after,
        }
        if getattr(args, "strict_cold", False) and "cold" in label:
            label_gates["cold_floor_ge_59"] = bool(after_tok_s is not None and after_tok_s >= 59.0)
            label_gates["cold_regression_within_tolerance"] = bool(
                tok_s_delta_pct is not None and tok_s_delta_pct >= -tolerance_pct
            )
        if getattr(args, "strict", False) and ("flappy" in label or "10k" in label or "6k" in label):
            label_gates["sustained_tok_s_ge_45"] = bool(after_tok_s is not None and after_tok_s >= 45.0)
            label_gates["decay_ratio_ge_0_85"] = bool(after_ratio is not None and after_ratio >= 0.85)
        if getattr(args, "strict_exactness", False):
            label_gates["strict_exactness_after"] = exactness_after
        gates[label] = all(label_gates.values())
        comparisons.append(
            {
                "label": label,
                "before_tok_s": before_tok_s,
                "after_tok_s": after_tok_s,
                "tok_s_delta": tok_s_delta,
                "tok_s_delta_pct": tok_s_delta_pct,
                "before_last64_over_first64": before_ratio,
                "after_last64_over_first64": after_ratio,
                "gates": label_gates,
                "passed": gates[label],
            }
        )
    report = {
        "action": "bench compare envelopes",
        "before": str(args.before),
        "after": str(args.after),
        "strict": bool(args.strict),
        "strict_cold": bool(args.strict_cold),
        "strict_exactness": bool(getattr(args, "strict_exactness", False)),
        "comparisons": comparisons,
        "passed": all(gates.values()),
    }
    output = Path(args.output) if args.output else None
    if output is not None:
        write_json(output, report)
    _print(report)
    return 0 if report["passed"] else EXIT_STRICT_GATE


def _cmd_bench_compare(args: Any) -> int:
    if getattr(args, "before", None) or getattr(args, "after", None):
        return _cmd_bench_compare_envelopes(args)
    models = args.models or []
    if not models:
        raise SystemExit("bench compare requires --models PATH_A PATH_B [...]")
    suite = args.suite or "champion-bakeoff"
    run_id = args.run_id or f"cli-compare-{time.strftime('%Y%m%d-%H%M%S')}"
    if suite == "champion-bakeoff":
        tasks = [
            {
                "label": "flappy-10k",
                "suite": "flappy",
                "max_tokens": 10000,
                "strict": bool(args.strict),
                "strict_cold": False,
            },
            {
                "label": "python-modules-long",
                "suite": "python_modules_long",
                "max_tokens": 6000,
                "strict": False,
                "strict_cold": False,
            },
            {
                "label": "cold-long-code-192",
                "suite": "cold-long-code-192",
                "max_tokens": 192,
                "strict": False,
                "strict_cold": True,
            },
        ]
    else:
        tasks = [
            {
                "label": suite,
                "suite": suite,
                "max_tokens": int(args.max_tokens),
                "strict": bool(args.strict),
                "strict_cold": bool(args.strict_cold),
            }
        ]
    if args.dry_run:
        _print(
            {
                "dry_run": True,
                "action": "bench compare",
                "models": models,
                "suite": suite,
                "tasks": tasks,
                "exactness_smoke_per_model": True,
            }
        )
        return 0
    results = []
    worst_exit = 0
    for model in models:
        for task in tasks:
            child = type("BenchArgs", (), vars(args).copy())()
            child.model = model
            child.suite = task["suite"]
            child.max_tokens = task["max_tokens"]
            child.strict = task["strict"]
            child.strict_cold = task["strict_cold"]
            child.run_id = f"{run_id}-{Path(model).name}-{task['label']}"
            child.output_dir = str(Path(args.output_dir or "outputs/cli/compare") / run_id)
            code = _cmd_bench_run(child)
            envelope_path = Path(child.output_dir) / child.run_id / "envelope.json"
            envelope = (
                json.loads(envelope_path.read_text(encoding="utf-8"))
                if envelope_path.exists()
                else {}
            )
            results.append(
                {
                    "model": model,
                    "task": task,
                    "exit_code": code,
                    "envelope": envelope,
                }
            )
            worst_exit = max(worst_exit, code)

    scorecards = []
    for model in models:
        model_rows = [row for row in results if row["model"] == model]
        by_label = {row["task"]["label"]: row for row in model_rows}
        quality_passed = all(
            row.get("envelope", {}).get("quality", {}).get("passed") for row in model_rows
        )
        cold_row = by_label.get("cold-long-code-192", {})
        cold_gate = (
            cold_row.get("envelope", {})
            .get("strict_gates", {})
            .get("cold_tok_s_ge_55")
        )
        metrics = {
            label: row.get("envelope", {}).get("runtime", {})
            for label, row in by_label.items()
        }
        eligible = bool(model_rows) and quality_passed and cold_gate is not False
        scorecards.append(
            {
                "model": model,
                "eligible": eligible,
                "quality_passed": quality_passed,
                "cold_tok_s_ge_55": cold_gate,
                "metrics": metrics,
            }
        )
    eligible_scorecards = [row for row in scorecards if row["eligible"]]
    recommended = None
    if eligible_scorecards:
        recommended = max(
            eligible_scorecards,
            key=lambda row: (
                float(row["metrics"].get("flappy-10k", {}).get("tok_s") or 0.0),
                float(row["metrics"].get("python-modules-long", {}).get("tok_s") or 0.0),
                float(row["metrics"].get("cold-long-code-192", {}).get("tok_s") or 0.0),
            ),
        )
    summary = {
        "run_id": run_id,
        "suite": suite,
        "results": results,
        "scorecards": scorecards,
        "recommended_champion": recommended["model"] if recommended else None,
        "champion_policy": "No-fan sustained and quality outrank fanmax tie-breakers.",
    }
    champion_path = Path("mtplx/champion.json")
    write_json(Path(args.output or Path("outputs/cli/compare") / run_id / "summary.json"), summary)
    if args.record_champion and recommended:
        write_json(
            champion_path,
            {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "model": recommended["model"],
                "suite": suite,
                "reason": "highest eligible no-fan champion-bakeoff scorecard",
                "run_id": run_id,
                "scorecard": recommended,
            },
        )
        summary["recorded_champion"] = str(champion_path)
    _print(summary)
    return worst_exit


def _http_json(url: str, *, timeout: float = 15.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc), "url": url}


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 15.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            return {
                "ok": 200 <= int(response.status) < 300,
                "status": int(response.status),
                "json": json.loads(text),
            }
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(text)
        except json.JSONDecodeError:
            parsed = text
        return {"ok": False, "status": exc.code, "error": parsed, "url": url}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc), "url": url}


def _http_post_text(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 15.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            return {
                "ok": 200 <= int(response.status) < 300,
                "status": int(response.status),
                "preview": text[:1200],
            }
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "error": text[:1200], "url": url}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"ok": False, "error": str(exc), "url": url}


def _cmd_bench_serve(args: Any) -> int:
    base = args.url.rstrip("/")
    health = _http_json(base + "/health")
    metrics = _http_json(base + "/metrics")
    report = {
        "suite": args.suite or "multiturn-flappy",
        "turns": args.turns,
        "health": health,
        "metrics": metrics,
        "required_cache_hit_ratio_turn_2_plus": 0.85,
        "note": "This smoke validates the server surface; full multi-turn generation remains the serving QA expansion point.",
    }
    _print(report)
    return 0 if health.get("ok") else EXIT_STRICT_GATE


def _cmd_bench_reference(args: Any) -> int:
    _print(
        {
            "action": "bench reference",
            "champion": args.champion,
            "references": args.references,
            "suite": args.suite,
            "max_tokens": args.max_tokens,
            "product_gate": False,
            "note": "Reference commands are diagnostic floors and do not promote MTPLX runs.",
        }
    )
    return 0


def _ssh_command(host: str, remote_script: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        host,
        f"bash -lc {shlex.quote(remote_script)}",
    ]


def _run_ssh(host: str, remote_script: str, *, timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _ssh_command(host, remote_script),
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout_s,
    )


def _remote_read_text(host: str, path: str, *, timeout_s: int = 30) -> tuple[str | None, str | None]:
    proc = _run_ssh(host, f"cat {shlex.quote(path)}", timeout_s=timeout_s)
    if proc.returncode != 0:
        return None, proc.stdout[-2000:]
    return proc.stdout, None


def _remote_stat(host: str, path: str, *, timeout_s: int = 30) -> dict[str, Any] | None:
    proc = _run_ssh(host, f"stat -c '%Y %s' {shlex.quote(path)}", timeout_s=timeout_s)
    if proc.returncode != 0:
        return None
    parts = proc.stdout.strip().split()
    if len(parts) < 2:
        return None
    try:
        return {"mtime_epoch": int(parts[0]), "size_bytes": int(parts[1])}
    except ValueError:
        return None


def _remote_epoch(host: str) -> int | None:
    proc = _run_ssh(host, "date +%s", timeout_s=10)
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def _remote_probe_script(args: Any) -> str:
    venv_python = str(Path(args.remote_venv) / "bin" / "python")
    return "\n".join(
        [
            "set -e",
            "echo host=$(hostname)",
            "date -Is",
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            f"{shlex.quote(venv_python)} - <<'PY'",
            "import importlib.metadata as md",
            "for pkg in ('vllm', 'torch', 'transformers', 'flashinfer-python'):",
            "    try:",
            "        print(f'{pkg}=' + md.version(pkg))",
            "    except Exception as exc:",
            "        print(f'{pkg}=unavailable:{exc}')",
            "PY",
            "command -v nsys",
            f"test -d {shlex.quote(args.remote_phase_dir)}",
            f"test -f {shlex.quote(str(Path(args.remote_phase_dir) / args.remote_run_script))}",
        ]
    )


def _write_remote_artifact(local_dir: Path, name: str, text: str | None) -> str | None:
    if text is None:
        return None
    path = local_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _remote_reference_prompt_line(args: Any) -> str:
    prompt_path = Path(prompt_suite_path(args.suite or "flappy"))
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                break
        else:
            raise SystemExit(f"empty prompt suite: {prompt_path}")
    row["suite"] = args.suite or row.get("suite") or "reference"
    row["max_tokens"] = int(args.max_tokens)
    row.setdefault("id", f"reference_{args.suite or 'prompt'}")
    if "prompt" not in row:
        raise SystemExit(f"remote vLLM reference requires prompt field in {prompt_path}")
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _remote_capture_script(args: Any, *, remote_run_cmd: str) -> str:
    prompt_line = _remote_reference_prompt_line(args)
    prompt_file = str(Path(args.remote_phase_dir) / "profile_prompt.jsonl")
    backup_file = str(Path(args.remote_phase_dir) / ".profile_prompt.jsonl.mtplx_backup")
    return "\n".join(
        [
            "set -e",
            f"if [ -f {shlex.quote(prompt_file)} ]; then cp {shlex.quote(prompt_file)} {shlex.quote(backup_file)}; fi",
            "restore_prompt() {",
            f"  if [ -f {shlex.quote(backup_file)} ]; then mv {shlex.quote(backup_file)} {shlex.quote(prompt_file)}; fi",
            "}",
            "trap restore_prompt EXIT",
            f"printf '%s\\n' {shlex.quote(prompt_line)} > {shlex.quote(prompt_file)}",
            remote_run_cmd,
        ]
    )


def _remote_offline_capture_script(args: Any, *, remote_out_dir: str) -> str:
    prompt_line = _remote_reference_prompt_line(args)
    offline_python = r'''from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["no-mtp", "mtp5"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.prompt_file.read_text().splitlines() if line.strip()]
    row = rows[0]
    prompt = row["prompt"]
    spec_summary = None
    if args.mode == "mtp5":
        spec_summary = {"method": "mtp", "num_speculative_tokens": 5}
    llm = LLM(
        model=args.model,
        served_model_name="qwen3.6-27b",
        tokenizer=args.model,
        quantization="compressed-tensors",
        tensor_parallel_size=2,
        max_model_len=32768,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        block_size=32,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        trust_remote_code=False,
        dtype="bfloat16",
        speculative_config=dict(spec_summary) if spec_summary else None,
        profiler_config={"profiler": "cuda"},
        disable_log_stats=True,
    )
    warm = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=min(64, max(1, args.max_tokens)),
        seed=42,
    )
    llm.generate([prompt], warm, use_tqdm=False)
    sampling = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_tokens,
        seed=42,
    )
    llm.start_profile()
    started = time.perf_counter()
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    elapsed = time.perf_counter() - started
    completion = outputs[0].outputs[0]
    completion_tokens = len(completion.token_ids)
    result = {
        "summary": {
            "label": f"offline-nsys-{args.mode}",
            "model": "qwen3.6-27b",
            "prompt_count": 1,
            "ok_count": 1,
            "total_completion_tokens": completion_tokens,
            "mean_decode_tok_s": completion_tokens / elapsed if elapsed else None,
            "mean_end_to_end_tok_s": completion_tokens / elapsed if elapsed else None,
        },
        "rows": [
            {
                "id": row.get("id"),
                "suite": row.get("suite"),
                "status": "ok",
                "max_tokens": args.max_tokens,
                "completion_tokens": completion_tokens,
                "wall_s": elapsed,
                "decode_tok_s": completion_tokens / elapsed if elapsed else None,
                "end_to_end_tok_s": completion_tokens / elapsed if elapsed else None,
                "text_prefix": completion.text[:240],
            }
        ],
        "sampler": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 42},
        "speculative_config": spec_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    llm.stop_profile()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    return f"""set -e
OUT_DIR={shlex.quote(remote_out_dir)}
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cleanup_vllm_children() {{
  for pattern in "[V]LLM::EngineCore" "[V]LLM::Worker" "[v]llm serve" "offline_capture.py" "multiprocessing.resource_tracker"; do
    pids="$(pgrep -f "$pattern" 2>/dev/null | awk -v self="$$" '$1 != self {{print}}' || true)"
    if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi
  done
  sleep 2
  for pattern in "[V]LLM::EngineCore" "[V]LLM::Worker" "[v]llm serve" "offline_capture.py" "multiprocessing.resource_tracker"; do
    pids="$(pgrep -f "$pattern" 2>/dev/null | awk -v self="$$" '$1 != self {{print}}' || true)"
    if [ -n "$pids" ]; then kill -KILL $pids 2>/dev/null || true; fi
  done
}}
trap cleanup_vllm_children EXIT
PROMPT_FILE="$OUT_DIR/profile_prompt.jsonl"
OFFLINE_SCRIPT="$OUT_DIR/offline_capture.py"
BENCH_JSON="$OUT_DIR/bench.json"
STDOUT_LOG="$OUT_DIR/stdout.log"
REPORT_BASE="$OUT_DIR/nsys-{shlex.quote(args.remote_mode)}"
printf '%s\\n' {shlex.quote(prompt_line)} > "$PROMPT_FILE"
cat > "$OFFLINE_SCRIPT" <<'PY'
{offline_python}
PY
source {shlex.quote(str(Path(args.remote_venv) / "bin" / "activate"))}
export CUDA_VISIBLE_DEVICES=0,1
export RAY_memory_monitor_refresh_ms=0
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export VLLM_SLEEP_WHEN_IDLE=0
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
set +e
nsys profile --force-overwrite=true --wait=primary --trace=cuda,nvtx --cuda-graph-trace=graph --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown -o "$REPORT_BASE" \\
  {shlex.quote(str(Path(args.remote_venv) / "bin" / "python"))} "$OFFLINE_SCRIPT" --mode {shlex.quote(args.remote_mode)} --model /home/youssof/ai/models/Qwen3.6-27B-AWQ-BF16-INT4 --prompt-file "$PROMPT_FILE" --output "$BENCH_JSON" --max-tokens {int(args.max_tokens)} > "$STDOUT_LOG" 2>&1
NSYS_STATUS=$?
cleanup_vllm_children
set -e
REP="${{REPORT_BASE}}.nsys-rep"
if [ -f "$REP" ]; then
  nsys stats --force-overwrite=true --force-export=true --report cuda_gpu_kern_sum "$REP" > "$OUT_DIR/cuda_gpu_kern_sum.txt" 2>&1 || true
  nsys stats --force-overwrite=true --force-export=true --report cuda_api_sum "$REP" > "$OUT_DIR/cuda_api_sum.txt" 2>&1 || true
  nsys stats --force-overwrite=true --force-export=true --report cuda_gpu_trace "$REP" > "$OUT_DIR/cuda_gpu_trace.txt" 2>&1 || true
  nsys stats --force-overwrite=true --force-export=true --report nvtx_sum "$REP" > "$OUT_DIR/nvtx_sum.txt" 2>&1 || true
fi
find "$OUT_DIR" -maxdepth 1 -type f -printf '%f %s\\n' | sort
exit "$NSYS_STATUS"
"""


def _cmd_bench_reference_vllm(args: Any) -> int:
    run_id = args.run_id or f"vllm-reference-{args.remote_mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    local_dir = Path(args.remote_output_dir or "outputs/cli/reference-vllm") / run_id
    remote_kind = getattr(args, "remote_capture_kind", "offline")
    remote_out_dir = str(Path(args.remote_phase_dir) / f"nsys-v4-{remote_kind}-{args.remote_mode}")
    remote_script = str(Path(args.remote_phase_dir) / args.remote_run_script)
    remote_run_cmd = (
        f"cd {shlex.quote(args.remote_phase_dir)} && "
        f"bash {shlex.quote(remote_script)} {shlex.quote(args.remote_mode)} {int(args.remote_port)}"
    )
    capture_script = (
        _remote_offline_capture_script(args, remote_out_dir=remote_out_dir)
        if remote_kind == "offline"
        else _remote_capture_script(args, remote_run_cmd=remote_run_cmd)
    )
    if args.dry_run:
        _print(
            {
                "dry_run": True,
                "action": "bench reference-vllm",
                "ssh_host": args.ssh_host,
                "remote_capture_kind": remote_kind,
                "probe_command": _ssh_command(args.ssh_host, _remote_probe_script(args)),
                "capture_command": _ssh_command(args.ssh_host, capture_script),
                "remote_prompt_override": json.loads(_remote_reference_prompt_line(args)),
                "local_output_dir": str(local_dir),
                "product_gate": False,
            }
        )
        return 0

    local_dir.mkdir(parents=True, exist_ok=True)
    probe = _run_ssh(args.ssh_host, _remote_probe_script(args), timeout_s=60)
    _write_remote_artifact(local_dir, "remote-probe.txt", probe.stdout)
    if probe.returncode != 0:
        report = {
            "run_id": run_id,
            "action": "bench reference-vllm",
            "passed": False,
            "error": "remote 3090 probe failed",
            "returncode": probe.returncode,
            "stdout_tail": probe.stdout[-4000:],
        }
        write_json(local_dir / "summary.json", report)
        _print(report)
        return EXIT_STRICT_GATE

    capture_error = None
    capture_started_epoch = None
    if args.capture_dispatch:
        capture_started_epoch = _remote_epoch(args.ssh_host)
        try:
            capture = _run_ssh(args.ssh_host, capture_script, timeout_s=int(args.remote_timeout_s))
        except subprocess.TimeoutExpired as exc:
            capture_error = {
                "error": "remote capture timed out",
                "timeout_s": args.remote_timeout_s,
                "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else None,
            }
        else:
            _write_remote_artifact(local_dir, "remote-capture.txt", capture.stdout)
            if capture.returncode != 0:
                capture_error = {
                    "error": "remote Nsight capture failed",
                    "returncode": capture.returncode,
                    "stdout_tail": capture.stdout[-4000:],
                }

    kernel_text, kernel_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "cuda_gpu_kern_sum.txt"),
    )
    cuda_api_text, cuda_api_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "cuda_api_sum.txt"),
    )
    bench_text, bench_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "bench.json"),
    )
    client_log, _client_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "client.log"),
    )
    stdout_log, _stdout_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "stdout.log"),
    )
    kernel_stat = _remote_stat(args.ssh_host, str(Path(remote_out_dir) / "cuda_gpu_kern_sum.txt"))
    cuda_api_stat = _remote_stat(args.ssh_host, str(Path(remote_out_dir) / "cuda_api_sum.txt"))
    bench_stat = _remote_stat(args.ssh_host, str(Path(remote_out_dir) / "bench.json"))
    kernel_stale = bool(
        args.capture_dispatch
        and capture_started_epoch is not None
        and kernel_stat
        and int(kernel_stat["mtime_epoch"]) < int(capture_started_epoch)
    )
    cuda_api_stale = bool(
        args.capture_dispatch
        and capture_started_epoch is not None
        and cuda_api_stat
        and int(cuda_api_stat["mtime_epoch"]) < int(capture_started_epoch)
    )
    _write_remote_artifact(local_dir, "cuda_gpu_kern_sum.txt", kernel_text)
    _write_remote_artifact(local_dir, "cuda_api_sum.txt", cuda_api_text)
    _write_remote_artifact(local_dir, "bench.json", bench_text)
    _write_remote_artifact(local_dir, "client.log", client_log)
    _write_remote_artifact(local_dir, "stdout.log", stdout_log)

    reference = summarize_vllm_reference(
        cuda_kernel_summary_text=kernel_text,
        cuda_api_summary_text=cuda_api_text,
        bench_json_text=bench_text,
    )
    capture_artifacts_valid = bool(
        kernel_text
        and bench_text
        and not kernel_stale
        and (not cuda_api_text or not cuda_api_stale)
    )
    capture_warning = None
    if capture_error and capture_artifacts_valid:
        capture_warning = {
            "warning": "remote Nsight command returned nonzero but fresh artifacts were recovered",
            "capture_error": capture_error,
        }
    report = {
        "run_id": run_id,
        "action": "bench reference-vllm",
        "ssh_host": args.ssh_host,
        "remote_mode": args.remote_mode,
        "remote_capture_kind": remote_kind,
        "remote_phase_dir": args.remote_phase_dir,
        "remote_out_dir": remote_out_dir,
        "suite": args.suite,
        "max_tokens": args.max_tokens,
        "capture_dispatch": bool(args.capture_dispatch),
        "capture_started_epoch": capture_started_epoch,
        "product_gate": False,
        "remote_prompt_override": json.loads(_remote_reference_prompt_line(args)),
        "capture_error": None if capture_warning else capture_error,
        "capture_warning": capture_warning,
        "probe_stdout_tail": probe.stdout[-4000:],
        "artifact_errors": {
            "cuda_gpu_kern_sum": kernel_error,
            "cuda_api_sum": cuda_api_error,
            "bench_json": bench_error,
        },
        "artifact_stats": {
            "cuda_gpu_kern_sum": kernel_stat,
            "cuda_api_sum": cuda_api_stat,
            "bench_json": bench_stat,
            "cuda_gpu_kern_sum_stale_for_capture": kernel_stale,
            "cuda_api_sum_stale_for_capture": cuda_api_stale,
        },
        "reference": reference,
        "local_output_dir": str(local_dir),
    }
    write_json(local_dir / "summary.json", report)
    _print(report)
    if args.capture_dispatch:
        return 0 if capture_artifacts_valid else EXIT_STRICT_GATE
    return 0 if bench_text else EXIT_STRICT_GATE


def cmd_qa_public(args: Any) -> int:
    if args.qa_action == "exactness":
        return _cmd_qa_exactness(args)
    if args.qa_action == "distribution":
        return _cmd_qa_distribution(args)
    raise SystemExit(f"unknown qa action: {args.qa_action}")


def _run_exactness_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(repo_root() / "scripts" / "phase0h_paged_verifier_exactness.py"), *args],
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _cmd_qa_exactness(args: Any) -> int:
    output = Path(args.output) if args.output else default_output_path("qa-exactness")
    cmd = [
        "--model",
        args.model,
        "--contexts",
        args.contexts,
        "--attention-impl",
        str(args.exactness_attention_impl),
        "--block-size",
        str(args.exactness_block_size),
        "--num-blocks",
        str(args.exactness_num_blocks),
        "--partition-threshold",
        str(args.exactness_partition_threshold),
        "--partition-size",
        str(args.exactness_partition_size),
        "--output",
        str(output),
    ]
    if args.exactness_no_partitioned:
        cmd.append("--no-partitioned")
    else:
        cmd.append("--partitioned")
    if args.prompt_suite:
        cmd.extend(["--prompt-suite", args.prompt_suite])
    proc = _run_exactness_command(cmd)
    if proc.returncode == 0:
        print(proc.stdout, end="")
        return 0
    output_text = proc.stdout or ""
    print("error: exactness check failed")
    if "float_to_fp8_e4m3" in output_text:
        print(
            "detail: the selected paged-attention exactness path failed to "
            "compile in the Metal backend on this machine."
        )
    else:
        candidate_lines = [
            line.strip()
            for line in output_text.splitlines()
            if any(
                marker in line.lower()
                for marker in ("runtimeerror:", "error:", "failed", "undeclared identifier")
            )
        ]
        for line in reversed(candidate_lines or output_text.splitlines()):
            stripped = line.strip()
            if stripped:
                print(f"detail: {stripped[:240]}")
                break
    print(f"output: {output}")
    print("try: mtplx qa exactness --exactness-attention-impl mlx_vector_paged")
    return EXIT_EXACTNESS


def _cmd_qa_distribution(args: Any) -> int:
    suite_names = distribution_suite_names(args.suite)
    rows = []
    worst = 0
    for suite_name in suite_names:
        output = Path(args.output_dir or "outputs/cli/qa-distribution") / f"{suite_name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
        proc = _run_exactness_command(
            [
                "--model",
                args.model,
                "--contexts",
                args.contexts,
                "--attention-impl",
                str(args.exactness_attention_impl),
                "--block-size",
                str(args.exactness_block_size),
                "--num-blocks",
                str(args.exactness_num_blocks),
                "--partition-threshold",
                str(args.exactness_partition_threshold),
                "--partition-size",
                str(args.exactness_partition_size),
                "--prompt-suite",
                prompt_suite_path(suite_name),
                "--output",
                str(output),
                *(
                    ["--no-partitioned"]
                    if args.exactness_no_partitioned
                    else ["--partitioned"]
                ),
            ]
        )
        rows.append(
            {
                "suite": suite_name,
                "output": str(output),
                "returncode": proc.returncode,
                "passed": proc.returncode == 0,
                "stdout_tail": proc.stdout[-2000:],
            }
        )
        worst = max(worst, proc.returncode)
    report = {
        "model": args.model,
        "reference_stack": args.reference_stack,
        "contexts": args.contexts,
        "tolerance": args.tolerance,
        "rows": rows,
        "passed": worst == 0,
    }
    _print(report)
    return 0 if worst == 0 else EXIT_EXACTNESS


def cmd_profile_public(args: Any) -> int:
    if args.profile_action == "dispatch":
        return _cmd_profile_dispatch(args)
    if args.profile_action == "thermal":
        return _cmd_profile_thermal(args)
    if args.profile_action == "compile-audit":
        return _cmd_profile_compile_audit(args)
    if args.profile_action == "eval-attribution":
        return _cmd_profile_eval_attribution(args)
    raise SystemExit(f"unknown profile action: {args.profile_action}")


def _cmd_profile_dispatch(args: Any) -> int:
    if args.trace:
        out_dir = Path(args.output_dir or "outputs/cli/dispatch") / time.strftime("%Y%m%d-%H%M%S")
        proc = subprocess.run(
            [
                sys.executable,
                str(repo_root() / "scripts" / "analyze_metal_command_trace.py"),
                args.trace,
                "--out-dir",
                str(out_dir),
            ],
            cwd=repo_root(),
            text=True,
            check=False,
        )
        return proc.returncode
    _print(
        {
            "action": "profile dispatch",
            "model": args.model,
            "suite": args.suite,
            "max_tokens": args.max_tokens,
            "implemented_capture": False,
            "next": "Run with --trace PATH to analyze an existing MLX Metal command trace.",
        }
    )
    return 0


def _cmd_profile_thermal(args: Any) -> int:
    cmd = [
        sys.executable,
        str(repo_root() / "scripts" / "run_flappy_smc_thermal_diagnostics.py"),
        "--model",
        args.model,
        "--run-id",
        args.run_id or f"cli-thermal-{time.strftime('%Y%m%d-%H%M%S')}",
        "--output-dir",
        args.output_dir or "outputs/cli/thermal",
    ]
    if args.dry_run:
        _print({"dry_run": True, "command": cmd})
        return 0
    return subprocess.call(cmd, cwd=repo_root())


def _cmd_profile_compile_audit(args: Any) -> int:
    inspection, gate_exit = _model_gate(args.model)
    output = Path(args.output) if args.output else (
        Path(args.output_dir or "outputs/cli/compile-audit")
        / f"compile-audit-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    cmd = [
        sys.executable,
        str(repo_root() / "scripts" / "probe_mx_compile_buckets.py"),
        "--model",
        args.model,
        "--prompts",
        args.prompts,
        "--prompt-index",
        str(args.prompt_index),
        "--prefill-chunks",
        args.prefill_chunks,
        "--depths",
        args.depths,
        "--max-tokens",
        str(args.max_tokens),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--verify-core",
        args.verify_core,
        "--output",
        str(output),
    ]
    if args.disable_thinking:
        cmd.append("--disable-thinking")
    if args.skip_prefill:
        cmd.append("--skip-prefill")
    if args.skip_verify:
        cmd.append("--skip-verify")
    if args.dry_run:
        _print(
            {
                "dry_run": True,
                "action": "profile compile-audit",
                "model": inspection,
                "exactness_smoke": {
                    "automatic": not args.skip_exactness_smoke,
                    "context": 2048,
                    "profile": _exactness_profile_kwargs(args),
                },
                "exact_paged_env": _exact_paged_env_from_args(args),
                "command": cmd,
                "output": str(output),
            }
        )
        return 0
    if gate_exit is not None:
        _print({"error": "model failed MTP primary gate", "model": inspection})
        return gate_exit
    if not args.skip_exactness_smoke:
        smoke = run_exactness_smoke(
            args.model,
            context=2048,
            prompt_suite=prompt_suite_path("flappy"),
            output=output.with_name(output.stem + "-exactness-smoke.json"),
            **_exactness_profile_kwargs(args),
        )
        if not smoke["passed"]:
            write_json(
                output.with_name(output.stem + "-failed.json"),
                {"error": "Phase 0H exactness smoke failed", "exactness_smoke": smoke},
            )
            _print({"error": "Phase 0H exactness smoke failed", "exactness_smoke": smoke})
            return EXIT_EXACTNESS
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        env={**os.environ, **_exact_paged_env_from_args(args)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="")
    return proc.returncode


def _cmd_profile_eval_attribution(args: Any) -> int:
    inspection, gate_exit = _model_gate(args.model)
    output = Path(args.output) if args.output else (
        Path(args.output_dir or "outputs/cli/eval-attribution")
        / f"eval-attribution-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    cmd = [
        sys.executable,
        str(repo_root() / "scripts" / "probe_eval_attribution.py"),
        "--model",
        args.model,
        "--prefix-tokens",
        str(args.prefix_tokens),
        "--verify-tokens",
        str(args.verify_tokens),
        "--seed",
        str(args.seed),
        "--depth",
        str(args.depth),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--verify-strategy",
        args.verify_strategy,
        "--verify-core",
        args.verify_core,
        "--mtp-history-policy",
        args.mtp_history_policy,
        "--orders",
        args.orders,
        "--output",
        str(output),
    ]
    if args.no_serving_fast_defaults:
        cmd.append("--no-serving-fast-defaults")
    if args.prompt:
        cmd.extend(["--prompt", args.prompt])
    payload = {
        "action": "profile eval-attribution",
        "model": inspection,
        "command": cmd,
        "output": str(output),
        "purpose": (
            "Attribute verify-cycle first-eval debt across verifier outputs, "
            "attention cache, and recurrent GDN/conv state before committing "
            "to a larger owned kernel boundary."
        ),
    }
    if args.dry_run:
        _print({"dry_run": True, **payload})
        return 0
    if gate_exit is not None:
        _print({"error": "model failed MTP primary gate", "model": inspection})
        return gate_exit
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="")
    return proc.returncode


def cmd_thermal_public(args: Any) -> int:
    if args.thermal_action != "fanmax-run":
        raise SystemExit(f"unknown thermal action: {args.thermal_action}")
    run_id = args.run_id or f"cli-fanmax-{time.strftime('%Y%m%d-%H%M%S')}"
    child = [
        sys.executable,
        "-m",
        "mtplx.cli",
        "bench",
        "run",
        "--model",
        args.model,
        "--suite",
        args.suite,
        "--max-tokens",
        str(args.max_tokens),
        "--run-id",
        run_id,
        "--fanmax",
    ]
    cmd = [
        sys.executable,
        str(repo_root() / "scripts" / "run_fanmax_command.py"),
        "--output-dir",
        args.output_dir or "outputs/cli/fanmax",
        "--",
        *child,
    ]
    if args.dry_run:
        _print({"dry_run": True, "command": cmd})
        return 0
    return subprocess.call(cmd, cwd=repo_root())


def cmd_max_public(args: Any) -> int:
    from optiq.runtime.mtp.thermal import (
        _read_max_marker,
        check_and_recover_stale_max,
        install_passwordless_sudoers_rule,
        install_thermal_control_homebrew,
        remove_passwordless_sudoers_rule,
        set_thermal_profile,
        thermal_status,
    )

    action = args.max_action
    if action == "install":
        payload = install_thermal_control_homebrew(
            install_daemon=not bool(getattr(args, "no_daemon", False)),
        )
        code = 0 if payload.get("ok") else 1
    elif action == "grant_sudo":
        payload = install_passwordless_sudoers_rule()
        code = 0 if payload.get("ok") else 1
    elif action == "revoke_sudo":
        payload = remove_passwordless_sudoers_rule()
        code = 0 if payload.get("ok") else 1
    elif action == "status":
        # Run stale-max recovery before reporting status so `mtplx max --status`
        # also doubles as the "fix my fans!" command after a crash.
        recovery = check_and_recover_stale_max()
        payload = thermal_status()
        payload["max_marker"] = _read_max_marker()
        payload["recovered_stale_max"] = recovery
        code = 0
    elif action == "silent":
        # `mtplx max --off` — explicit user request to restore fans now.
        # Also clear the marker so `--status` doesn't report stale state.
        from optiq.runtime.mtp.thermal import _clear_max_marker

        payload = set_thermal_profile("silent", dry_run=bool(getattr(args, "dry_run", False)))
        if payload.get("ok") and not getattr(args, "dry_run", False):
            _clear_max_marker()
        code = 0 if payload.get("ok") or getattr(args, "dry_run", False) else 1
    else:
        payload = set_thermal_profile(action, dry_run=bool(getattr(args, "dry_run", False)))
        code = 0 if payload.get("ok") or getattr(args, "dry_run", False) else 1
    if getattr(args, "json", False):
        _print(payload)
    else:
        if action in ("install", "grant_sudo", "revoke_sudo"):
            print(f"thermal {action} ok: {str(bool(payload.get('ok'))).lower()}")
            if payload.get("message"):
                print(payload["message"])
        else:
            selected = (payload.get("detection") or {}).get("selected") or {}
            tool = selected.get("kind", "none")
            print(f"thermal tool: {tool}")
            if action == "status":
                print(f"available: {str(bool((payload.get('detection') or {}).get('available'))).lower()}")
            else:
                print(f"profile: {action}")
                print(f"ok: {str(bool(payload.get('ok'))).lower()}")
            if payload.get("message"):
                print(payload["message"])
            elif (
                not bool((payload.get("detection") or {}).get("available"))
                and (payload.get("detection") or {}).get("instructions")
            ):
                print((payload.get("detection") or {})["instructions"])
    return code


def _server_url(host: str, port: int) -> str:
    return local_url_for_bind(host, int(port))


def _chat_url(host: str, port: int) -> str:
    return _server_url(host, port) + "/"


def _openwebui_docker_api_base_url(port: int) -> str:
    return f"http://host.docker.internal:{int(port)}/v1"


def _openwebui_docker_command(
    *,
    mtplx_port: int,
    webui_port: int = 3000,
    single_user: bool = False,
    api_key: str = "mtplx-local",
) -> list[str]:
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        "open-webui",
        "--restart",
        "unless-stopped",
        "-p",
        f"{int(webui_port)}:8080",
        "--add-host=host.docker.internal:host-gateway",
        "-e",
        "ENABLE_OPENAI_API=True",
        "-e",
        f"OPENAI_API_BASE_URL={_openwebui_docker_api_base_url(mtplx_port)}",
        "-e",
        f"OPENAI_API_KEY={api_key}",
    ]
    if single_user:
        command.extend(["-e", "WEBUI_AUTH=False"])
    command.extend(
        [
            "-v",
            "open-webui:/app/backend/data",
            "ghcr.io/open-webui/open-webui:main",
        ]
    )
    return command


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _open_browser_url(url: str) -> None:
    try:
        webbrowser.open(url, new=2, autoraise=True)
    except Exception as exc:
        _print_serve_start_line(f"warning: could not open browser automatically: {exc}")


def _connect_host_for_bind(host: str) -> str:
    return connect_host_for_bind(host)


def _port_is_busy(host: str, port: int) -> bool:
    try:
        with socket.create_connection((_connect_host_for_bind(host), int(port)), timeout=0.2):
            return True
    except OSError:
        return False


def _active_mlx_fork_status(*, expected_fragment: str, expected_commit: str | None) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec("mlx.core")
    except Exception as exc:
        return {
            "ok": False,
            "error": repr(exc),
            "expected_path_fragment": expected_fragment,
            "expected_commit": expected_commit,
        }
    if spec is None or not spec.origin:
        return {
            "ok": False,
            "error": "mlx.core is not installed",
            "expected_path_fragment": expected_fragment,
            "expected_commit": expected_commit,
        }
    path = Path(spec.origin).resolve()
    try:
        version = importlib.metadata.version("mlx")
    except Exception:
        version = None
    commit = None
    if expected_fragment in str(path):
        for parent in [path.parent, *path.parents]:
            if expected_fragment in parent.name or expected_fragment in str(parent):
                try:
                    commit = subprocess.check_output(
                        ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    ).strip()
                except Exception:
                    commit = None
                break
    ok = expected_fragment in str(path) and (
        expected_commit is None or commit in {None, expected_commit}
    )
    return {
        "ok": ok,
        "path": str(path),
        "version": version,
        "expected_path_fragment": expected_fragment,
        "expected_commit": expected_commit,
        "observed_commit": commit,
    }


def _apple_hardware_context() -> dict[str, Any]:
    mem_bytes = _sysctl_int("hw.memsize")
    chip = _sysctl_text("machdep.cpu.brand_string")
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "macos_version": _command_text(["sw_vers", "-productVersion"]),
        "macos_build": _command_text(["sw_vers", "-buildVersion"]),
        "chip": chip,
        "chip_family": _apple_chip_family(chip),
        "hw_model": _sysctl_text("hw.model"),
        "memory_gib": (mem_bytes / (1024**3)) if mem_bytes else None,
        "logical_cpu": _sysctl_int("hw.logicalcpu"),
        "physical_cpu": _sysctl_int("hw.physicalcpu"),
        "perf_cores": _sysctl_int("hw.perflevel0.physicalcpu"),
        "efficiency_cores": _sysctl_int("hw.perflevel1.physicalcpu"),
        "arm64": _sysctl_int("hw.optional.arm64"),
    }

def _software_context() -> dict[str, Any]:
    env = collect_environment(".").to_dict()
    return {
        "python_executable": env.get("python_executable"),
        "python_version": env.get("python_version"),
        "mtplx_version": _package_version("mtplx"),
        "mlx_version": _package_version("mlx"),
        "mlx_lm_version": _package_version("mlx-lm"),
        "numpy_version": _package_version("numpy"),
        "git_branch": env.get("git_branch"),
        "git_status": env.get("git_status"),
    }

def _mlx_backend_context(profile: Any) -> dict[str, Any]:
    path = None
    try:
        spec = importlib.util.find_spec("mlx.core")
        path = str(Path(spec.origin).resolve()) if spec and spec.origin else None
    except Exception as exc:  # pragma: no cover - host dependent
        path = f"ERROR: {exc}"
    fork_status = None
    if getattr(profile, "required_mlx_fork_fragment", None):
        fork_status = _active_mlx_fork_status(
            expected_fragment=profile.required_mlx_fork_fragment,
            expected_commit=profile.required_mlx_fork_commit,
        )
    custom_env = {
        key: os.environ.get(key)
        for key in (
            "MTPLX_GDN_OUT_QMV8",
            "MTPLX_SOURCE_QMM_MANY",
            "MTPLX_SOURCE_QMM_MANY_STRICT",
            "MTPLX_NATIVE_MLP",
            "MTPLX_VERIFY_MLP_FUSED",
        )
        if os.environ.get(key)
    }
    fork_active = bool(fork_status and fork_status.get("ok"))
    return {
        "mlx_core_path": path,
        "mlx_version": _package_version("mlx"),
        "mlx_lm_version": _package_version("mlx-lm"),
        "optional_fast_mlx_fork_active": fork_active,
        "optional_fast_mlx_fork": fork_status,
        "stock_mlx_likely": not fork_active,
        "custom_qmv_or_qmm_env": custom_env,
    }

def _ms_per_call(seconds: Any, calls: int) -> float | None:
    if not isinstance(seconds, (int, float)) or not calls:
        return None
    return 1000.0 * float(seconds) / calls

def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)

def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None

def _command_text(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            args,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
    except Exception:
        return None

def _sysctl_text(name: str) -> str | None:
    return _command_text(["sysctl", "-n", name])

def _sysctl_int(name: str) -> int | None:
    text = _sysctl_text(name)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None

def _apple_chip_family(chip: str | None) -> str | None:
    if not chip:
        return None
    words = str(chip).replace("(TM)", "").split()
    if len(words) >= 2 and words[0] == "Apple" and words[1].startswith("M"):
        return " ".join(words[:3]) if len(words) >= 3 else " ".join(words[:2])
    return chip

def _rate_lists(numerators: list[Any], denominators: list[Any]) -> list[float | None]:
    rates: list[float | None] = []
    for numerator, denominator in zip(numerators, denominators):
        den = float(denominator or 0)
        rates.append((float(numerator) / den) if den else None)
    return rates

def _format_depth_acceptance(row: dict[str, Any]) -> str:
    accepted = row.get("accepted_by_depth") or []
    drafted = row.get("drafted_by_depth") or []
    percentages = row.get("acceptance_percent_by_depth") or []
    parts = []
    for index, (acc, draft) in enumerate(zip(accepted, drafted), start=1):
        pct = percentages[index - 1] if index - 1 < len(percentages) else None
        pct_text = _fmt_metric(pct, digits=2)
        parts.append(f"MTP{index} {acc}/{draft} ({pct_text}%)")
    return " | ".join(parts) if parts else "n/a"

def _fmt_metric(value: Any, *, digits: int) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "n/a"

def _print_serve_start_line(text: str = "") -> None:
    print(text, flush=True)


_PROFILE_SHORT_SUMMARIES = {
    "safe": "Stable: exact/staged long-reply path, no fan control",
    "performance-cold": "Burst: max-fan short-context lane, not recommended beyond 8K context",
    "sustained": "Sustained: long-context native-MTP path with bounded memory",
    "exact": "QA-only exact paged verifier",
    "max-diagnostic": "Diagnostic fan-control profile",
}


def _runtime_mode_display(
    profile_name: str,
    *,
    max_mode: bool = False,
    generation_mode: str | None = None,
) -> str:
    mode = "AR" if str(generation_mode or "").lower() == GENERATION_MODE_AR else "MTP"
    if profile_name == "sustained" and max_mode:
        return f"Sustained Max {mode}"
    if profile_name == "sustained":
        return f"Sustained {mode}"
    if profile_name == "performance-cold" and max_mode:
        return f"Burst {mode}"
    if profile_name == "performance-cold":
        return f"Performance-cold {mode}"
    return f"{profile_name} {mode}"


def _print_serve_start_banner(args: Any) -> None:
    from optiq.runtime.mtp.version import DISPLAY_VERSION
    from optiq.runtime.mtp.ui import render_banner, render_startup_panel

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    profile_name = getattr(args, "profile", None) or DEFAULT_PROFILE_NAME
    warmup_tokens = int(getattr(args, "warmup_tokens", 16) or 0)
    generation_mode = _generation_mode_from_args(args)
    mode_label = _runtime_mode_display(
        profile_name,
        max_mode=bool(getattr(args, "max", False)),
        generation_mode=generation_mode,
    )
    model_label = getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID
    runtime_model = getattr(args, "model", DEFAULT_RUNTIME_MODEL_DIR)
    api_url = f"{_server_url(host, port)}/v1"
    chat_url = _chat_url(host, port)
    api_key = getattr(args, "api_key", None)
    api_note = "API key required" if api_key else "API key: leave blank for localhost"

    extra_lines: list[tuple[str, str]] = [
        ("Listening", bind_label(host, port)),
        ("Loading", str(runtime_model)),
        ("Warmup", f"{warmup_tokens} tokens"),
        ("Auth", api_note),
    ]

    render_banner()
    render_startup_panel(
        version=DISPLAY_VERSION,
        model=model_label,
        profile=profile_name,
        profile_summary=_PROFILE_SHORT_SUMMARIES.get(profile_name),
        api_url=api_url,
        chat_url=chat_url,
        mode_label=mode_label,
        extra_lines=extra_lines,
    )


def _print_serve_handoff(args: Any, runtime_model: str, profile_name: str) -> None:
    if is_wildcard_bind(getattr(args, "host", None)):
        _print_serve_start_line(
            "[1/6] Server config ready: listening on "
            f"{bind_label(args.host, int(args.port))}"
        )
        _print_serve_start_line(
            f"      Local API Base URL: {_server_url(args.host, int(args.port))}/v1"
        )
    else:
        _print_serve_start_line(
            f"[1/6] Server config ready: {_server_url(args.host, int(args.port))}/v1"
        )
    _print_serve_start_line(f"[2/6] Model resolved: {runtime_model}")
    _print_serve_start_line("[3/6] Runtime contract verified")
    _print_serve_start_line("      Loading the model can take about a minute on first start.")
    _print_serve_start_line()


def _server_command_name(args: Any) -> str:
    command = str(getattr(args, "command", None) or "quickstart")
    if command == "quick-start":
        return "quickstart"
    if command in {"quickstart", "serve"}:
        return command
    return "quickstart"


def _serve_should_onboard(args: Any) -> bool:
    """Return whether bare interactive ``mtplx serve`` should run setup."""

    if getattr(args, "command", None) != "serve":
        return False
    if bool(getattr(args, "yes", False)):
        return False
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if {"model", "profile", "max"} & set(cli_flags):
        return False
    return True


def cmd_serve_public(args: Any) -> int:
    api_key = getattr(args, "api_key", None)
    if not _is_localhost_bind(getattr(args, "host", None)) and not api_key:
        payload = {
            "error": "--api-key is required when --host is not localhost",
            "host": getattr(args, "host", None),
        }
        if getattr(args, "json", False):
            _print(payload)
        else:
            print("error: --api-key is required when --host is not localhost")
            print(f"host: {getattr(args, 'host', None)}")
            server_command = _server_command_name(args)
            print(f"try: mtplx {server_command} --host 127.0.0.1")
            print(f"try: mtplx {server_command} --host 0.0.0.0 --api-key $MTPLX_AUTH")
        return 2
    if _serve_should_onboard(args):
        from optiq.runtime.mtp.ui.onboarding import run_serve_flow

        choice = run_serve_flow(
            configured_model=getattr(args, "model", None),
            host=str(getattr(args, "host", "127.0.0.1")),
            port=int(getattr(args, "port", 8000)),
            default_open_browser=bool(getattr(args, "open_browser", False)),
        )
        if choice is None:
            _print_serve_start_line("aborted")
            return 130
        chosen_model = choice.get("model")
        if chosen_model:
            args.model = chosen_model
            try:
                from optiq.runtime.mtp.hf_loader import repo_id_from_model_ref

                if repo_id_from_model_ref(chosen_model):
                    args.download = True
            except Exception:
                pass
        chosen_profile = choice.get("profile")
        if chosen_profile:
            args.profile = chosen_profile
        args.max = bool(choice.get("max"))
        args.open_browser = bool(choice.get("open_browser"))
        args._onboarded = True
    depth_error = _validate_public_depth(args, printer=_print_serve_start_line)
    if depth_error is not None:
        return depth_error
    args.model_id = _public_model_id_for_args(
        args,
        str(getattr(args, "model", "")),
    )
    _print_serve_start_banner(args)
    if _port_is_busy(str(getattr(args, "host", "127.0.0.1")), int(getattr(args, "port", 8000))):
        if bool(getattr(args, "quickstart_pi", False)):
            base = _server_url(str(getattr(args, "host", "127.0.0.1")), int(getattr(args, "port", 8000)))
            health = _http_json(base + "/health", timeout=1.5)
            if health.get("ok"):
                from optiq.runtime.mtp.pi import pi_launch_command, pi_model_ref

                model_id = health.get("model") or getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID
                _print_serve_start_line("MTPLX is already running.")
                _print_serve_start_line(f"OpenAI API Base URL: {base}/v1")
                _print_serve_start_line(f"Pi model: {pi_model_ref(str(model_id))}")
                _quickstart_launch_pi_now(model_id=str(model_id))
                _print_serve_start_line(f"Manual fallback: {pi_launch_command(str(model_id))}")
                _print_serve_start_line("Use the existing server, or stop that terminal with Ctrl-C to restart.")
                return 0
        if bool(getattr(args, "quickstart_openwebui", False)):
            base = _server_url(str(getattr(args, "host", "127.0.0.1")), int(getattr(args, "port", 8000)))
            health = _http_json(base + "/health", timeout=1.5)
            if health.get("ok"):
                model_id = health.get("model") or getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID
                chat_url = _chat_url(str(getattr(args, "host", "127.0.0.1")), int(getattr(args, "port", 8000)))
                _print_serve_start_line("MTPLX is already running.")
                _print_serve_start_line(f"Chat URL: {chat_url}")
                _print_serve_start_line(f"OpenAI API Base URL: {base}/v1")
                _print_serve_start_line(f"Model: {model_id}")
                _print_serve_start_line("API key: leave blank for localhost")
                _print_serve_start_line("Opening chat UI in your browser...")
                _open_browser_url(chat_url)
                _print_serve_start_line("Use the existing server, or stop that terminal with Ctrl-C to restart.")
                return 0
        if bool(getattr(args, "quickstart_opencode", False)):
            base = _server_url(str(getattr(args, "host", "127.0.0.1")), int(getattr(args, "port", 8000)))
            health = _http_json(base + "/health", timeout=1.5)
            if health.get("ok"):
                model_id = health.get("model") or getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID
                _print_serve_start_line("MTPLX is already running.")
                _print_serve_start_line(f"OpenAI API Base URL: {base}/v1")
                _print_serve_start_line(f"OpenCode model: mtplx/{model_id}")
                _quickstart_launch_opencode_now()
                _print_serve_start_line("Use the existing server, or stop that terminal with Ctrl-C to restart.")
                return 0
        if bool(getattr(args, "quickstart_swival", False)):
            base = _server_url(str(getattr(args, "host", "127.0.0.1")), int(getattr(args, "port", 8000)))
            health = _http_json(base + "/health", timeout=1.5)
            if health.get("ok"):
                from optiq.runtime.mtp.swival import shell_swival_command

                model_id = health.get("model") or getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID
                context_window = int(health.get("context_window") or 262144)
                _print_serve_start_line("MTPLX is already running.")
                _print_serve_start_line(f"OpenAI API Base URL: {base}/v1")
                _print_serve_start_line(
                    "Swival command: "
                    + shell_swival_command(
                        base_url=base,
                        model_id=str(model_id),
                        context_window=context_window,
                    )
                )
                _print_serve_start_line("Use the existing server, or stop that terminal with Ctrl-C to restart.")
                return 0
        _print_serve_start_line(f"error: port {int(args.port)} is already in use")
        _print_serve_start_line("try: mtplx status")
        server_command = _server_command_name(args)
        _print_serve_start_line(f"try: stop the old mtplx {server_command} terminal with Ctrl-C")
        profile_arg = (
            f" --profile {getattr(args, 'profile')}"
            if getattr(args, "profile", None)
            else ""
        )
        max_arg = " --max" if bool(getattr(args, "max", False)) else ""
        _print_serve_start_line(
            f"try: mtplx {server_command}{profile_arg}{max_arg} --port {int(args.port) + 1}"
        )
        return 2
    profile = get_profile(getattr(args, "profile", None) or DEFAULT_PROFILE_NAME)
    cache_dir = getattr(args, "cache_dir", None)
    if bool(getattr(args, "download", False)):
        try:
            runtime_model, resolution = _quickstart_resolve_model(
                args.model,
                cache_dir=cache_dir,
                download=True,
            )
        except KeyboardInterrupt:
            _print_serve_start_line("download cancelled")
            return 130
        except Exception as exc:
            _print_serve_start_line(f"error: {exc}")
            return 1
        if runtime_model is None:
            if resolution.get("cancelled"):
                _print_serve_start_line("download cancelled")
                return 130
            gate_inspection = resolution.get("gate_inspection")
            if isinstance(gate_inspection, dict):
                _print_model_gate_error(
                    gate_inspection,
                    printer=_print_serve_start_line,
                    json_output=bool(getattr(args, "json", False)),
                )
                compatibility = gate_inspection.get("compatibility") or {}
                return int(compatibility.get("exit_code") or 1)
            resolution_error = resolution.get("error")
            if isinstance(resolution_error, dict):
                _print_command_error(
                    resolution_error,
                    command=_server_command_name(args),
                    json_output=bool(getattr(args, "json", False)),
                )
            else:
                _print_serve_start_line("error: model is not available locally")
            return 1
        if resolution.get("downloaded"):
            _print_serve_start_line(f"downloaded: {resolution.get('download_ref')}")
    else:
        runtime_model, resolve_error = _resolve_runtime_model_path(
            args.model,
            cache_dir=cache_dir,
        )
        if resolve_error is not None:
            _print_command_error(
                resolve_error,
                command=_server_command_name(args),
                json_output=bool(getattr(args, "json", False)),
            )
            return 1
    inspection, gate_exit = _model_gate(
        runtime_model,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=bool(getattr(args, "yes", False)),
    )
    if gate_exit is not None:
        _print_model_gate_error(inspection, printer=_print_serve_start_line)
        return gate_exit
    _apply_model_contract_depth_default(args, inspection)
    draft_lm_head = _model_draft_lm_head_spec(inspection, profile) or {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    draft_sampler = _model_draft_sampler_spec(inspection, profile)
    strict_fast_path = bool(getattr(args, "strict_fast_path", False))
    relax_mlx_fork_assert = False
    if profile.required_mlx_fork_fragment:
        fork_status = _active_mlx_fork_status(
            expected_fragment=profile.required_mlx_fork_fragment,
            expected_commit=profile.required_mlx_fork_commit,
        )
        if not fork_status.get("ok"):
            if strict_fast_path:
                _print_serve_start_line("[3/6] Fast MLX fork is required but not active")
                _print_serve_start_line(
                    f"      Expected: {profile.required_mlx_fork_fragment}"
                    + (f" @ {profile.required_mlx_fork_commit}" if profile.required_mlx_fork_commit else "")
                )
                observed = fork_status.get("path") or fork_status.get("error") or "unknown"
                _print_serve_start_line(f"      Found: {observed}")
                server_command = _server_command_name(args)
                _print_serve_start_line(f"try: mtplx {server_command} --profile sustained")
                _print_serve_start_line(f"try: mtplx {server_command} --profile stable")
                _print_serve_start_line(f"try: mtplx {server_command} --profile performance-cold --max")
                _print_serve_start_line("     (without --strict-fast-path, MTPLX starts in stock-MLX compatibility)")
                return 2
            relax_mlx_fork_assert = True
    model_id = _public_model_id_for_args(args, str(runtime_model))
    args.model_id = model_id
    _print_serve_handoff(args, runtime_model, profile.name)
    cmd = [
        sys.executable,
        "-m",
        "mtplx.server.openai",
        "--model",
        runtime_model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--depth",
        str(args.depth),
        "--generation-mode",
        _generation_mode_from_args(args),
        "--profile",
        profile.name,
        "--reasoning-mode",
        _reasoning_mode(args, default="auto"),
        "--preserve-thinking",
        _preserve_thinking_policy(args),
        "--verify-strategy",
        "capture_commit",
        "--verify-core",
        "linear-gdn-from-conv-tape",
        "--draft-lm-head-bits",
        str(draft_lm_head["bits"]),
        "--draft-lm-head-group-size",
        str(draft_lm_head["group_size"]),
        "--draft-lm-head-mode",
        str(draft_lm_head["mode"]),
        "--rate-limit",
        str(getattr(args, "rate_limit", 0)),
        "--stream-interval",
        str(getattr(args, "stream_interval", 1)),
        "--warmup-tokens",
        str(getattr(args, "warmup_tokens", 16)),
        "--model-id",
        str(model_id),
    ]
    if draft_sampler is not None:
        cmd.extend(
            [
                "--draft-temperature",
                str(float(draft_sampler["temperature"])),
                "--draft-top-p",
                str(float(draft_sampler["top_p"])),
                "--draft-top-k",
                str(int(draft_sampler["top_k"])),
            ]
        )
    if bool(getattr(args, "open_browser", False)):
        cmd.append("--open-browser")
    if bool(getattr(args, "quickstart_pi", False)):
        from optiq.runtime.mtp.pi import pi_launch_command

        cmd.extend(
            [
                "--launch-pi",
                "--server-console",
                "--pi-launch-command",
                pi_launch_command(str(getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID)),
            ]
        )
    if bool(getattr(args, "quickstart_opencode", False)):
        cmd.extend(["--launch-opencode", "--server-console"])
    if bool(getattr(args, "quickstart_swival", False)):
        cmd.append("--server-console")
    if bool(getattr(args, "stock_ar", False)):
        cmd.append("--stock-ar")
    if relax_mlx_fork_assert:
        cmd.append("--no-strict-mlx-fork-assert")
    if api_key:
        cmd.extend(["--api-key", str(api_key)])
    if getattr(args, "max_response_tokens", None) is not None:
        cmd.extend(["--max-response-tokens", str(args.max_response_tokens)])
    if getattr(args, "temperature", None) is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if getattr(args, "top_p", None) is not None:
        cmd.extend(["--top-p", str(args.top_p)])
    if getattr(args, "reasoning", None) is not None:
        reasoning_mode = _reasoning_mode(args, default="auto")
        if reasoning_mode == "on":
            cmd.append("--enable-thinking")
        elif reasoning_mode == "off":
            cmd.append("--no-enable-thinking")
    if getattr(args, "reasoning_parser", None):
        cmd.extend(["--reasoning-parser", str(args.reasoning_parser)])
    if not getattr(args, "stats_footer", True):
        cmd.append("--no-stats-footer")
    if getattr(args, "strict_warmup", False):
        cmd.append("--strict-warmup")
    if getattr(args, "max", False):
        from optiq.runtime.mtp.thermal import MaxSession

        def _emit(line: str) -> None:
            print(line, file=sys.stderr, flush=True)

        max_session = MaxSession(log=_emit)
        if not max_session.start():
            verified = max_session.thermal.get("verified") or {}
            _emit("")
            _emit("[max] !!! FAN CONTROL DID NOT TAKE EFFECT !!!")
            _emit(f"[max]   reason: {verified.get('message')}")
            actionable = verified.get("actionable")
            if actionable:
                _emit(f"[max]   action: {actionable}")
            _emit("[max] continuing the server WITHOUT fan boost.")
            _emit("")
            args.max = False  # don't lie to the watchdog about fan state
        # Only spin up the idle watchdog when verification confirmed fans
        # are actually pinned. If args.max was just disabled above, fall
        # through to the no-fan-control path.
        if getattr(args, "max", False):
            child_env = os.environ.copy()
            child_env["MTPLX_FAN_MODE"] = "max"
            idle_minutes = int(getattr(args, "max_idle_min", 15))
            watchdog = _MaxIdleWatchdog(
                host=str(getattr(args, "host", "127.0.0.1")),
                port=int(getattr(args, "port", 8000)),
                idle_seconds=max(60, idle_minutes * 60),
            )
            watchdog.start()
            try:
                proc = subprocess.run(cmd, env=child_env, cwd=repo_root(), check=False)
                return int(proc.returncode)
            finally:
                watchdog.stop()
                max_session.stop()  # belt-and-suspenders alongside atexit
    os.execvpe(sys.executable, cmd, os.environ.copy())
    return 0


class _MaxIdleWatchdog:
    """Background thread that drops fans to auto after ``idle_seconds`` of no
    chat activity, then ramps back to max on the next request.

    Polls the server's ``/health`` endpoint every 30s to read
    ``last_request_at`` / ``idle_seconds`` published by the OpenAI server.
    """

    def __init__(self, *, host: str, port: int, idle_seconds: int, poll_seconds: int = 30) -> None:
        self.url = f"http://{host}:{port}/health"
        self.idle_seconds = int(idle_seconds)
        self.poll_seconds = max(1, int(poll_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_max = True  # the parent set fans to max before spawning us

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="mtplx-max-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        from optiq.runtime.mtp.thermal import set_thermal_profile

        last_completed: int | None = None
        last_started_at: float | None = None
        last_active = time.time()
        while not self._stop.wait(self.poll_seconds):
            payload = _http_json(self.url, timeout=2.0)
            if not payload.get("ok"):
                continue
            current = int(payload.get("requests_completed") or 0)
            started_at = float(payload.get("last_request_started_at") or 0.0)
            active_requests = int(
                payload.get("foreground_active")
                if payload.get("foreground_active") is not None
                else payload.get("active_requests") or 0
            )
            if last_completed is None:
                last_completed = current
            if last_started_at is None:
                last_started_at = started_at
            now = time.time()
            if active_requests > 0:
                last_active = now
                last_started_at = started_at
                if not self._is_max:
                    set_thermal_profile("performance")
                    self._is_max = True
                continue
            if started_at and started_at != last_started_at:
                last_started_at = started_at
                last_active = now
                if not self._is_max:
                    set_thermal_profile("performance")
                    self._is_max = True
                continue
            if current != last_completed:
                last_completed = current
                last_active = now
                if not self._is_max:
                    set_thermal_profile("performance")
                    self._is_max = True
                continue
            if (now - last_active) >= self.idle_seconds and self._is_max:
                set_thermal_profile("silent")
                self._is_max = False


def _generate_one_shot_public(args: Any, *, command: str) -> tuple[int, dict[str, Any], list[Any]]:
    prompt = getattr(args, "prompt", None) or getattr(args, "prompt_arg", None)
    if not prompt:
        raise SystemExit(f"mtplx {command} requires a prompt")
    depth_error = _validate_public_depth(args, printer=lambda _line: None)
    if depth_error is not None:
        return depth_error, {
            "error": "invalid depth",
            "detail": (
                "--depth must be between "
                f"1 and {MAX_PUBLIC_SPECULATIVE_DEPTH} for the current MTPLX runtime"
            ),
        }, []
    runtime_model, resolve_error = _resolve_runtime_model_path(
        args.model,
        cache_dir=getattr(args, "cache_dir", None),
    )
    if resolve_error is not None:
        return EXIT_TELEMETRY, resolve_error, []
    inspection, gate_exit = _model_gate(
        runtime_model,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=bool(getattr(args, "yes", False)),
    )
    if gate_exit is not None:
        return gate_exit, {"error": "model failed MTP primary gate", "model": inspection}, []
    profile = get_profile(getattr(args, "profile", None) or DEFAULT_PROFILE_NAME)
    apply_profile_env(profile.name)
    generation_mode = _generation_mode_from_args(args)
    draft_lm_head = (
        _model_draft_lm_head_spec(inspection, profile)
        if generation_mode == GENERATION_MODE_MTP
        else None
    )
    draft_sampler = (
        _model_draft_sampler_spec(inspection, profile)
        if generation_mode == GENERATION_MODE_MTP
        else None
    )

    max_session: Any | None = None
    thermal: dict[str, Any] | None = None
    if getattr(args, "max", False):
        from optiq.runtime.mtp.thermal import MaxSession

        def _emit(line: str) -> None:
            print(line, file=sys.stderr, flush=True)

        max_session = MaxSession(log=_emit)
        if max_session.start():
            thermal = max_session.thermal
        else:
            verified = max_session.thermal.get("verified") or {}
            _emit("")
            _emit("[max] fan boost unavailable; continuing this run without fan boost.")
            if verified.get("message"):
                _emit(f"[max] reason: {verified.get('message')}")
            _emit("")
            thermal = max_session.thermal
            max_session = None

    from optiq.runtime.mtp.benchmarks.schema import PromptCase, encode_prompt_case
    from optiq.runtime.mtp.generation import generate_ar, generate_mtpk
    from optiq.runtime.mtp.runtime import load
    from optiq.runtime.mtp.sampling import SamplerConfig

    try:
        rt = load(runtime_model, mtp=True)
        draft_report = None
        if (
            draft_lm_head is not None
            and hasattr(rt, "model")
            and bool(getattr(rt, "mtp_enabled", True))
        ):
            from optiq.runtime.mtp.draft_lm_head import _install_draft_lm_head

            draft_report = _install_draft_lm_head(
                rt,
                bits=int(draft_lm_head["bits"]),
                group_size=int(draft_lm_head["group_size"]),
                mode=str(draft_lm_head["mode"]),
            )
        messages = None
        system = getattr(args, "system", None)
        if system:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        requested_max_tokens = getattr(args, "max_tokens", None)
        case = PromptCase(
            id=f"cli_{command}",
            category=command,
            prompt=prompt,
            max_tokens=int(requested_max_tokens or 0),
            messages=messages,
        )
        reasoning_mode = _reasoning_mode(args)
        enable_thinking = _enable_thinking_for_reasoning(reasoning_mode)
        prompt_ids = encode_prompt_case(
            rt.tokenizer,
            case,
            chat_template=True,
            enable_thinking=enable_thinking,
        )
        budget = _cli_generation_budget(
            tokenizer=rt.tokenizer,
            model_path=runtime_model,
            prompt_token_count=len(prompt_ids),
            explicit_max_tokens=requested_max_tokens,
        )
        max_tokens_value = int(budget["effective_max_tokens"])
        sampler = SamplerConfig(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
        if generation_mode == GENERATION_MODE_AR:
            out = generate_ar(
                rt,
                prompt_ids,
                max_tokens=max_tokens_value,
                sampler=sampler,
                seed=args.seed,
            )
        else:
            out = generate_mtpk(
                rt,
                prompt_ids,
                max_tokens=max_tokens_value,
                sampler=sampler,
                draft_sampler=_draft_sampler_from_spec(draft_sampler),
                speculative_depth=args.depth,
                seed=args.seed,
                mtp_hidden_variant="post_norm",
                mtp_cache_policy="persistent",
                mtp_history_policy="committed",
                verify_strategy="capture_commit",
                verify_core="linear-gdn-from-conv-tape",
            )
    finally:
        if max_session is not None:
            max_session.stop()
            thermal = max_session.thermal
    validations = [
        validate_no_degenerate_loop(out.text),
        validate_balanced_delimiters(out.text),
    ]
    if args.expect_python:
        validations.append(validate_python_syntax(out.text))
    payload = {
        "text": out.text,
        "model": _compact_model_summary(inspection),
        "profile": profile.to_dict(),
        "draft_lm_head": draft_report,
        "draft_sampler": draft_sampler,
        "stats": {
            "generated_tokens": out.stats.generated_tokens,
            "max_tokens": max_tokens_value,
            "request_max_tokens": budget["request_max_tokens"],
            "context_window": budget["context_window"],
            "remaining_context_tokens": budget["remaining_context_tokens"],
            "context_cap_applied": budget["context_cap_applied"],
            "reasoning": reasoning_mode,
            "generation_mode": generation_mode,
            "mtp_depth": (
                0
                if generation_mode == GENERATION_MODE_AR
                else int(getattr(out.stats, "speculative_depth", args.depth) or args.depth)
            ),
            "requested_mtp_depth": 0
            if generation_mode == GENERATION_MODE_AR
            else int(
                getattr(out.stats, "requested_speculative_depth", args.depth)
                or args.depth
            ),
            "long_context_mtp_depth_policy": (
                {}
                if generation_mode == GENERATION_MODE_AR
                else dict(getattr(out.stats, "long_context_mtp_depth_policy", {}) or {})
            ),
            "tok_s": out.stats.tok_s,
            "verify_ms_per_call": (
                None
                if generation_mode == GENERATION_MODE_AR
                else
                1000.0 * out.stats.verify_time_s / out.stats.verify_calls
                if out.stats.verify_calls
                else None
            ),
            "verify_calls": (
                0
                if generation_mode == GENERATION_MODE_AR
                else int(getattr(out.stats, "verify_calls", 0) or 0)
            ),
            "verify_time_s": (
                0.0
                if generation_mode == GENERATION_MODE_AR
                else float(getattr(out.stats, "verify_time_s", 0.0) or 0.0)
            ),
            "draft_time_s": (
                0.0
                if generation_mode == GENERATION_MODE_AR
                else float(getattr(out.stats, "draft_time_s", 0.0) or 0.0)
            ),
            "accepted_by_depth": (
                []
                if generation_mode == GENERATION_MODE_AR
                else list(getattr(out.stats, "accepted_by_depth", []) or [])
            ),
            "drafted_by_depth": (
                []
                if generation_mode == GENERATION_MODE_AR
                else list(getattr(out.stats, "drafted_by_depth", []) or [])
            ),
        },
        "validations": [v.__dict__ for v in validations],
    }
    if thermal is not None:
        payload["thermal"] = thermal
    return 0 if all(v.passed for v in validations) else EXIT_QUALITY, payload, validations


def cmd_run_public(args: Any) -> int:
    code, payload, _validations = _generate_one_shot_public(args, command="run")
    if "error" in payload:
        _print_command_error(
            payload,
            command="run",
            json_output=bool(getattr(args, "json", False)),
        )
        return code
    if getattr(args, "json", False):
        _print(payload)
    else:
        text = payload["text"]
        print(text, end="" if text.endswith("\n") else "\n")
        if not getattr(args, "quiet", False):
            stats = payload["stats"]
            tok_s = stats.get("tok_s")
            tok_s_text = f"{tok_s:.2f}" if isinstance(tok_s, (int, float)) else "n/a"
            mode = str(stats.get("generation_mode") or "mtp").upper()
            mtp_depth = int(stats.get("mtp_depth") or 0)
            print(
                f"\n[mtplx] profile={payload['profile']['name']} "
                f"mode={mode} mtp_depth={mtp_depth} "
                f"tokens={stats.get('generated_tokens')} tok_s={tok_s_text}"
            )
    return code


def cmd_chat_public(args: Any) -> int:
    # Still a one-shot smoke path until the interactive REPL lands in Phase 5.
    code, payload, _validations = _generate_one_shot_public(args, command="chat")
    if "error" in payload:
        _print_command_error(
            payload,
            command="chat",
            json_output=bool(getattr(args, "json", False)),
        )
        return code
    if getattr(args, "json", False):
        _print(payload)
    else:
        text = payload["text"]
        print(text, end="" if text.endswith("\n") else "\n")
    return code


def _quickstart_line(text: str = "") -> None:
    print(text, flush=True)


def _start_command_name(args: Any) -> str:
    command = str(getattr(args, "command", None) or "start")
    return "start" if command in {"start", "quickstart", "quick-start"} else command


def _start_invocation(args: Any, suffix: str = "") -> str:
    command = _start_command_name(args)
    return f"mtplx {command}{suffix}"


def _handle_quickstart_reasoning_command(args: Any, prompt: str) -> bool:
    parts = prompt.strip().split()
    if not parts or parts[0].lower() not in {"/reasoning", "--reasoning"}:
        return False
    if len(parts) == 1:
        _quickstart_line(f"Reasoning: {_reasoning_mode(args)}")
        _quickstart_line("try: /reasoning on")
        _quickstart_line("try: /reasoning off")
        _quickstart_line("try: /reasoning auto")
        return True
    if len(parts) == 2 and parts[1].lower() in {"auto", "on", "off"}:
        setattr(args, "reasoning", parts[1].lower())
        _quickstart_line(f"Reasoning: {_reasoning_mode(args)}")
        return True
    _quickstart_line("usage: /reasoning on|off|auto")
    return True


def _handle_quickstart_mtp_command(args: Any, prompt: str, *, runtime: Any | None = None) -> bool:
    parts = prompt.strip().split()
    if not parts or parts[0].lower() not in {"/mtp", "--mtp"}:
        return False
    if len(parts) == 1 or parts[1].lower() == "status":
        mode = _generation_mode_from_args(args)
        _quickstart_line(
            f"MTP: {'on' if mode == GENERATION_MODE_MTP else 'off'} "
            f"({_generation_mode_label(mode)})"
        )
        return True
    if len(parts) == 2 and parts[1].lower() in {"on", "off"}:
        requested = GENERATION_MODE_MTP if parts[1].lower() == "on" else GENERATION_MODE_AR
        if (
            requested == GENERATION_MODE_MTP
            and runtime is not None
            and not bool(getattr(runtime, "mtp_enabled", False))
        ):
            _quickstart_line("MTP: unavailable for this loaded runtime")
            return True
        _set_generation_mode_on_args(args, requested)
        _quickstart_line(
            "MTP: on for the next turn"
            if requested == GENERATION_MODE_MTP
            else "MTP: off for the next turn (target-only AR generation)"
        )
        return True
    _quickstart_line("usage: /mtp on|off|status")
    return True


def _quickstart_console() -> Any:
    """Return a ``rich.console.Console`` if available and stdout is a TTY."""

    if not sys.stdout.isatty():
        return None
    try:
        from rich.console import Console
    except ImportError:
        return None
    try:
        return Console()
    except Exception:
        return None


def _chat_input_prompt() -> str:
    """Prompt string for the chat REPL ``input()`` call.

    Uses ANSI color when the terminal supports it; falls back to plain text.
    """

    if not sys.stdout.isatty():
        return "\nyou> "
    # ANSI bold cyan for "you", reset, then bold "> ", then reset.
    # Avoid using rich here; ``input()`` does not interact well with rich Live.
    return "\n\033[1;36myou\033[0m\033[1m>\033[0m "


def _print_assistant_fallback(label: str, text: str) -> None:
    """Print a non-streamed assistant message with the same styled label."""

    console = _quickstart_console()
    if console is None:
        _quickstart_line(f"{label}:")
        print(text, end="" if text.endswith("\n") else "\n")
        return
    console.print(f"[bold cyan]{label}[/bold cyan]", highlight=False)
    console.print(text, soft_wrap=True, highlight=False, markup=False)


def _print_stats_line(text: str) -> None:
    """Print the stats footer with a dim/colored treatment when possible."""

    console = _quickstart_console()
    if console is None:
        _quickstart_line(text)
        return
    console.print(f"  [dim]{text}[/dim]", highlight=False)


class _QuickstartHeartbeat:
    def __init__(self, label: str, *, interval_s: float = 5.0) -> None:
        script = (
            "import signal,sys,time\n"
            "label=sys.argv[1]\n"
            "interval=float(sys.argv[2])\n"
            "running=True\n"
            "def stop(_signum,_frame):\n"
            "    global running\n"
            "    running=False\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "elapsed=0.0\n"
            "while running:\n"
            "    time.sleep(interval)\n"
            "    if not running:\n"
            "        break\n"
            "    elapsed += interval\n"
            "    print(f'      {label}... {elapsed:.0f}s elapsed', flush=True)\n"
        )
        self.proc = subprocess.Popen(
            [sys.executable, "-c", script, label, str(float(interval_s))],
            stdout=None,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )

    def set(self) -> None:
        proc = self.proc
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)


def _quickstart_heartbeat(label: str, *, interval_s: float = 5.0) -> _QuickstartHeartbeat:
    return _QuickstartHeartbeat(label, interval_s=interval_s)


def _quickstart_current_model(args: Any) -> str:
    model = getattr(args, "model", None)
    explicit_model = bool(getattr(args, "_model_explicit", False))
    if not explicit_model and is_verified_default_model_ref(model):
        selection = select_default_model()
        args._mtplx_default_model_selection = selection.to_dict()
        return selection.model
    return str(model or DEFAULT_MODEL_ID)


def _quickstart_download_ref(model: str) -> str:
    from optiq.runtime.mtp.hf_loader import repo_id_from_model_ref

    if repo_id_from_model_ref(model):
        return model
    selection = select_default_model()
    default_local_refs = {
        DEFAULT_MODEL_ID,
        selection.model,
        str(DEFAULT_RUNTIME_MODEL_DIR),
        DEFAULT_CHAMPION,
        str((repo_root() / DEFAULT_MODEL_ID).resolve()),
        str((repo_root() / selection.model).resolve()),
        str((repo_root() / str(DEFAULT_RUNTIME_MODEL_DIR)).resolve()),
    }
    if model in default_local_refs:
        return selection.hf_model
    raise ValueError(
        "cannot download a local model path. Re-run with --model HF_ORG/HF_REPO "
        "or choose a folder that already exists."
    )


def _quickstart_choose_model(args: Any, *, target: str = "terminal") -> tuple[str, bool]:
    model = _quickstart_current_model(args)
    download = bool(getattr(args, "download", False))
    if (
        target == "openwebui"
        or getattr(args, "yes", False)
        or getattr(args, "prompt", None)
        or getattr(args, "dry_run", False)
        or getattr(args, "_onboarded", False)
        or not sys.stdin.isatty()
    ):
        return model, download

    _quickstart_line(f"MTPLX {_start_command_name(args)}")
    selection = select_default_model()
    _quickstart_line("Choose a model:")
    _quickstart_line(f"  1. Use verified default for this Mac ({selection.label})")
    quality_ref = optimized_quality_model_ref()
    _quickstart_line(f"  2. Optimized Quality ({OPTIMIZED_QUALITY_DESCRIPTION})")
    _quickstart_line("  3. Choose a local model folder")
    _quickstart_line(f"  4. Download verified default from Hugging Face ({selection.hf_model})")
    choice = input("Select [1]: ").strip()
    if choice == "2":
        return quality_ref, download
    if choice == "3":
        chosen = input("Model folder: ").strip()
        return (chosen or model), False
    if choice == "4":
        return selection.hf_model, True
    return model, download


def _quickstart_resolve_model(model: str, *, cache_dir: str | None, download: bool) -> tuple[str | None, dict[str, Any]]:
    runtime_model, resolve_error = _resolve_runtime_model_path(model, cache_dir=cache_dir)
    if resolve_error is None:
        return runtime_model, {
            "model": model,
            "runtime_model": runtime_model,
            "downloaded": False,
            "download_ref": None,
        }
    if not download:
        return None, {
            "model": model,
            "runtime_model": None,
            "downloaded": False,
            "download_ref": None,
            "error": resolve_error,
        }

    download_ref = _quickstart_download_ref(model)
    try:
        inspection = inspect_model(download_ref).to_dict()
    except Exception as exc:
        return None, {
            "model": model,
            "runtime_model": None,
            "downloaded": False,
            "download_ref": download_ref,
            "error": {
                "error": "model failed Hugging Face preflight",
                "model": download_ref,
                "detail": str(exc),
            },
        }
    compatibility = inspection.get("compatibility") or {}
    if not bool(compatibility.get("can_run")):
        return None, {
            "model": model,
            "runtime_model": None,
            "downloaded": False,
            "download_ref": download_ref,
            "gate_inspection": inspection,
            "error": {
                "error": "model failed MTPLX compatibility gate",
                "model": inspection,
            },
        }

    from optiq.runtime.mtp.hf_loader import pull_model

    _quickstart_line(f"[1/4] Downloading model: {download_ref}")
    callback, finalize = _rich_download_progress_callback(repo_id=download_ref)
    try:
        try:
            result = pull_model(
                download_ref,
                cache_dir=cache_dir,
                progress_callback=callback,
                progress_interval_s=0.4,
            )
        except KeyboardInterrupt:
            return None, {
                "model": model,
                "runtime_model": None,
                "downloaded": False,
                "download_ref": download_ref,
                "cancelled": True,
                "error": {
                    "error": "download cancelled",
                    "model": download_ref,
                },
            }
    finally:
        finalize()
    runtime_model, resolve_error = _resolve_runtime_model_path(download_ref, cache_dir=cache_dir)
    if resolve_error is not None:
        return None, {
            "model": model,
            "runtime_model": None,
            "downloaded": True,
            "download_ref": download_ref,
            "download_result": result,
            "error": resolve_error,
        }
    return runtime_model, {
        "model": model,
        "runtime_model": runtime_model,
        "downloaded": True,
        "download_ref": download_ref,
        "download_result": result,
    }


def _quickstart_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _quickstart_decode_timing(stats: dict[str, Any]) -> tuple[float | None, float | None]:
    generated_tokens = int(stats.get("generated_tokens") or 0)
    elapsed_s = _quickstart_number(stats.get("elapsed_s")) or 0.0
    prompt_eval_time_s = _quickstart_number(stats.get("prompt_eval_time_s"))
    if prompt_eval_time_s is None:
        target_forward_time_s = _quickstart_number(stats.get("target_forward_time_s")) or 0.0
        verify_time_s = _quickstart_number(stats.get("verify_time_s")) or 0.0
        repair_time_s = _quickstart_number(stats.get("repair_time_s")) or 0.0
        prompt_eval_time_s = max(0.0, target_forward_time_s - verify_time_s - repair_time_s)
    decode_elapsed_s = max(0.0, elapsed_s - prompt_eval_time_s)
    if generated_tokens <= 0 or decode_elapsed_s <= 0.0:
        return None, decode_elapsed_s
    return generated_tokens / decode_elapsed_s, decode_elapsed_s


def _quickstart_token_window_rate(token_times: list[float]) -> float | None:
    if len(token_times) < 2:
        return None
    elapsed_s = float(token_times[-1]) - float(token_times[0])
    if elapsed_s <= 0.0:
        return None
    return (len(token_times) - 1) / elapsed_s


class _QuickstartIncrementalTokenDecoder:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._token_cache: list[int] = []
        self._print_len = 0

    def _decode(self, tokens: list[int]) -> str:
        try:
            return self._tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        except TypeError:
            return self._tokenizer.decode(tokens)

    def feed(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        self._token_cache.extend(int(token) for token in tokens)
        text = self._decode(self._token_cache)
        if text.endswith("\n"):
            printable = text[self._print_len :]
            self._token_cache = []
            self._print_len = 0
            return printable
        if text and self._is_cjk_char(ord(text[-1])):
            printable = text[self._print_len :]
            self._print_len += len(printable)
            return printable
        close_index = text.find("</think>", self._print_len)
        if close_index >= 0:
            boundary = close_index + len("</think>")
            printable = text[self._print_len : boundary]
            self._print_len = boundary
            return printable
        boundary = -1
        for index in range(len(text) - 1, -1, -1):
            if text[index].isspace():
                boundary = index + 1
                break
        if boundary <= self._print_len:
            return ""
        printable = text[self._print_len : boundary]
        self._print_len = boundary
        return printable

    def finish(self) -> str:
        if not self._token_cache:
            return ""
        text = self._decode(self._token_cache)
        printable = text[self._print_len :]
        self._token_cache = []
        self._print_len = 0
        return printable

    @staticmethod
    def _is_cjk_char(cp: int) -> bool:
        return (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF)
            or (0x2A700 <= cp <= 0x2B73F)
            or (0x2B740 <= cp <= 0x2B81F)
            or (0x2B820 <= cp <= 0x2CEAF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x2F800 <= cp <= 0x2FA1F)
        )


class _QuickstartTerminalStreamer:
    def __init__(self, tokenizer: Any, *, label: str) -> None:
        self._decoder = _QuickstartIncrementalTokenDecoder(tokenizer)
        self._label = label
        self.started = False
        self._console: Any = None
        try:
            from rich.console import Console
        except ImportError:
            self._console = None
        else:
            try:
                self._console = Console()
            except Exception:
                self._console = None

    def feed(self, tokens: list[int]) -> None:
        text = self._decoder.feed(tokens)
        if text:
            self._write(text)

    def finish(self) -> None:
        text = self._decoder.finish()
        if text:
            self._write(text)
        if self.started:
            print(flush=True)

    def _write(self, text: str) -> None:
        if not self.started:
            self._print_label()
            self.started = True
        if self._console is not None:
            self._console.print(text, end="", soft_wrap=True, highlight=False, markup=False)
        else:
            print(text, end="", flush=True)

    def _print_label(self) -> None:
        if self._console is not None:
            self._console.print(f"[bold cyan]{self._label}[/bold cyan]", highlight=False)
        else:
            print(f"{self._label}:")


def _quickstart_stats_line(payload: dict[str, Any]) -> str:
    stats = payload.get("stats") or {}
    generated_tokens = int(stats.get("generated_tokens") or 0)
    stream_tok_s = _quickstart_number(stats.get("stream_tok_s"))
    total_tok_s = _quickstart_number(stats.get("end_to_end_tok_s", stats.get("tok_s")))
    decode_tok_s = _quickstart_number(stats.get("decode_tok_s"))
    decode_elapsed_s = _quickstart_number(stats.get("decode_elapsed_s"))
    if decode_tok_s is None:
        decode_tok_s, decode_elapsed_s = _quickstart_decode_timing(stats)
    verify_ms = stats.get("verify_ms_per_call")
    verify_text = f"{verify_ms:.1f} ms/verify" if isinstance(verify_ms, (int, float)) else "verify n/a"
    if decode_tok_s is not None:
        speed_text = f"{decode_tok_s:.2f} tok/s"
        if stream_tok_s is not None and abs(decode_tok_s - stream_tok_s) >= 0.1:
            speed_text = f"{speed_text} | live_window={stream_tok_s:.2f}"
        if total_tok_s is not None and abs(decode_tok_s - total_tok_s) >= 0.1:
            speed_text = f"{speed_text} | total={total_tok_s:.2f}"
    elif stream_tok_s is not None:
        speed_text = f"{stream_tok_s:.2f} tok/s"
        if total_tok_s is not None and abs(stream_tok_s - total_tok_s) >= 0.1:
            speed_text = f"{speed_text} | total={total_tok_s:.2f}"
    elif total_tok_s is not None:
        speed_text = f"{total_tok_s:.2f} total tok/s"
    else:
        speed_text = "TPS n/a"
    elapsed_text = (
        f"{generated_tokens} tokens in {decode_elapsed_s:.2f}s decode"
        if decode_elapsed_s is not None and decode_elapsed_s > 0.0
        else f"{generated_tokens} tokens"
    )
    detail_parts = []
    generation_mode = str(stats.get("generation_mode") or "").lower()
    if generation_mode in GENERATION_MODES:
        mode_text = "AR" if generation_mode == GENERATION_MODE_AR else "MTP"
        mtp_depth = int(stats.get("mtp_depth") or 0)
        detail_parts.append(f"mode={mode_text}")
        detail_parts.append(f"mtp_depth={mtp_depth}")
    detail_parts.append(verify_text)
    verify_calls = stats.get("verify_calls")
    if isinstance(verify_calls, int) and verify_calls:
        detail_parts.append(f"{verify_calls} verify calls")
    accepted = stats.get("accepted_by_depth")
    if isinstance(accepted, list) and accepted:
        detail_parts.append(f"accept={accepted}")
    corrections = stats.get("correction_tokens")
    if isinstance(corrections, int) and corrections:
        detail_parts.append(f"corr={corrections}")
    ttft_s = _quickstart_number(stats.get("ttft_s"))
    if ttft_s is not None:
        detail_parts.append(f"ttft={ttft_s:.2f}s")
    return (
        f"[mtplx] {elapsed_text} | "
        f"{speed_text} | {' | '.join(detail_parts)} | profile={payload['profile']['name']}"
    )


def _quickstart_generate(
    *,
    rt: Any,
    inspection: dict[str, Any],
    profile: Any,
    args: Any,
    prompt: str,
    history: list[dict[str, str]],
    turn_index: int,
    max_tokens: int | None = None,
    include_history: bool = True,
    stream_label: str | None = None,
    draft_sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from optiq.runtime.mtp.benchmarks.schema import PromptCase, encode_prompt_case
    from optiq.runtime.mtp.generation import generate_ar, generate_mtpk
    from optiq.runtime.mtp.sampling import SamplerConfig

    messages: list[dict[str, str]] = []
    system = getattr(args, "system", None)
    if system:
        messages.append({"role": "system", "content": system})
    if include_history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    requested_max_tokens = max_tokens if max_tokens is not None else getattr(args, "max_tokens", None)
    case = PromptCase(
        id=f"quickstart_{turn_index}",
        category="quickstart",
        prompt=prompt,
        max_tokens=int(requested_max_tokens or 0),
        messages=messages,
    )
    reasoning_mode = _reasoning_mode(args)
    enable_thinking = _enable_thinking_for_reasoning(reasoning_mode)
    prompt_ids = encode_prompt_case(
        rt.tokenizer,
        case,
        chat_template=True,
        enable_thinking=enable_thinking,
    )
    budget = _cli_generation_budget(
        tokenizer=rt.tokenizer,
        model_path=getattr(rt, "model_path", getattr(args, "model", "")),
        prompt_token_count=len(prompt_ids),
        explicit_max_tokens=requested_max_tokens,
    )
    max_tokens_value = int(budget["effective_max_tokens"])
    request_started_s = time.perf_counter()
    token_times: list[float] = []
    terminal_streamer = (
        _QuickstartTerminalStreamer(rt.tokenizer, label=stream_label)
        if stream_label
        else None
    )

    def record_tokens(new_tokens: list[int]) -> None:
        now = time.perf_counter()
        token_times.extend([now for _token in new_tokens])
        if terminal_streamer is not None:
            terminal_streamer.feed(new_tokens)

    sampler = SamplerConfig(
        temperature=float(getattr(args, "temperature", 0.6)),
        top_p=float(getattr(args, "top_p", 0.95)),
        top_k=int(getattr(args, "top_k", 20)),
    )
    seed = int(getattr(args, "seed", 0)) + turn_index
    generation_mode = _generation_mode_from_args(args)
    try:
        if generation_mode == GENERATION_MODE_AR:
            out = generate_ar(
                rt,
                prompt_ids,
                max_tokens=max_tokens_value,
                sampler=sampler,
                seed=seed,
                token_callback=record_tokens,
            )
        else:
            out = generate_mtpk(
                rt,
                prompt_ids,
                max_tokens=max_tokens_value,
                sampler=sampler,
                draft_sampler=_draft_sampler_from_spec(draft_sampler),
                speculative_depth=int(getattr(args, "depth", 3)),
                seed=seed,
                mtp_hidden_variant="post_norm",
                mtp_cache_policy="persistent",
                mtp_history_policy="committed",
                verify_strategy="capture_commit",
                verify_core="linear-gdn-from-conv-tape",
                token_callback=record_tokens,
            )
    finally:
        if terminal_streamer is not None:
            terminal_streamer.finish()
    stats = {
        "generated_tokens": out.stats.generated_tokens,
        "max_tokens": max_tokens_value,
        "request_max_tokens": budget["request_max_tokens"],
        "context_window": budget["context_window"],
        "remaining_context_tokens": budget["remaining_context_tokens"],
        "context_cap_applied": budget["context_cap_applied"],
        "reasoning": reasoning_mode,
        "generation_mode": generation_mode,
        "mtp_depth": (
            0
            if generation_mode == GENERATION_MODE_AR
            else int(
                getattr(out.stats, "speculative_depth", getattr(args, "depth", 3))
                or getattr(args, "depth", 3)
            )
        ),
        "requested_mtp_depth": (
            0
            if generation_mode == GENERATION_MODE_AR
            else int(
                getattr(
                    out.stats,
                    "requested_speculative_depth",
                    getattr(args, "depth", 3),
                )
                or getattr(args, "depth", 3)
            )
        ),
        "long_context_mtp_depth_policy": (
            {}
            if generation_mode == GENERATION_MODE_AR
            else dict(getattr(out.stats, "long_context_mtp_depth_policy", {}) or {})
        ),
        "tok_s": out.stats.tok_s,
        "end_to_end_tok_s": out.stats.tok_s,
        "elapsed_s": out.stats.elapsed_s,
        "prompt_eval_time_s": out.stats.prompt_eval_time_s,
        "verify_time_s": 0.0 if generation_mode == GENERATION_MODE_AR else out.stats.verify_time_s,
        "target_forward_time_s": out.stats.target_forward_time_s,
        "repair_time_s": out.stats.repair_time_s,
        "draft_time_s": 0.0 if generation_mode == GENERATION_MODE_AR else out.stats.draft_time_s,
        "verify_calls": 0 if generation_mode == GENERATION_MODE_AR else out.stats.verify_calls,
        "accepted_by_depth": (
            []
            if generation_mode == GENERATION_MODE_AR
            else list(getattr(out.stats, "accepted_by_depth", []) or [])
        ),
        "drafted_by_depth": (
            []
            if generation_mode == GENERATION_MODE_AR
            else list(getattr(out.stats, "drafted_by_depth", []) or [])
        ),
        "correction_tokens": out.stats.correction_tokens,
        "bonus_tokens": out.stats.bonus_tokens,
        "stream_tok_s": _quickstart_token_window_rate(token_times),
        "ttft_s": (token_times[0] - request_started_s) if token_times else None,
        "verify_ms_per_call": (
            None
            if generation_mode == GENERATION_MODE_AR
            else
            1000.0 * out.stats.verify_time_s / out.stats.verify_calls
            if out.stats.verify_calls
            else None
        ),
    }
    decode_tok_s, decode_elapsed_s = _quickstart_decode_timing(stats)
    stats["decode_tok_s"] = decode_tok_s
    stats["decode_elapsed_s"] = decode_elapsed_s
    return {
        "text": out.text,
        "model": _compact_model_summary(inspection),
        "profile": profile.to_dict(),
        "draft_sampler": draft_sampler,
        "stats": stats,
        "streamed": bool(terminal_streamer and terminal_streamer.started),
        "validations": [
            validate_no_degenerate_loop(out.text).__dict__,
            validate_balanced_delimiters(out.text).__dict__,
        ],
    }


def _public_model_id_for_ref(model_ref: str, *, default_model_id: str) -> str:
    return public_model_id_for_ref(model_ref, default_model_id=default_model_id)


def _public_model_id_for_args(args: Any, model_ref: str | None) -> str:
    model_id = str(getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID)
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if "model-id" in cli_flags or model_id != DEFAULT_PUBLIC_MODEL_ID:
        return model_id
    return _public_model_id_for_ref(
        str(model_ref or ""),
        default_model_id=DEFAULT_PUBLIC_MODEL_ID,
    )


def _quickstart_openwebui_payload(args: Any) -> dict[str, Any]:
    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    model_id = _public_model_id_for_args(args, str(getattr(args, "model", "")))
    base = f"http://{_connect_host_for_bind(host)}:{port}"
    profile = str(getattr(args, "profile", None) or DEFAULT_PROFILE_NAME)
    return {
        "integration": "openwebui",
        "server_url": base,
        "base_url": base + "/v1",
        "api_base_url": base + "/v1",
        "chat_url": base + "/",
        "model_id": model_id,
        "api_key": "not required for localhost",
        "server_command": (
            f"mtplx quickstart --host {host} --port {port} "
            f"--model {shlex.quote(str(getattr(args, 'model', DEFAULT_RUNTIME_MODEL_DIR)))} "
            f"--profile {profile} "
            f"{'--max ' if bool(getattr(args, 'max', False)) else ''}"
            f"{'--no-mtp ' if _generation_mode_from_args(args) == GENERATION_MODE_AR else ''}"
            "--no-stats-footer --open-browser"
        ),
        "openwebui_steps": [
            f"Open chat UI: {base}/",
            f"OpenAI-compatible API base URL: {base}/v1",
            f"Model: {model_id}",
        ],
    }


def _quickstart_pi_payload(args: Any, *, write_config: bool = False) -> dict[str, Any]:
    from optiq.runtime.mtp.pi import (
        PI_LOCAL_API_KEY,
        build_pi_provider_config,
        pi_launch_command,
        pi_model_ref,
        pi_models_json_path,
        write_pi_models_config,
    )

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    model_id = _public_model_id_for_args(args, str(getattr(args, "model", "")))
    base_url = f"http://{_connect_host_for_bind(host)}:{port}/v1"
    api_key = str(getattr(args, "api_key", None) or PI_LOCAL_API_KEY)
    provider = build_pi_provider_config(
        base_url=base_url,
        model_id=model_id,
        model_name=f"MTPLX {model_id}",
        api_key=api_key,
    )
    payload = {
        "integration": "pi",
        "server_url": f"http://{_connect_host_for_bind(host)}:{port}",
        "base_url": base_url,
        "api_base_url": base_url,
        "model_id": model_id,
        "model_ref": pi_model_ref(model_id),
        "api_key": api_key,
        "config_path": str(pi_models_json_path()),
        "provider": provider,
        "no_hidden_max_tokens": "maxTokens" not in json.dumps(
            provider.get("models", [])
        ),
        "launch_command": pi_launch_command(model_id),
        "server_console": True,
        "server_controls": [
            "/reasoning on|off|auto|status",
            "/mtp on|off|status",
            "/stats",
            "/help",
        ],
        "server_command": (
            f"mtplx quickstart --host {host} --port {port} "
            f"--model {shlex.quote(str(getattr(args, 'model', DEFAULT_RUNTIME_MODEL_DIR)))} "
            f"--profile {str(getattr(args, 'profile', None) or DEFAULT_PROFILE_NAME)} "
            f"{'--max ' if bool(getattr(args, 'max', False)) else ''}"
            f"{'--no-mtp ' if _generation_mode_from_args(args) == GENERATION_MODE_AR else ''}"
            f"--api-key {shlex.quote(api_key)} --no-stats-footer"
        ),
        "pi_steps": [
            f"Pi config: {pi_models_json_path()}",
            f"Model in Pi: {pi_model_ref(model_id)}",
            f"Start Pi: {pi_launch_command(model_id)}",
        ],
    }
    if write_config:
        payload["config_write"] = write_pi_models_config(
            base_url=base_url,
            model_id=model_id,
            model_name=f"MTPLX {model_id}",
            api_key=api_key,
        )
    return payload


def _inspection_context_window(inspection: dict[str, Any] | None) -> int:
    if not isinstance(inspection, dict):
        return 262_144
    candidates: list[int] = []
    for key in (
        "context_window",
        "context_length",
        "max_context_length",
        "max_model_len",
        "model_max_length",
    ):
        value = inspection.get(key)
        if isinstance(value, int):
            candidates.append(value)
    compatibility = inspection.get("compatibility")
    if isinstance(compatibility, dict):
        for key in ("context_window", "context_length", "max_context_length"):
            value = compatibility.get(key)
            if isinstance(value, int):
                candidates.append(value)
    sane = [value for value in candidates if 0 < value <= 1_000_000]
    return max(sane) if sane else 262_144


def _quickstart_opencode_payload(
    args: Any,
    *,
    write_config: bool = False,
    inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from optiq.runtime.mtp.opencode import (
        build_opencode_provider_config,
        detect_opencode_desktop,
        opencode_config_path,
        opencode_model_ref,
        write_opencode_config,
    )

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    model_id = _public_model_id_for_args(args, str(getattr(args, "model", "")))
    base_url = f"http://{_connect_host_for_bind(host)}:{port}/v1"
    context_window = _inspection_context_window(inspection)
    provider_fragment = build_opencode_provider_config(
        base_url=base_url,
        model_id=model_id,
        model_name=f"MTPLX {model_id}",
        context_window=context_window,
        output_limit=context_window,
        enable_thinking=True,
        top_p=float(getattr(args, "top_p", 0.95)),
    )
    payload = {
        "integration": "opencode",
        "server_url": f"http://{_connect_host_for_bind(host)}:{port}",
        "base_url": base_url,
        "api_base_url": base_url,
        "model_id": model_id,
        "model_ref": opencode_model_ref(model_id),
        "config_path": str(opencode_config_path()),
        "provider": provider_fragment["provider"]["mtplx"],
        "config": provider_fragment,
        "detected": detect_opencode_desktop(),
        "context_window": context_window,
        "output_limit": context_window,
        "reasoning_field": "reasoning_content",
        "no_hidden_max_tokens": True,
        "server_console": True,
        "server_controls": [
            "/reasoning on|off|auto|status",
            "/mtp on|off|status",
            "/stats",
            "/help",
        ],
        "server_command": (
            f"mtplx start opencode --host {host} --port {port} "
            f"--model {shlex.quote(str(getattr(args, 'model', DEFAULT_RUNTIME_MODEL_DIR)))} "
            f"--profile {str(getattr(args, 'profile', None) or DEFAULT_PROFILE_NAME)} "
            f"{'--max ' if bool(getattr(args, 'max', False)) else ''}"
            f"{'--no-mtp ' if _generation_mode_from_args(args) == GENERATION_MODE_AR else ''}"
            "--reasoning on --no-stats"
        ),
        "opencode_steps": [
            f"OpenCode config: {opencode_config_path()}",
            f"Model in OpenCode: {opencode_model_ref(model_id)}",
            f"OpenAI-compatible API base URL: {base_url}",
            "Reasoning stream field: reasoning_content",
        ],
    }
    if write_config:
        payload["config_write"] = write_opencode_config(
            base_url=base_url,
            model_id=model_id,
            model_name=f"MTPLX {model_id}",
            context_window=context_window,
            output_limit=context_window,
            enable_thinking=True,
            top_p=float(getattr(args, "top_p", 0.95)),
        )
    return payload


def _quickstart_swival_payload(
    args: Any,
    *,
    inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from optiq.runtime.mtp.swival import (
        build_swival_command,
        detect_swival_cli,
        shell_swival_command,
    )

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    model_id = _public_model_id_for_args(args, str(getattr(args, "model", "")))
    server_url = f"http://{_connect_host_for_bind(host)}:{port}"
    context_window = _inspection_context_window(inspection)
    command_argv = build_swival_command(
        base_url=server_url,
        model_id=model_id,
        context_window=context_window,
    )
    launch_command = shell_swival_command(
        base_url=server_url,
        model_id=model_id,
        context_window=context_window,
    )
    return {
        "integration": "swival",
        "server_url": server_url,
        "base_url": server_url,
        "api_base_url": server_url.rstrip("/") + "/v1",
        "model_id": model_id,
        "context_window": context_window,
        "detected": detect_swival_cli(),
        "launch_command": launch_command,
        "command_argv": command_argv,
        "no_hidden_max_tokens": True,
        "server_console": True,
        "server_controls": [
            "/reasoning on|off|auto|status",
            "/mtp on|off|status",
            "/stats",
            "/help",
        ],
        "server_command": (
            f"mtplx start swival --host {host} --port {port} "
            f"--model {shlex.quote(str(getattr(args, 'model', DEFAULT_RUNTIME_MODEL_DIR)))} "
            f"--profile {str(getattr(args, 'profile', None) or DEFAULT_PROFILE_NAME)} "
            f"{'--max ' if bool(getattr(args, 'max', False)) else ''}"
            f"{'--no-mtp ' if _generation_mode_from_args(args) == GENERATION_MODE_AR else ''}"
            "--no-stats"
        ),
        "swival_steps": [
            f"Start MTPLX server: {server_url}",
            f"Run Swival: {launch_command}",
            "Provider: generic",
        ],
    }


def _quickstart_print_openwebui_handoff(args: Any, *, runtime_model: str) -> None:
    # The full banner + status panel are rendered by `_print_serve_start_banner`
    # inside `cmd_serve_public`, so this hand-off only emits a brief progress
    # marker. Keeping it minimal avoids visual duplication of the panel.
    _quickstart_line("[2/3] Starting local MTPLX server for the browser chat...")
    _quickstart_line(f"      Loading model: {runtime_model}")
    _quickstart_line("      Keep this terminal open. The next step is the model load.")
    _quickstart_line()


def _quickstart_print_pi_handoff(args: Any, *, runtime_model: str, pi: dict[str, Any]) -> None:
    _quickstart_line("[2/3] Connecting MTPLX to Pi...")
    config_write = pi.get("config_write") if isinstance(pi.get("config_write"), dict) else {}
    config_path = config_write.get("config_path") or pi.get("config_path")
    backup_path = config_write.get("backup_path")
    _quickstart_line(f"      Pi config: {config_path}")
    if backup_path:
        _quickstart_line(f"      Backed up unreadable old Pi config: {backup_path}")
    _quickstart_line(f"      Pi model: {pi.get('model_ref')}")
    _quickstart_line(f"      Loading model: {runtime_model}")
    _quickstart_line("      Keep this terminal open for the MTPLX server.")
    _quickstart_line("      Pi will open automatically when MTPLX is ready.")
    _quickstart_line("      Then type /help here for reasoning and MTP controls.")
    _quickstart_line(f"      Manual fallback: {pi.get('launch_command')}")
    _quickstart_line()


def _quickstart_launch_pi_now(*, model_id: str) -> None:
    from optiq.runtime.mtp.pi import launch_pi_in_terminal, pi_launch_command, pi_model_ref

    model_ref = pi_model_ref(str(model_id))
    command = pi_launch_command(str(model_id))
    result = launch_pi_in_terminal(command, model_ref=model_ref)
    if result.get("ok"):
        _print_serve_start_line("Opening Pi in Terminal...")
    else:
        _print_serve_start_line(f"Could not open Pi automatically: {result.get('error')}")
        _print_serve_start_line(f"Run manually: {command}")


def _quickstart_print_opencode_handoff(
    args: Any,
    *,
    runtime_model: str,
    opencode: dict[str, Any],
) -> None:
    _quickstart_line("[2/3] Connecting MTPLX to OpenCode Desktop...")
    config_write = (
        opencode.get("config_write")
        if isinstance(opencode.get("config_write"), dict)
        else {}
    )
    config_path = config_write.get("config_path") or opencode.get("config_path")
    backup_path = config_write.get("backup_path")
    _quickstart_line(f"      OpenCode config: {config_path}")
    if backup_path:
        _quickstart_line(f"      Backed up unreadable old OpenCode config: {backup_path}")
    _quickstart_line(f"      OpenCode model: {opencode.get('model_ref')}")
    _quickstart_line(f"      API base URL: {opencode.get('api_base_url')}")
    _quickstart_line("      Reasoning: raw reasoning_content stream")
    _quickstart_line("      Response cap: none hidden by MTPLX")
    _quickstart_line(f"      Loading model: {runtime_model}")
    _quickstart_line("      Keep this terminal open for the MTPLX server.")
    _quickstart_line("      OpenCode will open automatically when MTPLX is ready.")
    _quickstart_line()


def _quickstart_launch_opencode_now() -> None:
    from optiq.runtime.mtp.opencode import launch_opencode_app

    result = launch_opencode_app()
    if result.get("ok"):
        _print_serve_start_line("Opening OpenCode Desktop...")
    else:
        _print_serve_start_line(f"Could not open OpenCode automatically: {result.get('error')}")
        _print_serve_start_line("Open OpenCode manually and select the MTPLX model.")


def _quickstart_print_swival_handoff(
    args: Any,
    *,
    runtime_model: str,
    swival: dict[str, Any],
) -> None:
    _quickstart_line("[2/3] Preparing Swival generic provider command...")
    _quickstart_line(f"      MTPLX server: {swival.get('server_url')}")
    _quickstart_line(f"      Swival model: {swival.get('model_id')}")
    _quickstart_line(f"      Context window: {swival.get('context_window')} tokens")
    _quickstart_line("      Provider: generic")
    _quickstart_line("      Response cap: none hidden by MTPLX")
    _quickstart_line(f"      Loading model: {runtime_model}")
    _quickstart_line("      Keep this terminal open for the MTPLX server.")
    _quickstart_line(f"      Run Swival in another terminal: {swival.get('launch_command')}")
    _quickstart_line()


def _quickstart_require_pi_cli(args: Any) -> bool:
    """Return True when the Pi CLI is available, or install it interactively."""

    if shutil.which("pi"):
        return True

    from optiq.runtime.mtp.pi import pi_install_command

    _quickstart_line()
    _quickstart_line("[2/3] Pi is not installed")
    _quickstart_line("      MTPLX has not loaded the model yet.")
    _quickstart_line("      Install Pi first, then MTPLX will connect it to the local server.")
    _quickstart_line()
    _quickstart_line(f"      Install command: {pi_install_command()}")
    _quickstart_line(f"      Then re-run: {_start_invocation(args, ' pi')}")
    _quickstart_line()

    if not sys.stdin.isatty():
        return False

    try:
        answer = input("  Install Pi now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        _quickstart_line("aborted")
        return False
    if answer not in {"", "y", "yes"}:
        _quickstart_line("Pi install skipped. Re-run `mtplx start pi` after installing Pi.")
        return False

    npm = shutil.which("npm")
    if not npm:
        _quickstart_line("error: npm is not on PATH, so MTPLX cannot install Pi automatically.")
        _quickstart_line("Install Node.js/npm, then run:")
        _quickstart_line(f"  {pi_install_command()}")
        return False

    _quickstart_line(f"Running: {pi_install_command()}")
    try:
        result = subprocess.run([npm, "install", "-g", "@earendil-works/pi-coding-agent"], check=False)
    except KeyboardInterrupt:
        _quickstart_line("Pi install cancelled.")
        return False
    except OSError as exc:
        _quickstart_line(f"error: Pi install failed to start: {exc}")
        return False
    if result.returncode != 0:
        _quickstart_line(f"error: Pi install failed with exit code {result.returncode}")
        return False
    if not shutil.which("pi"):
        _quickstart_line("Pi installed, but `pi` is still not visible on this PATH.")
        _quickstart_line("Open a new terminal, then run:")
        _quickstart_line(f"  {_start_invocation(args, ' pi')}")
        return False
    _quickstart_line("Pi installed. Continuing with MTPLX setup.")
    _quickstart_line()
    return True


def _quickstart_run_openwebui(args: Any, *, runtime_model: str, inspection: dict[str, Any]) -> int:
    _quickstart_print_openwebui_handoff(args, runtime_model=runtime_model)
    serve_args = SimpleNamespace(
        model=runtime_model,
        cache_dir=getattr(args, "cache_dir", None),
        profile=getattr(args, "profile", None) or DEFAULT_PROFILE_NAME,
        model_id=getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=True,
        host=str(getattr(args, "host", "127.0.0.1")),
        port=int(getattr(args, "port", 8000)),
        api_key=getattr(args, "api_key", None),
        depth=int(getattr(args, "depth", 3)),
        no_mtp=bool(getattr(args, "no_mtp", False)),
        rate_limit=int(getattr(args, "rate_limit", 0)),
        stream_interval=int(getattr(args, "stream_interval", 1)),
        warmup_tokens=int(getattr(args, "warmup_tokens", 16)),
        max_response_tokens=getattr(args, "max_response_tokens", None),
        temperature=float(getattr(args, "temperature", 0.6)),
        top_p=float(getattr(args, "top_p", 0.95)),
        reasoning=getattr(args, "reasoning", None),
        preserve_thinking=_preserve_thinking_policy(args),
        reasoning_parser=getattr(args, "reasoning_parser", "qwen3"),
        stats_footer=False,
        strict_warmup=bool(getattr(args, "strict_warmup", False)),
        strict_fast_path=bool(getattr(args, "strict_fast_path", False)),
        quickstart_openwebui=True,
        open_browser=True,
        max=bool(getattr(args, "max", False)),
        max_idle_min=int(getattr(args, "max_idle_min", 15)),
    )
    return cmd_serve_public(serve_args)


def _quickstart_run_pi(args: Any, *, runtime_model: str, inspection: dict[str, Any]) -> int:
    if not getattr(args, "api_key", None):
        from optiq.runtime.mtp.pi import PI_LOCAL_API_KEY

        args.api_key = PI_LOCAL_API_KEY
    pi = _quickstart_pi_payload(args, write_config=True)
    _quickstart_print_pi_handoff(args, runtime_model=runtime_model, pi=pi)
    serve_args = SimpleNamespace(
        model=runtime_model,
        cache_dir=getattr(args, "cache_dir", None),
        profile=getattr(args, "profile", None) or DEFAULT_PROFILE_NAME,
        model_id=getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=True,
        host=str(getattr(args, "host", "127.0.0.1")),
        port=int(getattr(args, "port", 8000)),
        api_key=getattr(args, "api_key", None),
        depth=int(getattr(args, "depth", 3)),
        no_mtp=bool(getattr(args, "no_mtp", False)),
        rate_limit=int(getattr(args, "rate_limit", 0)),
        stream_interval=int(getattr(args, "stream_interval", 1)),
        warmup_tokens=int(getattr(args, "warmup_tokens", 16)),
        max_response_tokens=getattr(args, "max_response_tokens", None),
        temperature=float(getattr(args, "temperature", 0.6)),
        top_p=float(getattr(args, "top_p", 0.95)),
        reasoning=getattr(args, "reasoning", None),
        preserve_thinking=_preserve_thinking_policy(args),
        reasoning_parser=getattr(args, "reasoning_parser", "qwen3"),
        stats_footer=False,
        strict_warmup=bool(getattr(args, "strict_warmup", False)),
        strict_fast_path=bool(getattr(args, "strict_fast_path", False)),
        quickstart_openwebui=False,
        quickstart_pi=True,
        open_browser=False,
        max=bool(getattr(args, "max", False)),
        max_idle_min=int(getattr(args, "max_idle_min", 15)),
    )
    return cmd_serve_public(serve_args)


def _quickstart_run_opencode(args: Any, *, runtime_model: str, inspection: dict[str, Any]) -> int:
    if _reasoning_mode(args, default="auto") in {"auto", "on"}:
        args.reasoning = "on"
    opencode = _quickstart_opencode_payload(
        args,
        write_config=True,
        inspection=inspection,
    )
    _quickstart_print_opencode_handoff(
        args,
        runtime_model=runtime_model,
        opencode=opencode,
    )
    serve_args = SimpleNamespace(
        model=runtime_model,
        cache_dir=getattr(args, "cache_dir", None),
        profile=getattr(args, "profile", None) or DEFAULT_PROFILE_NAME,
        model_id=getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=True,
        host=str(getattr(args, "host", "127.0.0.1")),
        port=int(getattr(args, "port", 8000)),
        api_key=getattr(args, "api_key", None),
        depth=int(getattr(args, "depth", 3)),
        no_mtp=bool(getattr(args, "no_mtp", False)),
        rate_limit=int(getattr(args, "rate_limit", 0)),
        stream_interval=int(getattr(args, "stream_interval", 1)),
        warmup_tokens=int(getattr(args, "warmup_tokens", 16)),
        max_response_tokens=getattr(args, "max_response_tokens", None),
        temperature=float(getattr(args, "temperature", 0.6)),
        top_p=float(getattr(args, "top_p", 0.95)),
        reasoning="on",
        preserve_thinking=_preserve_thinking_policy(args),
        reasoning_parser=getattr(args, "reasoning_parser", "qwen3"),
        stats_footer=False,
        strict_warmup=bool(getattr(args, "strict_warmup", False)),
        strict_fast_path=bool(getattr(args, "strict_fast_path", False)),
        quickstart_openwebui=False,
        quickstart_pi=False,
        quickstart_opencode=True,
        open_browser=False,
        max=bool(getattr(args, "max", False)),
        max_idle_min=int(getattr(args, "max_idle_min", 15)),
    )
    return cmd_serve_public(serve_args)


def _quickstart_run_swival(args: Any, *, runtime_model: str, inspection: dict[str, Any]) -> int:
    swival = _quickstart_swival_payload(args, inspection=inspection)
    _quickstart_print_swival_handoff(
        args,
        runtime_model=runtime_model,
        swival=swival,
    )
    serve_args = SimpleNamespace(
        model=runtime_model,
        cache_dir=getattr(args, "cache_dir", None),
        profile=getattr(args, "profile", None) or DEFAULT_PROFILE_NAME,
        model_id=getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=True,
        host=str(getattr(args, "host", "127.0.0.1")),
        port=int(getattr(args, "port", 8000)),
        api_key=getattr(args, "api_key", None),
        depth=int(getattr(args, "depth", 3)),
        no_mtp=bool(getattr(args, "no_mtp", False)),
        rate_limit=int(getattr(args, "rate_limit", 0)),
        stream_interval=int(getattr(args, "stream_interval", 1)),
        warmup_tokens=int(getattr(args, "warmup_tokens", 16)),
        max_response_tokens=getattr(args, "max_response_tokens", None),
        temperature=float(getattr(args, "temperature", 0.6)),
        top_p=float(getattr(args, "top_p", 0.95)),
        reasoning=getattr(args, "reasoning", None),
        preserve_thinking=_preserve_thinking_policy(args),
        reasoning_parser=getattr(args, "reasoning_parser", "qwen3"),
        stats_footer=False,
        strict_warmup=bool(getattr(args, "strict_warmup", False)),
        strict_fast_path=bool(getattr(args, "strict_fast_path", False)),
        quickstart_openwebui=False,
        quickstart_pi=False,
        quickstart_opencode=False,
        quickstart_swival=True,
        open_browser=False,
        max=bool(getattr(args, "max", False)),
        max_idle_min=int(getattr(args, "max_idle_min", 15)),
    )
    return cmd_serve_public(serve_args)


def _quickstart_run_terminal_chat(args: Any, *, runtime_model: str, inspection: dict[str, Any]) -> int:
    max_session: Any | None = None
    if getattr(args, "max", False):
        from optiq.runtime.mtp.thermal import MaxSession

        max_session = MaxSession(log=_quickstart_line)
        if not max_session.start():
            verified = max_session.thermal.get("verified") or {}
            _quickstart_line()
            _quickstart_line("[max] fan boost unavailable; terminal chat will continue without fan boost.")
            if verified.get("message"):
                _quickstart_line(f"[max] reason: {verified.get('message')}")
            _quickstart_line()
            args.max = False
            max_session = None
    try:
        return _quickstart_run_terminal_chat_body(
            args,
            runtime_model=runtime_model,
            inspection=inspection,
        )
    finally:
        if max_session is not None:
            max_session.stop()


def _quickstart_run_terminal_chat_body(args: Any, *, runtime_model: str, inspection: dict[str, Any]) -> int:
    profile = get_profile(getattr(args, "profile", None) or DEFAULT_PROFILE_NAME)
    apply_profile_env(profile.name)
    generation_mode = _generation_mode_from_args(args)
    draft_lm_head = _model_draft_lm_head_spec(inspection, profile)
    draft_sampler = _model_draft_sampler_spec(inspection, profile)

    from optiq.runtime.mtp.runtime import load
    from optiq.runtime.mtp.ui import ModelLoadProgress, render_banner, render_startup_panel
    from optiq.runtime.mtp.version import DISPLAY_VERSION

    if sys.stdout.isatty():
        render_banner()
        render_startup_panel(
            version=DISPLAY_VERSION,
            model=str(runtime_model),
            profile=profile.name,
            profile_summary=_PROFILE_SHORT_SUMMARIES.get(profile.name),
            api_url="terminal chat (no server)",
            mode_label=(
                "AR target-only"
                if generation_mode == GENERATION_MODE_AR
                else
                _runtime_mode_display(
                    profile.name,
                    max_mode=bool(getattr(args, "max", False)),
                    generation_mode=generation_mode,
                )
            ),
            extra_lines=[
                ("Sampler", (
                    f"temp={float(getattr(args, 'temperature', 0.6)):.2f} "
                    f"top_p={float(getattr(args, 'top_p', 0.95)):.2f} "
                    f"top_k={int(getattr(args, 'top_k', 20))} "
                    f"depth={int(getattr(args, 'depth', 3))}"
                )),
                ("Reasoning", _reasoning_mode(args)),
            ],
        )

    started = time.perf_counter()
    quiet_progress = not sys.stdout.isatty()
    with ModelLoadProgress("Loading model", quiet=quiet_progress) as progress:
        progress.set_subtitle(f"profile {profile.name}")
        rt = load(runtime_model, mtp=True)
        progress.set_subtitle("ready")
    _quickstart_line(f"Model ready in {time.perf_counter() - started:.1f}s")
    _quickstart_line(f"Generation mode: {_generation_mode_label(generation_mode)}")
    draft_report = None
    if draft_lm_head is not None:
        _quickstart_line(
            "[3/4] Installing fast draft head: "
            f"{int(draft_lm_head['bits'])}-bit gs{int(draft_lm_head['group_size'])}"
        )
        draft_started = time.perf_counter()
        from optiq.runtime.mtp.draft_lm_head import _install_draft_lm_head

        draft_report = _install_draft_lm_head(
            rt,
            bits=int(draft_lm_head["bits"]),
            group_size=int(draft_lm_head["group_size"]),
            mode=str(draft_lm_head["mode"]),
        )
        _quickstart_line(f"      draft head ready in {time.perf_counter() - draft_started:.1f}s")
    _quickstart_line(
        f"Sampler: temp={float(getattr(args, 'temperature', 0.6)):.2f} "
        f"top_p={float(getattr(args, 'top_p', 0.95)):.2f} "
        f"top_k={int(getattr(args, 'top_k', 20))} depth={int(getattr(args, 'depth', 3))}"
    )
    _quickstart_line(f"Reasoning: {_reasoning_mode(args)}")
    if draft_report is not None:
        if generation_mode == GENERATION_MODE_AR:
            _quickstart_line("Draft-only LM head loaded for /mtp on; current AR mode bypasses it.")
        else:
            _quickstart_line("Native-MTP speed path: draft-only LM head is active.")
    if draft_sampler is not None and generation_mode == GENERATION_MODE_MTP:
        _quickstart_line(
            "Draft sampler: "
            f"temp={float(draft_sampler['temperature']):.2f} "
            f"top_p={float(draft_sampler['top_p']):.2f} "
            f"top_k={int(draft_sampler['top_k'])}"
        )
    _quickstart_line()

    history: list[dict[str, str]] = []
    last_payload: dict[str, Any] | None = None

    def run_turn(
        prompt: str,
        turn_index: int,
        *,
        max_tokens: int | None = None,
        include_history: bool = True,
        record_history: bool = True,
        quality_gate: bool = True,
        response_label: str = "MTPLX",
    ) -> int:
        nonlocal last_payload
        _quickstart_line("[4/4] generating response...")
        active_mode = _generation_mode_from_args(args)
        turn_label = (
            f"MTPLX {_generation_mode_label(active_mode)}"
            if response_label == "MTPLX"
            else response_label
        )
        payload = _quickstart_generate(
            rt=rt,
            inspection=inspection,
            profile=profile,
            args=args,
            prompt=prompt,
            history=history,
            turn_index=turn_index,
            max_tokens=max_tokens,
            include_history=include_history,
            stream_label=turn_label,
            draft_sampler=draft_sampler,
        )
        last_payload = payload
        text = str(payload["text"])
        if not payload.get("streamed"):
            _print_assistant_fallback(turn_label, text)
        if bool(getattr(args, "show_stats", True)):
            _quickstart_line()
            _print_stats_line(_quickstart_stats_line(payload))
        if record_history:
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": text})
        failures = [
            row for row in payload["validations"]
            if not row.get("passed")
        ]
        if failures and quality_gate:
            _quickstart_line("[mtplx] quality warning: response failed a basic output validator")
            return EXIT_QUALITY
        return 0

    first_prompt = getattr(args, "prompt", None)
    if first_prompt:
        return run_turn(str(first_prompt), 0)

    if not sys.stdin.isatty():
        _quickstart_line("error: no interactive terminal detected")
        prompt_hint = _start_invocation(args, ' --prompt "Say hi"')
        _quickstart_line(f"try: {prompt_hint}")
        return 2

    _quickstart_line(
        "Chat is ready. Type /mtp on|off|status, /stats, /speed, /reasoning on|off|auto, or /exit."
    )
    turn_index = 0
    worst_code = 0
    while True:
        try:
            prompt = input(_chat_input_prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            _quickstart_line()
            _quickstart_line("bye")
            return worst_code
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "exit", "quit", "/quit"}:
            _quickstart_line("bye")
            return worst_code
        if _handle_quickstart_reasoning_command(args, prompt):
            continue
        if _handle_quickstart_mtp_command(args, prompt, runtime=rt):
            continue
        if prompt.lower() in {"/stats", "stats"}:
            if last_payload is None:
                _quickstart_line("No stats yet.")
            else:
                _print_stats_line(_quickstart_stats_line(last_payload))
            continue
        if prompt.lower() in {"/speed", "speed", "/bench", "/benchmark"}:
            _quickstart_line(
                f"Running a {QUICKSTART_SPEED_MAX_TOKENS}-token speed sample without chat history."
            )
            code = run_turn(
                QUICKSTART_SPEED_PROMPT,
                turn_index,
                max_tokens=QUICKSTART_SPEED_MAX_TOKENS,
                include_history=False,
                record_history=False,
                quality_gate=False,
                response_label="MTPLX speed sample",
            )
            worst_code = max(worst_code, code)
            turn_index += 1
            continue
        code = run_turn(prompt, turn_index)
        worst_code = max(worst_code, code)
        turn_index += 1


def _quickstart_apply_tuned_depth(
    args: Any,
    *,
    runtime_model: str,
    target: str,
    can_prompt: bool,
) -> None:
    if target not in {"openwebui", "terminal"}:
        return
    if bool(getattr(args, "_explicit_depth", False)):
        return
    settings = {
        "profile": "performance-cold",
        "suite": TUNE_DEFAULT_SUITE,
        "depths": TUNE_DEFAULT_DEPTHS,
        "max_tokens": TUNE_DEFAULT_MAX_TOKENS,
        "limit": TUNE_DEFAULT_LIMIT,
        "seed": TUNE_DEFAULT_SEED,
        "thinking": "disabled",
    }
    profile = get_profile("performance-cold")
    hardware = _apple_hardware_context()
    software = _software_context()
    backend = _mlx_backend_context(profile)
    state_key, _key_material = _tune_state_key(
        runtime_model,
        settings=settings,
        hardware=hardware,
        software=software,
        backend=backend,
    )
    record = _load_tune_record(state_key)
    if record is not None:
        payload = record.get("payload") or {}
        best = payload.get("best") or {}
        depth = best.get("depth")
        if isinstance(depth, int):
            args.depth = depth
            _quickstart_line(
                f"tuned depth: D{depth} "
                f"({_fmt_metric(best.get('multiplier_vs_ar'), digits=2)}x AR)"
            )
            return
    if not can_prompt:
        return
    try:
        from optiq.runtime.mtp.ui.onboarding import screen_tuning_offer

        should_tune = screen_tuning_offer()
    except (KeyboardInterrupt, EOFError):
        _quickstart_line()
        _quickstart_line("tuning skipped")
        return
    if not should_tune:
        _quickstart_line("tuning skipped; using default depth")
        return
    tune_args = SimpleNamespace(
        command="tune",
        model=runtime_model,
        cache_dir=getattr(args, "cache_dir", None),
        depths=TUNE_DEFAULT_DEPTHS,
        max_tokens=TUNE_DEFAULT_MAX_TOKENS,
        limit=TUNE_DEFAULT_LIMIT,
        seed=TUNE_DEFAULT_SEED,
        run_id=None,
        output_dir=None,
        output=None,
        json=False,
        verbose=False,
        dry_run=False,
        no_save=False,
        retune=False,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=True,
    )
    code = _cmd_tune(
        tune_args,
        action="tune",
        save_default=True,
        verbose_default=False,
    )
    if code != 0:
        _quickstart_line("tuning failed; using default depth")
        return
    record = _load_tune_record(state_key)
    payload = record.get("payload") if isinstance(record, dict) else None
    best = payload.get("best") if isinstance(payload, dict) else None
    depth = best.get("depth") if isinstance(best, dict) else None
    if isinstance(depth, int):
        args.depth = depth

def cmd_quickstart_public(args: Any) -> int:
    raw_target = getattr(args, "target", None)

    # Start is the most common reason fans get blasted, so this is the
    # right place to scrub a stale --max marker left behind by a previously
    # killed session. No-op when the previous run exited cleanly.
    try:
        from optiq.runtime.mtp.thermal import check_and_recover_stale_max

        recovery = check_and_recover_stale_max()
        if recovery.get("recovered"):
            sys.stderr.write(
                "[max] restored fans from a previous --max session that "
                f"did not exit cleanly (stale pid {recovery.get('stale_pid')})\n"
            )
    except Exception:
        pass  # best-effort — never block start on cleanup

    # Onboarding flow: only when interactive, no explicit CLI overrides, and not
    # in dry-run / one-shot prompt / non-interactive automation. We detect
    # explicit CLI flags by scanning the raw argv tokens (stashed by main()),
    # because parser defaults and ``apply_user_config`` both overwrite the
    # parsed Namespace and would otherwise mask the user's actual intent.
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    has_explicit_target = raw_target is not None
    has_explicit_model = "model" in cli_flags
    args._model_explicit = has_explicit_model
    has_explicit_profile_flag = "profile" in cli_flags
    has_explicit_depth = "depth" in cli_flags
    args._explicit_depth = has_explicit_depth
    has_explicit_max = "max" in cli_flags
    has_prompt = bool(getattr(args, "prompt", None))
    is_dry_run = bool(getattr(args, "dry_run", False))
    is_yes = bool(getattr(args, "yes", False))
    fresh = bool(getattr(args, "fresh", False))

    skip_onboarding = (
        is_dry_run
        or has_prompt
        or is_yes
        or has_explicit_target
        or has_explicit_model
        or has_explicit_profile_flag
        or has_explicit_max
        or not is_tty
    )

    if not skip_onboarding:
        from optiq.runtime.mtp.ui.onboarding import run_quickstart_flow

        configured_model = getattr(args, "model", None)
        choice = run_quickstart_flow(fresh=fresh, configured_model=configured_model)
        if choice is None:
            _quickstart_line("aborted")
            return 130
        chosen_model = choice.get("model")
        if chosen_model:
            args.model = chosen_model
            # Auto-pull policy: the user has explicitly picked this model in
            # the onboarding wizard; if it isn't on disk we fetch it without
            # re-prompting. The legacy "Model is missing. Download? [Y/n]"
            # fallback below is reserved for non-onboarded shortcuts.
            try:
                from optiq.runtime.mtp.hf_loader import repo_id_from_model_ref

                if repo_id_from_model_ref(chosen_model):
                    args.download = True
            except Exception:
                # Best-effort: never let an import problem break the wizard.
                pass
        chosen_profile = choice.get("profile")
        if chosen_profile:
            args.profile = chosen_profile
        if choice.get("max"):
            args.max = True
        else:
            # If onboarding declined a fan-backed mode (e.g. ThermalForge
            # install was refused or failed), make sure --max from a stale
            # config or shell alias does not silently re-enable it.
            args.max = False
        chosen_target = choice.get("target")
        if chosen_target:
            raw_target = chosen_target
        # The new onboarding has already collected the user's choices, so the
        # legacy ``_quickstart_choose_model`` picker must not prompt again.
        args._onboarded = True
    else:
        # Skipped onboarding (explicit flags). If the user passed --max but
        # has no fan controller, offer to auto-install before MTPLX boots
        # rather than silently dumping the JSON warning later.
        if has_explicit_max and is_tty:
            from optiq.runtime.mtp.thermal import detect_thermal_control

            detection = detect_thermal_control()
            if not detection.get("available"):
                from optiq.runtime.mtp.ui.onboarding import ensure_thermal_control_installed

                if not ensure_thermal_control_installed():
                    args.max = False

    if raw_target is None:
        raw_target = "cli" if has_prompt else "web"
    raw_target = str(raw_target).lower()
    if raw_target in {"open-webui", "openwebui", "web"}:
        target = "openwebui"
    elif raw_target in {"cli", "terminal"}:
        target = "terminal"
    elif raw_target in {"pi", "pie"}:
        target = "pi"
    elif raw_target in {"opencode", "open-code", "oc"}:
        target = "opencode"
    elif raw_target in {"swival", "sv"}:
        target = "swival"
    else:
        target = raw_target
    if target not in QUICKSTART_TARGETS:
        _quickstart_line(f"error: unknown start target: {raw_target}")
        _quickstart_line(f"try: {_start_invocation(args)}")
        _quickstart_line(f"try: {_start_invocation(args, ' cli')}")
        return 2
    if target == "opencode" and "port" not in cli_flags:
        args.port = 18083
    if target == "swival" and "port" not in cli_flags:
        args.port = 18084
    depth_error = _validate_public_depth(args, printer=_quickstart_line)
    if depth_error is not None:
        return depth_error
    model, download = _quickstart_choose_model(args, target=target)
    default_selection = getattr(args, "_mtplx_default_model_selection", None)
    if isinstance(default_selection, dict) and model != default_selection.get("model"):
        default_selection = None
    cache_dir = getattr(args, "cache_dir", None)
    if getattr(args, "dry_run", False):
        openwebui = _quickstart_openwebui_payload(args) if target == "openwebui" else None
        pi = _quickstart_pi_payload(args) if target == "pi" else None
        opencode = _quickstart_opencode_payload(args) if target == "opencode" else None
        swival = _quickstart_swival_payload(args) if target == "swival" else None
        payload = {
            "action": _start_command_name(args),
            "target": target,
            "model": model,
            "cache_dir": cache_dir,
            "profile": getattr(args, "profile", DEFAULT_PROFILE_NAME),
            "generation_mode": _generation_mode_from_args(args),
            "max": bool(getattr(args, "max", False)),
            "download_if_missing": download,
            "default_model_selection": default_selection,
            "terminal_chat": target == "terminal",
            "openwebui": openwebui,
            "pi": pi,
            "opencode": opencode,
            "swival": swival,
            "stats_visible": bool(getattr(args, "show_stats", True)),
            "next": (
                _start_invocation(args)
                if target == "openwebui"
                else _start_invocation(args, " swival")
                if target == "swival"
                else _start_invocation(args, " opencode")
                if target == "opencode"
                else _start_invocation(args, " pi")
                if target == "pi"
                else _start_invocation(args, " cli")
            ),
        }
        if getattr(args, "json", False):
            _print(payload)
        else:
            _quickstart_line(f"MTPLX {_start_command_name(args)}")
            _quickstart_line(f"model: {model}")
            if isinstance(default_selection, dict):
                _quickstart_line(
                    "verified default for this Mac: "
                    f"{default_selection.get('display_name')} "
                    f"({default_selection.get('reason')})"
                )
            _quickstart_line(f"profile: {payload['profile']}")
            _quickstart_line(
                "mode: "
                + _runtime_mode_display(
                    str(payload["profile"]),
                    max_mode=bool(payload["max"]),
                    generation_mode=str(payload["generation_mode"]),
                )
            )
            _quickstart_line(f"generation: {_generation_mode_label(payload['generation_mode'])}")
            _quickstart_line(f"download if missing: {str(download).lower()}")
            if target == "openwebui":
                _quickstart_line(f"then: start local server -> open browser chat at {openwebui['chat_url']}")
            elif target == "opencode":
                _quickstart_line(
                    f"then: write OpenCode config -> start local server -> open {opencode['model_ref']}"
                )
            elif target == "swival":
                _quickstart_line(
                    f"then: start local server -> run {swival['launch_command']}"
                )
            elif target == "pi":
                _quickstart_line(
                    f"then: write Pi config -> start local server -> run {pi['launch_command']}"
                )
            else:
                _quickstart_line("then: load once -> chat in this terminal -> stream output -> show speed stats")
        return 0

    if target == "pi" and not _quickstart_require_pi_cli(args):
        return 2

    _quickstart_line(f"MTPLX {_start_command_name(args)}")
    if isinstance(default_selection, dict):
        _quickstart_line(
            "Verified default for this Mac: "
            f"{default_selection.get('display_name')} "
            f"({default_selection.get('reason')})"
        )
    _quickstart_line(f"[1/4] Checking model: {model}")
    try:
        runtime_model, resolution = _quickstart_resolve_model(model, cache_dir=cache_dir, download=download)
    except KeyboardInterrupt:
        _quickstart_line("download cancelled")
        return 130
    except Exception as exc:
        _quickstart_line(f"error: {exc}")
        return 1
    if runtime_model is None:
        if resolution.get("cancelled"):
            _quickstart_line("download cancelled")
            return 130
        gate_inspection = resolution.get("gate_inspection")
        if isinstance(gate_inspection, dict):
            _print_model_gate_error(
                gate_inspection,
                printer=_quickstart_line,
                json_output=bool(getattr(args, "json", False)),
            )
            compatibility = gate_inspection.get("compatibility") or {}
            return int(compatibility.get("exit_code") or 1)
        resolution_error = resolution.get("error")
        if (
            isinstance(resolution_error, dict)
            and resolution_error.get("error") not in {None, "model is not available locally"}
        ):
            _print_command_error(
                resolution_error,
                command="start",
                json_output=bool(getattr(args, "json", False)),
            )
            return 1
        if sys.stdin.isatty() and not getattr(args, "prompt", None):
            try:
                download_model = _quickstart_download_ref(model)
                label = "selected model" if download_model == model else "verified default"
            except ValueError:
                download_model = select_default_model().hf_model
                label = "verified default"
            answer = input(f"Model is missing. Download the {label} ({download_model}) now? [Y/n] ").strip().lower()
            if answer in {"", "y", "yes"}:
                try:
                    runtime_model, resolution = _quickstart_resolve_model(download_model, cache_dir=cache_dir, download=True)
                except KeyboardInterrupt:
                    _quickstart_line("download cancelled")
                    return 130
                except Exception as exc:
                    _quickstart_line(f"error: {exc}")
                    return 1
        if runtime_model is None and resolution.get("cancelled"):
            _quickstart_line("download cancelled")
            return 130
        gate_inspection = resolution.get("gate_inspection")
        if runtime_model is None and isinstance(gate_inspection, dict):
            _print_model_gate_error(
                gate_inspection,
                printer=_quickstart_line,
                json_output=bool(getattr(args, "json", False)),
            )
            compatibility = gate_inspection.get("compatibility") or {}
            return int(compatibility.get("exit_code") or 1)
        resolution_error = resolution.get("error")
        if (
            runtime_model is None
            and isinstance(resolution_error, dict)
            and resolution_error.get("error") not in {None, "model is not available locally"}
        ):
            _print_command_error(
                resolution_error,
                command="start",
                json_output=bool(getattr(args, "json", False)),
            )
            return 1
        if runtime_model is not None:
            _quickstart_line(f"model ready: {runtime_model}")
            if resolution.get("downloaded"):
                _quickstart_line(f"downloaded: {resolution.get('download_ref')}")
            inspection, gate_exit = _model_gate(
                runtime_model,
                unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
                yes=bool(getattr(args, "yes", False)),
            )
            if gate_exit is not None:
                _print_model_gate_error(
                    inspection,
                    printer=_quickstart_line,
                    json_output=bool(getattr(args, "json", False)),
                )
                return gate_exit
            _apply_model_contract_depth_default(args, inspection)
            _quickstart_apply_tuned_depth(
                args,
                runtime_model=runtime_model,
                target=target,
                can_prompt=bool(getattr(args, "_onboarded", False) and is_tty and not is_yes),
            )
            if target == "openwebui":
                args.model = runtime_model
                return _quickstart_run_openwebui(args, runtime_model=runtime_model, inspection=inspection)
            if target == "pi":
                args.model = runtime_model
                return _quickstart_run_pi(args, runtime_model=runtime_model, inspection=inspection)
            if target == "opencode":
                args.model = runtime_model
                return _quickstart_run_opencode(args, runtime_model=runtime_model, inspection=inspection)
            if target == "swival":
                args.model = runtime_model
                return _quickstart_run_swival(args, runtime_model=runtime_model, inspection=inspection)
            return _quickstart_run_terminal_chat(args, runtime_model=runtime_model, inspection=inspection)
        detail = (resolution.get("error") or {}).get("detail") if isinstance(resolution.get("error"), dict) else None
        _quickstart_line("model is not available locally")
        if detail:
            _quickstart_line(f"detail: {detail}")
        _quickstart_line(f"try: {_start_invocation(args, ' --download')}")
        _quickstart_line(
            "try: "
            f"{_start_invocation(args, ' cli' if target == 'terminal' else '')} "
            f"--model {shlex.quote(str(model))} --download"
        )
        _quickstart_line(f"try: {_start_invocation(args, ' --model /path/to/model')}")
        return 1

    inspection, gate_exit = _model_gate(
        runtime_model,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=bool(getattr(args, "yes", False)),
    )
    if gate_exit is not None:
        _print_model_gate_error(
            inspection,
            printer=_quickstart_line,
            json_output=bool(getattr(args, "json", False)),
        )
        return gate_exit
    _apply_model_contract_depth_default(args, inspection)
    _quickstart_apply_tuned_depth(
        args,
        runtime_model=runtime_model,
        target=target,
        can_prompt=bool(getattr(args, "_onboarded", False) and is_tty and not is_yes),
    )
    _quickstart_line(f"model ready: {runtime_model}")
    if resolution.get("downloaded"):
        _quickstart_line(f"downloaded: {resolution.get('download_ref')}")
    if target == "openwebui":
        args.model = runtime_model
        return _quickstart_run_openwebui(args, runtime_model=runtime_model, inspection=inspection)
    if target == "pi":
        args.model = runtime_model
        return _quickstart_run_pi(args, runtime_model=runtime_model, inspection=inspection)
    if target == "opencode":
        args.model = runtime_model
        return _quickstart_run_opencode(args, runtime_model=runtime_model, inspection=inspection)
    if target == "swival":
        args.model = runtime_model
        return _quickstart_run_swival(args, runtime_model=runtime_model, inspection=inspection)
    return _quickstart_run_terminal_chat(args, runtime_model=runtime_model, inspection=inspection)


def cmd_metrics_public(args: Any) -> int:
    if args.metrics_action != "watch":
        raise SystemExit(f"unknown metrics action: {args.metrics_action}")
    base = args.url.rstrip("/")
    count = int(getattr(args, "count", 0) or 0)
    interval = float(getattr(args, "interval", 1.0))
    seen = 0
    while True:
        payload = _http_json(base + "/metrics", timeout=float(getattr(args, "timeout", 5.0)))
        latest = payload.get("latest") if isinstance(payload, dict) else None
        row = {
            "url": base + "/metrics",
            "ok": bool(isinstance(payload, dict) and "error" not in payload),
            "latest": latest,
        }
        if getattr(args, "json", False):
            print(json.dumps(row, sort_keys=True))
        else:
            if not row["ok"]:
                print(f"metrics: cannot reach {base}/metrics")
                if isinstance(payload, dict) and payload.get("detail"):
                    print(f"detail: {payload.get('detail')}")
            elif latest:
                tok_s = latest.get("tok_s") or latest.get("decode_tok_s") or latest.get("server_tok_s")
                generated = latest.get("completion_tokens") or latest.get("generated_tokens")
                verify_ms = latest.get("verify_ms_per_call") or latest.get("late_verify_ms")
                cache_ratio = latest.get("cache_ratio")
                print(
                    "tok_s={tok_s} tokens={tokens} verify_ms={verify_ms} cache_ratio={cache_ratio}".format(
                        tok_s=tok_s if tok_s is not None else "n/a",
                        tokens=generated if generated is not None else "n/a",
                        verify_ms=verify_ms if verify_ms is not None else "n/a",
                        cache_ratio=cache_ratio if cache_ratio is not None else "n/a",
                    )
                )
            else:
                print("metrics: no generation recorded yet")
        seen += 1
        if count and seen >= count:
            break
        time.sleep(interval)
    return 0 if row.get("ok") else 1


def cmd_integrate_public(args: Any) -> int:
    action = args.integration
    server_url = f"http://{args.host}:{args.port}"
    api_base_url = server_url.rstrip("/") + "/v1"
    model_id = getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID
    if action == "openwebui":
        docker_command = _openwebui_docker_command(
            mtplx_port=int(args.port),
            webui_port=int(getattr(args, "webui_port", 3000) or 3000),
            single_user=bool(getattr(args, "single_user", False)),
            api_key=str(getattr(args, "api_key", None) or "mtplx-local"),
        )
        payload = {
            "integration": "openwebui",
            "server_url": server_url,
            "base_url": api_base_url,
            "api_base_url": api_base_url,
            "docker_api_base_url": _openwebui_docker_api_base_url(int(args.port)),
            "model_id": model_id,
            "server_command": (
                f"mtplx quickstart --profile sustained --host {args.host} --port {args.port} "
                "--no-stats-footer"
            ),
            "docker_command": _shell_join(docker_command),
            "docker_command_argv": docker_command,
            "single_user_warning": (
                "WEBUI_AUTH=False creates a single-user Open WebUI instance and "
                "cannot be safely toggled on an existing shared data volume."
                if getattr(args, "single_user", False)
                else None
            ),
            "api_key": {
                "required_for_localhost": False,
                "env": args.api_key_env,
            },
            "notes": [
                "Use the /v1 base URL as the OpenAI-compatible endpoint.",
                "Dockerized Open WebUI must use host.docker.internal, not 127.0.0.1, to reach MTPLX on the Mac host.",
                "Keep --no-stats-footer enabled for UI clients.",
            ],
        }
    elif action == "claude-code":
        payload = {
            "integration": "claude-code",
            "base_url": server_url,
            "model_id": model_id,
            "environment": {
                "ANTHROPIC_BASE_URL": server_url,
                "ANTHROPIC_API_KEY": f"${args.api_key_env}",
            },
            "server_command": (
                f"mtplx quickstart --profile sustained --host {args.host} --port {args.port} "
                "--no-stats-footer"
            ),
            "smoke": {
                "root_probe": f"curl {server_url}/",
                "messages": f"curl {server_url}/v1/messages",
            },
        }
    elif action == "opencode":
        payload = {
            "integration": "opencode",
            "server_url": server_url,
            "base_url": api_base_url,
            "api_base_url": api_base_url,
            "model_id": model_id,
            "config_path": "~/.config/opencode/opencode.json",
            "server_command": (
                f"mtplx quickstart --profile sustained --host {args.host} --port {args.port} "
                "--reasoning on --no-stats-footer"
            ),
            "config": {
                "provider": {
                    "mtplx": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "MTPLX (local)",
                        "options": {
                            "baseURL": api_base_url,
                            "apiKey": str(getattr(args, "api_key", None) or "mtplx-local"),
                            "timeout": False,
                            "chunkTimeout": 900000,
                        },
                        "models": {
                            model_id: {
                                "name": "MTPLX local",
                                "reasoning": True,
                                "interleaved": {"field": "reasoning_content"},
                                "tool_call": True,
                                "temperature": True,
                                "limit": {
                                    "context": 262144,
                                    "output": 262144,
                                },
                                "modalities": {
                                    "input": ["text"],
                                    "output": ["text"],
                                },
                                "options": {
                                    "enable_thinking": True,
                                },
                            }
                        },
                    }
                },
                "model": f"mtplx/{model_id}",
                "small_model": f"mtplx/{model_id}",
            },
            "notes": [
                "OpenCode's UI setting says reasoning summaries, but this config uses the raw interleaved reasoning_content stream.",
                "Do not add OpenAI reasoningSummary/reasoningEffort fields for MTPLX; those are provider-summary controls, not Qwen raw thinking.",
                "MTPLX now defaults reasoning on, and this config also sends enable_thinking=true explicitly for tool-active requests.",
            ],
        }
    elif action == "swival":
        from optiq.runtime.mtp.swival import build_swival_command, detect_swival_cli, shell_swival_command

        context_window = int(getattr(args, "context_window", None) or 262144)
        payload = {
            "integration": "swival",
            "server_url": server_url,
            "base_url": server_url,
            "api_base_url": api_base_url,
            "model_id": model_id,
            "context_window": context_window,
            "detected": detect_swival_cli(),
            "launch_command": shell_swival_command(
                base_url=server_url,
                model_id=model_id,
                context_window=context_window,
            ),
            "command_argv": build_swival_command(
                base_url=server_url,
                model_id=model_id,
                context_window=context_window,
            ),
            "server_command": (
                f"mtplx quickstart --profile sustained --host {args.host} --port {args.port} "
                "--no-stats-footer"
            ),
            "notes": [
                "Swival generic provider receives the root server URL; Swival handles the OpenAI-compatible /v1 path.",
                "No MTPLX config file is written for Swival.",
                "No hidden output cap is added.",
            ],
        }
    else:
        raise SystemExit(f"unknown integration: {action}")
    if getattr(args, "smoke", False):
        payload["smoke_result"] = {
            "health": _http_json(server_url + "/health", timeout=float(args.timeout)),
            "models": _http_json(api_base_url + "/models", timeout=float(args.timeout)),
        }
    if getattr(args, "json", False):
        _print(payload)
    else:
        print(f"MTPLX connect: {action}")
        print(f"base URL: {api_base_url if action in {'openwebui', 'opencode'} else server_url}")
        print(f"model: {model_id}")
        print(f"start server: {payload.get('server_command')}")
        if action == "openwebui":
            print("Open WebUI:")
            print("  Settings -> Connections -> OpenAI API")
            print(f"  API base URL: {api_base_url}")
            print("  API key: leave blank for localhost")
            if getattr(args, "docker", False):
                print("Docker:")
                print(f"  {_shell_join(payload['docker_command_argv'])}")
        elif action == "opencode":
            print("OpenCode:")
            print("  Config path: ~/.config/opencode/opencode.json")
            print("  Provider: mtplx")
            print("  Reasoning: raw reasoning_content stream, enable_thinking=true")
        elif action == "swival":
            print("Swival:")
            print(f"  {payload.get('launch_command')}")
        else:
            env = payload.get("environment") or {}
            print("Claude Code environment:")
            for key, value in env.items():
                print(f"  {key}={value}")
        smoke = payload.get("smoke_result")
        if isinstance(smoke, dict):
            health = smoke.get("health") if isinstance(smoke.get("health"), dict) else {}
            models = smoke.get("models") if isinstance(smoke.get("models"), dict) else {}
            print("smoke:")
            print(f"  health: {'ok' if health.get('ok') else 'failed'}")
            data = models.get("data") if isinstance(models, dict) else None
            if isinstance(data, list):
                print(f"  models endpoint: ok ({len(data)} model(s))")
            elif models.get("error"):
                print(f"  models endpoint: failed ({models.get('error')})")
            else:
                print("  models endpoint: unknown")
    return 0


def cmd_openwebui_public(args: Any) -> int:
    if args.openwebui_action != "docker-command":
        raise SystemExit(f"unknown openwebui action: {args.openwebui_action}")
    command = _openwebui_docker_command(
        mtplx_port=int(getattr(args, "mtplx_port", 8000)),
        webui_port=int(getattr(args, "webui_port", 3000)),
        single_user=bool(getattr(args, "single_user", False)),
        api_key=str(getattr(args, "api_key", None) or "mtplx-local"),
    )
    payload = {
        "action": "openwebui docker-command",
        "docker_command": _shell_join(command),
        "docker_command_argv": command,
        "openwebui_url": f"http://127.0.0.1:{int(getattr(args, 'webui_port', 3000))}",
        "mtplx_api_base_url_for_container": _openwebui_docker_api_base_url(
            int(getattr(args, "mtplx_port", 8000))
        ),
        "single_user_warning": (
            "WEBUI_AUTH=False creates a single-user Open WebUI instance and cannot be safely toggled on an existing shared data volume."
            if getattr(args, "single_user", False)
            else None
        ),
    }
    if getattr(args, "json", False):
        _print(payload)
    else:
        print(payload["docker_command"])
        print(f"Open WebUI: {payload['openwebui_url']}")
        print(f"MTPLX API from container: {payload['mtplx_api_base_url_for_container']}")
        if payload["single_user_warning"]:
            print(f"warning: {payload['single_user_warning']}")
    return 0


def _write_fixture_runtime_contract(path: Path, *, arch_id: str, profile: str = DEFAULT_PROFILE_NAME) -> None:
    write_json(
        path / "mtplx_runtime.json",
        {
            "mtplx_version": "0.2.0",
            "arch_id": arch_id,
            "mtp_depth_max": 3,
            "recommended_profile": profile,
            "exactness_baseline": {"phase0h": "synthetic-smoke", "max_abs_diff": 0.0},
            "verified_on": {
                "timestamp": "2026-05-02T00:00:00Z",
                "hardware": "synthetic-fixture",
                "macos": "synthetic-fixture",
            },
        },
    )


def _write_fixture_weights(path: Path, keys: list[str]) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    save_file({key: np.ones((1,), dtype=np.float32) for key in keys}, path)


def _architecture_qa_fixtures() -> list[dict[str, Any]]:
    from optiq.runtime.mtp.constants import EXPECTED_MTP_KEYS

    return [
        {
            "label": "qwen3-next-verified-sidecar",
            "config": {
                "architectures": ["Qwen3NextForCausalLM"],
                "model_type": "qwen3_next",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "vocab_size": 32,
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
            },
            "contract": "qwen3-next-mtp",
            "safetensors": {"mtp.safetensors": list(EXPECTED_MTP_KEYS)},
            "expect": {
                "tier": "verified",
                "arch_id": "qwen3-next-mtp",
                "can_run": True,
                "recommended_backend": "qwen3_next",
                "runtime_compatibility": "native",
            },
        },
        {
            "label": "deepseek-v3-contract-gated",
            "config": {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 61,
            },
            "contract": "deepseek-v3-mtp",
            "safetensors": {
                "model.safetensors": [
                    "model.layers.61.enorm.weight",
                    "model.layers.62.enorm.weight",
                ]
            },
            "expect": {
                "tier": "verified",
                "arch_id": "deepseek-v3-mtp",
                "can_run": True,
                "recommended_backend": "deepseek_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "deepseek-v32-contract-gated",
            "config": {
                "architectures": ["DeepseekV32ForCausalLM"],
                "model_type": "deepseek_v32",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 61,
            },
            "contract": "deepseek-v3-mtp",
            "safetensors": {
                "model.safetensors": [
                    "model.layers.61.enorm.weight",
                    "model.layers.62.enorm.weight",
                ]
            },
            "expect": {
                "tier": "verified",
                "arch_id": "deepseek-v3-mtp",
                "can_run": True,
                "recommended_backend": "deepseek_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "glm-moe-dsa-contract-gated",
            "config": {
                "architectures": ["GlmMoeDsaForCausalLM"],
                "model_type": "glm_moe_dsa",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 61,
            },
            "contract": "glm-moe-dsa-mtp",
            "safetensors": {
                "model.safetensors": [
                    "model.layers.61.enorm.weight",
                    "model.layers.62.enorm.weight",
                ]
            },
            "expect": {
                "tier": "verified",
                "arch_id": "glm-moe-dsa-mtp",
                "can_run": True,
                "recommended_backend": "deepseek_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "glm4-moe-contract-gated",
            "config": {
                "architectures": ["Glm4MoeForCausalLM"],
                "model_type": "glm4_moe",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 47,
            },
            "contract": "glm4-moe-mtp",
            "safetensors": {"model.safetensors": ["model.layers.47.enorm.weight"]},
            "expect": {
                "tier": "verified",
                "arch_id": "glm4-moe-mtp",
                "can_run": True,
                "recommended_backend": "glm_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "glm4-moe-family-gated",
            "config": {
                "architectures": ["Glm4MoeForCausalLM"],
                "model_type": "glm4_moe",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 47,
            },
            "safetensors": {"model.safetensors": ["model.layers.47.hnorm.weight"]},
            "expect": {
                "tier": "family-compatible-unverified",
                "arch_id": "glm4-moe-mtp",
                "can_run": True,
                "recommended_backend": "glm_mtp",
                "runtime_compatibility": "native-family-gated",
            },
        },
        {
            "label": "glm4-moe-lite-contract-gated",
            "config": {
                "architectures": ["Glm4MoeLiteForCausalLM"],
                "model_type": "glm4_moe_lite",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 47,
            },
            "contract": "glm4-moe-lite-mtp",
            "safetensors": {"model.safetensors": ["model.layers.47.enorm.weight"]},
            "expect": {
                "tier": "verified",
                "arch_id": "glm4-moe-lite-mtp",
                "can_run": True,
                "recommended_backend": "glm_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "glm4-moe-lite-family-gated",
            "config": {
                "architectures": ["Glm4MoeLiteForCausalLM"],
                "model_type": "glm4_moe_lite",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 47,
            },
            "safetensors": {"model.safetensors": ["model.layers.47.eh_proj.weight"]},
            "expect": {
                "tier": "family-compatible-unverified",
                "arch_id": "glm4-moe-lite-mtp",
                "can_run": True,
                "recommended_backend": "glm_mtp",
                "runtime_compatibility": "native-family-gated",
            },
        },
        {
            "label": "mimo-contract-gated",
            "config": {
                "architectures": ["MiMoForCausalLM"],
                "model_type": "mimo",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 46,
            },
            "contract": "mimo-mtp",
            "safetensors": {
                "model.safetensors": ["model.mtp_layers.0.token_layernorm.weight"]
            },
            "expect": {
                "tier": "verified",
                "arch_id": "mimo-mtp",
                "can_run": True,
                "recommended_backend": "mimo_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "mimo-family-gated",
            "config": {
                "architectures": ["MiMoForCausalLM"],
                "model_type": "mimo",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 46,
            },
            "safetensors": {
                "model.safetensors": ["model.mtp_layers.0.hidden_layernorm.weight"]
            },
            "expect": {
                "tier": "family-compatible-unverified",
                "arch_id": "mimo-mtp",
                "can_run": True,
                "recommended_backend": "mimo_mtp",
                "runtime_compatibility": "native-family-gated",
            },
        },
        {
            "label": "minimax-m2-num-mtp-modules-recognized-pending",
            "config": {
                "architectures": ["MiniMaxM2ForCausalLM"],
                "model_type": "minimax_m2",
                "num_mtp_modules": 2,
            },
            "expect": {
                "tier": "architecture-compatible-but-unverified",
                "arch_id": "minimax-m2-mtp",
                "can_run": False,
                "runtime_compatibility": "recognized-backend-pending",
            },
        },
        {
            "label": "nemotron-h-contract-gated",
            "config": {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 52,
                "mtp_hybrid_override_pattern": "*E",
            },
            "contract": "nemotron-h-mtp",
            "safetensors": {
                "mtp.safetensors": [
                    "mtp.layers.0.enorm.weight",
                    "mtp.layers.0.hnorm.weight",
                    "mtp.layers.0.eh_proj.weight",
                    "mtp.layers.0.norm.weight",
                    "mtp.layers.0.mixer.q_proj.weight",
                    "mtp.layers.1.norm.weight",
                    "mtp.layers.1.mixer.gate.weight",
                    "mtp.layers.1.final_layernorm.weight",
                ]
            },
            "expect": {
                "tier": "verified",
                "arch_id": "nemotron-h-mtp",
                "can_run": True,
                "recommended_backend": "nemotron_h_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "nemotron-h-family-gated",
            "config": {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 52,
                "mtp_hybrid_override_pattern": "*E",
            },
            "safetensors": {
                "mtp.safetensors": [
                    "mtp.layers.0.enorm.weight",
                    "mtp.layers.0.hnorm.weight",
                    "mtp.layers.0.eh_proj.weight",
                    "mtp.layers.0.norm.weight",
                    "mtp.layers.0.mixer.q_proj.weight",
                    "mtp.layers.1.norm.weight",
                    "mtp.layers.1.mixer.gate.weight",
                    "mtp.layers.1.final_layernorm.weight",
                ]
            },
            "expect": {
                "tier": "family-compatible-unverified",
                "arch_id": "nemotron-h-mtp",
                "can_run": True,
                "recommended_backend": "nemotron_h_mtp",
                "runtime_compatibility": "native-family-gated",
            },
        },
        {
            "label": "gemma4-without-mtp-stays-no-mtp",
            "config": {
                "architectures": ["Gemma4ForCausalLM"],
                "model_type": "gemma4",
            },
            "expect": {
                "tier": "no-MTP",
                "arch_id": None,
                "can_run": False,
                "runtime_compatibility": "unsupported",
            },
        },
        {
            "label": "gemma4-mtp-marker-recognized-pending",
            "config": {
                "architectures": ["Gemma4ForCausalLM"],
                "model_type": "gemma4",
                "num_nextn_predict_layers": 1,
            },
            "expect": {
                "tier": "architecture-compatible-but-unverified",
                "arch_id": "gemma-mtp",
                "can_run": False,
                "runtime_compatibility": "recognized-backend-pending",
            },
        },
    ]


def _compact_qa_observed(inspection: dict[str, Any]) -> dict[str, Any]:
    compatibility = inspection.get("compatibility") or {}
    return {
        "architecture": inspection.get("architecture"),
        "model_type": inspection.get("model_type"),
        "mtp_num_hidden_layers": inspection.get("mtp_num_hidden_layers"),
        "tier": compatibility.get("tier"),
        "arch_id": compatibility.get("arch_id"),
        "can_run": compatibility.get("can_run"),
        "recommended_backend": compatibility.get("recommended_backend"),
        "runtime_compatibility": compatibility.get("runtime_compatibility"),
        "support_level": compatibility.get("support_level"),
        "message": compatibility.get("message"),
    }


def _qa_expectation_passed(observed: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    for key, expected_value in expected.items():
        if observed.get(key) != expected_value:
            failures.append(f"{key}: expected {expected_value!r}, got {observed.get(key)!r}")
    return not failures, failures


def _run_architecture_fixture_qa() -> list[dict[str, Any]]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="mtplx-arch-qa-") as tmp:
        root = Path(tmp)
        for spec in _architecture_qa_fixtures():
            model_dir = root / spec["label"]
            model_dir.mkdir(parents=True, exist_ok=True)
            write_json(model_dir / "config.json", spec["config"])
            if spec.get("contract"):
                _write_fixture_runtime_contract(model_dir, arch_id=str(spec["contract"]))
            for filename, keys in (spec.get("safetensors") or {}).items():
                _write_fixture_weights(model_dir / filename, list(keys))
            try:
                inspection = inspect_model(str(model_dir)).to_dict()
                observed = _compact_qa_observed(inspection)
                passed, failures = _qa_expectation_passed(observed, spec["expect"])
            except Exception as exc:
                observed = {"error": str(exc)}
                passed = False
                failures = [str(exc)]
            rows.append(
                {
                    "label": spec["label"],
                    "expected": spec["expect"],
                    "observed": observed,
                    "passed": passed,
                    "failures": failures,
                }
            )
    return rows


def _run_runtime_import_smoke() -> list[dict[str, Any]]:
    smokes = []
    for module_name, class_name in (
        ("mtplx.backends.deepseek_mtp", "DeepSeekMTPBackend"),
        ("mtplx.backends.glm_mtp", "GLMMTPBackend"),
        ("mtplx.backends.mimo_mtp", "MiMoMTPBackend"),
        ("mtplx.backends.nemotron_h_mtp", "NemotronHMTPBackend"),
    ):
        try:
            module = importlib.import_module(module_name)
            backend = getattr(module, class_name)()
            health = backend.health()
            smokes.append(
                {
                    "module": module_name,
                    "class": class_name,
                    "passed": bool(health.get("contract_required")),
                    "health": health,
                }
            )
        except Exception as exc:
            smokes.append(
                {
                    "module": module_name,
                    "class": class_name,
                    "passed": False,
                    "error": str(exc),
                }
            )
    return smokes


def _cmd_model_qa_architectures(args: Any) -> int:
    catalog = architecture_catalog()
    ids = {row["arch_id"] for row in catalog}
    required_catalog = {
        "qwen3-next-mtp",
        "deepseek-v3-mtp",
        "glm-moe-dsa-mtp",
        "glm4-moe-mtp",
        "glm4-moe-lite-mtp",
        "mimo-mtp",
        "nemotron-h-mtp",
        "minimax-m2-mtp",
        "gemma-mtp",
    }
    required_verified = {
        "qwen3-next-mtp",
        "deepseek-v3-mtp",
        "glm-moe-dsa-mtp",
        "glm4-moe-mtp",
        "glm4-moe-lite-mtp",
        "mimo-mtp",
        "nemotron-h-mtp",
    }
    verified_ids = {row["arch_id"] for row in catalog if row.get("can_run_verified")}
    fixture_rows = _run_architecture_fixture_qa()
    runtime_smokes = (
        _run_runtime_import_smoke()
        if getattr(args, "runtime_import_smoke", False)
        else []
    )
    gates = {
        "catalog_has_main_families": required_catalog.issubset(ids),
        "verified_contract_gated_families_listed": required_verified.issubset(verified_ids),
        "fixture_inspections_passed": all(row["passed"] for row in fixture_rows),
        "runtime_import_smoke_passed": (
            all(row["passed"] for row in runtime_smokes)
            if runtime_smokes
            else True
        ),
    }
    payload = {
        "action": "model qa-architectures",
        "fixture_count": len(fixture_rows),
        "verified_runtime_arch_ids": sorted(verified_ids),
        "recognized_backend_pending_arch_ids": sorted(
            row["arch_id"]
            for row in catalog
            if row.get("runtime_compatibility") == "recognized-backend-pending"
        ),
        "gates": gates,
        "fixtures": fixture_rows,
        "runtime_import_smokes": runtime_smokes,
        "passed": all(gates.values()),
    }
    output = Path(args.output) if getattr(args, "output", None) else None
    if output is not None:
        write_json(output, payload)
    if getattr(args, "json", False):
        _print(payload)
    else:
        print("MTPLX architecture QA")
        for row in fixture_rows:
            status = "pass" if row["passed"] else "fail"
            print(f"- {row['label']}: {status}")
        print(f"passed: {str(payload['passed']).lower()}")
        if output is not None:
            print(f"output: {output}")
    return 0 if payload["passed"] else EXIT_STRICT_GATE


def cmd_model_public(args: Any) -> int:
    if args.model_action == "architectures":
        rows = architecture_catalog()
        payload = {
            "action": "model architectures",
            "verified_runtime_arch_ids": [
                row["arch_id"] for row in rows if row.get("can_run_verified")
            ],
            "recognized_backend_pending_arch_ids": [
                row["arch_id"]
                for row in rows
                if row.get("runtime_compatibility") == "recognized-backend-pending"
            ],
            "architectures": rows,
        }
        if getattr(args, "json", False):
            _print(payload)
        else:
            print("MTPLX architecture support")
            for row in rows:
                status = "runnable" if row.get("can_run_verified") else "backend-pending"
                print(
                    f"- {row['arch_id']}: {row['display_name']} "
                    f"({status}, backend={row.get('backend') or 'none'})"
                )
        return 0
    if args.model_action == "qa-architectures":
        return _cmd_model_qa_architectures(args)
    if args.model_action != "publish-check":
        raise SystemExit(f"unknown model action: {args.model_action}")
    staging = Path(args.staging_dir)
    manifest_path = staging / "MTPLX_PUBLISH_MANIFEST.json"
    runtime_contract_path = staging / "mtplx_runtime.json"
    symlinks = [str(path) for path in staging.iterdir() if path.is_symlink()] if staging.exists() else []
    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inspection: dict[str, Any] | None = None
    inspect_error = None
    if staging.exists():
        try:
            inspection = _compact_model_summary(inspect_model(str(staging)).to_dict())
        except Exception as exc:
            inspect_error = str(exc)
    gates = {
        "staging_exists": staging.exists(),
        "manifest_exists": manifest_path.exists(),
        "runtime_contract_exists": runtime_contract_path.exists(),
        "no_symlinks": not symlinks,
        "repo_id_explicit": bool(args.repo_id or (manifest or {}).get("repo_id")),
        "inspect_verified": bool(
            inspection
            and (inspection.get("compatibility") or {}).get("tier") == "verified"
            and (inspection.get("compatibility") or {}).get("can_run")
        ),
    }
    payload = {
        "action": "model publish-check",
        "staging_dir": str(staging),
        "repo_id": args.repo_id or (manifest or {}).get("repo_id"),
        "manifest": str(manifest_path),
        "runtime_contract": str(runtime_contract_path),
        "symlinks": symlinks,
        "inspection": inspection,
        "inspect_error": inspect_error,
        "size_bytes": (manifest or {}).get("size_bytes"),
        "weight_size_bytes": (manifest or {}).get("weight_size_bytes"),
        "uploaded": (manifest or {}).get("upload_policy", {}).get("uploaded"),
        "gates": gates,
        "passed": all(gates.values()),
    }
    _print(payload)
    return 0 if payload["passed"] else EXIT_STRICT_GATE


def cmd_config_public(args: Any) -> int:
    from optiq.runtime.mtp.config import load_user_config, user_config_path
    from optiq.runtime.mtp.profiles import resolve_profile_name

    path = user_config_path(getattr(args, "config", None))
    if args.config_action == "show":
        payload = load_user_config(path).to_dict()
        if getattr(args, "json", False):
            _print(payload)
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0
    if args.config_action != "set":
        raise SystemExit(f"unknown config action: {args.config_action}")
    current = load_user_config(path)
    values = {
        "model": current.model,
        "model_dir": current.model_dir,
        "profile": current.profile,
        "thermal_control": current.thermal_control,
    }
    key = str(args.key).strip()
    if key not in values:
        raise SystemExit("config set key must be one of: model, model_dir, profile, thermal_control")
    value = str(args.value).strip()
    if key == "profile":
        value = resolve_profile_name(value)
    if key == "thermal_control" and value not in {"auto", "none"}:
        raise SystemExit("thermal_control must be auto or none")
    values[key] = value
    if not getattr(args, "dry_run", False):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# MTPLX user configuration"]
        for item_key in ("model", "model_dir", "profile", "thermal_control"):
            item_value = values.get(item_key)
            if item_value:
                lines.append(f"{item_key} = {json.dumps(item_value)}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "path": str(path),
        "dry_run": bool(getattr(args, "dry_run", False)),
        "updated": {key: value},
        "config": values,
    }
    _print(payload)
    return 0


def _source_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _hotpath_boundary_report() -> dict[str, Any]:
    root = repo_root()
    paths = {
        "generation": root / "mtplx" / "generation.py",
        "gdn_capture": root / "mtplx" / "gdn_capture.py",
        "paged_cache": root / "mtplx" / "cache_state.py",
        "paged_sdpa": root / "mtplx" / "kernels" / "sdpa_2pass_paged.py",
        "verify_qmv": root / "mtplx" / "verify_qmv.py",
        "native_mlp_cpp": root / "native_extensions" / "verify_mlp" / "gate_up" / "gate_up.cpp",
        "native_gdn_cpp": root / "native_extensions" / "verify_mlp" / "gdn_tail" / "gdn_tail.cpp",
        "logits_topk": root / "mtplx" / "kernels" / "logits_topk.py",
    }
    boundaries = [
        {
            "name": "verify_output_eval",
            "file": str(paths["generation"].relative_to(root)),
            "status": "intentional-single-sync",
            "default_hot_path": True,
            "evidence": "_eval_verify_outputs materializes verifier logits/hidden once per verify cycle for exact sampling.",
            "next_action": "Only remove by moving sampler/residual correction onto device or owning a larger verify-cycle boundary.",
        },
        {
            "name": "gdn_capture_kernels",
            "file": str(paths["gdn_capture"].relative_to(root)),
            "status": "mlx-fast-metal-kernel",
            "default_hot_path": True,
            "evidence": "GDN capture backends are registered through mx.fast.metal_kernel and remain lazy until verifier eval.",
            "next_action": "Do not wrap again; only revisit as a larger GDN layer/state primitive.",
        },
        {
            "name": "paged_attention_mlx_vector",
            "file": str(paths["paged_sdpa"].relative_to(root)),
            "status": "mlx-fast-metal-kernel",
            "default_hot_path": True,
            "evidence": "The product long-response profile uses the local mlx_vector_paged SDPA kernels, not the raw external fallback.",
            "next_action": "Closed as the main decay target unless attention attribution changes.",
        },
        {
            "name": "qmv_custom_kernels",
            "file": str(paths["verify_qmv"].relative_to(root)),
            "status": "mlx-fast-metal-kernel",
            "default_hot_path": False,
            "evidence": "QMV probes are mx.fast.metal_kernel helpers; prior live screens show wrapper-level qmv changes regress or tie.",
            "next_action": "Keep default-off unless a larger primitive amortizes the boundary.",
        },
        {
            "name": "fused_logits_topk_distribution",
            "file": str(paths["logits_topk"].relative_to(root)),
            "status": "mlx-fast-metal-kernel-default-off-closed",
            "default_hot_path": False,
            "evidence": "Dense-logit tile top-k plus logsumexp preserved sparse target distributions but was slower than the stock batched MLX sampler on the 4x151936 verifier shape.",
            "next_action": "Do not promote; only revisit if the target-distribution boundary is fused with more of accept/reject or LM-head work.",
        },
        {
            "name": "native_rowwise_mlp",
            "file": str(paths["native_mlp_cpp"].relative_to(root)),
            "status": "cpp-mlx-primitive-default-off",
            "default_hot_path": False,
            "evidence": "Native MLP returns mx::array backed by GateUpSwiGLU* primitives, but the current rowwise boundary is closed live.",
            "next_action": "Next MLP work must be a larger verify-layer/MLX-source primitive, not the same rowwise wrapper.",
        },
        {
            "name": "native_residual_mlp",
            "file": str(paths["native_mlp_cpp"].relative_to(root)),
            "status": "cpp-mlx-primitive-default-off-closed",
            "default_hot_path": False,
            "evidence": "Residual+MLP layer-boundary primitive was exact and isolated-fast, but integrated D3/192 live generation regressed.",
            "next_action": "Do not tune this two-dispatch scratch design further; move to lower-level source work or a larger verify-cycle boundary.",
        },
        {
            "name": "native_gdn_tail",
            "file": str(paths["native_gdn_cpp"].relative_to(root)),
            "status": "cpp-mlx-primitive-default-off",
            "default_hot_path": False,
            "evidence": "Native GDN tail is a C++ MLX primitive with internal scratch, but isolated/live results were slower.",
            "next_action": "Only revisit as packed GDN family/state ownership, not tail-only.",
        },
        {
            "name": "external_vllm_partitioned_fallback",
            "file": str(paths["paged_cache"].relative_to(root)),
            "status": "raw-sync-hazard-default-off-or-fallback",
            "default_hot_path": False,
            "evidence": "Fallback path contains explicit mx.eval before raw op and mx.synchronize after it.",
            "next_action": "Never use as product path; require local primitive path or wrap the raw op before benchmarking.",
        },
        {
            "name": "state_root_eval_sync",
            "file": str(paths["generation"].relative_to(root)),
            "status": "diagnostic-boundary",
            "default_hot_path": True,
            "evidence": "Stable staged profiles use state-root eval; depends/async variants were exact but closed as product fixes.",
            "next_action": "Keep for exact state ownership until a larger owned verify-cycle boundary replaces it.",
        },
    ]
    raw_sync_markers = {
        "explicit_mx_synchronize_in_cache_state": _source_contains(paths["paged_cache"], "mx.synchronize()"),
        "external_partitioned_raw_call_present": _source_contains(paths["paged_cache"], "paged_attention_v2_online_partitioned"),
        "native_mlp_is_mlx_primitive": _source_contains(paths["native_mlp_cpp"], "std::make_shared<GateUpSwiGLU"),
        "native_gdn_is_mlx_primitive": _source_contains(paths["native_gdn_cpp"], "std::make_shared<GdnNormGateOutQMV8"),
    }
    return {
        "action": "debug hotpath",
        "boundaries": boundaries,
        "raw_sync_markers": raw_sync_markers,
        "verdict": {
            "do_not_loop": [
                "dynamic SDPA partition/block gridding",
                "existing native rowwise MLP",
                "native residual MLP layer-boundary fusion",
                "native GDN tail-only",
                "wrapper-level qmv/RMSNorm microkernels",
                "standalone dense-logit top-k distribution kernels",
                "state-root include/exclude or depends toggles",
            ],
            "highest_upside_next": [
                "larger owned verify-layer or verify-cycle primitive",
                "MLX-source/native primitive that fuses enough MLP/GDN work to amortize dispatch",
                "device-side sampler/residual boundary only if exactness can be preserved",
            ],
            "cold_speed_policy": "default-off diagnostics only; promote nothing without full exactness and same-night cold gate.",
        },
    }


def cmd_debug_public(args: Any) -> int:
    if args.debug_action == "hotpath":
        payload = _hotpath_boundary_report()
        output = Path(args.output) if getattr(args, "output", None) else None
        if output is not None:
            write_json(output, payload)
        _print(payload)
        return 0
    if args.debug_action != "bundle":
        raise SystemExit(f"unknown debug action: {args.debug_action}")
    bundle_id = args.run_id or f"debug-bundle-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.output_dir or "outputs/cli/debug-bundles") / bundle_id
    out_dir.mkdir(parents=True, exist_ok=True)
    doctor_args = type("DoctorArgs", (), vars(args).copy())()
    doctor_args.project_root = getattr(args, "project_root", ".")
    doctor_args.model_cache = getattr(args, "model_cache", None)
    doctor_args.deep = True
    env = collect_environment(doctor_args.project_root).to_dict()
    from optiq.runtime.mtp.hf_loader import hf_cache_report
    from optiq.runtime.mtp.thermal import detect_thermal_control

    doctor = {
        "environment": env,
        "huggingface": hf_cache_report(cache_dir=doctor_args.model_cache),
        "thermal_control": detect_thermal_control(),
        "tools": {
            "python": sys.executable,
            "powermetrics": shutil.which("powermetrics"),
            "sudo": shutil.which("sudo"),
        },
    }
    doctor = _deep_doctor_report(doctor_args, doctor)
    _write_json_redacted(out_dir / "doctor.json", doctor)
    _write_json_redacted(out_dir / "metrics.json", _http_json(args.url.rstrip("/") + "/metrics") if args.url else {})
    _write_json_redacted(
        out_dir / "summary.json",
        {
            "bundle_id": bundle_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "redacted": True,
            "files": ["doctor.json", "metrics.json", "summary.json"],
        },
    )
    archive = out_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(out_dir.iterdir()):
            tar.add(path, arcname=f"{bundle_id}/{path.name}")
    payload = {
        "action": "debug bundle",
        "bundle_dir": str(out_dir),
        "archive": str(archive),
        "redacted": True,
    }
    _print(payload)
    return 0
