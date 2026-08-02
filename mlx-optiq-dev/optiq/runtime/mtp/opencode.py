"""OpenCode Desktop integration helpers.

The public CLI uses this module to make ``mtplx start opencode`` a real
connection flow: merge an MTPLX OpenAI-compatible provider into OpenCode's
JSON config, start the local server with raw reasoning enabled, then launch
OpenCode when possible.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

OPENCODE_PROVIDER_ID = "mtplx"
OPENCODE_NPM_PACKAGE = "@ai-sdk/openai-compatible"
OPENCODE_DEFAULT_CONTEXT_WINDOW = 262_144
OPENCODE_DEFAULT_CHUNK_TIMEOUT_MS = 900_000


def opencode_config_path(path: str | Path | None = None) -> Path:
    """Return OpenCode's JSON config path.

    ``MTPLX_OPENCODE_CONFIG`` exists for tests and power users. Normal users
    get OpenCode's shared config path under ``~/.config/opencode``.
    """

    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get("MTPLX_OPENCODE_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".config" / "opencode" / "opencode.json"


def opencode_model_ref(model_id: str, *, provider_id: str = OPENCODE_PROVIDER_ID) -> str:
    return f"{provider_id}/{model_id}"


def detect_opencode_desktop() -> dict[str, Any]:
    """Best-effort OpenCode Desktop detection for UX messages.

    Launching through macOS ``open -a`` is still attempted even when this
    returns missing; Spotlight/app registration can know about apps outside
    the common Applications paths.
    """

    if sys.platform != "darwin":
        return {"available": False, "kind": "unsupported_platform"}
    candidates = [
        Path("/Applications/OpenCode.app"),
        Path.home() / "Applications" / "OpenCode.app",
        Path("/Applications/OpenCode Desktop.app"),
        Path.home() / "Applications" / "OpenCode Desktop.app",
    ]
    for candidate in candidates:
        if candidate.exists():
            return {"available": True, "kind": "app", "path": str(candidate)}
    if shutil.which("opencode"):
        return {"available": True, "kind": "cli", "path": shutil.which("opencode")}
    return {"available": False, "kind": "not_found"}


def launch_opencode_app() -> dict[str, Any]:
    """Open OpenCode Desktop without blocking the MTPLX server."""

    if sys.platform != "darwin":
        return {
            "ok": False,
            "status": "unsupported_platform",
            "error": "automatic OpenCode launch currently requires macOS",
        }
    for app_name in ("OpenCode", "OpenCode Desktop"):
        try:
            subprocess.Popen(
                ["open", "-a", app_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return {"ok": True, "status": "launched", "app": app_name}
        except OSError as exc:
            last_error = str(exc)
    return {"ok": False, "status": "launch_failed", "error": last_error}


def build_opencode_provider_config(
    *,
    base_url: str,
    model_id: str,
    model_name: str | None = None,
    context_window: int = OPENCODE_DEFAULT_CONTEXT_WINDOW,
    output_limit: int | None = None,
    chunk_timeout_ms: int = OPENCODE_DEFAULT_CHUNK_TIMEOUT_MS,
    enable_thinking: bool = True,
    top_p: float = 0.95,
) -> dict[str, Any]:
    """Build the OpenCode provider/config fragment MTPLX owns.

    OpenCode's `limit` object is model metadata, not a server-side generation
    cap. We intentionally do not write hidden maxTokens/maxOutput caps.
    """

    context = int(context_window or OPENCODE_DEFAULT_CONTEXT_WINDOW)
    output = int(output_limit if output_limit is not None else context)
    return {
        "provider": {
            OPENCODE_PROVIDER_ID: {
                "npm": OPENCODE_NPM_PACKAGE,
                "name": "MTPLX (local)",
                "options": {
                    "baseURL": str(base_url).rstrip("/"),
                    "timeout": False,
                    "chunkTimeout": int(chunk_timeout_ms),
                },
                "models": {
                    str(model_id): {
                        "name": model_name or f"MTPLX {model_id}",
                        "reasoning": True,
                        "interleaved": {"field": "reasoning_content"},
                        "tool_call": True,
                        "temperature": True,
                        "limit": {
                            "context": context,
                            "output": output,
                        },
                        "modalities": {
                            "input": ["text"],
                            "output": ["text"],
                        },
                        "options": {
                            "topP": float(top_p),
                            "enable_thinking": bool(enable_thinking),
                        },
                    }
                },
            }
        },
        "model": opencode_model_ref(str(model_id)),
        "small_model": opencode_model_ref(str(model_id)),
    }


def _backup_invalid_config(path: Path) -> Path:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.invalid-{stamp}.bak")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.invalid-{stamp}-{counter}.bak")
        counter += 1
    path.replace(backup)
    return backup


def merge_opencode_config(
    existing: dict[str, Any] | None,
    *,
    config_fragment: dict[str, Any],
    provider_id: str = OPENCODE_PROVIDER_ID,
) -> dict[str, Any]:
    """Merge or create OpenCode config while preserving unrelated providers."""

    payload = dict(existing or {})
    providers = payload.get("provider")
    if not isinstance(providers, dict):
        providers = {}
    else:
        providers = dict(providers)
    fragment_providers = config_fragment.get("provider")
    if not isinstance(fragment_providers, dict) or provider_id not in fragment_providers:
        raise ValueError(f"config_fragment must include provider.{provider_id}")
    providers[str(provider_id)] = fragment_providers[provider_id]
    payload["provider"] = providers
    payload["model"] = config_fragment["model"]
    payload["small_model"] = config_fragment["small_model"]
    return payload


def write_opencode_config(
    *,
    base_url: str,
    model_id: str,
    model_name: str | None = None,
    path: str | Path | None = None,
    provider_id: str = OPENCODE_PROVIDER_ID,
    context_window: int = OPENCODE_DEFAULT_CONTEXT_WINDOW,
    output_limit: int | None = None,
    chunk_timeout_ms: int = OPENCODE_DEFAULT_CHUNK_TIMEOUT_MS,
    enable_thinking: bool = True,
    top_p: float = 0.95,
) -> dict[str, Any]:
    """Write MTPLX into OpenCode config and return a handoff payload."""

    config_path = opencode_config_path(path)
    backup_path: Path | None = None
    existing: dict[str, Any] | None = None
    if config_path.exists():
        try:
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
            existing = parsed if isinstance(parsed, dict) else {}
        except (OSError, json.JSONDecodeError):
            backup_path = _backup_invalid_config(config_path)
            existing = {}

    fragment = build_opencode_provider_config(
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        context_window=context_window,
        output_limit=output_limit,
        chunk_timeout_ms=chunk_timeout_ms,
        enable_thinking=enable_thinking,
        top_p=top_p,
    )
    merged = merge_opencode_config(
        existing,
        config_fragment=fragment,
        provider_id=provider_id,
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    try:
        config_path.chmod(0o600)
    except OSError:
        pass
    return {
        "config_path": str(config_path),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "provider_id": provider_id,
        "base_url": str(base_url).rstrip("/"),
        "model_id": model_id,
        "model_ref": opencode_model_ref(model_id, provider_id=provider_id),
        "context_window": int(context_window),
        "output_limit": int(output_limit if output_limit is not None else context_window),
        "chunk_timeout_ms": int(chunk_timeout_ms),
        "reasoning_field": "reasoning_content",
        "no_hidden_max_tokens": True,
        "written": True,
    }
