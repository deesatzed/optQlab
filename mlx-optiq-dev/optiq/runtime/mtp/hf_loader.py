"""Hugging Face model resolution and local cache helpers."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from optiq.runtime.mtp.artifacts import _hf_repo_id_from_ref
from optiq.runtime.mtp.profiles import DEFAULT_PROFILE_NAME


DEFAULT_MODEL_CACHE = Path("~/.mtplx/models").expanduser()
DownloadProgressCallback = Callable[[dict[str, Any]], None]
REQUIRED_MTPLX_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
    "mtp.safetensors",
    "mtplx_runtime.json",
)


def _query_repo_total_bytes(repo_id: str, *, revision: str | None = None) -> int | None:
    """Best-effort estimate of the remote repo's total size."""

    try:
        from huggingface_hub import HfApi
    except Exception:
        return None
    try:
        info = HfApi().model_info(
            repo_id=repo_id,
            revision=revision,
            files_metadata=True,
        )
    except Exception:
        return None
    siblings = getattr(info, "siblings", None) or []
    total = 0
    for sibling in siblings:
        size = getattr(sibling, "size", None)
        if isinstance(size, int) and size > 0:
            total += size
    return total or None


@contextlib.contextmanager
def _suppress_hf_hub_progress() -> Iterator[None]:
    """Suppress Hugging Face tqdm bars while MTPLX owns download progress."""

    previous_env = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    disabled_via_helper = False
    try:
        try:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
            disabled_via_helper = True
        except Exception:
            pass
        yield
    finally:
        if disabled_via_helper:
            try:
                from huggingface_hub.utils import enable_progress_bars

                enable_progress_bars()
            except Exception:
                pass
        if previous_env is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = previous_env


def model_cache_dir(value: str | Path | None = None) -> Path:
    if value:
        return Path(value).expanduser()
    env = os.environ.get("MTPLX_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    return DEFAULT_MODEL_CACHE


def safe_model_name(repo_id: str) -> str:
    return repo_id.strip("/").replace("/", "--")


def repo_id_from_model_ref(value: str) -> str | None:
    return _hf_repo_id_from_ref(value)


def cached_model_path(repo_id: str, *, cache_dir: str | Path | None = None) -> Path:
    return model_cache_dir(cache_dir) / safe_model_name(repo_id)


def _complete_indexed_weights(path: Path, index_name: str) -> bool:
    index = path / index_name
    if not index.is_file():
        return False
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    weight_map = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(weight_map, dict):
        return False
    filenames = {
        name
        for name in weight_map.values()
        if isinstance(name, str) and name.strip()
    }
    if not filenames:
        return False
    for name in filenames:
        shard = path / name
        try:
            if not shard.is_file() or shard.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def _complete_unindexed_weights(path: Path) -> bool:
    for pattern in ("*.safetensors", "*.bin", "*.gguf"):
        for candidate in path.glob(pattern):
            try:
                if candidate.is_file() and candidate.stat().st_size > 0:
                    return True
            except OSError:
                continue
    return False


def cached_model_is_complete(path: Path) -> bool:
    """Return whether a Hub cache directory is ready to run.

    ``snapshot_download(local_dir=...)`` creates the destination early. An
    interrupted pull can therefore leave config/tokenizer files plus an index,
    which looks cached even though the weight shards are missing.
    """

    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return (
        _complete_indexed_weights(path, "model.safetensors.index.json")
        or _complete_indexed_weights(path, "pytorch_model.bin.index.json")
        or _complete_unindexed_weights(path)
    )


def resolve_model_path(model_ref: str, *, cache_dir: str | Path | None = None) -> Path:
    local = Path(model_ref).expanduser()
    if local.exists():
        return local
    repo_id = repo_id_from_model_ref(model_ref)
    if repo_id is None:
        return local
    cached = cached_model_path(repo_id, cache_dir=cache_dir)
    if cached_model_is_complete(cached):
        return cached
    raise FileNotFoundError(
        f"Model {repo_id} is not cached. Run: mtplx pull {repo_id}"
    )


def validate_mtplx_model_files(path: Path) -> dict[str, Any]:
    from ...sidecar_layout import exists as _sidecar_exists

    def _present(name: str) -> bool:
        # The MTP head may live at the root (legacy) or under optiq/ (new layout).
        if name == "mtp.safetensors":
            return _sidecar_exists(path, name)
        return (path / name).exists()

    missing = [name for name in REQUIRED_MTPLX_MODEL_FILES if not _present(name)]
    contract: dict[str, Any] | None = None
    contract_error: str | None = None
    contract_path = path / "mtplx_runtime.json"
    if contract_path.exists():
        try:
            loaded = json.loads(contract_path.read_text(encoding="utf-8"))
            contract = loaded if isinstance(loaded, dict) else None
        except Exception as exc:
            contract_error = str(exc)
    return {
        "ok": not missing and contract_error is None,
        "required_files": list(REQUIRED_MTPLX_MODEL_FILES),
        "missing_files": missing,
        "contract_present": contract_path.exists(),
        "contract_arch_id": contract.get("arch_id") if isinstance(contract, dict) else None,
        "contract_error": contract_error,
    }


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for child in path.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _emit_download_progress(callback: DownloadProgressCallback | None, payload: dict[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        # Progress reporting must never break a model download.
        return


def _start_download_heartbeat(
    destination: Path,
    *,
    callback: DownloadProgressCallback | None,
    interval_s: float,
) -> tuple[threading.Event | None, threading.Thread | None]:
    if callback is None or interval_s <= 0:
        return None, None

    stop = threading.Event()
    started_at = time.monotonic()
    last_at = started_at
    last_size = directory_size_bytes(destination)

    def run() -> None:
        nonlocal last_at, last_size
        while not stop.wait(interval_s):
            now = time.monotonic()
            current_size = directory_size_bytes(destination)
            delta = current_size - last_size
            payload = {
                "event": "progress",
                "path": str(destination),
                "size_bytes": current_size,
                "delta_bytes": delta,
                "elapsed_s": now - started_at,
                "interval_s": now - last_at,
            }
            last_at = now
            last_size = current_size
            _emit_download_progress(callback, payload)

    thread = threading.Thread(target=run, name="mtplx-download-heartbeat", daemon=True)
    thread.start()
    return stop, thread


@dataclass(frozen=True)
class CachedModel:
    repo_id: str
    path: Path
    size_bytes: int
    has_runtime_contract: bool
    has_config: bool
    validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_bytes / 1_000_000_000, 3),
            "has_runtime_contract": self.has_runtime_contract,
            "has_config": self.has_config,
            "validation": self.validation,
            "recommended_profile": DEFAULT_PROFILE_NAME if self.validation.get("ok") else None,
            "delete_command": f"mtplx remove {self.repo_id}",
        }


def list_cached_models(*, cache_dir: str | Path | None = None) -> list[CachedModel]:
    root = model_cache_dir(cache_dir)
    if not root.exists():
        return []
    rows: list[CachedModel] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        repo_id = child.name.replace("--", "/")
        rows.append(
            CachedModel(
                repo_id=repo_id,
                path=child,
                size_bytes=directory_size_bytes(child),
                has_runtime_contract=(child / "mtplx_runtime.json").exists(),
                has_config=(child / "config.json").exists(),
                validation=validate_mtplx_model_files(child),
            )
        )
    return rows


def pull_model(
    model_ref: str,
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    progress_callback: DownloadProgressCallback | None = None,
    progress_interval_s: float = 10.0,
) -> dict[str, Any]:
    repo_id = repo_id_from_model_ref(model_ref)
    if repo_id is None:
        raise ValueError(f"pull requires a Hugging Face repo id or URL, got: {model_ref}")
    root = model_cache_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = cached_model_path(repo_id, cache_dir=root)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub is required for mtplx pull: {exc}") from exc

    started_size = directory_size_bytes(destination)
    if destination.exists() and cached_model_is_complete(destination):
        resolved = destination
        reused_existing = True
        resumed_existing = False
        validation = validate_mtplx_model_files(resolved)
        if repo_id.lower().startswith("youssofal/qwen3.6-27b-mtplx") and not validation["ok"]:
            raise RuntimeError(
                "cached MTPLX model is incomplete: "
                + ", ".join(validation["missing_files"] or [str(validation.get("contract_error"))])
            )
    else:
        reused_existing = False
        resumed_existing = destination.exists() and started_size > 0
        destination.mkdir(parents=True, exist_ok=True)
        total_bytes = (
            _query_repo_total_bytes(repo_id, revision=revision)
            if progress_callback is not None
            else None
        )
        _emit_download_progress(
            progress_callback,
            {
                "event": "resume" if resumed_existing else "start",
                "repo_id": repo_id,
                "path": str(destination),
                "size_bytes": started_size,
                "total_bytes": total_bytes,
            },
        )
        stop, thread = _start_download_heartbeat(
            destination,
            callback=progress_callback,
            interval_s=progress_interval_s,
        )
        progress_suppression = (
            _suppress_hf_hub_progress()
            if progress_callback is not None
            else contextlib.nullcontext()
        )
        try:
            with progress_suppression:
                path = snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    revision=revision,
                    local_dir=str(destination),
                )
            resolved = Path(path)
            validation = validate_mtplx_model_files(resolved)
            if not cached_model_is_complete(resolved):
                raise RuntimeError(
                    "downloaded model is incomplete: weight shards are missing or still partial"
                )
            if repo_id.lower().startswith("youssofal/qwen3.6-27b-mtplx") and not validation["ok"]:
                raise RuntimeError(
                    "downloaded MTPLX model is incomplete: "
                    + ", ".join(validation["missing_files"] or [str(validation.get("contract_error"))])
                )
        finally:
            if stop is not None:
                stop.set()
            if thread is not None:
                thread.join(timeout=1.0)
        final_size = directory_size_bytes(resolved)
        _emit_download_progress(
            progress_callback,
            {
                "event": "complete",
                "repo_id": repo_id,
                "path": str(resolved),
                "size_bytes": final_size,
                "total_bytes": total_bytes if total_bytes else final_size,
                "delta_bytes": final_size - started_size,
            },
        )
    return {
        "repo_id": repo_id,
        "path": str(resolved),
        "cache_dir": str(root),
        "revision": revision,
        "reused_existing": reused_existing,
        "resumed_existing": resumed_existing,
        "started_size_bytes": started_size,
        "size_bytes": directory_size_bytes(resolved),
        "has_runtime_contract": (resolved / "mtplx_runtime.json").exists(),
        "has_config": (resolved / "config.json").exists(),
        "validation": validate_mtplx_model_files(resolved),
    }


def remove_cached_model(model_ref: str, *, cache_dir: str | Path | None = None) -> dict[str, Any]:
    repo_id = repo_id_from_model_ref(model_ref) or model_ref.replace("--", "/")
    path = cached_model_path(repo_id, cache_dir=cache_dir)
    existed = path.exists()
    size = directory_size_bytes(path) if existed else 0
    if existed:
        shutil.rmtree(path)
    return {
        "repo_id": repo_id,
        "path": str(path),
        "removed": existed,
        "size_bytes_removed": size,
    }


def hf_cache_report(*, cache_dir: str | Path | None = None) -> dict[str, Any]:
    root = model_cache_dir(cache_dir)
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    token_source = "environment" if token_present else None
    if not token_present:
        try:
            from huggingface_hub import get_token

            token_present = bool(get_token())
            token_source = "huggingface_hub" if token_present else None
        except Exception:
            token_present = False
    try:
        usage = shutil.disk_usage(root if root.exists() else root.parent)
        free_bytes: int | None = usage.free
    except OSError:
        free_bytes = None
    return {
        "cache_dir": str(root),
        "cache_exists": root.exists(),
        "cache_writable": os.access(root if root.exists() else root.parent, os.W_OK),
        "disk_free_bytes": free_bytes,
        "disk_free_gb": round(free_bytes / 1_000_000_000, 3) if free_bytes is not None else None,
        "cached_models": len(list_cached_models(cache_dir=root)),
        "token_present": token_present,
        "token_source": token_source,
    }
