"""Fine-tune wizard routes."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from flask import (
    Blueprint, current_app, jsonify, render_template, request,
)

from .. import auth, hf, job_bus, jobs, local_quants, train_job


bp = Blueprint("finetune", __name__)


@bp.route("/finetune")
def finetune_page():
    paths = current_app.config["OPTIQ_LAB_PATHS"]
    models = _list_local_models(paths.models_dir)
    return render_template(
        "finetune.html",
        page_title="Fine-tune",
        section="finetune",
        local_models=models,
    )


@bp.route("/api/finetune/inspect-dataset", methods=["POST"])
def inspect_dataset():
    """Validate a dataset directory: must contain train.jsonl (and ideally
    valid.jsonl). Returns row counts + sample of the first row."""
    data = request.get_json(force=True) or {}
    path = Path(data.get("path") or "").expanduser()
    if not path.is_dir():
        return jsonify({"ok": False, "error": f"not a directory: {path}"}), 400

    train_path = path / "train.jsonl"
    valid_path = path / "valid.jsonl"
    if not train_path.is_file():
        return jsonify({"ok": False, "error": "train.jsonl missing"}), 400

    n_train = _count_lines(train_path)
    n_valid = _count_lines(valid_path) if valid_path.is_file() else 0
    sample = None
    try:
        with train_path.open() as f:
            sample = json.loads(f.readline())
    except Exception:
        pass

    # Detect format
    fmt = "unknown"
    if isinstance(sample, dict):
        if "messages" in sample:
            fmt = "messages (chat)"
        elif "prompt" in sample and ("completion" in sample or "response" in sample):
            fmt = "prompt+completion"
        elif "text" in sample:
            fmt = "text"

    return jsonify({
        "ok": True,
        "n_train": n_train,
        "n_valid": n_valid,
        "format": fmt,
        "sample_keys": list(sample.keys()) if isinstance(sample, dict) else None,
    })


@bp.route("/api/finetune/submit", methods=["POST"])
def submit():
    data = request.get_json(force=True) or {}
    model_dir = (data.get("model_dir") or "").strip()
    data_dir = (data.get("data_dir") or "").strip()
    if not (model_dir and data_dir):
        return jsonify({"ok": False, "error": "model_dir + data_dir required"}), 400
    if not Path(model_dir).is_dir():
        return jsonify({"ok": False, "error": f"model_dir not found: {model_dir}"}), 400
    if not (Path(data_dir) / "train.jsonl").is_file():
        return jsonify({"ok": False, "error": "data_dir missing train.jsonl"}), 400

    paths = current_app.config["OPTIQ_LAB_PATHS"]
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(model_dir).name)
    adapter_path = paths.models_dir / f"{name}-lora-{int(time.time())}"

    method = (data.get("method") or "sft").lower()
    if method not in ("sft", "dpo", "vision"):
        return jsonify({"ok": False,
                        "error": f"unknown method: {method}"}), 400
    # Vision (image+text) data check: each row must carry an image.
    if method == "vision":
        try:
            with open(Path(data_dir) / "train.jsonl") as f:
                first = json.loads(f.readline())
            if not (first.get("image") or first.get("images")):
                return jsonify({"ok": False,
                                "error": "vision data missing 'image' key — "
                                "build it with the 'VLM image+text' dataset "
                                "template"}), 400
        except Exception as e:
            return jsonify({"ok": False,
                            "error": f"failed to read vision train.jsonl: {e}"}), 400
    # DPO data check: each row must have prompt/chosen/rejected.
    if method == "dpo":
        try:
            with open(Path(data_dir) / "train.jsonl") as f:
                first = json.loads(f.readline())
            missing = [k for k in ("prompt", "chosen", "rejected")
                       if k not in first]
            if missing:
                return jsonify({"ok": False,
                                "error": f"DPO data missing keys: {missing}"}), 400
        except Exception as e:
            return jsonify({"ok": False,
                            "error": f"failed to read DPO train.jsonl: {e}"}), 400

    # Resolve iters + LR through OptiqLoraConfig so Lab, CLI, and direct
    # construction all agree. Omitting iters/num_epochs/learning_rate (or
    # sending null) falls back to the config's method-aware defaults:
    # SFT = 3 epochs @ 2e-4, DPO = 1 epoch @ 5e-5. An explicit value wins.
    from optiq.lora.config import OptiqLoraConfig as _Cfg

    def _optnum(key, cast):
        v = data.get(key)
        return None if v in (None, "", "null") else cast(v)

    _iters_in = _optnum("iters", int)
    _epochs_in = _optnum("num_epochs", float)
    _lr_in = _optnum("learning_rate", float)
    _bs = int(data.get("batch_size", 1))
    try:
        with open(Path(data_dir) / "train.jsonl") as f:
            n_examples = sum(1 for ln in f if ln.strip())
    except Exception:
        n_examples = 0
    _probe = _Cfg(method=("dpo" if method == "dpo" else "sft"),
                  iters=_iters_in, num_epochs=_epochs_in,
                  learning_rate=_lr_in, batch_size=_bs)
    # Concrete iters so the job's progress bar has a denominator.
    iters = _probe.effective_iters(n_examples) if n_examples > 0 else (_iters_in or 500)
    learning_rate = _probe.effective_learning_rate()

    # DPO warmup / schedule pass-through. ``None`` keeps the resolver
    # default (10% of iters, floor 10; cosine decay).
    dpo_warmup_raw = data.get("dpo_warmup_iters")
    dpo_warmup_iters = (
        None if dpo_warmup_raw in (None, "", "null")
        else max(0, int(dpo_warmup_raw))
    )
    dpo_lr_schedule = data.get("dpo_lr_schedule") or "cosine"
    if dpo_lr_schedule not in ("constant", "cosine"):
        return jsonify({"ok": False,
                        "error": f"dpo_lr_schedule must be 'constant' "
                                 f"or 'cosine', got {dpo_lr_schedule!r}"}), 400

    job_config = {
        "model_dir": model_dir,
        "data_dir": data_dir,
        "adapter_path": str(adapter_path),
        "method": method,
        "dpo_beta": float(data.get("dpo_beta", 0.1)),
        "dpo_warmup_iters": dpo_warmup_iters,
        "dpo_lr_schedule": dpo_lr_schedule,
        "dpo_loss": (data.get("dpo_loss") or "sigmoid"),
        "dpo_label_smoothing": float(data.get("dpo_label_smoothing") or 0.0),
        "rank": int(data.get("rank", 8)),
        "scale": float(data.get("scale", 20.0)),
        "dropout": float(data.get("dropout", 0.0)),
        "rank_scaling": data.get("rank_scaling", "by_bits"),
        "target_modules": data.get("target_modules") or ["q_proj", "v_proj"],
        "num_layers": int(data.get("num_layers", 16)),
        "iters": iters,
        "batch_size": _bs,
        "learning_rate": learning_rate,
        "fused_dpo": bool(data.get("fused_dpo", False)),
        "max_seq_length": int(data.get("max_seq_length", 1024)),
        "grad_accumulation_steps": int(data.get("grad_accumulation_steps", 1)),
        # Vision LoRA: uniform letterbox canvas (bounded memory on Apple Silicon).
        "image_size": int(data.get("image_size", 512)),
    }
    # Vision uses scale 8 by default (the hybrid family collapses at 20).
    if method == "vision" and data.get("scale") in (None, "", "null"):
        job_config["scale"] = 8.0

    job_id = job_bus.submit(
        "finetune",
        train_job.run,
        config=job_config,
        resource_class="memory_heavy",
    )
    return jsonify({"ok": True, "job_id": job_id, "adapter_path": str(adapter_path)})


@bp.route("/api/finetune/push", methods=["POST"])
def push():
    """Push a trained LoRA adapter (or merged adapter, or exported
    model directory) to HF as a model repo.

    The wizard's Step 5 may produce up to three artifacts: the trained
    adapter (always), an optional merged adapter (rank-concat with a
    second adapter the user selects), and an optional exported model
    directory (base model + adapter, ready for `optiq serve`). The
    request includes a ``source_path`` hint when the user wants to
    push one of the derived artifacts; if missing, falls back to the
    trained adapter from the job config.
    """
    data = request.get_json(force=True) or {}
    job_id = (data.get("job_id") or "").strip()
    repo_id = (data.get("repo_id") or "").strip()
    private = bool(data.get("private", True))
    password = data.get("password") or ""
    source_path = (data.get("source_path") or "").strip() or None

    if not (job_id and repo_id and password):
        return jsonify({"ok": False, "error": "job_id, repo_id, password are all required"}), 400
    if not auth.verify_password(password):
        return jsonify({"ok": False, "error": "wrong Lab password"}), 400

    job = jobs.get(job_id)
    if job is None or job["status"] != "done":
        return jsonify({"ok": False, "error": "job not done or not found"}), 400

    if source_path:
        # The user passed an explicit path (merged adapter or exported
        # model dir from /api/finetune/merge / /api/finetune/export).
        folder = source_path
    else:
        try:
            cfg = json.loads(job["config_json"])
        except Exception:
            cfg = {}
        folder = cfg.get("adapter_path")
    if not folder or not Path(folder).is_dir():
        return jsonify({"ok": False, "error": f"path missing: {folder!r}"}), 400

    token_pair = hf.get_first_token_decrypted(password)
    if token_pair is None:
        return jsonify({"ok": False,
                        "error": "no HF token saved. Add one in Settings → Hugging Face."}), 400
    _, plain_token = token_pair

    try:
        url = hf.push_folder(
            folder=folder,
            repo_id=repo_id,
            token=plain_token,
            repo_type="model",
            private=private,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": f"push failed: {e}"}), 500
    return jsonify({"ok": True, "url": url, "pushed_path": str(folder)})


@bp.route("/api/finetune/list-adapters", methods=["GET"])
def list_local_adapters():
    """Return every locally-discoverable PEFT-style adapter directory.

    Walks the Lab's models dir + the standard ``./adapters/`` and
    ``~/.optiq/lab/adapters/`` locations, looking for any directory
    that contains an ``adapters.safetensors`` (or
    ``best/adapters.safetensors``). Used by the Combine UI in the
    fine-tune wizard's Step 5.
    """
    paths = current_app.config["OPTIQ_LAB_PATHS"]
    search_roots: list[Path] = []
    if hasattr(paths, "models_dir"):
        search_roots.append(Path(paths.models_dir))
    # Common project-local convention
    search_roots.append(Path.cwd() / "adapters")
    # Per-user lab data
    search_roots.append(Path.home() / ".optiq" / "lab" / "adapters")

    seen: set[str] = set()
    adapters: list[dict] = []
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        # Two-deep walk to catch ./adapters/<name>/ and
        # ./adapters/<name>/best/.
        for cand in list(root.iterdir()):
            if not cand.is_dir():
                continue
            sf = cand / "adapters.safetensors"
            best_sf = cand / "best" / "adapters.safetensors"
            if not (sf.exists() or best_sf.exists()):
                continue
            resolved = str(cand.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            # Try to pull rank distribution from the OptiQ sidecar.
            rank_dist = ""
            optiq_sidecar = cand / "optiq_lora_config.json"
            if optiq_sidecar.exists():
                try:
                    cfg = json.loads(optiq_sidecar.read_text())
                    applied = cfg.get("applied_ranks") or {}
                    from collections import Counter
                    counts = Counter(applied.values())
                    rank_dist = ", ".join(
                        f"rank {r}: {n}" for r, n in sorted(counts.items())
                    )
                except Exception:
                    pass
            adapters.append({
                "path": resolved,
                "name": cand.name,
                "rank_dist": rank_dist or "rank info unavailable",
            })
    adapters.sort(key=lambda a: a["name"])
    return jsonify({"adapters": adapters})


@bp.route("/api/finetune/merge", methods=["POST"])
def merge_adapters_route():
    """Rank-concat merge two local LoRA adapters into one.

    Wraps ``optiq.lora.merge.merge_adapters`` with a sensible default
    output path under ``<models_dir>/merged-<a>-<b>-<ts>/``.
    """
    data = request.get_json(force=True) or {}
    adapter_a = (data.get("adapter_a") or "").strip()
    adapter_b = (data.get("adapter_b") or "").strip()
    if not (adapter_a and adapter_b):
        return jsonify({"ok": False,
                        "error": "adapter_a and adapter_b are both required"}), 400
    if not Path(adapter_a).exists() or not Path(adapter_b).exists():
        return jsonify({"ok": False,
                        "error": "one or both adapter paths do not exist"}), 400

    paths = current_app.config["OPTIQ_LAB_PATHS"]
    out_root = Path(getattr(paths, "models_dir", Path.cwd() / "adapters"))
    name_a = Path(adapter_a).name
    name_b = Path(adapter_b).name
    out_dir = out_root / f"merged-{name_a}-{name_b}-{int(time.time())}"

    try:
        from optiq.lora.merge import merge_adapters
        stats = merge_adapters(
            adapter_paths=[adapter_a, adapter_b],
            output_dir=out_dir,
        )
    except Exception as exc:
        return jsonify({"ok": False,
                        "error": f"merge failed: {exc}"}), 500

    return jsonify({
        "ok": True,
        "merged_path": str(out_dir),
        "stats": {
            "layers_merged": stats["layers_merged"],
            "layers_only_in_one": stats["layers_only_in_one"],
        },
    })


@bp.route("/api/finetune/export", methods=["POST"])
def export_model_route():
    """Bundle a base model + one adapter into a deployable model dir.

    Wraps ``optiq.lora.merge.merge_adapters`` indirectly via the export
    path used by the CLI's ``optiq lora export``. Output goes under
    ``<models_dir>/export-<adapter_name>-<ts>/``.
    """
    import shutil

    data = request.get_json(force=True) or {}
    base_model = (data.get("base_model") or "").strip()
    adapter_path = (data.get("adapter_path") or "").strip()
    if not (base_model and adapter_path):
        return jsonify({"ok": False,
                        "error": "base_model and adapter_path are both required"}), 400
    if not Path(adapter_path).exists():
        return jsonify({"ok": False,
                        "error": f"adapter not found: {adapter_path}"}), 400

    paths = current_app.config["OPTIQ_LAB_PATHS"]
    out_root = Path(getattr(paths, "models_dir", Path.cwd() / "adapters"))
    out_dir = out_root / f"export-{Path(adapter_path).name}-{int(time.time())}"

    # Resolve base model: try local path first, then HF snapshot.
    base_dir = Path(base_model)
    if not base_dir.exists():
        try:
            from huggingface_hub import snapshot_download
            base_dir = Path(snapshot_download(repo_id=base_model))
        except Exception as exc:
            return jsonify({"ok": False,
                            "error": f"could not resolve base model "
                                     f"{base_model!r}: {exc}"}), 400

    out_dir.mkdir(parents=True, exist_ok=True)
    skip_names = {"adapter_config.json", "adapters.safetensors"}
    for src in sorted(base_dir.iterdir()):
        if src.name in skip_names or src.name.startswith("."):
            continue
        if src.is_file():
            shutil.copy2(src, out_dir / src.name)
        elif src.is_dir() and src.name != "source_adapters":
            shutil.copytree(src, out_dir / src.name)

    # Copy the adapter at the top level.
    adp = Path(adapter_path)
    sf = (adp / "best" / "adapters.safetensors"
          if (adp / "best" / "adapters.safetensors").exists()
          else adp / "adapters.safetensors")
    cfg = adp / "adapter_config.json"
    shutil.copy2(sf, out_dir / "adapters.safetensors")
    if cfg.exists():
        shutil.copy2(cfg, out_dir / "adapter_config.json")

    # Preserve the source adapter for reference.
    src_root = out_dir / "source_adapters" / adp.name
    if not src_root.exists():
        shutil.copytree(adp, src_root)

    (out_dir / "optiq_export.json").write_text(json.dumps({
        "base_model": base_model,
        "base_model_local": str(base_dir),
        "adapter": str(adp),
    }, indent=2) + "\n")

    return jsonify({"ok": True, "export_path": str(out_dir)})


# ---------------------------------------------------------------------------


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def _list_local_models(root: Path) -> list[dict]:
    """Every OptiQ quant available to fine-tune: locally-built AND pulled into
    the HuggingFace cache. Fine-tune loads the model directly (MLX-native), so
    anything mlx-lm can load is fair game — plus the user can type any HF repo
    id on the page. Mirrors the Server page's discovery so the two agree."""
    built = [{"name": q.display_name, "path": q.path, "source": "local"}
             for q in local_quants.discover(root)]
    seen = {q["path"] for q in built}
    cached = [{"name": q.display_name, "path": q.path, "source": "hf_cache"}
              for q in local_quants.discover_hf_cache()
              if q.path not in seen]
    return built + cached
