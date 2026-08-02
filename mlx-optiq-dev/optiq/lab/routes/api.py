"""JSON API: /api/status, /api/jobs, /api/integrations."""

from __future__ import annotations

import urllib.error
import urllib.request
from urllib.parse import urlparse

from flask import Blueprint, current_app, jsonify, request

from .. import integrations, jobs


bp = Blueprint("api", __name__, url_prefix="/api")


@bp.route("/status")
def status():
    """Live status for the sidebar — API reachability, current model,
    MTP state, port, integration token."""
    api_url = current_app.config["OPTIQ_API_URL"]
    port = urlparse(api_url).port

    # When a cluster is serving, IT is the backend — not the Lab's own supervised
    # API server, which has no model loaded. Report the cluster, or the sidebar
    # says "Model —" and Chat claims nothing is being served while the ring answers.
    try:
        from .cluster import _serve, _endpoint_live
        if _serve.get("activated") and _serve.get("model"):
            live = _endpoint_live(_serve["port"])
            return jsonify({
                "api_url": api_url,
                "api_port": port,
                "api_reachable": live,
                "api_status": "ready" if live else "starting",
                "api_last_error": None,
                "model": _serve["model"],
                "backend": "cluster",
                "cluster_nodes": _serve.get("nodes"),
                "mtp_enabled": False,
                "mtp_depth": 0,
                "drafter_id": None,
                "prompt_cache_bytes": None,
                "sampler": None,
                "sampler_defaults": {},
                "adapters": [],
                "auth_token": "sk-optiq-local",
            })
    except Exception:
        pass

    supervisor = current_app.config.get("OPTIQ_API_SUPERVISOR")
    if supervisor is not None:
        s = supervisor.state()
        # Resolve the model's recommended sampler so the UI can show it
        # as the default (and as a placeholder in the override inputs).
        recommended_sampler = {}
        if s.model:
            try:
                from optiq.runtime.gen_config import read_recommended_sampling
                rec = read_recommended_sampling(s.model)
                # Use mlx_lm.server's flag naming so the UI matches the CLI.
                recommended_sampler = {
                    "temp":  rec.get("temperature"),
                    "top_p": rec.get("top_p"),
                    "top_k": rec.get("top_k"),
                    "min_p": rec.get("min_p"),
                }
                # Strip Nones so the JSON is small and consistent.
                recommended_sampler = {
                    k: v for k, v in recommended_sampler.items() if v is not None
                }
            except Exception:
                pass
        return jsonify({
            "api_url": api_url,
            "api_port": port,
            "api_reachable": s.status == "ready",
            "api_status": s.status,            # idle / starting / ready / stopping / error / crashed
            "api_last_error": s.last_error,
            "model": s.model,
            "mtp_enabled": s.mtp_enabled,
            "mtp_depth": s.mtp_depth,
            "drafter_id": s.drafter_id,
            "prompt_cache_bytes": s.prompt_cache_bytes,
            "sampler": s.sampler,                       # user overrides (dict or None)
            "sampler_defaults": recommended_sampler,    # model's recommended values
            "adapters": s.adapters,                     # list of mounted LoRA dir paths
            "auth_token": "sk-optiq-local",
        })

    # Legacy path (no supervisor injected) — fall back to probing the API
    state = _probe_api(api_url)
    loaded = current_app.config.get("OPTIQ_LOADED_MODEL")
    return jsonify({
        "api_url": api_url,
        "api_port": port,
        "api_reachable": state["reachable"],
        "api_status": "ready" if state["reachable"] else "idle",
        "api_last_error": None,
        "model": loaded or state["model"],
        "mtp_enabled": current_app.config.get("OPTIQ_MTP_ENABLED", False),
        "mtp_depth": current_app.config.get("OPTIQ_MTP_DEPTH", 0),
        "auth_token": "sk-optiq-local",
    })


@bp.route("/server/apply", methods=["POST"])
def server_apply():
    """Hot-swap the loaded model and/or MTP setting.

    Body: {"model": "<abs path>", "mtp": bool, "mtp_depth": int}
    Returns immediately; the supervisor probes readiness in a worker
    thread. Caller should poll /api/status until status == 'ready' OR
    one of {'error', 'crashed'}.
    """
    supervisor = current_app.config.get("OPTIQ_API_SUPERVISOR")
    if supervisor is None:
        return jsonify({"ok": False, "error": "supervisor not available"}), 500

    data = request.get_json(force=True) or {}
    model = (data.get("model") or "").strip()
    if not model:
        return jsonify({"ok": False, "error": "model is required"}), 400

    import os
    if not os.path.isdir(model) and "/" not in model:
        return jsonify({"ok": False, "error": f"model not a local dir or HF id: {model!r}"}), 400

    mtp = bool(data.get("mtp", False))
    mtp_depth = int(data.get("mtp_depth", 2))
    drafter_id = (data.get("drafter_id") or "").strip() or None
    if mtp and drafter_id:
        return jsonify({"ok": False,
                        "error": "mtp and drafter_id are mutually exclusive"}), 400

    # SSD expert streaming mode for large MoE quants: auto (default), on, off.
    stream_experts = (data.get("stream_experts") or "auto").strip().lower()
    if stream_experts not in ("auto", "on", "off"):
        return jsonify({"ok": False,
                        "error": "stream_experts must be auto, on, or off"}), 400

    pcb_raw = data.get("prompt_cache_bytes")
    prompt_cache_bytes: int | None
    if pcb_raw is None or pcb_raw == "":
        prompt_cache_bytes = None  # api_runner's default kicks in
    else:
        try:
            prompt_cache_bytes = int(pcb_raw)
            if prompt_cache_bytes < 256 * 1024**2:  # 256 MB floor — too low is meaningless
                return jsonify({"ok": False,
                                "error": "prompt_cache_bytes must be >= 256 MB"}), 400
        except (TypeError, ValueError):
            return jsonify({"ok": False,
                            "error": f"prompt_cache_bytes not an int: {pcb_raw!r}"}), 400

    # Sampler overrides — partial dict of {temp, top_p, top_k, min_p}.
    # Each key is optional; missing keys fall through to the model's
    # generation_config.json defaults inside api_runner.
    sampler_raw = data.get("sampler") or {}
    if not isinstance(sampler_raw, dict):
        return jsonify({"ok": False, "error": "sampler must be an object"}), 400
    sampler: dict = {}
    for key, parser, low, high in [
        ("temp", float, 0.0, 5.0),
        ("top_p", float, 0.0, 1.0),
        ("top_k", int, 0, 100000),
        ("min_p", float, 0.0, 1.0),
    ]:
        raw = sampler_raw.get(key)
        if raw is None or raw == "":
            continue
        try:
            v = parser(raw)
        except (TypeError, ValueError):
            return jsonify({"ok": False,
                            "error": f"sampler.{key} not a {parser.__name__}: {raw!r}"}), 400
        if not (low <= v <= high):
            return jsonify({"ok": False,
                            "error": f"sampler.{key}={v} out of range [{low}, {high}]"}), 400
        sampler[key] = v
    if not sampler:
        sampler = None

    # Adapter list — each entry is a local directory containing
    # adapters.safetensors + adapter_config.json (PEFT-compatible) or the
    # optiq_lora_config.json sidecar from `optiq lora train`. One adapter
    # routes through mlx-lm's classic single-adapter boot; two or more
    # activate OptiQ's mounted-LoRA mode (request-side switching via the
    # 'adapters' body field).
    adapters_raw = data.get("adapters") or []
    if not isinstance(adapters_raw, list):
        return jsonify({"ok": False,
                        "error": "adapters must be a list of directory paths"}), 400
    adapters: list[str] = []
    for entry in adapters_raw:
        path = (str(entry) if not isinstance(entry, str) else entry).strip()
        if not path:
            continue
        if not os.path.isdir(path):
            return jsonify({"ok": False,
                            "error": f"adapter path is not a directory: {path!r}"}), 400
        # Sanity: at least one of the recognized adapter weight files must exist.
        if not (os.path.isfile(os.path.join(path, "adapters.safetensors"))
                or os.path.isfile(os.path.join(path, "adapter_model.safetensors"))):
            return jsonify({"ok": False,
                            "error": f"no adapter weights in {path!r} "
                                     "(expected adapters.safetensors or adapter_model.safetensors)"}), 400
        adapters.append(path)

    try:
        if supervisor.is_alive():
            supervisor.restart(model=model, mtp=mtp, mtp_depth=mtp_depth,
                               drafter_id=drafter_id,
                               prompt_cache_bytes=prompt_cache_bytes,
                               sampler=sampler,
                               adapters=adapters,
                               stream_experts=stream_experts)
        else:
            supervisor.start(model=model, mtp=mtp, mtp_depth=mtp_depth,
                             drafter_id=drafter_id,
                             prompt_cache_bytes=prompt_cache_bytes,
                             sampler=sampler,
                             adapters=adapters,
                             stream_experts=stream_experts)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    s = supervisor.state()
    return jsonify({"ok": True, "status": s.status, "model": s.model,
                    "adapters": s.adapters})


@bp.route("/server/inspect-sampling", methods=["POST"])
def server_inspect_sampling():
    """Return recommended sampler settings for a model without loading it.

    Reads ``generation_config.json`` from the local snapshot cache if
    available, otherwise pulls just that one small file from HF. Used
    by the server-settings UI to pre-fill the sampler inputs as soon
    as the user picks a model in the dropdown, instead of waiting for
    the model to load.

    Body: ``{"model": "<HF repo id or local path>"}``
    """
    data = request.get_json(force=True) or {}
    model = (data.get("model") or "").strip()
    if not model:
        return jsonify({"ok": False, "error": "model required"}), 400
    from optiq.runtime.gen_config import read_recommended_sampling
    try:
        defaults = read_recommended_sampling(model, allow_hf_fetch=True)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    # Map HF's ``temperature`` to the UI's ``temp`` key to match the
    # mlx_lm.server CLI surface the user types overrides into.
    out: dict = {}
    if "temperature" in defaults:
        out["temp"] = defaults["temperature"]
    for k in ("top_p", "top_k", "min_p"):
        if k in defaults:
            out[k] = defaults[k]
    return jsonify({"ok": True, "defaults": out})


@bp.route("/server/stop", methods=["POST"])
def server_stop():
    supervisor = current_app.config.get("OPTIQ_API_SUPERVISOR")
    if supervisor is None:
        return jsonify({"ok": False, "error": "supervisor not available"}), 500
    supervisor.stop()
    return jsonify({"ok": True})


@bp.route("/server/adapters", methods=["GET"])
def server_adapters():
    """List LoRA adapter directories the Lab can find on disk.

    Walks two roots:
      * ``<lab_root>/models`` — adapters dropped in by the Fine-tune wizard.
      * ``<lab_root>/cache`` — adapters under the cache dir (rare).

    A directory counts as an adapter if it has either
    ``adapters.safetensors`` or ``adapter_model.safetensors``. Returns
    ``[{name, path, has_optiq_sidecar}]`` so the UI can show a clickable
    list users add to the mount set with one click.
    """
    import os
    paths = current_app.config["OPTIQ_LAB_PATHS"]
    roots = [paths.models_dir]
    cache_dir = getattr(paths, "cache_dir", None)
    if cache_dir:
        roots.append(cache_dir)

    found: list[dict] = []
    seen = set()
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        try:
            entries = list(os.scandir(root))
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            path = entry.path
            if path in seen:
                continue
            has_weights = (
                os.path.isfile(os.path.join(path, "adapters.safetensors"))
                or os.path.isfile(os.path.join(path, "adapter_model.safetensors"))
            )
            if not has_weights:
                continue
            seen.add(path)
            found.append({
                "name": entry.name,
                "path": path,
                "has_optiq_sidecar": os.path.isfile(
                    os.path.join(path, "optiq_lora_config.json")
                ),
            })
    found.sort(key=lambda a: a["name"].lower())
    return jsonify({"adapters": found})


@bp.route("/server/published")
def server_published():
    """List published mlx-community OptiQ quants. Cached server-side
    for 15 min; pass ?refresh=1 to bust."""
    from .. import optiq_models
    force = request.args.get("refresh") == "1"
    try:
        quants = optiq_models.list_published(force_refresh=force)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({
        "ok": True,
        "quants": [
            {
                "repo_id": q.repo_id,
                "family": q.family,
                "size": q.size_label,
                "bits": q.bits_label,
                "downloads": q.downloads,
            }
            for q in quants
        ],
    })


@bp.route("/jobs/recent")
def jobs_recent():
    return jsonify({"jobs": jobs.recent(limit=8)})


@bp.route("/integrations")
def integrations_snippets():
    api_url = current_app.config["OPTIQ_API_URL"]
    snippets = integrations.all_snippets(api_url)
    return jsonify({
        "snippets": [
            {
                "label": s.label,
                "language": s.language,
                "body": s.body,
                "description": s.description,
            }
            for s in snippets
        ],
    })


# ---------------------------------------------------------------------------


def _probe_api(api_url: str) -> dict:
    """Best-effort: ask the model server what's currently loaded."""
    try:
        req = urllib.request.Request(
            f"{api_url}/v1/models",
            headers={"Authorization": "Bearer sk-optiq-local"},
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            import json
            data = json.loads(resp.read())
            models = data.get("data") or []
            model = models[0]["id"] if models else None
    except Exception:
        return {"reachable": False, "model": None, "mtp_enabled": False, "mtp_depth": 0}

    # mlx_lm.server doesn't expose MTP state via the API. We can't tell
    # from outside whether --mtp was passed, so default to "unknown".
    return {
        "reachable": True,
        "model": model,
        "mtp_enabled": False,
        "mtp_depth": 0,
    }
