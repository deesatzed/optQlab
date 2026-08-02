"""Models surface — unified list + Fit-gated load (critical gap G1/G7)."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
)

from .. import db, fit_engine, local_quants, machine, spine
from ..config import ensure_lab_dirs

bp = Blueprint("models", __name__)


def _quant_rows():
    paths = current_app.config["OPTIQ_LAB_PATHS"]
    local_built = []
    for q in local_quants.discover(paths.models_dir):
        local_built.append(_row(q, "local"))
    hf_cached = []
    seen = {r["path"] for r in local_built}
    for q in local_quants.discover_hf_cache():
        if q.path not in seen:
            hf_cached.append(_row(q, "hf_cache"))
    # Spine builds (registered paths)
    try:
        conn = db.get_conn()
        for b in conn.execute("SELECT * FROM builds ORDER BY created_at DESC").fetchall():
            p = b["path"]
            if p and p not in seen:
                seen.add(p)
                local_built.append({
                    "name": b["name"] or p,
                    "path": p,
                    "bits_label": b["quant_profile"] or "—",
                    "bpw_detail": f"{b['bpw']} BPW" if b["bpw"] is not None else "",
                    "size_gb": b["weights_gb"],
                    "source": "spine",
                    "has_mtp": False,
                })
    except Exception:
        pass
    return local_built + hf_cached


def _row(q, source: str) -> dict:
    return {
        "name": q.display_name,
        "path": q.path,
        "bits_label": q.bits_label,
        "bpw_detail": q.bpw_detail,
        "size_gb": round(q.size_bytes / 1e9, 2) if q.size_bytes else None,
        "source": source,
        "has_mtp": q.has_mtp,
    }


@bp.route("/models")
def models_page():
    ensure_lab_dirs()
    rows = _quant_rows()
    loaded = current_app.config.get("OPTIQ_LOADED_MODEL")
    supervisor = current_app.config.get("OPTIQ_API_SUPERVISOR")
    if supervisor is not None:
        try:
            loaded = supervisor.state().model or loaded
        except Exception:
            pass
    return render_template(
        "models.html",
        page_title="Models",
        section="models",
        models=rows,
        loaded=loaded,
    )


@bp.route("/api/fit/predict", methods=["POST"])
def fit_predict():
    data = request.get_json(force=True) or {}
    path = (data.get("path") or data.get("model") or "").strip() or None
    weights_gb = data.get("weights_gb")
    if weights_gb is not None:
        try:
            weights_gb = float(weights_gb)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "weights_gb must be a number"}), 400
    ctx = int(data.get("ctx") or data.get("context") or 32768)
    kv_bits = int(data.get("kv_bits") or 8)
    try:
        result = fit_engine.predict(
            path=path,
            weights_gb=weights_gb,
            ctx=ctx,
            kv_bits=kv_bits,
            n_layers=int(data.get("n_layers") or 48),
            n_kv_heads=int(data.get("n_kv_heads") or 8),
            head_dim=int(data.get("head_dim") or 128),
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "fit": result.to_dict()})


@bp.route("/api/fit/calibrate", methods=["POST"])
def fit_calibrate():
    try:
        data = fit_engine.run_calibration_snapshot()
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "calibration": data})


@bp.route("/api/machine")
def api_machine():
    api_url = current_app.config["OPTIQ_API_URL"]
    lab_port = urlparse(request.host_url).port or 7860
    model = current_app.config.get("OPTIQ_LOADED_MODEL")
    api_reachable = None
    api_status = None
    adapters: list = []
    mtp = False
    supervisor = current_app.config.get("OPTIQ_API_SUPERVISOR")
    if supervisor is not None:
        try:
            s = supervisor.state()
            model = s.model or model
            api_reachable = s.status == "ready"
            api_status = s.status
            adapters = list(s.adapters or [])
            mtp = bool(s.mtp_enabled)
        except Exception:
            pass
    st = machine.machine_state(
        api_url=api_url,
        lab_port=lab_port,
        model=model,
        api_reachable=api_reachable,
        api_status=api_status,
        adapters=adapters,
        mtp_enabled=mtp,
    )
    return jsonify(st)


@bp.route("/api/models/load", methods=["POST"])
def models_load():
    """Fit-gated load — single primary load path (G1/G7).

    Body: {model|path, ctx?, kv_bits?, force?, mtp?, mtp_depth?, adapters?}
    If Fit blocks_load and force is not true → 409 with fit payload.
    Otherwise reuses ApiSupervisor start/restart.
    """
    data = request.get_json(force=True) or {}
    model = (data.get("model") or data.get("path") or "").strip()
    if not model:
        return jsonify({"ok": False, "error": "model required"}), 400

    ctx = int(data.get("ctx") or 32768)
    kv_bits = int(data.get("kv_bits") or 8)
    force = bool(data.get("force"))

    try:
        fit = fit_engine.predict(path=model if os.path.exists(model) else None,
                                 weights_gb=data.get("weights_gb"),
                                 ctx=ctx, kv_bits=kv_bits)
    except Exception as e:
        return jsonify({"ok": False, "error": f"fit failed: {e}"}), 500

    if fit.blocks_load and not force:
        return jsonify({
            "ok": False,
            "error": "fit_blocks_load",
            "message": fit.detail,
            "fit": fit.to_dict(),
        }), 409

    supervisor = current_app.config.get("OPTIQ_API_SUPERVISOR")
    if supervisor is None:
        # No supervisor (tests / legacy): record intent only
        current_app.config["OPTIQ_LOADED_MODEL"] = model
        try:
            spine.register_build(
                name=os.path.basename(model.rstrip("/")) or model,
                path=model if os.path.exists(model) else model,
                quant_profile=None,
                weights_gb=fit.weights_gb or None,
            )
        except Exception:
            pass
        return jsonify({
            "ok": True,
            "status": "configured",
            "model": model,
            "fit": fit.to_dict(),
            "note": "no ApiSupervisor — model path stored in Lab config only",
        })

    mtp = bool(data.get("mtp"))
    mtp_depth = int(data.get("mtp_depth") or 2)
    adapters = data.get("adapters") or []
    if not isinstance(adapters, list):
        return jsonify({"ok": False, "error": "adapters must be a list"}), 400

    try:
        if supervisor.is_alive():
            supervisor.restart(
                model=model, mtp=mtp, mtp_depth=mtp_depth, adapters=adapters,
            )
        else:
            supervisor.start(
                model=model, mtp=mtp, mtp_depth=mtp_depth, adapters=adapters,
            )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "fit": fit.to_dict()}), 500

    current_app.config["OPTIQ_LOADED_MODEL"] = model
    try:
        spine.register_build(
            name=os.path.basename(str(model).rstrip("/")) or str(model),
            path=str(model),
            weights_gb=fit.weights_gb or None,
        )
    except Exception:
        pass

    s = supervisor.state()
    return jsonify({
        "ok": True,
        "status": s.status,
        "model": s.model,
        "fit": fit.to_dict(),
        "forced": force and fit.blocks_load,
    })
