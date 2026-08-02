"""Eval GUI APIs and page (WP-3 / G2)."""

from __future__ import annotations

import json

from flask import Blueprint, current_app, jsonify, render_template, request

from .. import eval_job, eval_service, job_bus, local_quants
from ..config import ensure_lab_dirs

bp = Blueprint("eval_ui", __name__)


@bp.route("/eval")
def eval_page():
    ensure_lab_dirs()
    paths = current_app.config["OPTIQ_LAB_PATHS"]
    models = []
    seen = set()
    for q in local_quants.discover(paths.models_dir):
        if q.path not in seen:
            seen.add(q.path)
            models.append({"name": q.display_name, "path": q.path})
    for q in local_quants.discover_hf_cache():
        if q.path not in seen:
            seen.add(q.path)
            models.append({"name": q.display_name, "path": q.path})
    return render_template(
        "eval.html",
        page_title="Eval",
        section="eval",
        models=models,
        eval_sets=eval_service.list_eval_sets(),
        recent=eval_service.list_evals(limit=20),
    )


@bp.route("/api/eval/sets", methods=["GET"])
def api_list_sets():
    return jsonify({"sets": eval_service.list_eval_sets()})


@bp.route("/api/eval/sets", methods=["POST"])
def api_create_set():
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    items = data.get("items")
    if items is None and data.get("jsonl"):
        items = []
        for line in str(data["jsonl"]).splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    try:
        sid = eval_service.save_eval_set(name, items or [])
    except (ValueError, json.JSONDecodeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "id": sid, "set": eval_service.load_eval_set(sid)}), 201


@bp.route("/api/eval/sets/<set_id>", methods=["GET"])
def api_get_set(set_id: str):
    s = eval_service.load_eval_set(set_id)
    if not s:
        return jsonify({"error": "not found"}), 404
    return jsonify(s)


@bp.route("/api/eval/results", methods=["GET"])
def api_list_results():
    build_id = request.args.get("build_id") or None
    return jsonify({"evals": eval_service.list_evals(build_id=build_id, limit=50)})


@bp.route("/api/eval/results/<eval_id>", methods=["GET"])
def api_get_result(eval_id: str):
    e = eval_service.get_eval(eval_id)
    if not e:
        return jsonify({"error": "not found"}), 404
    return jsonify(e)


@bp.route("/api/eval/compare", methods=["POST"])
def api_compare():
    data = request.get_json(force=True) or {}
    a = data.get("a") or data.get("baseline")
    b = data.get("b") or data.get("candidate")
    if not a or not b:
        return jsonify({"ok": False, "error": "a and b eval ids required"}), 400
    try:
        cmp_ = eval_service.compare_evals(a, b)
    except KeyError:
        return jsonify({"ok": False, "error": "eval not found"}), 404
    return jsonify({"ok": True, "compare": cmp_})


@bp.route("/api/eval/promote", methods=["POST"])
def api_promote():
    """Regression gate: is candidate allowed vs baseline?"""
    data = request.get_json(force=True) or {}
    baseline = data.get("baseline")
    candidate = data.get("candidate")
    if not baseline or not candidate:
        return jsonify({"ok": False, "error": "baseline and candidate required"}), 400
    min_delta = float(data.get("min_delta") or 0.0)
    try:
        gate = eval_service.promote_allowed(baseline, candidate, min_delta=min_delta)
    except KeyError:
        return jsonify({"ok": False, "error": "eval not found"}), 404
    return jsonify({"ok": True, **gate})


@bp.route("/api/eval/run", methods=["POST"])
def api_run_eval():
    """Submit eval job (memory_heavy). Body: model_path, suite, eval_set_id?, n_samples?"""
    data = request.get_json(force=True) or {}
    model_path = (data.get("model_path") or data.get("model") or "").strip()
    suite = (data.get("suite") or "gsm8k-50").strip()
    if not model_path:
        return jsonify({"ok": False, "error": "model_path required"}), 400
    if suite == "byo" and not data.get("eval_set_id"):
        return jsonify({"ok": False, "error": "eval_set_id required for byo"}), 400

    # Pre-create job id so worker can store it in eval metadata
    from ..jobs import new_job_id
    # job_bus.submit creates its own id — we put a placeholder; store uses config job_id after patch
    config = {
        "model_path": model_path,
        "suite": suite,
        "eval_set_id": data.get("eval_set_id"),
        "n_samples": data.get("n_samples"),
        "build_id": data.get("build_id"),
        "skip_kl": bool(data.get("skip_kl", True)),
        "show_score": bool(data.get("show_score", True)),
        "max_tokens": data.get("max_tokens") or 128,
    }
    job_id = job_bus.submit(
        "eval",
        eval_job.run,
        config=config,
        resource_class="memory_heavy",
        build_id=data.get("build_id"),
    )
    return jsonify({"ok": True, "job_id": job_id, "suite": suite, "note": "Track progress under Runs"})


@bp.route("/api/fit/consequences", methods=["POST"])
def api_fit_consequences():
    """Knob consequence preview (WP-3D) — estimates labeled as estimates."""
    from .. import fit_engine
    data = request.get_json(force=True) or {}
    path = (data.get("path") or data.get("model") or "").strip() or None
    weights_gb = data.get("weights_gb")
    ctx = int(data.get("ctx") or 32768)
    kv_bits = int(data.get("kv_bits") or 8)
    base = fit_engine.predict(path=path, weights_gb=weights_gb, ctx=ctx, kv_bits=kv_bits)
    variants = []
    for label, c, k in (
        ("kv_4bit", ctx, 4),
        ("kv_8bit", ctx, 8),
        ("ctx_8k", 8192, kv_bits),
        ("ctx_32k", 32768, kv_bits),
        ("ctx_64k", 65536, kv_bits),
    ):
        alt = fit_engine.predict(path=path, weights_gb=weights_gb or base.weights_gb, ctx=c, kv_bits=k)
        variants.append({
            "label": label,
            "ctx": c,
            "kv_bits": k,
            "total_gb": alt.total_gb,
            "delta_gb": round(alt.total_gb - base.total_gb, 3),
            "verdict": alt.verdict,
            "blocks_load": alt.blocks_load,
            "estimate": True,
        })
    return jsonify({
        "ok": True,
        "baseline": base.to_dict(),
        "variants": variants,
        "notes": ["All ΔGB / verdicts are Fit Engine estimates, not measured tok/s or capability."],
    })
