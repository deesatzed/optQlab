"""Build-dataset wizard routes."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from flask import (
    Blueprint, current_app, jsonify, render_template, request,
)

from .. import auth, dataset_job, dataset_templates, hf, job_bus, jobs


bp = Blueprint("dataset", __name__)


@bp.route("/dataset")
def dataset_page():
    return render_template(
        "dataset.html",
        page_title="Build dataset",
        section="dataset",
        templates=[t.__dict__ for t in dataset_templates.TEMPLATES],
    )


@bp.route("/api/dataset/submit", methods=["POST"])
def submit():
    data = request.get_json(force=True) or {}
    template_id = (data.get("template_id") or "").strip()
    if not dataset_templates.get_template(template_id):
        return jsonify({"ok": False, "error": "unknown template"}), 400

    paths = current_app.config["OPTIQ_LAB_PATHS"]
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", template_id)
    out_dir = paths.cache_dir / f"dataset-{safe}-{int(time.time())}"

    job_id = job_bus.submit(
        "dataset",
        dataset_job.run,
        config={
            "template_id": template_id,
            "inputs": data.get("inputs") or {},
            "output_dir": str(out_dir),
            "api_url": current_app.config["OPTIQ_API_URL"],
            "auth_token": "sk-optiq-local",
            "model_name": data.get("model_name"),
        },
        resource_class="memory_heavy",
    )
    return jsonify({"ok": True, "job_id": job_id, "output_dir": str(out_dir)})


@bp.route("/api/dataset/push", methods=["POST"])
def push():
    data = request.get_json(force=True) or {}
    job_id = (data.get("job_id") or "").strip()
    repo_id = (data.get("repo_id") or "").strip()
    private = bool(data.get("private", True))
    password = data.get("password") or ""

    if not (job_id and repo_id and password):
        return jsonify({"ok": False, "error": "job_id, repo_id, password required"}), 400
    if not auth.verify_password(password):
        return jsonify({"ok": False, "error": "wrong Lab password"}), 400

    job = jobs.get(job_id)
    if job is None or job["status"] != "done":
        return jsonify({"ok": False, "error": "job not done or not found"}), 400

    try:
        cfg = json.loads(job["config_json"])
    except Exception:
        cfg = {}
    out_dir = cfg.get("output_dir")
    if not out_dir or not Path(out_dir).is_dir():
        return jsonify({"ok": False, "error": f"output dir missing: {out_dir!r}"}), 400

    token_pair = hf.get_first_token_decrypted(password)
    if token_pair is None:
        return jsonify({"ok": False,
                        "error": "no HF token saved. Add one in Settings → Hugging Face."}), 400
    _, plain_token = token_pair

    try:
        url = hf.push_folder(
            folder=out_dir,
            repo_id=repo_id,
            token=plain_token,
            repo_type="dataset",   # <-- dataset repo, not model
            private=private,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": f"push failed: {e}"}), 500
    return jsonify({"ok": True, "url": url})
