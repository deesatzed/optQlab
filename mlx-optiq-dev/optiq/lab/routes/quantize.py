"""Quantize wizard routes."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from flask import (
    Blueprint, current_app, flash, jsonify, redirect, render_template,
    request, url_for,
)

from .. import auth, hf, job_bus, jobs, quantize_job
from .. import db as labdb


bp = Blueprint("quantize", __name__)


SUPPORTED_ARCHS = {
    # arch_id_substring → friendly label
    "Qwen3": "Qwen3.5 / 3.6",
    "Gemma3": "Gemma-4",
    "Llama": "Llama",
    "Mistral": "Mistral",
}


@bp.route("/quantize")
def quantize_page():
    return render_template(
        "quantize.html",
        page_title="Quantize",
        section="quantize",
    )


@bp.route("/api/quantize/inspect", methods=["POST"])
def inspect():
    """Pre-flight check on an HF model id: arch, size, MTP presence."""
    data = request.get_json(force=True) or {}
    model_id = (data.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"ok": False, "error": "model_id is required"}), 400

    from huggingface_hub import HfApi
    try:
        # files_metadata=True populates sibling.size for our size-in-GB calc
        info = HfApi().model_info(model_id, files_metadata=True)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    arch = None
    has_mtp = False
    config = (info.config or {})
    arch_list = config.get("architectures") or []
    if arch_list:
        arch = arch_list[0]

    # Total weights size in bytes (sum across all safetensors files)
    bytes_total = 0
    for sib in (info.siblings or []):
        fname = sib.rfilename
        if not fname.endswith(".safetensors"):
            continue
        bytes_total += sib.size or 0
        if "mtp" in fname.lower():
            has_mtp = True

    # Detect supported / partial / unknown arch
    label = None
    for needle, friendly in SUPPORTED_ARCHS.items():
        if arch and needle in arch:
            label = friendly
            break
    support = "supported" if label else "untested"
    if not arch:
        support = "unknown"

    return jsonify({
        "ok": True,
        "model_id": model_id,
        "arch": arch,
        "arch_label": label,
        "support": support,
        "size_gb": round(bytes_total / 1e9, 2) if bytes_total else None,
        "mtp_in_source": has_mtp,
    })


@bp.route("/api/quantize/submit", methods=["POST"])
def submit():
    data = request.get_json(force=True) or {}
    model_id = (data.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"ok": False, "error": "model_id is required"}), 400

    target_bpw = float(data.get("target_bpw", 5.0))
    reference = data.get("reference", "auto")
    candidate_bits = data.get("candidate_bits") or [4, 8]
    calibration_mix = data.get("calibration_mix", "optiq")
    n_calibration = int(data.get("n_calibration", 8))

    paths = current_app.config["OPTIQ_LAB_PATHS"]
    # Follow the community naming convention: <base>-OptiQ-<lowest-bit>bit
    # e.g. Qwen/Qwen3.5-0.8B + candidate_bits=[4,8] → Qwen3.5-0.8B-OptiQ-4bit
    # matches mlx-community/Qwen3.5-9B-OptiQ-4bit etc.
    base = model_id.rsplit("/", 1)[-1]                       # strip org
    base = re.sub(r"[^a-zA-Z0-9._-]", "_", base)             # filesystem-safe
    lowest_bit = min(int(b) for b in candidate_bits)
    output_name = f"{base}-OptiQ-{lowest_bit}bit"
    output_dir = paths.models_dir / output_name

    job_id = job_bus.submit(
        "quantize",
        quantize_job.run,
        config={
            "model_name": model_id,
            "output_dir": str(output_dir),
            "target_bpw": target_bpw,
            "candidate_bits": candidate_bits,
            "reference": reference,
            "calibration_mix": calibration_mix,
            "n_calibration": n_calibration,
        },
        resource_class="memory_heavy",
    )
    return jsonify({"ok": True, "job_id": job_id,
                    "output_dir": str(output_dir)})


@bp.route("/api/jobs/<job_id>/stream")
def job_stream(job_id):
    """SSE stream of job events. Closes when job hits a terminal state."""
    from flask import Response, stream_with_context

    @stream_with_context
    def gen():
        # Send an initial open frame so the EventSource fires onopen
        yield "retry: 1500\n\n"
        for event in jobs.tail(job_id, follow=True):
            payload = json.dumps(event, default=str)
            yield f"data: {payload}\n\n"
        # Final marker
        yield "event: end\ndata: {}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@bp.route("/api/jobs/<job_id>")
def job_get(job_id):
    info = jobs.get(job_id)
    if info is None:
        return jsonify({"ok": False, "error": "not found"}), 404
    return jsonify({"ok": True, "job": info})


@bp.route("/api/quantize/push", methods=["POST"])
def push():
    """Push a completed quantize output to HF. Requires Lab password
    (needed to decrypt the saved HF token)."""
    data = request.get_json(force=True) or {}
    job_id = (data.get("job_id") or "").strip()
    repo_id = (data.get("repo_id") or "").strip()
    private = bool(data.get("private", True))
    password = data.get("password") or ""

    if not (job_id and repo_id and password):
        return jsonify({"ok": False, "error": "job_id, repo_id, password are all required"}), 400
    if not auth.verify_password(password):
        return jsonify({"ok": False, "error": "wrong Lab password"}), 400

    job = jobs.get(job_id)
    if job is None or job["status"] != "done":
        return jsonify({"ok": False, "error": "job not done or not found"}), 400

    output_dir = job.get("output_path") or _output_dir_from_config(job)
    if not output_dir or not Path(output_dir).is_dir():
        return jsonify({"ok": False, "error": f"output dir missing: {output_dir!r}"}), 400

    token_pair = hf.get_first_token_decrypted(password)
    if token_pair is None:
        return jsonify({"ok": False,
                        "error": "no HF token saved. Add one in Settings → Hugging Face."}), 400
    _, plain_token = token_pair

    try:
        url = hf.push_folder(
            folder=str(Path(output_dir) / "optiq_mixed"),
            repo_id=repo_id,
            token=plain_token,
            repo_type="model",
            private=private,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": f"push failed: {e}"}), 500
    return jsonify({"ok": True, "url": url})


def _output_dir_from_config(job: dict) -> str | None:
    try:
        cfg = json.loads(job.get("config_json") or "{}")
        return cfg.get("output_dir")
    except Exception:
        return None
