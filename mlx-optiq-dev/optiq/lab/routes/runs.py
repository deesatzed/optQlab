"""Global Runs list — non-blocking job history (WP-2D / G6)."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request

from .. import job_bus, jobs
from ..config import ensure_lab_dirs

bp = Blueprint("runs", __name__)


@bp.route("/runs")
def runs_page():
    ensure_lab_dirs()
    return render_template(
        "runs.html",
        page_title="Runs",
        section="runs",
    )


@bp.route("/api/runs")
def list_runs():
    """List jobs (and dual-written runs metadata). Query: kind, status, limit."""
    ensure_lab_dirs()
    kind = (request.args.get("kind") or "").strip() or None
    status = (request.args.get("status") or "").strip() or None
    try:
        limit = min(200, max(1, int(request.args.get("limit") or 50)))
    except ValueError:
        limit = 50

    rows = jobs.recent(limit=limit * 2 if status else limit, kind=kind)
    out = []
    for r in rows:
        if status and r.get("status") != status:
            continue
        out.append({
            "id": r["id"],
            "kind": r["kind"],
            "status": r["status"],
            "progress": r.get("progress"),
            "message": r.get("message"),
            "error": r.get("error"),
            "started_at": r.get("started_at"),
            "ended_at": r.get("ended_at"),
            "log_path": r.get("log_path"),
            "config_json": r.get("config_json"),
        })
        if len(out) >= limit:
            break
    return jsonify({"runs": out})


@bp.route("/api/runs/<job_id>/cancel", methods=["POST"])
def cancel_run(job_id: str):
    ok = job_bus.cancel(job_id)
    if not ok:
        # Fall back: mark zombie/failed cleanup via jobs table if already terminal
        row = jobs.get(job_id)
        if row is None:
            return jsonify({"ok": False, "error": "not found"}), 404
        if row["status"] in ("done", "failed", "cancelled", "zombie"):
            return jsonify({"ok": False, "error": "already terminal", "status": row["status"]}), 400
        return jsonify({"ok": False, "error": "could not cancel", "status": row["status"]}), 400
    row = jobs.get(job_id)
    return jsonify({"ok": True, "status": row["status"] if row else "cancelled"})


@bp.route("/api/runs/<job_id>/log")
def run_log(job_id: str):
    """Return last N log lines as text (for Runs detail)."""
    row = jobs.get(job_id)
    if row is None:
        return jsonify({"error": "not found"}), 404
    path = row.get("log_path")
    if not path:
        return jsonify({"lines": []})
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return jsonify({"lines": [], "error": "log unreadable"})
    try:
        tail = min(500, max(1, int(request.args.get("tail") or 200)))
    except ValueError:
        tail = 200
    return jsonify({"lines": [ln.rstrip("\n") for ln in lines[-tail:]]})
