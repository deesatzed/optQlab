"""Spine API: events bus, provenance export, workspace CRUD.

Endpoints (all under ``/api``):

* ``GET  /events`` — poll event log after a cursor
* ``GET  /bus/stream`` — SSE event bus
* ``GET  /messages/<id>/provenance`` — provenance envelope export
* ``GET|POST /workspaces`` — list / create
* ``GET|PATCH /workspaces/<id>`` — get / update
"""

from __future__ import annotations

import json
import time
from typing import Any

from flask import Blueprint, Response, jsonify, request, stream_with_context

from .. import db, events, spine


bp = Blueprint("spine_api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@bp.route("/events")
def list_events():
    """Return events with id > after_id (default 0), limited to ``limit``."""
    after_id = _int_arg("after_id", 0)
    limit = _int_arg("limit", 500)
    if limit < 1:
        limit = 1
    if limit > 5000:
        limit = 5000
    return jsonify({"events": events.iter_after(after_id=after_id, limit=limit)})


@bp.route("/bus/stream")
def bus_stream():
    """SSE stream of spine events.

    Polls ``events.iter_after`` every 0.2s and yields ``data: <json>\\n\\n``
    for each new row. Query params:

    * ``after_id`` (int, default 0) — cursor
    * ``max_events`` (int, optional) — close after N events (testability)
    * ``max_seconds`` (float, optional) — close after idle/elapsed window
    """
    after_id = _int_arg("after_id", 0)
    max_events = request.args.get("max_events", type=int)
    max_seconds = request.args.get("max_seconds", type=float)

    @stream_with_context
    def gen():
        nonlocal after_id
        sent = 0
        start = time.monotonic()
        # Initial retry hint for EventSource clients
        yield "retry: 1500\n\n"
        while True:
            if max_seconds is not None and (time.monotonic() - start) >= max_seconds:
                break
            batch = events.iter_after(after_id=after_id, limit=100)
            if not batch:
                time.sleep(0.2)
                continue
            for row in batch:
                yield f"data: {json.dumps(row, default=str)}\n\n"
                after_id = int(row["id"])
                sent += 1
                if max_events is not None and sent >= max_events:
                    return

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@bp.route("/messages/<message_id>/provenance")
def message_provenance(message_id: str):
    """Return the provenance envelope for a message, or 404."""
    envelope = spine.get_provenance(message_id)
    if envelope is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(envelope)


# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------


@bp.route("/workspaces")
def list_workspaces():
    """Return all workspaces ordered by updated_at DESC."""
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT id FROM workspaces ORDER BY updated_at DESC, created_at DESC"
    ).fetchall()
    workspaces: list[dict[str, Any]] = []
    for row in rows:
        ws = spine.get_workspace(row["id"])
        if ws is not None:
            workspaces.append(ws)
    return jsonify({"workspaces": workspaces})


@bp.route("/workspaces", methods=["POST"])
def create_workspace():
    """Create a workspace. Body: ``{name, ...optional fields}``."""
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name")
    if not name or not isinstance(name, str) or not name.strip():
        return jsonify({"error": "name is required"}), 400

    fields: dict[str, Any] = {}
    for key in (
        "system_prompt",
        "sampler_json",
        "tools_policy_json",
        "attached_files_json",
        "eval_set_id",
        "default_build_id",
        "id",
    ):
        if key in data:
            fields[key] = data[key]

    ws_id = spine.create_workspace(name.strip(), **fields)
    ws = spine.get_workspace(ws_id)
    return jsonify(ws), 201


@bp.route("/workspaces/<workspace_id>")
def get_workspace(workspace_id: str):
    """Return one workspace or 404."""
    ws = spine.get_workspace(workspace_id)
    if ws is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(ws)


@bp.route("/workspaces/<workspace_id>", methods=["PATCH"])
def patch_workspace(workspace_id: str):
    """Update workspace name/fields.

    If ``default_build_id`` is present in the body, routes through
    ``spine.set_workspace_build`` with optional ``resident_build_id``.
    """
    existing = spine.get_workspace(workspace_id)
    if existing is None:
        return jsonify({"error": "not found"}), 404

    data = request.get_json(force=True, silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "body must be a JSON object"}), 400

    # Build assignment goes through the coherence-aware path
    if "default_build_id" in data and data["default_build_id"]:
        build_id = data["default_build_id"]
        if not isinstance(build_id, str):
            return jsonify({"error": "default_build_id must be a string"}), 400
        resident = data.get("resident_build_id")
        if resident is not None and not isinstance(resident, str):
            return jsonify({"error": "resident_build_id must be a string"}), 400
        try:
            spine.set_workspace_build(
                workspace_id,
                build_id,
                resident_build_id=resident,
            )
        except KeyError:
            return jsonify({"error": "not found"}), 404

    # Scalar / JSON field updates
    col_map: list[tuple[str, Any]] = []
    if "name" in data:
        name = data["name"]
        if not isinstance(name, str) or not name.strip():
            return jsonify({"error": "name must be a non-empty string"}), 400
        col_map.append(("name", name.strip()))
    if "system_prompt" in data:
        sp = data["system_prompt"]
        if sp is None:
            sp = ""
        if not isinstance(sp, str):
            return jsonify({"error": "system_prompt must be a string"}), 400
        col_map.append(("system_prompt", sp))
    if "sampler_json" in data:
        col_map.append(("sampler_json", _as_json_text(data["sampler_json"], "{}")))
    if "tools_policy_json" in data:
        col_map.append(
            ("tools_policy_json", _as_json_text(data["tools_policy_json"], "{}"))
        )
    if "attached_files_json" in data:
        col_map.append(
            ("attached_files_json", _as_json_text(data["attached_files_json"], "[]"))
        )
    if "eval_set_id" in data:
        col_map.append(("eval_set_id", data["eval_set_id"]))

    if col_map:
        sets = ", ".join(f"{col} = ?" for col, _ in col_map)
        params = [val for _, val in col_map]
        params.append(workspace_id)
        with db.transaction() as conn:
            conn.execute(
                f"""
                UPDATE workspaces
                SET {sets}, updated_at = datetime('now')
                WHERE id = ?
                """,
                params,
            )

    updated = spine.get_workspace(workspace_id)
    return jsonify(updated)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _int_arg(name: str, default: int) -> int:
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _as_json_text(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return json.dumps(value)
