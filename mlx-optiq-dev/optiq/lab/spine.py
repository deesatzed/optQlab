"""Domain repositories for the Lab spine (Phase 0).

Single module (YAGNI) covering workspaces, builds, conversations, messages,
and message provenance. All persistence is real SQLite via ``optiq.lab.db``.
Side effects go through ``events.append``.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from . import db, events


# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------


def new_id(prefix: str) -> str:
    """Generate an id like ``ws_`` / ``bld_`` / ``msg_`` / ``conv_`` + hex."""
    if not prefix.endswith("_"):
        prefix = f"{prefix}_"
    return f"{prefix}{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _dumps(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _loads(raw: Any, default: Any) -> Any:
    if raw is None or raw == "":
        return default
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _row_to_dict(row: Any) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


_WORKSPACE_JSON_DEFAULTS = {
    "sampler_json": "{}",
    "tools_policy_json": "{}",
    "attached_files_json": "[]",
    "coherence_flags_json": "{}",
}


def create_workspace(name: str, **fields) -> str:
    """Insert workspace; return id.

    Optional fields: system_prompt, sampler_json (dict or str),
    tools_policy_json, attached_files_json, eval_set_id, default_build_id.
    """
    workspace_id = fields.pop("id", None) or new_id("ws_")
    system_prompt = fields.get("system_prompt", "")
    if system_prompt is None:
        system_prompt = ""
    sampler_json = _dumps(fields.get("sampler_json"), "{}")
    tools_policy_json = _dumps(fields.get("tools_policy_json"), "{}")
    attached_files_json = _dumps(fields.get("attached_files_json"), "[]")
    eval_set_id = fields.get("eval_set_id")
    default_build_id = fields.get("default_build_id")

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO workspaces (
                id, name, default_build_id, system_prompt,
                sampler_json, tools_policy_json, attached_files_json,
                eval_set_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workspace_id,
                name,
                default_build_id,
                system_prompt,
                sampler_json,
                tools_policy_json,
                attached_files_json,
                eval_set_id,
            ),
        )
    return workspace_id


def get_workspace(workspace_id: str) -> dict | None:
    """Return workspace dict with JSON columns parsed, or None."""
    conn = db.get_conn()
    row = conn.execute(
        "SELECT * FROM workspaces WHERE id = ?",
        (workspace_id,),
    ).fetchone()
    if row is None:
        return None
    return _decode_workspace(_row_to_dict(row))


def _decode_workspace(d: dict[str, Any]) -> dict[str, Any]:
    d["sampler_json"] = _loads(d.get("sampler_json"), {})
    d["tools_policy_json"] = _loads(d.get("tools_policy_json"), {})
    d["attached_files_json"] = _loads(d.get("attached_files_json"), [])
    d["coherence_flags_json"] = _loads(d.get("coherence_flags_json"), {})
    return d


def set_workspace_build(
    workspace_id: str,
    build_id: str,
    *,
    resident_build_id: str | None = None,
) -> dict:
    """Set default_build_id and update coherence flags for residency.

    If ``resident_build_id`` is not None and differs from ``build_id``, set
    ``model_not_resident=true`` and emit ``workspace.coherence``.
    If they match, clear ``model_not_resident``.
    Return the updated workspace dict.
    """
    conn = db.get_conn()
    row = conn.execute(
        "SELECT * FROM workspaces WHERE id = ?",
        (workspace_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"workspace not found: {workspace_id}")

    flags = _loads(row["coherence_flags_json"], {})
    if not isinstance(flags, dict):
        flags = {}

    emit_coherence = False
    if resident_build_id is not None and resident_build_id != build_id:
        flags["model_not_resident"] = True
        emit_coherence = True
    elif resident_build_id is not None and resident_build_id == build_id:
        flags.pop("model_not_resident", None)

    flags_text = json.dumps(flags)
    with db.transaction() as conn:
        conn.execute(
            """
            UPDATE workspaces
            SET default_build_id = ?,
                coherence_flags_json = ?,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (build_id, flags_text, workspace_id),
        )

    if emit_coherence:
        events.append(
            type="workspace.coherence",
            entity_type="workspace",
            entity_id=workspace_id,
            payload={
                "model_not_resident": True,
                "default_build_id": build_id,
                "resident_build_id": resident_build_id,
            },
            workspace_id=workspace_id,
        )

    updated = get_workspace(workspace_id)
    assert updated is not None
    return updated


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def register_build(
    *,
    name: str,
    path: str,
    source_hf_id: str | None = None,
    quant_profile: str | None = None,
    bpw: float | None = None,
    weights_gb: float | None = None,
    kv_bits_default: int | None = None,
    ctx_default: int | None = None,
    adapter_stack: list | None = None,
    metadata: dict | None = None,
    build_id: str | None = None,
) -> str:
    """Insert build; return id. Emit event ``build.registered``."""
    bid = build_id or new_id("bld_")
    adapter_stack_json = _dumps(adapter_stack if adapter_stack is not None else [], "[]")
    metadata_json = _dumps(metadata if metadata is not None else {}, "{}")

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO builds (
                id, name, source_hf_id, path, quant_profile, bpw, weights_gb,
                kv_bits_default, ctx_default, adapter_stack_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bid,
                name,
                source_hf_id,
                path,
                quant_profile,
                bpw,
                weights_gb,
                kv_bits_default,
                ctx_default,
                adapter_stack_json,
                metadata_json,
            ),
        )

    events.append(
        type="build.registered",
        entity_type="build",
        entity_id=bid,
        payload={
            "name": name,
            "path": path,
            "source_hf_id": source_hf_id,
            "quant_profile": quant_profile,
        },
    )
    return bid


def get_build(build_id: str) -> dict | None:
    """Return build dict with JSON columns parsed as adapter_stack/metadata."""
    conn = db.get_conn()
    row = conn.execute(
        "SELECT * FROM builds WHERE id = ?",
        (build_id,),
    ).fetchone()
    if row is None:
        return None
    d = _row_to_dict(row)
    d["adapter_stack"] = _loads(d.pop("adapter_stack_json", None), [])
    d["metadata"] = _loads(d.pop("metadata_json", None), {})
    return d


# ---------------------------------------------------------------------------
# Conversation / messages
# ---------------------------------------------------------------------------


def upsert_conversation_from_chat_payload(data: dict) -> str:
    """Upsert conversation + replace messages from a save_chat-shaped payload.

    data: ``{id?, title, messages:[{role,content,provenance?}], model, workspace_id?}``

    Emits ``conversation.upserted`` and per-message ``message.created``.
    Returns conversation id.
    """
    conv_id = data.get("id") or new_id("conv_")
    title = data.get("title") or "Untitled chat"
    model = data.get("model") or ""
    workspace_id = data.get("workspace_id")
    messages = data.get("messages") or []
    if not isinstance(messages, list):
        messages = []

    prepared: list[tuple[str, str, str, int, dict | None]] = []
    for seq, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        mid = msg.get("id") or new_id("msg_")
        role = msg.get("role") or "user"
        content = msg.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content)
        prov = msg.get("provenance")
        if prov is not None and not isinstance(prov, dict):
            prov = None
        prepared.append((mid, role, content, seq, prov))

    with db.transaction() as conn:
        existing = conn.execute(
            "SELECT id FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO conversations (id, workspace_id, title, model)
                VALUES (?, ?, ?, ?)
                """,
                (conv_id, workspace_id, title, model),
            )
        else:
            conn.execute(
                """
                UPDATE conversations
                SET workspace_id = ?,
                    title = ?,
                    model = ?,
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (workspace_id, title, model, conv_id),
            )

        old_msgs = conn.execute(
            "SELECT id FROM messages WHERE conversation_id = ?",
            (conv_id,),
        ).fetchall()
        old_ids = [r["id"] for r in old_msgs]
        if old_ids:
            placeholders = ",".join("?" * len(old_ids))
            conn.execute(
                f"DELETE FROM message_provenance WHERE message_id IN ({placeholders})",
                old_ids,
            )
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conv_id,),
            )

        for mid, role, content, seq, prov in prepared:
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, seq)
                VALUES (?, ?, ?, ?, ?)
                """,
                (mid, conv_id, role, content, seq),
            )
            if prov is not None and role == "assistant":
                complete = 1 if provenance_complete(prov) else 0
                conn.execute(
                    """
                    INSERT INTO message_provenance (message_id, envelope_json, complete)
                    VALUES (?, ?, ?)
                    """,
                    (mid, json.dumps(prov), complete),
                )

    events.append(
        type="conversation.upserted",
        entity_type="conversation",
        entity_id=conv_id,
        payload={
            "title": title,
            "model": model,
            "message_count": len(prepared),
        },
        workspace_id=workspace_id,
    )
    for mid, role, content, seq, _prov in prepared:
        events.append(
            type="message.created",
            entity_type="message",
            entity_id=mid,
            payload={
                "conversation_id": conv_id,
                "role": role,
                "seq": seq,
            },
            workspace_id=workspace_id,
        )

    return conv_id


def get_conversation(conversation_id: str) -> dict | None:
    """Return ``{id,title,model,messages:[{id,role,content,seq,provenance?}]}``."""
    conn = db.get_conn()
    row = conn.execute(
        "SELECT * FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    if row is None:
        return None

    msg_rows = conn.execute(
        """
        SELECT m.id, m.role, m.content, m.seq, p.envelope_json
        FROM messages m
        LEFT JOIN message_provenance p ON p.message_id = m.id
        WHERE m.conversation_id = ?
        ORDER BY m.seq ASC
        """,
        (conversation_id,),
    ).fetchall()

    messages: list[dict[str, Any]] = []
    for m in msg_rows:
        item: dict[str, Any] = {
            "id": m["id"],
            "role": m["role"],
            "content": m["content"],
            "seq": m["seq"],
        }
        if m["envelope_json"] is not None:
            item["provenance"] = _loads(m["envelope_json"], {})
        messages.append(item)

    return {
        "id": row["id"],
        "title": row["title"],
        "model": row["model"],
        "workspace_id": row["workspace_id"],
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def provenance_complete(envelope: dict) -> bool:
    """True if required keys present and non-null.

    Required: (build_id OR server_model_label), sampler, context_window, captured_at.
    """
    if not isinstance(envelope, dict):
        return False
    has_identity = (
        envelope.get("build_id") is not None
        or envelope.get("server_model_label") is not None
    )
    if not has_identity:
        return False
    for key in ("sampler", "context_window", "captured_at"):
        if envelope.get(key) is None:
            return False
    return True


def set_message_provenance(message_id: str, envelope: dict) -> None:
    """Upsert message_provenance; complete flag from provenance_complete()."""
    if not isinstance(envelope, dict):
        raise TypeError("envelope must be a dict")
    complete = 1 if provenance_complete(envelope) else 0
    envelope_json = json.dumps(envelope)
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO message_provenance (message_id, envelope_json, complete, captured_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(message_id) DO UPDATE SET
                envelope_json = excluded.envelope_json,
                complete = excluded.complete,
                captured_at = datetime('now')
            """,
            (message_id, envelope_json, complete),
        )


def get_provenance(message_id: str) -> dict | None:
    """Return envelope dict or None."""
    conn = db.get_conn()
    row = conn.execute(
        "SELECT envelope_json FROM message_provenance WHERE message_id = ?",
        (message_id,),
    ).fetchone()
    if row is None:
        return None
    return _loads(row["envelope_json"], {})
