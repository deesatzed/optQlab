"""SQLite schema + helpers for the Lab.

Single connection-per-thread via ``get_conn()``. WAL mode so SSE polling
doesn't fight the writer. Schema is created on first connect; migrations
are additive (new tables / new columns only; no destructive rewrites).
"""

from __future__ import annotations

import sqlite3
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import ensure_lab_dirs


_local = threading.local()


def get_conn() -> sqlite3.Connection:
    """Per-thread SQLite connection. Initialised lazily."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn
    paths = ensure_lab_dirs()
    conn = sqlite3.connect(paths.db, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    _migrate(conn)
    _local.conn = conn
    return conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Wrap a block in a transaction. Rolls back on exception."""
    conn = get_conn()
    try:
        conn.execute("BEGIN")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


SCHEMA_V1 = [
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS credentials (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        password_hash TEXT NOT NULL,
        salt BLOB NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hf_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        encrypted_token BLOB NOT NULL,
        username TEXT,
        orgs TEXT,
        scope TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        config_json TEXT NOT NULL,
        progress REAL NOT NULL DEFAULT 0.0,
        message TEXT,
        log_path TEXT NOT NULL,
        output_path TEXT,
        error TEXT,
        started_at TEXT NOT NULL DEFAULT (datetime('now')),
        ended_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS models_local (
        path TEXT PRIMARY KEY,
        source_hf_id TEXT,
        bpw REAL,
        mtp_present INTEGER NOT NULL DEFAULT 0,
        kv_config_path TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_jobs_status_started
        ON jobs(status, started_at DESC)
    """,
]


SCHEMA_V2 = [
    """
    CREATE TABLE IF NOT EXISTS workspaces (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        default_build_id TEXT,
        system_prompt TEXT NOT NULL DEFAULT '',
        sampler_json TEXT NOT NULL DEFAULT '{}',
        tools_policy_json TEXT NOT NULL DEFAULT '{}',
        attached_files_json TEXT NOT NULL DEFAULT '[]',
        eval_set_id TEXT,
        coherence_flags_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS builds (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        source_hf_id TEXT,
        path TEXT,
        quant_profile TEXT,
        bpw REAL,
        weights_gb REAL,
        kv_bits_default INTEGER,
        ctx_default INTEGER,
        adapter_stack_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS adapters (
        id TEXT PRIMARY KEY,
        build_id TEXT,
        name TEXT NOT NULL,
        path TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        path TEXT,
        workspace_id TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        workspace_id TEXT,
        build_id TEXT,
        config_json TEXT NOT NULL DEFAULT '{}',
        progress REAL NOT NULL DEFAULT 0.0,
        message TEXT,
        error TEXT,
        started_at TEXT NOT NULL DEFAULT (datetime('now')),
        ended_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        workspace_id TEXT,
        title TEXT,
        model TEXT,
        mode TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        seq INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS message_provenance (
        message_id TEXT PRIMARY KEY,
        envelope_json TEXT NOT NULL DEFAULT '{}',
        complete INTEGER NOT NULL DEFAULT 0,
        captured_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evals (
        id TEXT PRIMARY KEY,
        build_id TEXT,
        suite TEXT NOT NULL,
        scores_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        id TEXT PRIMARY KEY,
        run_id TEXT,
        kind TEXT NOT NULL,
        path TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL DEFAULT (datetime('now')),
        type TEXT NOT NULL,
        entity_type TEXT,
        entity_id TEXT,
        payload_json TEXT NOT NULL DEFAULT '{}',
        workspace_id TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_events_workspace_ts
        ON events(workspace_id, ts DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_events_entity
        ON events(entity_type, entity_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_events_type_ts
        ON events(type, ts DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_conversation_seq
        ON messages(conversation_id, seq)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_runs_status_started
        ON runs(status, started_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_adapters_build
        ON adapters(build_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_conversations_workspace
        ON conversations(workspace_id, updated_at DESC)
    """,
]


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply schema v1 then additive v2 spine tables. Always ends at version 2.

    After v2 tables exist, run a one-shot spine backfill of models_local and
    on-disk chats (schema_meta key ``spine_backfill_v1``). Backfill errors are
    logged to stderr and never fail startup.
    """
    for stmt in SCHEMA_V1:
        conn.execute(stmt)
    for stmt in SCHEMA_V2:
        conn.execute(stmt)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', '2')"
    )

    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'spine_backfill_v1'"
    ).fetchone()
    if row is None:
        try:
            # Backfill helpers call get_conn(); bind this connection first so
            # we do not re-enter _migrate via a second connect.
            _local.conn = conn
            from . import spine_migrate

            spine_migrate.run_all()
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) "
                "VALUES ('spine_backfill_v1', 'done')"
            )
        except Exception as exc:
            print(
                f"[optiq.lab] spine backfill skipped: {exc}",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Small typed accessors used everywhere
# ---------------------------------------------------------------------------


def credentials_exist() -> bool:
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM credentials WHERE id = 1").fetchone()
    return row is not None


def reset_for_tests() -> None:
    """Drop and rebuild — for test fixtures, not for production use."""
    conn = get_conn()
    conn.executescript(
        """
        DROP TABLE IF EXISTS events;
        DROP TABLE IF EXISTS artifacts;
        DROP TABLE IF EXISTS evals;
        DROP TABLE IF EXISTS message_provenance;
        DROP TABLE IF EXISTS messages;
        DROP TABLE IF EXISTS conversations;
        DROP TABLE IF EXISTS runs;
        DROP TABLE IF EXISTS datasets;
        DROP TABLE IF EXISTS adapters;
        DROP TABLE IF EXISTS builds;
        DROP TABLE IF EXISTS workspaces;
        DROP TABLE IF EXISTS schema_meta;
        DROP TABLE IF EXISTS credentials;
        DROP TABLE IF EXISTS hf_tokens;
        DROP TABLE IF EXISTS jobs;
        DROP TABLE IF EXISTS models_local;
        """
    )
    _migrate(conn)
