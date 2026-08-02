"""Schema v2 spine tables — Phase 0."""

from optiq.lab import db


SPINE_TABLES = [
    "workspaces",
    "builds",
    "adapters",
    "datasets",
    "runs",
    "conversations",
    "messages",
    "message_provenance",
    "evals",
    "artifacts",
    "events",
]

V1_TABLES = [
    "schema_meta",
    "credentials",
    "hf_tokens",
    "jobs",
    "models_local",
]


def _table_names(conn) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r["name"] if isinstance(r, dict) or hasattr(r, "keys") else r[0] for r in rows}


def test_schema_v2_creates_spine_tables(lab_home):
    conn = db.get_conn()
    names = _table_names(conn)
    for table in SPINE_TABLES:
        assert table in names, f"missing spine table: {table}"

    version = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'version'"
    ).fetchone()
    assert version is not None
    assert version["value"] == "2"


def test_schema_v2_preserves_v1_tables(lab_home):
    conn = db.get_conn()
    names = _table_names(conn)
    for table in V1_TABLES:
        assert table in names, f"v1 table missing after v2 migrate: {table}"
