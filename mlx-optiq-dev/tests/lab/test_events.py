"""Append-only event log repository — Phase 0 spine."""

from optiq.lab import events


def test_append_and_list_events(lab_home):
    id1 = events.append(
        type="run.started",
        entity_type="run",
        entity_id="run_abc",
        payload={"kind": "quantize"},
        workspace_id="ws_1",
    )
    id2 = events.append(
        type="run.progress",
        entity_type="run",
        entity_id="run_abc",
        payload={"progress": 0.5},
        workspace_id="ws_1",
    )

    assert isinstance(id1, int)
    assert isinstance(id2, int)
    assert id2 > id1

    rows = events.iter_after(after_id=0)
    assert len(rows) == 2
    assert rows[0]["id"] == id1
    assert rows[0]["type"] == "run.started"
    assert rows[0]["entity_type"] == "run"
    assert rows[0]["entity_id"] == "run_abc"
    assert rows[0]["payload"] == {"kind": "quantize"}
    assert rows[0]["workspace_id"] == "ws_1"
    assert rows[0]["ts"]  # non-empty timestamp from DB default

    assert rows[1]["id"] == id2
    assert rows[1]["type"] == "run.progress"
    assert rows[1]["payload"] == {"progress": 0.5}

    # Cursor: only events after id1
    after = events.iter_after(after_id=id1)
    assert len(after) == 1
    assert after[0]["id"] == id2

    # Limit
    limited = events.iter_after(after_id=0, limit=1)
    assert len(limited) == 1
    assert limited[0]["id"] == id1


def test_iter_after_empty(lab_home):
    rows = events.iter_after()
    assert rows == []

    rows = events.iter_after(after_id=999)
    assert rows == []


def test_append_null_payload(lab_home):
    eid = events.append(
        type="workspace.created",
        entity_type="workspace",
        entity_id="ws_x",
        payload=None,
    )
    assert isinstance(eid, int)

    rows = events.iter_after(after_id=0)
    assert len(rows) == 1
    assert rows[0]["id"] == eid
    assert rows[0]["payload"] == {}
    assert rows[0]["workspace_id"] is None
    assert rows[0]["type"] == "workspace.created"
    assert rows[0]["entity_type"] == "workspace"
    assert rows[0]["entity_id"] == "ws_x"
