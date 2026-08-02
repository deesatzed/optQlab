# Phase 0 Spine Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend installed mlx-optiq Lab (0.4.7) with SQLite domain objects, append-only events, sequential job bus, and message provenance — with **no Flask UI redesign** — verified by real SQLite tests under a temp `OPTIQ_HOME`.

**Architecture:** Bootstrap an editable package worktree from the pip install; migrate `lab.db` to schema v2 (workspaces, builds, conversations, messages, provenance, runs, artifacts, events); dual-write chats; wrap `jobs.submit` with sequential admission + event emit; expose bus SSE and provenance export APIs. Sequential residency only.

**Tech Stack:** Python 3.13 (conda `py313`), SQLite WAL, Flask, multiprocessing jobs, pytest, existing `optiq.lab.*` layout.

**Design:** `docs/plans/2026-08-02-phase-0-spine-design.md`  
**Baseline package:** `/Users/o2satz/miniforge3/envs/py313/lib/python3.13/site-packages/optiq` (mlx-optiq==0.4.7)  
**Worktree root:** `/Volumes/WS4TB/optiqlab/mlx-optiq-dev`

**Policy:** No mocks, no fake metrics, no calendar/time estimates on steps. Every task ends with a verification command and expected outcome.

---

## Task 0: Bootstrap editable worktree from pip install

**Files:**
- Create: `mlx-optiq-dev/pyproject.toml`
- Create: `mlx-optiq-dev/README.dev.md`
- Create: `mlx-optiq-dev/optiq/` (copy from site-packages)
- Create: `mlx-optiq-dev/tests/lab/conftest.py`
- Create: `mlx-optiq-dev/.gitignore`

**Step 1: Copy package sources**

```bash
SRC=/Users/o2satz/miniforge3/envs/py313/lib/python3.13/site-packages
DEST=/Volumes/WS4TB/optiqlab/mlx-optiq-dev
mkdir -p "$DEST"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$SRC/optiq/" "$DEST/optiq/"
# minimal package metadata for editable install
```

**Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "mlx-optiq"
version = "0.4.7+phase0dev"
description = "Editable worktree for OptiQ Lab Phase 0 spine (bootstrapped from pip 0.4.7)"
requires-python = ">=3.11"
dependencies = [
  "mlx>=0.20",
  "mlx-lm>=0.31.3",
  "transformers<5.13",
  "numpy",
  "scipy",
  "huggingface-hub",
  "lm-format-enforcer>=0.10",
  "pillow>=10",
  "textual>=0.60",
  "openai>=1.0",
  "ddgs>=9.0",
  "html2text>=2024.0",
  "flask",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov"]

[project.scripts]
optiq = "optiq.cli:cli"

[tool.setuptools.packages.find]
where = ["."]
include = ["optiq*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 3: `conftest.py` with real temp home**

```python
# tests/lab/conftest.py
import os
import pytest

@pytest.fixture()
def lab_home(tmp_path, monkeypatch):
    home = tmp_path / "optiq_home"
    home.mkdir()
    monkeypatch.setenv("OPTIQ_HOME", str(home))
    # Reset thread-local DB if already imported
    from optiq.lab import db
    if getattr(db._local, "conn", None) is not None:
        try:
            db._local.conn.close()
        except Exception:
            pass
        db._local.conn = None
    yield home
    if getattr(db._local, "conn", None) is not None:
        try:
            db._local.conn.close()
        except Exception:
            pass
        db._local.conn = None
```

**Step 4: Install editable + smoke import**

```bash
cd /Volumes/WS4TB/optiqlab/mlx-optiq-dev
/Users/o2satz/miniforge3/envs/py313/bin/pip install -e ".[dev]"
/Users/o2satz/miniforge3/envs/py313/bin/python -c "import optiq; print(optiq.__file__)"
```

Expected: path under `mlx-optiq-dev/optiq/...`

**Step 5: Git init worktree**

```bash
cd /Volumes/WS4TB/optiqlab/mlx-optiq-dev
git init
printf '%s\n' '__pycache__/' '*.pyc' '.pytest_cache/' '*.egg-info/' 'dist/' 'build/' > .gitignore
git add -A
git commit -m "chore: bootstrap editable mlx-optiq 0.4.7 worktree for Phase 0"
```

---

## Task 1: Schema v2 migration (tables only)

**Files:**
- Modify: `mlx-optiq-dev/optiq/lab/db.py`
- Create: `mlx-optiq-dev/tests/lab/test_schema_v2.py`

**Step 1: Failing test — version becomes 2 and tables exist**

```python
# tests/lab/test_schema_v2.py
from optiq.lab import db
from optiq.lab.config import ensure_lab_dirs

def test_schema_v2_creates_spine_tables(lab_home):
    ensure_lab_dirs()
    conn = db.get_conn()
    ver = conn.execute(
        "SELECT value FROM schema_meta WHERE key='version'"
    ).fetchone()["value"]
    assert ver == "2"
    for table in (
        "workspaces", "builds", "adapters", "datasets", "runs",
        "conversations", "messages", "message_provenance",
        "evals", "artifacts", "events",
    ):
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        assert row is not None, table

def test_schema_v2_preserves_v1_tables(lab_home):
    ensure_lab_dirs()
    conn = db.get_conn()
    for table in ("credentials", "hf_tokens", "jobs", "models_local"):
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
```

**Step 2: Run test — expect FAIL** (version still `1`, tables missing)

```bash
cd /Volumes/WS4TB/optiqlab/mlx-optiq-dev
/Users/o2satz/miniforge3/envs/py313/bin/pytest tests/lab/test_schema_v2.py -v
```

**Step 3: Implement `SCHEMA_V2` + migration in `db.py`**

- Keep `SCHEMA_V1` apply.
- Add `SCHEMA_V2` list of `CREATE TABLE IF NOT EXISTS` matching design §4.1.
- `_migrate`: after v1, apply v2 if version < 2; set `version='2'`.
- Extend `reset_for_tests()` to drop new tables too.

**Workspace DDL (exact):**

```sql
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
);
```

(Remaining tables per design doc — include FKs as TEXT ids without mandatory FK enforcement if circular; use indexes.)

**Step 4: pytest PASS**

```bash
/Users/o2satz/miniforge3/envs/py313/bin/pytest tests/lab/test_schema_v2.py -v
```

**Step 5: Commit**

```bash
git add optiq/lab/db.py tests/lab/test_schema_v2.py
git commit -m "feat(lab): schema v2 spine tables for Phase 0"
```

---

## Task 2: Event log repository

**Files:**
- Create: `mlx-optiq-dev/optiq/lab/events.py`
- Create: `mlx-optiq-dev/tests/lab/test_events.py`
- Modify: `mlx-optiq-dev/optiq/lab/db.py` (if helpers needed)

**Step 1: Failing tests**

```python
from optiq.lab import events
from optiq.lab.config import ensure_lab_dirs
from optiq.lab import db

def test_append_and_list_events(lab_home):
    ensure_lab_dirs()
    db.get_conn()
    e1 = events.append(
        type="job.started",
        entity_type="job",
        entity_id="job_abc",
        payload={"kind": "quantize"},
        workspace_id=None,
    )
    e2 = events.append(
        type="job.progress",
        entity_type="job",
        entity_id="job_abc",
        payload={"progress": 0.5},
    )
    assert e2 > e1
    all_e = list(events.iter_after(0))
    assert len(all_e) == 2
    assert all_e[0]["type"] == "job.started"
    after = list(events.iter_after(e1))
    assert len(after) == 1
    assert after[0]["id"] == e2
```

**Step 2: pytest FAIL**

**Step 3: Implement `events.py`**

```python
def append(*, type: str, entity_type: str, entity_id: str,
           payload: dict | None = None, workspace_id: str | None = None) -> int:
    ...

def iter_after(after_id: int = 0, limit: int = 500) -> list[dict]:
    ...
```

Use `transaction()`; return last_insert_rowid.

**Step 4: pytest PASS → commit**

```bash
git commit -am "feat(lab): append-only events repository"
```

---

## Task 3: Domain repositories (workspace, build, conversation, message, provenance)

**Files:**
- Create: `mlx-optiq-dev/optiq/lab/spine.py` (or split `repos/*.py` — prefer single `spine.py` for Phase 0 YAGNI)
- Create: `mlx-optiq-dev/tests/lab/test_spine_repos.py`

**Step 1: Tests covering**

- `create_workspace(name, **fields) -> id`
- `set_workspace_build(ws_id, build_id, resident_build_id=None)` → if build != resident, `coherence_flags_json` includes `model_not_resident: true` and event `workspace.coherence`
- `register_build(path, source_hf_id, ...)`
- `upsert_conversation_from_chat_payload(data)` → conversation + messages
- `set_message_provenance(message_id, envelope)` + `get_provenance(message_id)`
- `provenance_complete(envelope) -> bool` for required keys

**Step 2: Implement minimal CRUD in `spine.py` using real SQL.**

**Step 3: pytest PASS → commit**

```bash
git commit -am "feat(lab): spine repositories for workspace/build/conversation/provenance"
```

---

## Task 4: Migrate-on-read + dual-write for chats

**Files:**
- Modify: `mlx-optiq-dev/optiq/lab/routes/chat.py`
- Create: `mlx-optiq-dev/tests/lab/test_chat_dual_write.py`

**Step 1: Test with Flask test client**

```python
from optiq.lab.app import create_app
from optiq.lab import db, spine
from optiq.lab.config import ensure_lab_dirs
import json

def test_save_chat_dual_writes(lab_home, monkeypatch):
    ensure_lab_dirs()
    app = create_app(secret_key=b"test-secret-key-32bytes-long!!")
    # bypass auth for test: mark credentials or patch before_request
    ...
```

**Auth bypass for tests:** Prefer injecting a test-only config flag or inserting credentials + session. If auth is hard, unit-test a extracted function `persist_chat(data) -> chat_id` called by the route (extract if needed — preferred).

**Preferred approach (cleaner):**

1. Add `optiq/lab/chat_store.py` with `save_chat_record`, `load_chat_record`, `list_chat_records`.
2. Route calls `chat_store`; tests call `chat_store` directly (no Flask auth).
3. Route thin wrapper stays.

**Step 2: `save_chat_record` must:**

1. Write JSON file (same shape as today).  
2. Upsert `conversations` + delete/reinsert `messages`.  
3. For each assistant message with `provenance` key, write `message_provenance`.  
4. Emit `message.created` / `conversation.upserted` events.

**Step 3: `load_chat_record`:** if conversation in DB, return DB view shaped like old JSON; else read file and import into DB (migrate-on-read).

**Step 4: pytest PASS → commit**

```bash
git commit -am "feat(lab): dual-write chat store with migrate-on-read"
```

---

## Task 5: Sequential JobBus + events + cancel

**Files:**
- Create: `mlx-optiq-dev/optiq/lab/job_bus.py`
- Modify: `mlx-optiq-dev/optiq/lab/jobs.py` (optional: keep low-level; bus wraps)
- Modify: `mlx-optiq-dev/optiq/lab/routes/quantize.py`, `finetune.py`, `dataset.py` to call bus
- Create: `mlx-optiq-dev/tests/lab/test_job_bus.py`

**Step 1: Tests with a tiny real target (no MLX)**

```python
import time
from optiq.lab import job_bus, db, events
from optiq.lab.config import ensure_lab_dirs

def _slow_job(emit, config):
    emit({"type": "progress", "progress": 0.1, "message": "hi"})
    time.sleep(config.get("sleep", 0.3))
    emit({"type": "progress", "progress": 1.0, "message": "done"})

def test_sequential_queues_second_heavy(lab_home):
    ensure_lab_dirs()
    db.get_conn()
    j1 = job_bus.submit("test_heavy", _slow_job, {"sleep": 0.8}, resource_class="memory_heavy")
    j2 = job_bus.submit("test_heavy", _slow_job, {"sleep": 0.1}, resource_class="memory_heavy")
    # j2 must not be running while j1 running
    row2 = db.get_conn().execute("SELECT status FROM jobs WHERE id=?", (j2,)).fetchone()
    assert row2["status"] in ("queued", "running")
    # wait for both done
    ...
    # events include job.started for both in order
    types = [e["type"] for e in events.iter_after(0) if e["entity_id"] in (j1, j2)]
    assert "job.started" in types
    assert "job.done" in types

def test_cancel_running_job(lab_home):
    ...
```

**Admission policy (locked):** second `memory_heavy` while one is queued/running → status stays **`queued`** until the previous finishes; a parent **dispatcher thread** starts the next process. Light jobs start immediately.

**Step 2: Implement `job_bus.py`**

- Process registry dict + lock  
- Dispatcher for queue  
- On emit: update jobs row (existing), append `events`, update `runs`  
- `cancel(job_id)`: `proc.terminate()`, status `cancelled`, event  

**Step 3: Wire routes**

Replace `jobs.submit(...)` with `job_bus.submit(...)` in quantize/finetune/dataset (same args + resource_class=`memory_heavy`).

**Step 4: pytest PASS → commit**

```bash
git commit -am "feat(lab): sequential job bus with events and cancel"
```

---

## Task 6: Bus SSE + provenance export routes

**Files:**
- Create: `mlx-optiq-dev/optiq/lab/routes/spine_api.py` (or add to `routes/api.py`)
- Modify: `mlx-optiq-dev/optiq/lab/app.py` register blueprint
- Create: `mlx-optiq-dev/tests/lab/test_spine_api.py`

**Endpoints (JSON/SSE only; no templates):**

| Method | Path | Behavior |
|--------|------|----------|
| GET | `/api/events?after_id=&limit=` | JSON list |
| GET | `/api/bus/stream?after_id=` | SSE `data: {event}\n\n` until client disconnect; poll events |
| GET | `/api/messages/<message_id>/provenance` | JSON envelope or 404 |
| GET/POST/PATCH | `/api/workspaces` | minimal CRUD |

Auth: same session gate as other Lab APIs (use existing auth patterns from `routes/api.py`).

**Tests:** use `chat_store` + `events` for data; Flask test client with auth bypass or credentials fixture.

**Commit:** `feat(lab): spine API for events bus and provenance export`

---

## Task 7: Server-side partial provenance on tool chat stream end

**Files:**
- Modify: `mlx-optiq-dev/optiq/lab/chat_orchestrator.py` and/or `routes/chat.py`
- Create: `mlx-optiq-dev/tests/lab/test_stream_provenance.py`

**Behavior:**

- If request body includes `chat_id` (optional new field), after stream `done`, upsert messages and attach envelope with known fields: `server_model_label`, `tools_called` (from orchestrator), `healed`/`retry_hits` if available, `sampler` from request, `captured_at`.  
- Do **not** fabricate tok/s or peak memory if not measured.  
- If `chat_id` absent, no-op (existing clients unchanged).

**Test:** unit-test a pure function `build_partial_provenance(request_ctx, tool_trace) -> dict` rather than full SSE integration.

**Commit:** `feat(lab): partial provenance capture from chat stream context`

---

## Task 8: Import existing models_local + optional chat backfill helper

**Files:**
- Create: `mlx-optiq-dev/optiq/lab/spine_migrate.py`
- Call from `db._migrate` after v2 tables created (idempotent)
- Create: `mlx-optiq-dev/tests/lab/test_spine_migrate.py`

**Behavior:**

1. For each `models_local` row → `builds` if path missing.  
2. Function `backfill_chats_from_disk()` import `chat_*.json` not in DB (called from migrate once, or CLI).  
3. Emit `build.registered` / `conversation.imported` events.

**Commit:** `feat(lab): backfill builds and chats into spine`

---

## Task 9: Full verification harness + handoff note

**Files:**
- Create: `mlx-optiq-dev/tests/lab/test_phase0_acceptance.py`
- Create: `docs/build/07-phase0-plan-ready.md` (status for parent optiqlab workspace)

**Acceptance tests (all must pass):**

1. Schema v2 on fresh home  
2. Schema v2 on DB that already had v1 tables populated  
3. Dual-write chat round-trip  
4. Provenance export round-trip  
5. Sequential bus: two heavies never both `running`  
6. Events after_id resume  
7. Workspace coherence flag when build not resident  
8. `mark_zombies` still works after bus changes  

**Run:**

```bash
cd /Volumes/WS4TB/optiqlab/mlx-optiq-dev
/Users/o2satz/miniforge3/envs/py313/bin/pytest tests/lab -v
```

Expected: **100% pass**. If any fail, do not mark Phase 0 done; open action plan for the gap.

**Commit:** `test(lab): Phase 0 acceptance suite`

**Handoff note content:** how to run tests, that UI is unchanged, next is Phase 1 Machine & Models.

---

## Anti-drift safeguards

**DO:**
- Additive migrations only  
- Real SQLite under `OPTIQ_HOME`  
- Keep JSON chat files working  
- Sequential memory_heavy policy  
- Nulls over invented metrics  

**DO NOT:**
- Change Flask tab templates / redesign IA  
- Add mock Capability Scores  
- Co-reside two models  
- Skip tests “because prototype”  
- Edit site-packages in place (only worktree)  

**Checkpoint every task:** pytest for that task green before next task.

---

## Verification harness summary

| Level | Command | Pass criteria |
|-------|---------|---------------|
| Unit | `pytest tests/lab/test_*.py -v` | all green |
| Integration | acceptance suite | all 8 scenarios |
| Manual (optional) | `OPTIQ_HOME=/tmp/p0 optiq lab` | existing tabs load; no template change |

---

## Operator checklist

- [ ] Task 0 — editable worktree + pytest import path  
- [ ] Task 1 — schema v2  
- [ ] Task 2 — events  
- [ ] Task 3 — spine repos  
- [ ] Task 4 — chat dual-write  
- [ ] Task 5 — sequential job bus  
- [ ] Task 6 — bus/provenance API  
- [ ] Task 7 — stream partial provenance  
- [ ] Task 8 — backfill  
- [ ] Task 9 — acceptance 100% + handoff note  

---

## Open items (not blocking Phase 0 start)

1. Upstream merge path when official mlx-optiq git becomes available (re-apply commits).  
2. WebSocket upgrade of `/api/bus/stream` deferred.  
3. Full Fit Engine / multi-model policy deferred to later phases.

---

## Execution options

Plan complete and saved to:

- `docs/plans/2026-08-02-phase-0-spine-design.md`  
- `docs/plans/2026-08-02-phase-0-spine.md`  

**1. Subagent-Driven (this session)** — dispatch fresh subagent per task, review between tasks  

**2. Parallel Session (separate)** — new session runs executing-plans on this file in the worktree  

**Which approach?**
