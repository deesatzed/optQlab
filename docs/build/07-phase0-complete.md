# Phase 0 Complete — OptiQ Lab Spine

**Date:** 2026-08-02  
**Repo:** `/Volumes/WS4TB/optiqlab/mlx-optiq-dev`  
**Status:** 🟢 Green — full lab suite 100% pass

**Next (gaps):** User-visible product gaps are tracked in `docs/plans/2026-08-02-gap-mitigation-plan.md` and `docs/build/08-gap-board.md`. Phase 0 closes spine only — not Fit/eval UI/thesis.

---

## What Phase 0 delivered

Phase 0 established the **Lab spine**: durable SQLite domain model, dual-write chat, sequential job admission, append-only events, provenance capture/export, and migrate/backfill from existing Lab data. No UI redesign; backend-only foundation for later phases.

| Area | Delivery |
|------|----------|
| Schema v2 | Additive spine tables (`workspaces`, `builds`, `adapters`, `datasets`, `runs`, `conversations`, `messages`, `message_provenance`, `evals`, `artifacts`, `events`) on top of v1 (`jobs`, `models_local`, credentials, etc.) |
| Events | Append-only event log with `after_id` cursor resume |
| Domain repos | Workspace / build / conversation / provenance APIs (`optiq.lab.spine`) |
| Chat dual-write | JSON file + spine DB; migrate-on-read for file-only chats |
| Job bus | Sequential `memory_heavy` admission; light can overlap; cancel; lifecycle events; dual-write to `runs` |
| Spine API | Workspaces CRUD, events list, bus SSE stream, provenance export |
| Provenance | Partial capture from chat stream; never fakes `tok_per_sec` / `peak_mem_gb` |
| Backfill | `models_local` → builds; on-disk `chat_*.json` → conversations (one-shot meta key) |
| Acceptance | End-to-end suite under real SQLite (`tests/lab/test_phase0_acceptance.py`) |

---

## How to run tests

```bash
cd /Volumes/WS4TB/optiqlab/mlx-optiq-dev
/Users/o2satz/miniforge3/envs/py313/bin/pytest tests/lab -v
```

**Verified result:** `43 passed` (100%).

Acceptance-only:

```bash
/Users/o2satz/miniforge3/envs/py313/bin/pytest tests/lab/test_phase0_acceptance.py -v
```

Fixtures use real SQLite under a temp `OPTIQ_HOME` (`tests/lab/conftest.py`). No mock DB.

---

## Key modules

| Module | Role |
|--------|------|
| `optiq/lab/db.py` | Schema v1+v2 migrate, connection, one-shot backfill hook |
| `optiq/lab/events.py` | Append-only events + `iter_after` |
| `optiq/lab/spine.py` | Workspace, build, conversation, provenance repositories |
| `optiq/lab/spine_migrate.py` | Backfill builds/chats into spine |
| `optiq/lab/chat_store.py` | Dual-write chat file + DB, migrate-on-read |
| `optiq/lab/job_bus.py` | Sequential job admission, cancel, lifecycle events |
| `optiq/lab/jobs.py` | Process jobs table + `mark_zombies` |
| `optiq/lab/provenance_capture.py` | Partial stream provenance (honest nulls for unmeasured metrics) |
| `optiq/lab/routes/spine_api.py` | HTTP: workspaces, events, bus stream, provenance |
| `optiq/lab/routes/chat.py` | Chat routes wired to dual-write + provenance apply |
| `tests/lab/test_phase0_acceptance.py` | Phase 0 acceptance gate |

---

## git log summary (Phase 0)

```
eef1915 feat(lab): backfill builds and chats into spine
d23c0d7 feat(lab): partial provenance capture from chat stream context
d8a1f59 feat(lab): spine API for events bus and provenance export
f7a97fd feat(lab): sequential job bus with events and cancel
e682eae feat(lab): dual-write chat store with migrate-on-read
54569f7 feat(lab): spine repositories for workspace/build/conversation/provenance
3f6e67c feat(lab): append-only events repository
8cf0e5f feat(lab): schema v2 spine tables for Phase 0
```

Acceptance commit (this task):

```
4b92768 test(lab): Phase 0 acceptance suite
```

Bootstrap baseline: `806e0e0 Bootstrap editable worktree from mlx-optiq 0.4.7`

---

## UI unchanged confirmation

**Confirmed:** Phase 0 commits touch **no** Lab UI presentation assets.

- No changes under `optiq/lab/templates/`
- No changes under `optiq/lab/static/` (CSS, JS, fonts, images)
- No HTML/CSS redesign work

Backend-only paths: `optiq/lab/*.py`, route modules, and `tests/lab/*`. Chat route and app wiring are API-level, not template/static.

---

## Acceptance coverage

| # | Check | Test |
|---|-------|------|
| 1 | Schema v2 on fresh home | `test_schema_v2_on_fresh_home` |
| 2 | Schema v2 preserves jobs/models across reconnect | `test_schema_v2_preserves_data_across_reconnect` |
| 3 | Dual-write chat round-trip | `test_dual_write_chat_round_trip` |
| 4 | Provenance export (spine + API) | `test_provenance_export_round_trip` |
| 5 | Two heavies never both running | `test_sequential_two_heavies_never_both_running` |
| 6 | Events `after_id` resume | `test_events_after_id_resume` |
| 7 | Workspace coherence `model_not_resident` | `test_workspace_coherence_model_not_resident` |
| 8 | `mark_zombies` still works | `test_mark_zombies_still_works` |

---

## Next: Phase 1 — Machine & Models

Phase 1 should build on the spine for **Machine & Models** product surface:

1. Build registry UX backed by `builds` + `models_local` backfill
2. Resident model / workspace coherence flags in the real UI
3. Provenance completeness gaps filled where measurable (still no faked metrics)
4. Wire job bus status into Lab job panels (quantize / dataset / finetune already dual-path capable)

Do **not** claim product completeness until Phase 1+ UI and machine-facing flows land.

---

## Status

| Field | Value |
|-------|--------|
| Status | 🟢 **Green** |
| Lab tests | **43 passed**, 0 failed |
| UI | Unchanged (templates/static) |
| Mock DB | None — real SQLite under `lab_home` |
| Blockers for Phase 0 close | None |

```json
{
  "phase": 0,
  "status": "green",
  "tests_passed": 43,
  "tests_failed": 0,
  "ui_changed": false,
  "handoff": "docs/build/07-phase0-complete.md",
  "next": "Phase 1 Machine & Models"
}
```
