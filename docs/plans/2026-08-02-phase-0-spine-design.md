# Phase 0 — Spine Design

**Date:** 2026-08-02  
**Status:** Ready for implementation plan  
**Spec source:** `OptiQ Lab interactive prototype/uploads/optiq-lab-ux-redesign.md` §6.2 A1/A2/A5, §6.5 Phase 0, §6.7  
**Runtime evidence:** installed `mlx-optiq==0.4.7` (`optiq.lab.*` under site-packages)

---

## 1. Outcome

Ship a **durable Lab spine** under `~/.optiq/lab/lab.db` such that:

1. First-class domain objects exist in SQLite (not only JSON chat files + thin `jobs` / `models_local` tables).  
2. Every state change appends an immutable **event**.  
3. Long work runs through a **sequential job bus** with admission control (at most one memory-heavy job at a time; single model residency).  
4. Every assistant message can carry a complete **provenance envelope** (stored, exportable as JSON).  
5. **No user-visible UI redesign** — existing Flask tabs and routes keep working; dual-write and additive APIs only.

Success is proven by **automated tests + a real DB on disk**, not demos.

---

## 2. Locked decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Multi-model residency (§6.7.1) | **Sequential only** | Phase 0 job bus never co-resides two models; Arena/dual-eval remain sequential. |
| Workspace (§6.7.2) | **Coherent ownership in schema** | Workspace owns default build, system prompt, sampler, tools policy, file refs, eval set; incoherence is an event, not silent. |
| Code land | **Editable worktree bootstrapped from pip install** | No upstream git checkout available; extract `optiq` package into `/Volumes/WS4TB/optiqlab/mlx-optiq-dev` and `pip install -e`. |
| Transport for bus | **SSE multiplex first** (`/api/bus/stream`) + events table | Matches current Flask stack; redesign “WebSocket” is Phase 1 transport upgrade if needed. Per-job SSE remains for compatibility. |
| UI | **No template/IA change** | Capture + APIs only. |

---

## 3. Current Lab spine (baseline)

| Concern | Today (0.4.7) | Evidence |
|---------|---------------|----------|
| DB | SQLite WAL `~/.optiq/lab/lab.db` | `optiq/lab/db.py`, live tables |
| Schema v1 | `schema_meta`, `credentials`, `hf_tokens`, `jobs`, `models_local` | live `.schema` |
| Jobs | `multiprocessing.Process` + JSONL log + status row | `optiq/lab/jobs.py` |
| Job kinds | `quantize`, `finetune`, dataset | routes `submit()` |
| Job stream | Per-job SSE `/api/jobs/<id>/stream` | `routes/quantize.py` |
| Chats | JSON files under `chats/` — no provenance | `routes/chat.py` `save_chat` |
| Models | `models_local` path/bpw only — not full Build | `db.py` |
| Cancel job process | No first-class cancel on job bus | jobs.py end |
| Event log | None | — |
| Workspace | None | — |

Phase 0 **extends** this; it does not replace Flask or mlx_lm.server.

---

## 4. Target architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Flask Lab (unchanged tabs)                                  │
│  chat / quantize / finetune / dataset / hub / …             │
└───────────────┬─────────────────────────────┬───────────────┘
                │ dual-write / submit          │ subscribe (new, optional)
                ▼                              ▼
┌──────────────────────────┐     ┌────────────────────────────┐
│ Spine repository         │     │ JobBus                     │
│  workspaces, builds,     │     │  SequentialAdmission       │
│  conversations, messages,│     │  Process workers           │
│  provenance, runs,       │     │  emit → events + job log   │
│  artifacts, events       │     └────────────────────────────┘
└────────────┬─────────────┘
             ▼
      ~/.optiq/lab/lab.db  (+ jobs/*.log kept for back-compat)
```

### 4.1 Schema v2 (additive)

New tables (names fixed for Phase 0):

| Table | Purpose |
|-------|---------|
| `workspaces` | Coherent unit of work |
| `builds` | Model + quant profile + KV defaults + path lineage (supersedes *usage* of bare `models_local` for new code; keep `models_local` for back-compat reads) |
| `adapters` | Adapter artifacts linked to a build |
| `datasets` | Dataset artifacts |
| `runs` | Logical run record (1:1 with `jobs.id` initially; richer fields) |
| `conversations` | Chat thread metadata (workspace_id, title, mode) |
| `messages` | Ordered turns |
| `message_provenance` | 1:1 with assistant message; JSON envelope |
| `evals` | Eval result rows linked to build (shell for Phase 3; schema only) |
| `artifacts` | File paths / blobs metadata for run outputs |
| `events` | Append-only event log |

**Workspace columns (coherent ownership):**

- `id`, `name`, `created_at`, `updated_at`  
- `default_build_id` (nullable FK → builds)  
- `system_prompt` TEXT  
- `sampler_json` TEXT (temp, top_p, …)  
- `tools_policy_json` TEXT  
- `attached_files_json` TEXT (paths / doc ids)  
- `eval_set_id` TEXT nullable  
- `coherence_flags_json` TEXT (e.g. `{"model_not_resident": true}`)

**Build columns (minimum):**

- `id`, `name`, `source_hf_id`, `path`, `quant_profile`, `bpw`, `weights_gb`, `kv_bits_default`, `ctx_default`, `adapter_stack_json`, `created_at`, `metadata_json`

**Message provenance envelope (A5) — required keys when present:**

```json
{
  "build_id": "bld_…",
  "quant_profile": "4-bit mixed",
  "adapter_stack": ["clinical-v3"],
  "kv_bits": 8,
  "sampler": {"temperature": 0.7, "max_tokens": 1024},
  "context_used": 6200,
  "context_window": 32768,
  "tools_enabled": ["python", "web_search"],
  "tools_called": [{"name": "python", "healed": false}],
  "retrieved_chunk_ids": [],
  "healed": false,
  "retry_hits": 0,
  "thinking_used": 0,
  "thinking_budget": 0,
  "tok_per_sec": null,
  "peak_mem_gb": null,
  "server_model_label": "…",
  "captured_at": "ISO-8601"
}
```

Missing fields store `null`; completeness is measurable (`provenance_complete` boolean computed from required set for Phase 0: `build_id` OR `server_model_label`, `sampler`, `context_window`, `captured_at`).

**Events row:**

- `id` INTEGER PK AUTOINCREMENT  
- `ts` TEXT DEFAULT datetime('now')  
- `type` TEXT (e.g. `job.started`, `job.progress`, `job.done`, `message.created`, `workspace.coherence`, `build.registered`)  
- `entity_type` TEXT  
- `entity_id` TEXT  
- `payload_json` TEXT  
- `workspace_id` TEXT nullable  

Index: `(type, ts)`, `(entity_type, entity_id)`, `(workspace_id, ts)`.

Schema version key: `schema_meta.version = '2'`.

### 4.2 Job bus (A2, sequential)

Keep `jobs` table + log files. Add:

1. **`JobBus.submit`** — wrapper over current `jobs.submit` with:
   - `resource_class`: `memory_heavy` | `light`  
   - Sequential policy: reject or queue if another `memory_heavy` job is `queued`/`running`  
   - Always write `events` on start/progress/done/fail  
2. **`JobBus.cancel(job_id)`** — terminate process if tracked; mark `cancelled`; emit event  
3. **Process registry** — in-memory map `job_id → Process` in parent (lost on restart → zombies as today, plus event)  
4. **`runs` row** created at submit (kind, workspace_id, build_id, status mirror)

**Resource classes (Phase 0):**

| Kind | Class |
|------|-------|
| quantize, finetune, dataset (LLM gen), eval (future) | `memory_heavy` |
| bookkeeping, export | `light` |

Deep research / chat streams stay **request-scoped** (not job-bus) in Phase 0 — only background multiproc jobs use the bus. Provenance still attaches to messages.

### 4.3 Dual-write paths (no UI break)

| Existing API | Phase 0 behavior |
|--------------|------------------|
| `POST /api/chats` | Write JSON file **and** upsert `conversations` + replace `messages` (+ optional provenance if body includes `provenance` per assistant message) |
| `GET /api/chats`, `GET /api/chats/<id>` | Prefer DB if conversation exists; else file (migrate-on-read) |
| `jobs.submit` callers | Call `JobBus.submit` (same return job_id) |
| `/api/jobs/<id>/stream` | Unchanged; still tails log |
| **New** `GET /api/bus/stream` | SSE of `events` since `?after_id=` (UI not required to use yet) |
| **New** `GET /api/messages/<id>/provenance` | Export envelope JSON |
| **New** `POST /api/workspaces` etc. | CRUD minimal JSON API (no templates) |

### 4.4 Provenance capture points

1. **Client-supplied** on `POST /api/chats` when messages include `provenance` object (forward-compatible with future React UI).  
2. **Server-side on tool chat stream end** — when `/api/chat/stream` finishes, if chat_id provided, attach partial envelope from known server fields (model label, tools called/healed from orchestrator events).  
3. Never invent Capability Scores or fake tok/s.

### 4.5 Migration of existing data

On schema v2 apply:

1. Create tables.  
2. Optionally scan `chats_dir` and import files not already in `conversations` (id = file stem).  
3. For each `models_local` row, insert a `builds` row if path not present.  
4. Do **not** delete old tables or files.

---

## 5. Non-goals (Phase 0)

- React SPA / redesign IA (Phase 1–2)  
- Fit Engine A4  
- Eval GUI / BYO sets (schema shell only)  
- Multi-model concurrent residency  
- WebSocket transport  
- Changing Flask tab templates or CSS  
- Mock/demo seed data in production DB  

---

## 6. Testing contract

All tests use a temp `OPTIQ_HOME` and real SQLite files.

| Suite | Must pass |
|-------|-----------|
| Schema | migrate v1→v2 on empty and on v1-populated DB |
| Events | every job lifecycle emits ordered events; after_id resume |
| Admission | second memory_heavy while one running → queued or rejected (document chosen: **queue**) |
| Cancel | running light stub job cancelled → status cancelled + event |
| Dual-write chats | save_chat creates conversation+messages; GET returns same content |
| Provenance | envelope round-trip export; incomplete allowed with nulls |
| Workspace coherence | setting default_build_id that is not “resident” sets flag + event (resident stub = in-process flag for tests) |
| Regression | existing job submit still returns id; mark_zombies still works |

---

## 7. Risks

| Risk | Mitigation |
|------|------------|
| No upstream git → hard to upstream patches | Worktree documents origin version 0.4.7; keep patch series small under `optiq/lab/` |
| Dual-write drift JSON vs DB | migrate-on-read; DB wins after first write |
| Sequential queue starves | FIFO queue; light jobs never block |
| Provenance incomplete from server alone | allow nulls; measure completeness % in tests |

---

## 8. Approval

Decisions locked with user:

- Sequential multi-model for Phase 0  
- Coherent workspace schema  
- Pip-installed package → local editable worktree  

Implementation plan: `docs/plans/2026-08-02-phase-0-spine.md`
