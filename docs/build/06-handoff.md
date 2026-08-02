# Handoff Packet — OptiQ Lab interactive prototype

**Command:** `docs-context --mode=handoff`  
**Date:** 2026-08-02  
**Session:** `/build` Stage 2 onboard (role=SWE, checkpoints=stage defaults)

---

## Snapshot

| Field | Value |
|-------|--------|
| Date | 2026-08-02 |
| Branch | **n/a** (no git) |
| Commit | **n/a** |
| Env | local design prototype |
| Status | 🟡 Yellow |

**Why not 🟢:** no tests, intentional demo stubs, no backend, open product decisions, CDN dependency unvalidated offline.  
**Why not 🔴:** artifact is coherent; IA matches redesign brief; chain discovery complete with evidence.

---

## What I Was Doing

- **Goal:** Stage 2 onboard — truth map of this workspace + debt baseline + assumption registry + handoff  
- **Progress:** Steps 1–6 of planned chain complete (artifacts under `docs/build/`)  
- **Last verified working state:** Code and docs read-only analyzed; prototype **not** browser-smoke-tested in this session (A-0004 remains unvalidated)

---

## What's In Flight

| Area | State |
|------|--------|
| Staged | n/a (no git) |
| Unstaged | New: `docs/build/*`, `.governance/assumptions.json` |
| WIP | None in prototype HTML/JS |

---

## Blockers

| Blocker | Owner / unblock path |
|---------|----------------------|
| Production mlx-optiq Lab repo path unknown | User provides path or clone location → validate A-0005 |
| Phase 0 stack decision (dc-runtime vs React SPA A3) | User/product → validate A-0006 |
| Multi-model residency decision | User/product → validate A-0007 |
| Workspace coherence rules | User/product → validate A-0008 |

---

## Risks & Unknowns

| Risk | Evidence | Mitigation |
|------|----------|------------|
| Seed metrics treated as real | Forensics D-003 / A-0002 | Demo labels; never productize seeds |
| Empty Send confuses demos | D-001 | Disable or toast |
| Fit mistaken for calibrated engine | A-0003 | Rename / watermark |
| Offline open fails | A-0004 unvalidated | Vendor React; run offline test |
| Scope = full product rebuild | D-002 | Work only against redesign phasing in real repo |

---

## Key Files Reference

| Role | Path |
|------|------|
| Entry | `OptiQ Lab interactive prototype/OptiQ Lab.dc.html` |
| Runtime | `OptiQ Lab interactive prototype/support.js` |
| Design tokens | `OptiQ Lab interactive prototype/_ds/organic-…/styles.css` |
| Product spec | `OptiQ Lab interactive prototype/uploads/optiq-lab-ux-redesign.md` |
| Project load | `docs/build/01-project-load.md` |
| Deepdive | `docs/build/02-discovery-deepdive.md` |
| Forensics | `docs/build/03-discovery-forensics.md` |
| Debt | `docs/build/04-governance-debt.md` |
| Assumptions (human) | `docs/build/05-governance-assumptions.md` |
| Assumptions (machine) | `.governance/assumptions.json` |
| This handoff | `docs/build/06-handoff.md` |

---

## Next Actions (Prioritized)

### P0
1. **Decide product path** — prototype polish only vs start Phase 0 in mlx-optiq (answers A-0006).  
2. **Resolve open redesign decisions** — multi-model residency (A-0007), workspace coherence (A-0008).  
3. **Honest-demo pass** — sample-data watermark; disable dead primary actions (Send, Export, New build).  
4. **`git init` + baseline** if this tree will be edited further.  

### P1
5. Root **README**: what this is / is not / how to serve.  
6. **Offline React** check (A-0004) + vendor if needed.  
7. Point next session at **production Lab repo** for real wiring.  

### P2
8. Minimal smoke tests for `computeFit` boundaries.  
9. Trim or complete Organic DS export.  
10. Stage 3 (enhance) only if treating prototype as product UI scaffold; else Stage 1/2 planning against mlx-optiq.

---

## Open Questions / Decisions Needed

1. Where is the live mlx-optiq / OptiQ Lab source of truth?  
2. Is the next build slice **prototype fidelity** or **Phase 0 spine in production**?  
3. Confirm multi-model vs sequential queue (redesign §6.7).  
4. How far does “workspace” ownership go (redesign §6.7)?  

---

## Verification Commands

```bash
# Inventory
find "/Volumes/WS4TB/optiqlab" -type f ! -name '.DS_Store' | sort

# Open prototype (network required for React unless vendored)
cd "/Volumes/WS4TB/optiqlab/OptiQ Lab interactive prototype"
python3 -m http.server 8765
# → http://127.0.0.1:8765/OptiQ%20Lab.dc.html

# Confirm demo stub still empty
rg -n "sendMessage" "OptiQ Lab.dc.html"

# Confirm no backend
rg -n "create_app|FastAPI|fetch\\(|EventSource" "OptiQ Lab interactive prototype" || true

# Assumptions
cat "/Volumes/WS4TB/optiqlab/.governance/assumptions.json" | head -40

# Spec
wc -l "uploads/optiq-lab-ux-redesign.md"
```

---

## Chain outputs summary

| Step | Command | Artifact | One-line result |
|------|---------|----------|-----------------|
| 1 | project-load | `01-project-load.md` | Prototype + Organic DS + redesign MD; no git/backend/tests |
| 2 | deepdive | `02-discovery-deepdive.md` | Maturity 3/10 product / 7/10 design; IA screens present |
| 3 | forensics | `03-discovery-forensics.md` | 0 live backend; demo/synth/dead cataloged |
| 4 | debt | `04-governance-debt.md` | 18 items; 4 critical (honesty + spine + CDN + scores) |
| 5 | assumptions | `05-…` + `.governance/assumptions.json` | 12 tracked; 6 unvalidated incl. stack decisions |
| 6 | handoff | `06-handoff.md` | This packet |

---

## Next suggested stage

- **If continuing design fidelity:** Stage 3 UX repair on prototype (labels, dead controls) or Stage 5 handoff-only.  
- **If building the real Lab:** Stage 1/2 against **mlx-optiq production repo** with redesign MD as spec → `planning --mode=plan` for Phase 0.  
- **Do not** claim production readiness from this workspace.

---

```json
{
  "mode": "handoff",
  "branch": null,
  "commit": null,
  "status": "yellow",
  "blockers": 4,
  "next_actions": [
    "product-path-decision",
    "redesign-6.7-decisions",
    "honest-demo-pass",
    "git-init-baseline",
    "readme",
    "locate-mlx-optiq"
  ],
  "unknowns": 4,
  "artifacts": [
    "docs/build/01-project-load.md",
    "docs/build/02-discovery-deepdive.md",
    "docs/build/03-discovery-forensics.md",
    "docs/build/04-governance-debt.md",
    "docs/build/05-governance-assumptions.md",
    "docs/build/06-handoff.md",
    ".governance/assumptions.json"
  ]
}
```
