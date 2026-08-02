# Gap Board — OptiQ Lab

**Date:** 2026-08-02  
**Master plan:** `docs/plans/2026-08-02-gap-mitigation-plan.md`  
**Rule:** Status is only 🟢 when verification evidence exists. No estimates.

---

## Legend

| Status | Meaning |
|--------|---------|
| 🟢 Closed | Verification passed |
| 🟡 Partial | Plumbing or subset only |
| 🔴 Open | Not mitigated for users |
| ⬜ Not started | Future phase |

---

## Board

| Gap ID | Gap (user impact) | Tier | WP | Phase | Status | Exit metric / verification |
|--------|-------------------|------|-----|-------|--------|----------------------------|
| C3 | Demo looks like live product | P0 | WP-0 | Now | 🟢 | Prototype banner + disabled controls + README |
| G1 | Loads crash/thrash without warning | P0 | WP-1A | 1 | 🟢 | Fit Engine + 409 on load; tests green |
| G7 | Three ways to load model | P0 | WP-1C | 1 | 🟡 | Primary path Models/Fit; Server advanced remains |
| — | Machine invisible | P0 | WP-1B | 1 | 🟢 | Strip + /api/machine real RAM/probes |
| G5 | Jobs fight for RAM/model | P0 | WP-1 + bus | 0–1 | 🟡 | Sequential bus backend 🟢; UI admission 🔴 |
| G6 | Lose multi-hour jobs in UI | P0 | WP-2D | 0–2 | 🟡 | Bus 🟢; global Runs UI 🔴 |
| G3 | Tabs ≠ real work loop | P0 | WP-1–2 | 1–2 | 🔴 | Work/Models/Runs shell live |
| G4 | Can’t prove what made an answer | P0 | WP-2B | 0–2 | 🟡 | Spine/partial capture 🟢; full UI+complete 🔴 |
| G8 | Silent “model is dumb” | P0 | WP-2B | 2 | 🔴 | Health chips for truncation/retrieval/thinking/… |
| G2 | Moat (measurement) invisible | P0 | WP-3 | 3 | 🔴 | Eval GUI + real scores vs CLI identity |
| T3 | No “my tests” / promote gate | P0 | WP-3B/C | 3 | 🔴 | BYO set + promote blocked on regression |
| G9 | Global settings only | P1 | WP-2A | 2 | 🟡 | Schema 🟢; workspace UX 🔴 |
| T4 | Workspace lies about model | P1 | WP-2A | 2 | 🟡 | Coherence banner when not resident |
| G10 | Chat behind Msty/Open WebUI | P1 | WP-2C | 2 | 🔴 | Edit/resend, branch, search, regen |
| C1 | Model discovery UX lag | P1 | WP-1C | 1 | 🔴 | Unified Models surface |
| C2 | Slow path to first useful token | P1 | WP-4F | 4 | 🔴 | Documented + measured clean-machine path |
| G11 | Best knobs CLI-only | P1 | WP-3E | 3–4 | 🔴 | 5 crown jewels have GUI homes |
| G12 | Remote/auth friction | P1 | WP-4C | 4 | 🔴 | Auth-safe non-localhost |
| T5 | No guided optimize / recipes | P2 | WP-4A/B | 4 | ⬜ | Recipe round-trip; guided flow |
| T6 | Integration copy-paste only | P2 | WP-4D | 4 | ⬜ | Probed health only |
| T7 | Ship path = pip worktree | P1 eng | WP-5 | Cont. | 🟡 | CI + merge strategy written |

---

## Phase rollup

| Phase | Intent | Gap focus | Foundation |
|-------|--------|-----------|------------|
| **0** | Spine | Data, jobs, events, dual-write, partial provenance | **Complete** (`07-phase0-complete.md`) — 43 tests |
| **1** | Machine & Models | G1, G7, strip, Models, C1 | WP-1 |
| **2** | Conversation | G3/4/8/9/10, Runs UI, T2/T4 | WP-2 |
| **3** | Measurement | G2, T1/T3, G11 eval side | WP-3 |
| **4** | Reach | G12, T5/T6, C2, recipes | WP-4 |

---

## P0 open count

| | Count |
|--|------:|
| P0 gaps total | 11 (C3, G1–G8, G2, T3 — G5/G6/G4 partial) |
| P0 fully closed | 0 (user-visible) |
| P0 partial (eng only) | G4, G5, G6 |

---

## Next action (single)

**Start WP-0** (demo hygiene) in parallel with **WP-1A Fit Engine** design/implementation on `mlx-optiq-dev`.  
Do not claim Phase 1 done until Fit + strip + single Load pass verification on a real Mac.
