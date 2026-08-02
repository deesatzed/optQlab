# Gap Mitigation Plan — OptiQ Lab

**Date:** 2026-08-02  
**Status:** Active plan (executable)  
**Audience:** Product owner + engineering  
**Sources:** UX redesign §5–§6, Stage 2 discovery, Phase 0 spine complete (`mlx-optiq-dev`), non‑SWE gap analysis  

**Policy:** No calendar/time estimates. No mock/demo data in shipped product paths. Each mitigation has verification (pass/fail). Real measurement only.

---

## 0. Purpose

Close every **important** product gap so OptiQ Lab becomes what the thesis claims:

> Know if it fits → load it → use it → measure it against *my* work → improve it → prove what produced every answer — on one Mac.

This plan maps **gap → mitigation → work package → verification → dependency**, in execution order.

---

## 1. Current baseline (honest)

| Layer | State |
|-------|--------|
| Engine (quantize, serve, eval CLI, tools, research) | Real (mlx-optiq / Lab 0.4.7-class) |
| Shipping Lab UI | Flask tabs — powerful, fragmented (G1–G12 open) |
| Redesign prototype | High-fidelity **demo** IA (not live backend) |
| Phase 0 spine | **Done** in `mlx-optiq-dev`: schema v2, events, sequential job bus, dual-write chats, partial provenance, APIs — **no user-visible redesign** |
| Multi-model | **Sequential only** (locked) |
| Workspace coherence | **Schema-ready**; UI not shipped |

**Implication:** Mitigations below assume Phase 0 is the foundation. Do not re-build spine unless verification fails.

---

## 2. Importance tiers

| Tier | Meaning | Must mitigate? |
|------|---------|----------------|
| **P0** | Product thesis fails or non‑SWE trust fails without it | Yes — blocking “product” |
| **P1** | Loses demos/deals to LM Studio/Msty/Open WebUI despite better engine | Yes — near-term |
| **P2** | Category-defining / reach | Yes — after P0/P1 |
| **P3** | Nice / ops | Track; not blocking core thesis |

All **G1–G12** and the five non‑SWE box features are **P0 or P1**.

---

## 3. Master gap register

| ID | Gap (short) | Tier | Redesign | Phase 0 status | Target phase |
|----|-------------|------|----------|----------------|--------------|
| G1 | No fit prediction | P0 | A4, N1 | Open | **1** |
| G2 | Measurement severed from use | P0 | A6, N8–N10 | Open | **3** |
| G3 | Tab IA ≠ workflow | P0 | §6.3 | Open | **1–2** |
| G4 | No product provenance | P0 | A5, N11 | Plumbing partial | **2** (UI + complete capture) |
| G5 | No admission control for RAM/model | P0 | A2 | Sequential bus **backend** | **1** (surface + memory-aware later) |
| G6 | Blocking / lost jobs | P0 | N3 | Bus + events | **1–2** (Runs UI) |
| G7 | Triple load paths | P0 | N4 | Open | **1** |
| G8 | Silent failures | P0 | N5 | Partial tools only | **2** |
| G9 | Global config | P1 | N6 | Schema | **2** |
| G10 | Chat table stakes | P1 | N7 | Open | **2** |
| G11 | Crown jewels CLI-only | P1 | N10, W* | Open | **3–4** |
| G12 | Auth / remote friction | P1 | W3 | Open | **4** |
| C1 | Download/discover UX lag | P1 | competitive | Open | **1** |
| C2 | Install → first token | P1 | metrics | Open | **1 + packaging** |
| C3 | Demo honesty (prototype vs product) | P0 GTM | forensics | Open | **Immediate** |
| T1 | Moat not in GUI (eval) | P0 | =G2 | Open | **3** |
| T2 | Non‑SWE “why this answer” | P0 | =G4+G8 | Partial | **2** |
| T3 | BYO eval + promote gate | P0 | N9, W4 | Schema shell | **3** |
| T4 | Workspace coherence UX | P1 | §6.7.2 | Schema | **2** |
| T5 | Guided optimize / recipes | P2 | W1–W2 | Open | **4** |
| T6 | Integration health | P2 | N13 | Open | **4** |
| T7 | Upstream ship path | P1 eng | — | pip worktree | **Continuous** |

---

## 4. Locked decisions (do not reopen without written change)

1. **Multi-model:** sequential residency for Phase 1–2; dual-resident only after explicit Fit-aware policy.  
2. **Workspace:** coherent ownership (model, prompt, sampler, tools, files, eval set); incoherence is loud.  
3. **No fake metrics** in production UI (no seed Capability Scores as truth).  
4. **Phase 0 spine** is the data/job substrate; UI layers on top.

---

## 5. Mitigation program — work packages

Execution order is dependency-safe. Each package lists **gaps closed**, **deliverables**, **verification**, **DO NOT**.

---

### WP-0 — Immediate: truth & packaging hygiene (start now)

**Closes:** C3, parts of GTM trust  
**Depends on:** nothing  

| # | Action | Verification |
|---|--------|--------------|
| 0.1 | Label redesign prototype UI “Sample data — not live Lab” (banner or README) | Visual on open; no bare Capability Score without label |
| 0.2 | Disable or toast dead prototype controls (Send, Export, Promote) | Click does not imply live action |
| 0.3 | Root README in `optiqlab` + `mlx-optiq-dev`: what is prototype vs spine vs shipping Lab | New contributor can answer in 2 minutes |
| 0.4 | Document: editable install points at `mlx-optiq-dev` | `python -c "import optiq; print(optiq.__file__)"` documented |

**DO NOT:** Demo prototype scores as measured results.

**Exit:** Stakeholder can no longer confuse demo with product.

---

### WP-1 — Phase 1: Machine & Models (Fit, one load path, visibility)

**Closes:** G1, G7, G5 (visible), C1 (Models surface), parts of C2  
**Depends on:** Phase 0 spine (builds, runs, events)  

#### 1A — Real Fit Engine (not prototype formula)

| Deliverable | Detail |
|-------------|--------|
| Fit service | `weights + kv(ctx, bits) + activation headroom + overhead` vs free unified memory |
| Apple cliffs | compressed-memory threshold; MTLResource-style hard-fail priors |
| Calibration | one-time measure pass on machine; store under `~/.optiq/lab/` |
| API | `POST /api/fit/predict` → verdict + breakdown + what-if |
| UI | Load dialog: budget bar + 4 verdicts (comfortable / degraded / will not fit / hard-fail) |

**Verification:**
- [ ] Unit tests with real arithmetic fixtures (published ceiling table as constants)  
- [ ] On a real Mac: load blocked when verdict is will-not-fit / hard-fail  
- [ ] Never invent Capability Score inside Fit  

**DO NOT:** Ship `computeFit` toy formula as calibrated Fit.

#### 1B — Machine strip + panel (truthful)

| Deliverable | Detail |
|-------------|--------|
| Machine state API | loaded build, RAM used/free (real OS read), ports, adapter, running run count |
| UI strip | always visible in Lab shell (Flask incremental or new SPA shell — pick one path in 1D) |
| Health probes | TCP/HTTP to lab + serve ports; show down/up without lying “healthy” when down |

**Verification:**
- [ ] Kill serve process → strip shows unhealthy within one poll interval  
- [ ] RAM numbers move when a model loads (real, not seed)  

#### 1C — Unified Models surface + single load path

| Deliverable | Detail |
|-------------|--------|
| Models list | hub published + local builds + HF search (existing backend, one UI) |
| Build card | lineage, fit badge (from 1A), actions: Load / Eval / Fine-tune / Quantize |
| Kill triple load | remove or redirect Settings→Server load and Hub-only load to **one** Load dialog |

**Verification:**
- [ ] Only one primary “Load” affordance in IA  
- [ ] Quantize completion registers a `builds` row (spine) and appears on Models without manual re-entry  

#### 1D — Shell decision (must pick one)

| Option | When |
|--------|------|
| **A — Incremental Flask shell** | Faster to ship strip + Models; higher long-term cost |
| **B — New SPA shell** (redesign A3) | Aligns with Phase 2–3; more upfront |

**Recommendation:** **B** if Phase 2 starts immediately after; **A** only if a Fit+strip hotfix must land before SPA.

**Exit WP-1:** Non‑SWE can open Lab, see machine state, pass Fit, load once, find the model in one place.

---

### WP-2 — Phase 2: Conversation, workspaces, health, provenance UI

**Closes:** G3 (Work), G4, G8, G9, G10, T2, T4, G6 (Runs UX)  
**Depends on:** WP-1 shell or interim Work route; Phase 0 chat dual-write + provenance  

#### 2A — Work surface (modes, not tabs)

| Deliverable | Detail |
|-------------|--------|
| Modes | Chat · Research · Compare (fork thread across two builds **sequentially** if dual-resident not allowed) |
| Right rail | workspace-scoped prompt, sampler, tools, files, eval set |
| Dual-write | all saves go through spine (already dual-write; make UI only path) |

**Verification:**
- [ ] Switch workspace changes system prompt/files without global bleed  
- [ ] Coherence flag when workspace default build ≠ resident build (visible banner)  

#### 2B — Run health + provenance productization

| Deliverable | Detail |
|-------------|--------|
| Capture | context_used/window, healed, retry, tools_called, retrieval empty, thinking used/budget, build_id, sampler |
| UI chips | under each assistant message |
| Inspector | full envelope; Export JSON works (real file download) |
| Completeness | target: required fields non-null for local tool-path streams; null only when unmeasured |

**Verification:**
- [ ] Empty retrieval shows warning chip (real RAG miss)  
- [ ] Export JSON validates against envelope schema  
- [ ] No fabricated tok/s  

#### 2C — Chat table stakes

| Deliverable | Detail |
|-------------|--------|
| Edit & resend, regenerate | |
| Branch / fork | |
| Search chats | over spine conversations |
| Stop | already exists; keep |

**Verification:**
- [ ] Feature checklist vs Msty baseline: edit, regen, search present  
- [ ] Automated UI or API tests for branch parent_id linkage  

#### 2D — Runs list UI (global)

| Deliverable | Detail |
|-------------|--------|
| Subscribe | `/api/bus/stream` or poll events + runs table |
| Actions | open log, cancel, artifacts |
| Survive navigation | leave Quantize-equivalent action, job still visible under Runs |

**Verification:**
- [ ] Start quantize, navigate to Work, Runs still shows progress  
- [ ] Cancel from Runs terminates process (Phase 0 cancel)  

#### 2E — Deep Research as first-class mode

| Deliverable | Detail |
|-------------|--------|
| Trace UI | live stages from real events |
| Report library | artifacts table + download |

**Verification:**
- [ ] Cancel at stage boundary works  
- [ ] Report artifact path exists on disk after run  

**Exit WP-2:** User can work in workspaces, diagnose bad answers, leave long jobs, and match baseline chat UX.

---

### WP-3 — Phase 3: Measurement (thesis delivery)

**Closes:** G2, T1, T3, G11 (eval-related), knob consequences  
**Depends on:** Builds as first-class (WP-1), Runs (WP-2)  

#### 3A — Eval as a Run type

| Deliverable | Detail |
|-------------|--------|
| Wire | `optiq eval` (smoketest + full suite) as job_bus kind `eval` |
| Store | `evals` rows linked to `build_id` with real scores_json |
| UI | Run eval from Build card; progress in Runs |

**Verification:**
- [ ] Scores in DB match CLI output for same build (identity check on one machine)  
- [ ] Zero scores invented by frontend  

#### 3B — BYO eval set

| Deliverable | Detail |
|-------------|--------|
| Import | prompt JSONL / lift from conversation  
| Run | score build against user set  
| Diff | two builds, delta column  

**Verification:**
- [ ] User set of N prompts produces N results stored  
- [ ] Diff UI uses stored scores only  

#### 3C — Regression / promote gate

| Deliverable | Detail |
|-------------|--------|
| Baseline | pin eval result on build  
| Promote | disabled if delta &lt; 0 on primary metric or user set  

**Verification:**
- [ ] Artificial worse score blocks promote  
- [ ] Better score allows promote  

#### 3D — Knob consequence preview

| Deliverable | Detail |
|-------------|--------|
| Predictions | ΔGB / Δtok-s / Δcapability estimate for KV bits, BPW, ctx (use Fit + sensitivity priors; label **estimate** until calibrated) |
| UI | next to knobs in Load / Build settings |

**Verification:**
- [ ] UI labels “estimate” when not measured  
- [ ] Fit GB delta consistent with Fit Engine  

#### 3E — Surface CLI crown jewels (minimum set)

| Feature | GUI home |
|---------|----------|
| KV cache bits | Build / Load knobs + consequence |
| MTP / drafter | Build advanced + machine strip indicator |
| Adapter stack | Build card + load |
| Idle timeout | Machine panel |
| Context scale | Workspace / Load |

**Verification:**
- [ ] Each has a GUI path; no “docs only” for these five  

**Exit WP-3:** Capability Score and *my* evals drive load/promote decisions; moat is visible.

---

### WP-4 — Phase 4: Reach & category

**Closes:** G12, T5, T6, remaining G11, C2 polish  
**Depends on:** WP-1–3  

| Package | Deliverable | Verification |
|---------|-------------|--------------|
| 4A Recipes | Export/import full stack file (model+quant+KV+adapter+sampler+tools+prompt) | Round-trip file → identical workspace config |
| 4B Guided optimize | “I have X GB and need Y” → proposal + Fit + smoketest run | Completes without CLI |
| 4C Remote auth | Non-localhost bind + session/auth suitable for LAN/Tailscale | Password required; no open LAN without auth |
| 4D Integration health | Claude Code/Codex last request / error rate if observable; else connection probe | Never show “connected” without probe |
| 4E Cluster view | Thunderbolt/multi-node status if engine supports | Hidden if single node |
| 4F Install path | Documented first-token path; reduce steps where possible | Measured checklist for clean machine |

**Exit WP-4:** Non‑SWE remote use + shareable recipes + honest integration status.

---

### WP-5 — Engineering continuous (parallel)

**Closes:** T7  

| Action | Verification |
|--------|--------------|
| Maintain `mlx-optiq-dev` as ship vehicle or merge to upstream git when available | CI: `pytest tests/lab` 100% |
| No edit site-packages | Import path under worktree or official package |
| Claim gate before any “production ready” language | verification --mode=claim style checklist |

---

## 6. Dependency graph (execution order)

```
WP-0 (hygiene)
   │
   ▼
WP-1 Fit + Machine + Models + shell
   │
   ├──────────────► WP-2 Work + Health + Runs UI + table stakes
   │                      │
   │                      ▼
   │                 WP-3 Measurement (thesis)
   │                      │
   └──────────────────────┴──► WP-4 Reach
                                      │
WP-5 continuous ──────────────────────┘
```

**Do not start WP-3 UI before builds + runs are real (WP-1/2).**  
**Do not start WP-4 remote before auth model is fixed (G12).**

---

## 7. Success metrics (product, not calendar)

| Metric | Baseline (today) | Target when mitigations done |
|--------|------------------|------------------------------|
| Fit prediction before load | None | 100% of loads go through Fit; block on hard-fail |
| Unpredicted OOM/swap from Lab load | Unmeasured | Zero unpredicted (Fit said comfortable/degraded only) |
| Assistant messages with complete provenance (required fields) | ~0% product | 100% on local tool stream path |
| Users who can run eval without CLI | CLI-only minority | Primary path is GUI |
| Quantize → chat with same build | 12+ clicks / 3 tabs | ≤4 actions, no tab archaeology |
| Long job abandoned because UI lost it | Common fear | Runs always reachable |
| Demo confusion (prototype as live) | High risk | WP-0 done |
| Install → first useful token | ~heavy | Documented minimal path; measured on clean Mac |

**Rule:** If a metric is not measured, status is **UNKNOWN**, not green.

---

## 8. Gap → mitigation quick index

| Gap | Mitigation package |
|-----|-------------------|
| G1 | WP-1A |
| G2 | WP-3A–D |
| G3 | WP-1C, WP-2A |
| G4 | WP-2B (+ Phase 0 spine) |
| G5 | WP-1B + bus; dual-model policy later |
| G6 | WP-2D (+ Phase 0 bus) |
| G7 | WP-1C |
| G8 | WP-2B |
| G9 | WP-2A, T4 |
| G10 | WP-2C |
| G11 | WP-3E, WP-4 |
| G12 | WP-4C |
| C1 | WP-1C |
| C2 | WP-4F + packaging |
| C3 | WP-0 |
| T1–T3 | WP-3 |
| T4 | WP-2A |
| T5–T6 | WP-4 |
| T7 | WP-5 |

---

## 9. Operator checklist (mitigation backlog)

Copy into issue tracker; check only when verification passes.

### WP-0
- [ ] 0.1 Prototype sample-data labeling  
- [ ] 0.2 Dead control disarm  
- [ ] 0.3 READMEs (product map)  
- [ ] 0.4 Editable install documented  

### WP-1
- [ ] 1A Fit Engine + tests + API  
- [ ] 1A Load dialog wired to real Fit  
- [ ] 1B Machine API (real RAM + port probes)  
- [ ] 1B Machine strip UI  
- [ ] 1C Unified Models + single Load  
- [ ] 1C Quantize → Build registration path  
- [ ] 1D Shell path chosen and recorded  

### WP-2
- [ ] 2A Work modes + workspace rail  
- [ ] 2A Coherence banner  
- [ ] 2B Run health chips (full taxonomy minimum set)  
- [ ] 2B Provenance inspector + export  
- [ ] 2C Edit/resend, regen, branch, search  
- [ ] 2D Global Runs UI + cancel  
- [ ] 2E Research mode productized  

### WP-3
- [ ] 3A Eval job + real scores in DB  
- [ ] 3B BYO eval set  
- [ ] 3C Promote gate  
- [ ] 3D Knob consequence (labeled estimates)  
- [ ] 3E Crown-jewel GUI homes (5)  

### WP-4
- [ ] 4A Recipes  
- [ ] 4B Guided optimize  
- [ ] 4C Remote auth  
- [ ] 4D Integration health (probed)  
- [ ] 4E Cluster (or hide)  
- [ ] 4F First-token path doc + measure  

### WP-5
- [ ] CI green on spine tests  
- [ ] Upstream/merge strategy written  

---

## 10. Risk register (mitigation risks)

| Risk | Mitigation of the mitigation |
|------|------------------------------|
| Fit estimates wrong → false confidence | Conservative verdicts; hard-fail priors; calibration pass required |
| SPA rewrite stalls Fit hotfix | WP-1D option A interim strip only |
| Eval GUI shows wrong scores | Identity check vs CLI; no FE-side score math |
| Sequential Compare feels broken | Explicit queue UX: “Model B runs after A” |
| Scope explosion into P2 early | Gate WP-4 until WP-3 exit metrics move |

---

## 11. Related documents

| Doc | Role |
|-----|------|
| `docs/build/08-gap-board.md` | Gap → phase → metric board (one page) |
| `docs/build/09-gaps-nontechnical.md` | Non‑SWE language gaps |
| `docs/plans/2026-08-02-phase-0-spine.md` | Phase 0 (done) |
| `docs/plans/2026-08-02-phase-0-spine-design.md` | Spine design |
| `docs/build/07-phase0-complete.md` | Phase 0 evidence |
| `OptiQ Lab interactive prototype/uploads/optiq-lab-ux-redesign.md` | Product thesis + G1–G12 |

---

## 12. Definition of “important gaps mitigated”

All of the following are true with **evidence**:

1. Fit blocks unsafe loads (G1).  
2. Eval + BYO eval run from GUI; scores real (G2).  
3. One Models/Load path; machine strip truthful (G7, strip).  
4. Provenance + run health on answers; export works (G4, G8).  
5. Runs survive navigation (G6).  
6. Workspaces coherent and visible (G9).  
7. Chat table stakes present (G10).  
8. Prototype cannot be mistaken for live metrics (C3).  
9. No “production ready” claim while WP-1–3 open.

Until then: product status remains **workbench-in-progress**, not finished OptiQ Lab.
