# Discovery Forensics — Hostile-Committee Mapping

**Command:** `discovery --mode=forensics`  
**Date:** 2026-08-02  
**Role:** SWE | Stage 2 onboard chain step 3/6

**Classification legend**

| Label | Meaning |
|-------|---------|
| **PURE DEMO** | Hardcoded UI narrative; no real computation beyond presentation |
| **SYNTH** | Client-side synthetic logic (formulas, filters) over seed data — not live systems |
| **LIVE BACKEND** | Real server/API/process interaction |
| **HYBRID** | Mix of live + synth/demo |
| **DEAD/ORPHAN** | Visible or semi-visible control with no handler / no path |

---

## 1. Executive Summary

OptiQ Lab in this workspace is a **design-host interactive prototype**. There is **zero LIVE BACKEND** surface. Everything user-visible is either:

- **PURE DEMO** seed content (builds, chats, runs, health tags), or  
- **SYNTH** UI state machines over that seed (view routing, Fit calculator, filters, load selection), or  
- **DEAD/ORPHAN** chrome (Send, Export, Hub search, ⌘K, New build, ↓.md).

| Question | Answer |
|----------|--------|
| What is the system? | Single-page dc-runtime React prototype of redesigned Lab IA |
| Runtime entrypoints | `OptiQ Lab.dc.html` + `support.js` (+ unpkg React) |
| Production-like? | **No** — looks production; data is sample |
| Top risks | Fake metrics as truth; empty Send in demos; CDN React; confusion with real mlx-optiq Lab |
| Committee posture | Treat as **Figma-with-logic**, not as Lab vNext |

### Top 10 risks

1. Capability Scores / tok/s presented as measured (seed)
2. Port “healthy” and integration “connected” with no probes
3. Fit verdicts presented as real Fit Engine (simplified SYNTH)
4. Send button appears functional, is empty
5. Stakeholders confuse this folder with production Lab
6. Offline open fails without unpkg
7. Research mode shows completed TTD-DR with no job system
8. Promote button in eval compare has no promote pipeline
9. Workspace list not stateful (static HTML)
10. No provenance export despite button

### Top 10 hostile-committee questions

1. Which numbers on screen were ever measured on a Mac? **None in this repo.**
2. Can I chat with a model from this UI? **No.**
3. Does Load start mlx_lm.server? **No — only flips seed `machineLoadedId`.**
4. Is Fit calibrated? **No — formula + thresholds only.**
5. Where is the job bus? **Not here; redesign Phase 0.**
6. Is SQLite present? **No.**
7. What happens when I press Send? **Nothing.**
8. Are eval deltas real? **Seed scores; own-eval delta hardcoded +2.**
9. Why do ports show healthy? **Static tags.**
10. What’s the path to production? **Redesign MD phasing 0–4 — not started in this workspace.**

---

## 2. System Map

### Component inventory

| Component | Type | Present |
|-----------|------|---------|
| Frontend app | dc-runtime SPA-in-one-file | Yes |
| Backend services | Flask / FastAPI Lab | **No** |
| Workers / job bus | multiprocessing / WS | **No** |
| Database | SQLite / files | **No** |
| Auth | password / Fernet | **No** |
| Third-party APIs | HF Hub, search | **No** (UI only) |
| CDN deps | React, ReactDOM, Babel | Yes (unpkg) |
| Design system | Organic CSS | Yes |
| Spec document | UX redesign MD | Yes |

### Runtime topology

```
Browser
  ├─ OptiQ Lab.dc.html
  ├─ support.js (dc-runtime)
  │    └─ unpkg React 18.3.1 + ReactDOM (+ Babel optional)
  ├─ _ds/.../styles.css
  └─ in-memory seed: BUILDS, RUNS, MESSAGES, MACHINE
```

No localhost :7860 / :8080 processes are started by this prototype.

### Primary user flows (3–7)

1. **Browse Work chat** → inspect run-health → open provenance  
2. **Switch Research / Compare modes** → view static / seed-driven panels  
3. **Edit workspace rail** (prompt, temperature) → local state only  
4. **Models → Load** → adjust ctx/KV → Fit verdict → Confirm Load  
5. **Models → Details** → metrics bars → Load or open Eval compare  
6. **Runs → filter → Compare** on completed eval  
7. **Machine strip expand** → view seed memory breakdown / job list

---

## 3. Repo Navigation Guide

| Goal | Start here |
|------|------------|
| See the product vision | `uploads/optiq-lab-ux-redesign.md` |
| Understand UI chrome | `OptiQ Lab.dc.html` L25–532 |
| Trace behavior | `OptiQ Lab.dc.html` L537–794 (`DCLogic`) |
| Understand runtime | `support.js` (boot, CDN, compile) |
| Tokens / components | `_ds/.../styles.css`, `readme.md` |
| Prior chain context | `docs/build/01-project-load.md`, `02-discovery-deepdive.md` |

**Run:** `python3 -m http.server` from prototype directory; needs network for React.

---

## 4. Frontend UI-to-Behavior Catalog

| UI Location | UI Element | Purpose | Trigger | Classification | Evidence | Backend Mapping | Failure Mode | Test Coverage | Committee Note |
|-------------|------------|---------|---------|----------------|----------|-----------------|--------------|---------------|----------------|
| Nav rail | Work / Models / Runs buttons | Switch primary view | click | **SYNTH** | `goWork/goModels/goRuns` | none | none | none | Real SPA routing over one document |
| Nav rail | Workspace rows (clinical-eval, latency-bench) | Show workspaces | display | **PURE DEMO** | static HTML L52–59 | none | cannot switch | none | Not in `state` |
| Nav rail | ⌘K “Jump to anything” | Command palette | display | **DEAD/ORPHAN** | no handler | none | does nothing | none | Affordance only |
| Machine strip | Strip bar | Show resident model / RAM | click toggle | **HYBRID→SYNTH** | seed MACHINE + `machineLoadedId` | none | RAM bar from `sizeGB/totalRam` | none | Looks live |
| Machine panel | Port health tags | Endpoint status | display | **PURE DEMO** | hardcoded “healthy” L119–120 | none | always healthy | none | Hostile hot seat |
| Machine panel | Integrations Claude/Codex | Connection status | display | **PURE DEMO** | static tags | none | fake connected | none | |
| Machine panel | Active jobs list | Show running | display | **PURE DEMO** | seed RUNS filter | none | static progress | none | |
| Work header | Mode segmented control | Chat/Research/Compare | change | **SYNTH** | `setMode*` | none | none | none | |
| Chat | Message list | Conversation | render | **PURE DEMO** | `MESSAGES` const | none | fixed thread | none | |
| Chat | Run-health chips | Failure taxonomy | click → provenance | **PURE DEMO** data + **SYNTH** open | seed msg fields + `openProvenance` | none | no real health | none | Core differentiator as theater |
| Chat | Draft textarea | Compose | change | **SYNTH** | `draft` state | none | not sent | none | |
| Chat | Send button | Send | click | **DEAD/ORPHAN** | `sendMessage=()=>{}` | none | silent no-op | none | **Must disclose in demos** |
| Research | Trace card + report | Deep Research | display | **PURE DEMO** | static HTML L185–206 | none | no cancel/start | none | TTD-DR theater |
| Research | ↓ .md | Export report | click | **DEAD/ORPHAN** | no handler | none | nothing | none | |
| Compare | Side-by-side | Arena mode | select B | **HYBRID→SYNTH** | seed builds + `arenaBuildB` | none | static left text | none | Not dual-load |
| Right rail | System prompt | Workspace config | change | **SYNTH** | state only | none | not applied to model | none | |
| Right rail | Temperature slider | Sampler | change | **SYNTH** | state only | none | no generation | none | |
| Right rail | Tools / files / eval tags | Policy display | display | **PURE DEMO** | static tags | none | cannot toggle tools | none | |
| Models | Search input | Hub search | type | **DEAD/ORPHAN** | unbound input | none | no filter | none | |
| Models | + New build | Create build | click | **DEAD/ORPHAN** | no handler | none | nothing | none | |
| Models | Build cards | Lifecycle | click | **PURE DEMO** data + **SYNTH** open | BUILDS + handlers | none | fake scores | none | |
| Models | Capability Score big number | Differentiation | display | **PURE DEMO** | seed `overall` | none | not measured | none | **Highest deception risk** |
| Models | Load button | Open fit dialog | click | **SYNTH** | `openLoad` | none | no server | none | |
| Load dialog | Ctx / KV controls | Fit knobs | change | **SYNTH** | `setLoadCtx/KvBits` | none | | none | Good UX demo |
| Load dialog | Fit budget bar + verdict | Fit Engine stand-in | compute | **SYNTH** | `computeFit` | none | not calibrated | none | Not A4 engine |
| Load dialog | Load model CTA | Commit load | click | **SYNTH** | `confirmLoad` sets id | none | no process load | none | Label carefully |
| Build detail | Metrics bars | Six-metric suite | display | **PURE DEMO** | seed metrics | none | | none | |
| Build detail | Eval / Fine-tune / Publish | Actions | partial | **DEAD/ORPHAN** (except Load, open eval) | no full handlers | none | | none | Spot-check each button |
| Eval compare | A/B selects + table | Diff builds | change | **SYNTH** over seed | `evalRows` | none | deltas from seed | none | |
| Eval compare | Own eval +2 | BYO set | display | **PURE DEMO** | `evalOwnDelta = 2` | none | always +2 | none | |
| Eval compare | Promote | Gate promote | click | **DEAD/ORPHAN** | disabled only; no promote | none | | none | |
| Runs | Table + filters | Job history | filter/click | **SYNTH** filter + **PURE DEMO** rows | RUNS + filters | none | | none | |
| Provenance | Drawer fields | Envelope | open | **PURE DEMO** | seed msg fields | none | Export dead | none | |
| Provenance | Export JSON | Export | click | **DEAD/ORPHAN** | no handler | none | | none | |
| Global | React runtime | Render | boot | **LIVE** (CDN) | unpkg fetch | CDN | offline fail | none | Only true network live path |

**Backend mapping summary:** every domain feature maps to **∅**. Only external live path is CDN script load.

---

## 5. Backend Service & API Inventory

| Category | Finding |
|----------|---------|
| HTTP routes | None in workspace |
| WebSocket / SSE | None |
| Domain services | None |
| Jobs / queues | None |
| Integrations | Display-only tags |

**Claim:** No backend.  
**Evidence:** file inventory; no Python/TS server; no fetch to Lab ports in component script.  
**Confidence:** high.

---

## 6. Data & State: Where Truth Lives

| Store | Contents | Durable? |
|-------|----------|----------|
| `BUILDS` const | 4 builds + metrics | No (reload resets) |
| `RUNS` const | 6 jobs | No |
| `MESSAGES` const | 4 messages | No |
| `MACHINE` const | RAM/ports/adapter | No |
| `Component.state` | view, modes, dialogs, draft, prompt, temp | Session only |
| Browser localStorage | unused | — |
| SQLite / files | absent | — |

**Object contracts (seed shapes):** Build `{id,name,source,quantProfile,bpw,weightsGB,sizeGB,adapters,metrics[],fit,overall,tokPerSec}`; Run `{id,type,target,status,progress?,started,duration}`; Message assistant fields include ctx/healed/thinking/tools/retrieval for provenance theater.

---

## 7. Synth / Demo / Live Classification Report

### Counts (UI components cataloged)

| Class | Count (approx) | Controls |
|-------|----------------|----------|
| PURE DEMO | 18 | Seed content, static tags, research panel, capability numbers |
| SYNTH | 14 | Navigation, filters, fit math, load selection, local form state |
| LIVE BACKEND | 0 | — |
| HYBRID | 0 true hybrid | (CDN React is infrastructure, not product hybrid) |
| DEAD/ORPHAN | 9 | Send, Export, ⌘K, Hub search, New build, ↓.md, Promote, tool toggles, most action buttons |

**Controls / feature flags:** none. Environment switches: none. Demo is the only mode.

---

## 8. Drift Report

| Drift type | Finding |
|------------|---------|
| Spec vs prototype | Redesign A1–A6 (spine, job bus, React SPA product, real Fit Engine, real provenance, eval service) **not implemented** — only visual IA |
| Duplicated load paths | Redesign called out Hub/Settings/Quantize triple load; prototype has **single** Load dialog (good IA direction, still non-live) |
| Abandoned deps | N/A |
| Alternative versions | Production Lab described in MD only; not in tree |
| Manifest vs filesystem | `_ds_manifest.json` references `components/*.html`, `templates/*` — **those files are not in this export** (partial DS bundle) |

---

## 9. Risk Register & Mitigations

| Risk | Impact | Likelihood | Evidence | Mitigation | Quick verify |
|------|--------|------------|----------|------------|--------------|
| Fake Capability Scores | High | High | seed overall | Watermark “Sample data” | View source BUILDS |
| Empty Send in live demo | Medium | High | L616 | Disable button or toast “prototype” | Click Send |
| Fit as real engine | High | Med | computeFit | Rename “illustrative fit” | Change ctx to 64k |
| Port healthy lie | Medium | High | static tags | Remove or mark simulated | No process on :8080 |
| CDN offline break | Medium | High | unpkg URLs | Vendor React | Disconnect network |
| Scope = full product | High | Med | missing Phase 0 | Plan against phasing | Search for sqlite |
| DS incomplete export | Low | Med | missing component HTML | Accept CSS-only | ls _ds |

---

## 10. Demo Script + Committee Q&A Cards

### 10–15 min demo flow

1. **Open** Work → show clinical-eval workspace framing (30s)  
2. **Chat** → click run-health chips → Provenance drawer (2 min) — *disclose sample data*  
3. **Research mode** → show TTD-DR trace UI concept (1 min) — *disclose static*  
4. **Compare mode** → switch build B (1 min)  
5. **Machine strip** expand → memory breakdown concept (1 min)  
6. **Models** → open Load on b4 will-not-fit → tweak knobs → show verdicts (3 min) — *disclose illustrative*  
7. **Build detail** → metrics + Eval compare deltas (2 min)  
8. **Runs** → filter Eval → Compare (1 min)  
9. **Close** with redesign thesis: loop not tabs (1 min)

### Q&A cards

| Q | A |
|---|---|
| What is this? | Interactive IA prototype for redesigned OptiQ Lab |
| Why not tabs? | Redesign thesis: Work/Models/Runs + Machine strip |
| Is inference live? | No |
| Is Fit real? | Illustrative client formula only |
| Where’s the Lab backend? | mlx-optiq (not this folder); described in redesign MD |
| What breaks if we remove seed? | All domain content disappears |
| Next to productionize? | Phase 0 spine per redesign §6.5 |

---

```json
{
  "mode": "forensics",
  "components_total": 41,
  "classification": {
    "pure_demo": 18,
    "synth": 14,
    "live": 0,
    "hybrid": 0,
    "dead": 9
  },
  "top_risks": [
    "seed-capability-scores-as-truth",
    "empty-send-button",
    "illustrative-fit-as-engine",
    "static-port-healthy",
    "cdn-react-offline",
    "scope-confusion-with-production-lab"
  ],
  "committee_hot_seats": [
    "Which metrics were measured?",
    "What does Send do?",
    "Does Load touch mlx_lm?",
    "Where is job bus / SQLite?",
    "Why are ports healthy?"
  ],
  "confidence": "high"
}
```
