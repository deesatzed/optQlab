# Discovery Deepdive — OptiQ Lab interactive prototype

**Command:** `discovery --mode=deepdive`  
**Date:** 2026-08-02  
**Role:** SWE | Stage 2 onboard chain step 2/6

---

## 1. Executive Summary

This workspace is **not** the production mlx-optiq Lab. It is a **high-fidelity interactive design prototype** that implements the redesigned OptiQ Lab information architecture (Work / Models / Runs + Machine strip) as a single `*.dc.html` document driven by a vendored **dc-runtime** (`support.js`) and the **Organic** design system.

| Dimension | Assessment |
|-----------|------------|
| What it is | UI/UX prototype + product redesign brief |
| What it is not | Backend, package, deployable Lab, git repo |
| Maturity | **3/10** as a product codebase; **7/10** as a design artifact |
| Primary value | Spec-faithful IA mock with working client-side state + illustrative Fit calculator |
| Biggest risk | Treating seed metrics / “healthy” ports / Capability Scores as real measurements |

**Claim:** The prototype exercises navigation, load dialog Fit bar, provenance drawer, eval compare, and machine strip with live React state.  
**Evidence:** `OptiQ Lab.dc.html` lines 594–792 (`class Component extends DCLogic`, `computeFit`, `renderVals`).  
**Confidence:** high  
**Unknowns:** whether this was authored inside a proprietary design tool (dc-runtime / parent postMessage design mode) and how it will merge into mlx-optiq.

---

## 2. Repository Overview

### Structure

```
/Volumes/WS4TB/optiqlab/
├── docs/build/                          # this /build chain (created 2026-08-02)
│   ├── 01-project-load.md
│   └── 02-discovery-deepdive.md
└── OptiQ Lab interactive prototype/
    ├── OptiQ Lab.dc.html                # sole application surface
    ├── support.js                       # dc-runtime (bundled)
    ├── .thumbnail
    ├── uploads/optiq-lab-ux-redesign.md # product thesis + phasing
    └── _ds/organic-ed51b8ee-…/
        ├── styles.css                   # design tokens + components
        ├── _ds_bundle.js                # empty namespace bootstrap
        ├── _ds_manifest.json            # DS cards/templates/tokens index
        ├── _adherence.oxlintrc.json     # token-adherence lint rules
        └── readme.md
```

### Scale & languages

| Metric | Value | Evidence |
|--------|-------|----------|
| Content files | ~9 | find + listing |
| Disk | ~236 KB | `du -sh` |
| LOC (approx) | ~2,977 | `wc -l` on html/js/css |
| Languages | HTML template dialect, JS, CSS, Markdown, JSON | path inventory |
| Git | **absent** | `git status` exit 128 / no `.git` |

### Stack

| Layer | Technology | Evidence |
|-------|------------|----------|
| Shell | `x-dc` template + `sc-if` / `sc-for` control tags | HTML |
| Logic | `class Component extends DCLogic` | script block L594 |
| Runtime | dc-runtime `support.js` (GENERATED) | header comment L1 |
| UI library | React 18.3.1 + ReactDOM via unpkg (SRI) | support.js L1143–1147 |
| Optional transpile | @babel/standalone 7.29.0 | support.js L1147–1148 |
| Design | Organic tokens (Caprasimo/Figtree, terracotta/sage) | styles.css + readme |
| Data | In-memory const arrays | BUILDS/RUNS/MESSAGES/MACHINE |

---

## 3. Development History & Iterations

| Signal | Finding |
|--------|---------|
| Commits | None — no repository |
| Authors | Unknown |
| Hotspots | N/A |
| Velocity | Single-drop artifact (~2026-08-02 08:56 timestamps) |
| Branches | N/A |

**Archaeological note:** `support.js` is a monorepo-style bundle (`// GENERATED from dc-runtime/src/*.ts`). Source TS is **not** present. Parent-frame `postMessage` hooks (`__dc_design_mode`) imply this HTML was authored for a design-tool host, not primarily for bare `file://` browsing.

---

## 4. Entry Points, Config & Runtime Topology

### Boot path

1. Browser loads `OptiQ Lab.dc.html`
2. `support.js` loads; pulls React/ReactDOM (and Babel if needed) from unpkg **or** `window.__resources` blob map
3. Runtime parses `<x-dc>`, evaluates `<script type="text/x-dc" data-dc-script>`, requires `class Component extends DCLogic`
4. Renders into a host `div` via `ReactDOM.createRoot` / legacy `render`
5. Stylesheet: `_ds/.../styles.css`; empty DS namespace from `_ds_bundle.js`

### Config surface

| Config | Present? |
|--------|----------|
| Environment variables | No |
| package.json / lockfile | No |
| CLI flags | No |
| Feature flags | No |
| Secrets | No |

### Network / ports (prototype *display only*)

| Port | Role in UI copy | Wired? |
|------|-----------------|--------|
| :7860 | Lab UI | Display tag only (`machine.labPort`) |
| :8080 | mlx_lm.server | Display tag only (`machine.servePort`) |
| unpkg.com | React runtime | **Live** dependency for standalone open |

### How to run

```bash
cd "/Volumes/WS4TB/optiqlab/OptiQ Lab interactive prototype"
python3 -m http.server 8765
# open http://127.0.0.1:8765/OptiQ%20Lab.dc.html
# Requires network for unpkg unless React is pre-injected
```

---

## 5. Feature & Capability Map (what actually works)

### Working client features (stateful UI)

| Feature | Behavior | Evidence |
|---------|----------|----------|
| Nav: Work / Models / Runs | Switches `state.view` | `goWork/goModels/goRuns` |
| Machine strip expand | Toggles panel | `toggleMachinePanel` |
| Chat / Research / Compare modes | Switches `workMode` | `setMode*` |
| Run-health chips on assistant msgs | Rendered from seed fields | `msgRow` + tags |
| Provenance drawer | Opens per message; shows seed envelope | `openProvenance` |
| Workspace right rail | Editable system prompt + sampler temp in state | `setSystemPrompt`, `setSamplerTemp` |
| Models cards + Load | Opens Fit dialog | `openLoad` |
| Fit bar | Recalculates on ctx / KV knobs; can block Load | `computeFit`, `blocksLoad` |
| Confirm Load | Sets `machineLoadedId` | `confirmLoad` |
| Build detail | Lineage string, metrics, Load/Eval actions | `openBuildDetail` |
| Eval compare | Delta rows from seed metrics; Promote disabled if gate fails | `evalRows`, `evalGateBlocks` |
| Runs filter | all / eval / quantize | `setFilter*` |
| Active job badge | Count of `status==='running'` | `runningJobCount` |

### Non-working / decorative

| Feature | Reality |
|---------|---------|
| Send message | `sendMessage = () => {}` empty |
| Export JSON | Button, no handler |
| ⌘K command palette | Visual chrome only |
| Port health / integration status | Hardcoded tags (“healthy”, “connected”, “idle”) |
| Progress of running quantize | Static `progress:64` in seed |
| HF Hub search | Placeholder input, no search logic |
| Fine-tune / quantize / dataset actions | Labels only |
| Persistence | None across reload |
| Any mlx-optiq / Flask / SSE | **Absent from workspace** |

### Spec coverage (redesign six screens)

| Screen (§6.4) | Prototype coverage |
|---------------|-------------------|
| 1. Load dialog + Fit bar | **Present** (illustrative formula) |
| 2. Machine strip + panel | **Present** (seed machine) |
| 3. Conversation + right rail + run-health | **Present** (seed thread) |
| 4. Build card | **Present** (4 seed builds) |
| 5. Eval compare | **Present** (seed metrics; own-eval delta hardcoded `+2`) |
| 6. Runs list | **Present** (6 seed runs) |

Phase 0 spine (SQLite, event log, job bus, real provenance capture) is **not** in this workspace.

---

## 6. AI Breadcrumbs & Technical Debt Inventory

| Marker | Count | Notes |
|--------|-------|-------|
| `TODO` / `FIXME` / `HACK` / `XXX` | 0 in content files | clean of markers |
| Explicit demo stub | 1 | `sendMessage = () => {}; // demo composer…` |
| Hardcoded clinical/demo narrative | Throughout | seed BUILDS/MESSAGES/RUNS |
| Inline styles density | Very high | almost all layout via style= attrs |
| No module split | 1 file owns UI + domain seed + fit math | maintenance risk if grown |
| Generated runtime without source | support.js only | cannot patch dc-runtime here |
| Design-system adherence lint | oxlintrc present | not wired to CI (no CI) |

Debt hotspots (conceptual):
1. **Seed-as-truth risk** — scores look real
2. **Monolithic HTML** — 798 lines template + 250 lines script
3. **CDN hard dependency** for React
4. **Fit model oversimplification** — not the Fit Engine described in A4
5. **Zero automated tests**

---

## 7. Testing, Coverage & Quality Signals

| Signal | Status |
|--------|--------|
| Unit / integration / E2E tests | **None** |
| Test runner | None |
| CI | None |
| Typecheck | None (JS in HTML script) |
| Lint | oxlintrc for DS only; not applied in pipeline |
| Coverage | N/A |

---

## 8. Dependency & Security Surface

| Dependency | Version | Source | Risk |
|------------|---------|--------|------|
| react | 18.3.1 | unpkg UMD + SRI | Supply-chain; offline break |
| react-dom | 18.3.1 | unpkg UMD + SRI | same |
| @babel/standalone | 7.29.0 | unpkg (on demand) | same; `new Function` eval path in runtime |
| Organic DS | vendored | local | low |
| Backend APIs | none | — | — |

**Security notes:**
- No auth, no secrets stored — appropriate for prototype
- Runtime uses `new Function(...)` for module evaluation (dc-runtime design) — not user input here
- `postMessage` with `"*"` target origin for design-mode parent — only relevant in embedded host
- Prototype **must not** be mistaken for a production surface with live inference

---

## 9. Hidden Complexity & Archaeological Notes

1. **dc-runtime is non-trivial** (~1900 LOC): template compile, `sc-if`/`sc-for`, streaming placeholders, external x-import, design-mode bridge. The product logic is a thin layer on top.
2. **Redesign MD is the real product document** — 312 lines covering current Lab (Flask), competitive critique, P0–P2 needs, architecture moves A1–A6, phasing 0–4, open decisions.
3. **Production Lab is described but not present** — `optiq lab` Flask :7860, mlx_lm.server :8080, multiprocessing + SSE, etc. live in another codebase (mlx-optiq), not this folder.
4. **`evalOwnDelta = 2` is hardcoded** — the BYO eval “+2” delta is not computed from data.
5. **Fit MTLResource rule** is a boolean threshold (`ctx >= 65536 && kvBits <= 6`), not a real resource-count model.
6. **Workspace list** is static HTML (clinical-eval / latency-bench), not driven by state array.

---

## 10. Risk Register (evidence-linked)

| # | Risk | Impact | Likelihood | Evidence | Mitigation |
|---|------|--------|------------|----------|------------|
| R1 | Stakeholders treat prototype metrics as measured | High | Medium | Seed `overall`/`metrics` look production-grade | Label UI “prototype / sample data”; never ship these numbers |
| R2 | Offline / air-gapped open fails (no React) | Medium | High | unpkg CDN load | Bundle React or inject `__resources` |
| R3 | Scope confusion: redesign vs implement Phase 0 | High | High | No SQLite/job bus; only UI | Explicit next-slice plan against phasing table |
| R4 | Fit formula diverges from real Fit Engine | High | Certain if productized as-is | `computeFit` simplified | Replace with A4 model + calibration |
| R5 | No git → unrecoverable edits | Medium | Medium | no `.git` | `git init` + baseline commit before changes |
| R6 | Dual truth: brief vs prototype drift | Medium | Medium | both may evolve independently | Drift audit against redesign MD when IA changes |
| R7 | Composer stub confuses demo audiences | Low | High | empty `sendMessage` | Wire local-only append or hide Send |

---

## 11. Appendices

### A. Seed inventory

- **BUILDS:** b1–b4 (Qwen 4-bit, Qwen 6-bit, Llama-4-Scout 3-bit, DeepSeek-Coder 4-bit)
- **RUNS:** r1–r6 (running quantize, done eval, done fine-tune, queued dataset, done research, failed quantize)
- **MESSAGES:** m1–m4 clinical-eval narrative
- **MACHINE:** 64 GB RAM, 9.2 reserved, ports 8080/7860, adapter clinical-v3

### B. File size ranking

| Bytes | File |
|------:|------|
| 69150 | support.js |
| 55522 | OptiQ Lab.dc.html |
| 31324 | optiq-lab-ux-redesign.md |
| 26206 | .thumbnail |
| 10793 | styles.css |
| 7353 | _ds_manifest.json |
| 7019 | readme.md |
| 4042 | _adherence.oxlintrc.json |
| 297 | _ds_bundle.js |

### C. Commands used

```bash
find /Volumes/WS4TB/optiqlab -type f
wc -l …; du -sh …
rg -n patterns on html/js
read script block L537–794; support.js CDN section
```

---

```json
{
  "mode": "deepdive",
  "repo_scale": {
    "files": 9,
    "loc": 2977,
    "languages": ["html", "javascript", "css", "markdown", "json"]
  },
  "maturity_score": "3",
  "maturity_as_design_artifact": "7",
  "debt_hotspots": [
    "seed-data-as-truth",
    "monolithic-dc-html",
    "cdn-react-dependency",
    "illustrative-fit-engine",
    "no-tests-no-git"
  ],
  "security_flags": [
    "unpkg-cdn-dependency",
    "new-Function-eval-in-runtime",
    "postMessage-wildcard-design-mode"
  ],
  "unknown_count": 3,
  "unknowns": [
    "location of production mlx-optiq Lab repo",
    "intended next stack (keep dc-runtime vs React SPA per A3)",
    "host design tool that generated this artifact"
  ],
  "confidence": "high"
}
```
