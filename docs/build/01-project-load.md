# Project Load — OptiQ Lab interactive prototype

**Command:** `docs-context --mode=project-load`  
**Date:** 2026-08-02  
**Role:** SWE | Stage 2 onboard chain step 1/6

---

## Project: OptiQ Lab interactive prototype

**Purpose:** High-fidelity interactive UI prototype of the redesigned OptiQ Lab (Work / Models / Runs + persistent Machine strip), implementing the IA and six priority screens from the UX redesign brief — not a production app and not wired to mlx-optiq.

**Stack:**
- Single-page **dc-runtime** document (`*.dc.html`) — Design Component / React template runtime
- Vendored runtime: `support.js` (GENERATED from `dc-runtime/src/*.ts`; rebuild note: `cd dc-runtime && bun run build`)
- **Organic** design system (CSS tokens, Caprasimo + Figtree, Lucide icons)
- In-document `DCLogic` class with hardcoded seed data (BUILDS, RUNS, MESSAGES, MACHINE)
- No package manager, no backend, no git, no test harness

**Status:** dirty / non-repo | no commits | no branch  
**Maturity:** design prototype (Stage 2)

### Key Paths

| Role | Path |
|------|------|
| Entry | `OptiQ Lab interactive prototype/OptiQ Lab.dc.html` |
| Logic / seed data | same file, `<script type="text/x-dc" data-dc-script>` (~lines 537–794) |
| Runtime | `OptiQ Lab interactive prototype/support.js` |
| Design system | `OptiQ Lab interactive prototype/_ds/organic-ed51b8ee-d25f-4a85-ab7c-2e7adba6b498/` (`styles.css`, `_ds_manifest.json`, `readme.md`) |
| Spec / redesign brief | `OptiQ Lab interactive prototype/uploads/optiq-lab-ux-redesign.md` (312 lines) |
| Tests | **none** |
| Config | **none** (no env, no package.json) |
| Docs (this chain) | `docs/build/` |

### Scale (evidence)

| Metric | Value |
|--------|-------|
| Tracked content files | ~8 (excl. `.DS_Store`, thumbnail) |
| Disk | ~236 KB |
| HTML prototype | 798 lines |
| support.js | 1911 lines |
| styles.css | 257 lines |
| UX redesign MD | 312 lines |
| Total LOC (source-ish) | ~2,977 |

### Architecture snapshot

```
optiqlab/
└── OptiQ Lab interactive prototype/
    ├── OptiQ Lab.dc.html     # UI shell + DCLogic + seed arrays
    ├── support.js            # dc-runtime (React bootstrap, template compile)
    ├── .thumbnail
    ├── uploads/
    │   └── optiq-lab-ux-redesign.md   # product thesis + IA + phasing
    └── _ds/organic-…/
        ├── styles.css
        ├── _ds_bundle.js
        ├── _ds_manifest.json
        ├── _adherence.oxlintrc.json
        └── readme.md
```

**IA implemented in the prototype (mirrors redesign §6.3):**
1. **Nav rail** — Work / Models / Runs + workspace list + ⌘K affordance
2. **Machine strip** — loaded build, RAM bar, ports, adapter, job count; expands to Machine panel
3. **Work** — modes Chat / Research / Compare; run-health chips; provenance drawer; workspace right rail
4. **Models** — build cards, fit badges, load dialog with Fit bar (`computeFit`)
5. **Runs** — job list with filters; eval compare modal
6. **Modals** — Load w/ fit, Build detail, Eval compare, Provenance

**Interactivity that is real (client-only state):**
- View switching (`view`: work|models|runs)
- Work mode switching (chat|research|compare)
- Machine panel toggle
- Load dialog open/close, ctx/KV knobs, `computeFit` live recalculation, `confirmLoad` swaps resident build
- Build detail open/close
- Eval compare open + build A/B selectors
- Provenance open/close
- Draft textarea + system prompt + sampler temp (UI state only)
- Runs filter (all|eval|quantize)

**Explicitly non-functional (demo stubs):**
- `sendMessage = () => {}; // demo composer — non-persisting in this prototype`
- All BUILDS / RUNS / MESSAGES / MACHINE are static arrays
- Ports shown as "healthy" tags with no network calls
- ⌘K is visual only
- No fetch to Flask / mlx_lm.server

### Recent activity

- No git repository at `/Volumes/WS4TB/optiqlab`
- Files stamped 2026-08-02 08:56 (prototype + DS assets)
- No prior handoff packets found under this project
- Error reference (`~/.claude/state/error-reference.json`): **0 OptiQ-related entries**

### Known issues / flags

1. **PURE DEMO** — UI prototype only; no mlx-optiq / Lab backend present in this workspace
2. **No README / CLAUDE.md / package.json** at repo root
3. **No git** — no history, no branch hygiene
4. **dc-runtime dependency** — `support.js` expects `window.React` / `window.ReactDOM`; how those load in standalone browser open is via runtime bootstrap (confirm at deepdive)
5. **Fit engine is illustrative** — `computeFit` uses simplified formulas + hardcoded `MACHINE.totalRamGB: 64`, not measured calibration
6. **UX brief is the product source of truth** for the redesign; prototype implements Phase 1–3 *screens*, not Phase 0 spine (SQLite/job bus)

### Quick commands

```bash
# Open the interactive prototype in a browser (local file)
open "/Volumes/WS4TB/optiqlab/OptiQ Lab interactive prototype/OptiQ Lab.dc.html"

# Or serve to avoid file:// quirks
cd "/Volumes/WS4TB/optiqlab/OptiQ Lab interactive prototype" && python3 -m http.server 8765
# then open http://127.0.0.1:8765/OptiQ%20Lab.dc.html

# Spec
open "/Volumes/WS4TB/optiqlab/OptiQ Lab interactive prototype/uploads/optiq-lab-ux-redesign.md"

# Tests / Build
# none in this workspace
```

### Prior handoff

**None found.** No `Handoff*.md` / `HANDOFF*.md` under this project.

### Open questions for later chain steps

- Is the production mlx-optiq Lab repo elsewhere, and should this prototype eventually merge into it?
- Target stack confirmation (redesign A3: React SPA + FastAPI) vs. keeping dc-runtime for longer?
- Phase 0 decisions still open in the brief: multi-model residency vs sequential queue; workspace coherence rules

---

```json
{
  "mode": "project-load",
  "project": "OptiQ Lab interactive prototype",
  "stack": ["dc-runtime", "React (via support.js)", "Organic DS", "static seed data"],
  "status": "yellow",
  "git": false,
  "tests": false,
  "backend": false,
  "prior_handoff": false,
  "primary_spec": "OptiQ Lab interactive prototype/uploads/optiq-lab-ux-redesign.md",
  "entry": "OptiQ Lab interactive prototype/OptiQ Lab.dc.html",
  "confidence": "high"
}
```
