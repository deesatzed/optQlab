# Technical Debt Report — OptiQ Lab interactive prototype

**Command:** `governance --mode=debt`  
**Date:** 2026-08-02  
**Role:** SWE | Stage 2 onboard chain step 4/6  
**Note:** Effort classes used instead of calendar/time estimates (project policy).

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total debt items tracked | 18 |
| 🔴 Critical | 4 |
| 🟡 High | 7 |
| 🟢 Medium/Low | 7 |
| Debt velocity (30d) | **UNKNOWN** — no git history (baseline first measurement) |
| Worst hotspots | `OptiQ Lab.dc.html` (monolith + stubs); missing DS tree; no repo hygiene |

**Context:** This is a **design prototype**, so much “debt” is **intentional incompleteness**. Debt is framed as *what blocks productization / honest demos*, not as failed production code.

---

## Critical Debt

| ID | Category | Location | Age | Severity | Effort class | Owner hint |
|----|----------|----------|-----|----------|--------------|------------|
| D-001 | Completeness Gaps | `OptiQ Lab.dc.html:616` `sendMessage = () => {}` | ≤1d artifact | 🔴 Critical (demo honesty) | S | product/UX |
| D-002 | Architecture Debt | No Phase 0 spine (SQLite, job bus, provenance capture) vs redesign §6.2 | open | 🔴 Critical (product path) | XL | eng |
| D-003 | Completeness Gaps | All Capability Score / metric / tok/s values are seed — no measurement pipeline | open | 🔴 Critical (truth) | XL | eng (mlx-optiq) |
| D-004 | Security / Ops | React/ReactDOM loaded from unpkg; offline break | open | 🔴 Critical (repro) | S | eng |

---

## High Priority Debt

| ID | Category | Location | Severity | Effort | Notes |
|----|----------|----------|----------|--------|-------|
| D-005 | Completeness Gaps | Dead controls: Export JSON, ↓.md, Hub search, + New build, ⌘K, Promote | 🟡 High | M | Forensics catalog |
| D-006 | Architecture Debt | Fit engine is illustrative (`computeFit`), not Fit Engine A4 | 🟡 High | L | Must not ship as calibrated |
| D-007 | Doc Debt | No README / CLAUDE.md / run instructions at root | 🟡 High | S | Onboarding friction |
| D-008 | Architecture Debt | No git repository | 🟡 High | S | Init + baseline |
| D-009 | Test Debt | Zero tests of any kind | 🟡 High | M | At least smoke: Fit verdicts, view routing |
| D-010 | Doc Debt | `_ds_manifest.json` references ~14 component/template paths not present in export | 🟡 High | M | Partial DS bundle |
| D-011 | Completeness Gaps | Port/integration “healthy/connected” static tags | 🟡 High | S | Label as simulated |

---

## Standard Debt

| ID | Category | Location | Severity | Notes |
|----|----------|----------|----------|-------|
| D-012 | Architecture Debt | Monolithic `OptiQ Lab.dc.html` (template + domain + fit) | 🟢 Medium | Fine for prototype; split before growth |
| D-013 | Architecture Debt | Inline style sprawl vs DS classes | 🟢 Medium | Many tokens used correctly; layout is style= soup |
| D-014 | Deferred Work | Workspace list static HTML (not state) | 🟢 Medium | Cannot switch workspaces |
| D-015 | Deferred Work | Tools policy tags not interactive | 🟢 Medium | Display only |
| D-016 | Completeness Gaps | `evalOwnDelta = 2` hardcoded | 🟢 Medium | BYO eval theater |
| D-017 | Architecture Debt | `support.js` generated without source tree | 🟢 Low | Accept as vendored runtime |
| D-018 | Doc Debt | Redesign MD phasing “Weeks” table vs prototype only | 🟢 Low | Spec ahead of code (expected) |

---

## Debt Age Distribution

| Bucket | Count |
|--------|-------|
| <7d | 18 (whole workspace is a single drop) |
| 7–30d | 0 |
| 30–90d | 0 |
| >90d | 0 |

## Trend Analysis

- Net debt change past 30d: **baseline established** (no prior tracker)
- New vs resolved: 18 / 0
- First tracker file: this document

## Debt Hotspots

1. `OptiQ Lab interactive prototype/OptiQ Lab.dc.html` — stubs, seed truth, monolith  
2. `OptiQ Lab interactive prototype/_ds/.../_ds_manifest.json` vs missing files  
3. Workspace root — no git, no README, no package metadata  
4. `support.js` CDN coupling  

## Recommended Actions

**P0**
- Label prototype UI as sample data (or gate demo script) for scores/ports  
- Disable or toast empty actions (Send, Export, New build)  
- Vendor React or document network requirement  
- `git init` + baseline commit of current tree  

**P1**
- Root README: what this is / is not / how to open  
- Minimal smoke checks for `computeFit` verdict boundaries  
- Resolve DS export completeness or trim manifest  

**P2**
- Decide product path: keep dc-runtime vs redesign A3 React SPA  
- Only then open Phase 0 spine work in the **production** repo (not as fake completion here)

## Debt Budget (points)

| | Points |
|--|--------|
| Current | 42 (Critical×4 + High×3 + Med×1 weighted: 4×5 + 7×3 + 7×1 = 20+21+7) |
| Target for “honest prototype” | ≤15 (clear labels + dead controls fixed + README + git) |
| Over target | +27 |

---

```json
{
  "mode": "debt",
  "total_items": 18,
  "critical": 4,
  "high": 7,
  "medium_low": 7,
  "hotspots": [
    "OptiQ Lab.dc.html",
    "_ds incomplete export",
    "no git / no README",
    "cdn react"
  ],
  "baseline": true,
  "confidence": "high"
}
```
