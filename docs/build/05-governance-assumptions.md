# Assumption Registry — OptiQ Lab interactive prototype

**Command:** `governance --mode=assumptions` (list)  
**Date:** 2026-08-02  
**Role:** SWE | Stage 2 onboard chain step 5/6  
**Persistence:** `.governance/assumptions.json`

---

**Total:** 12  
**Unvalidated:** 6 (⚠️ RISK)  
**Validated:** 6  
**Invalidated:** 0  
**Stale (>30d unreviewed):** 0  

---

## High-Risk Unvalidated

- **[A-0006]** Target production stack is React SPA + FastAPI + SQLite spine (A1–A3), not long-term dc-runtime.  
  — source: redesign §6.2 — verify: explicit product decision before Phase 0  

- **[A-0007]** Multi-model residency vs sequential queue for Arena/eval is still open.  
  — source: redesign §6.7 decision 1 — verify: written owner decision  

## Medium-Risk Unvalidated

- **[A-0004]** Standalone open requires unpkg for React unless `__resources` injects.  
  — source: support.js L1143+ — verify: offline open test  

- **[A-0005]** Production Lab shape (Flask :7860 + mlx_lm :8080) matches redesign §1.1.  
  — source: redesign MD — verify: against real mlx-optiq repo  

- **[A-0008]** Workspace coherence rules undecided.  
  — source: redesign §6.7 decision 2 — verify: written contract  

## Low-Risk Unvalidated

- **[A-0009]** Partial Organic DS export is intentional for this prototype.  
  — verify: render with styles.css only  

- **[A-0012]** Artifact may prefer design-tool host over bare browser.  
  — verify: standalone vs host boot comparison  

---

## Validated

- **[A-0001]** This folder is prototype-only, not production Lab.  
- **[A-0002]** Scores/metrics/tok-s/progress are seed fiction.  
- **[A-0003]** `computeFit` is illustrative, not Fit Engine A4.  
- **[A-0010]** Empty `sendMessage` is intentional demo stub.  
- **[A-0011]** MACHINE RAM figures are demo constants, not host memory.  

---

## Invalidated

_(none)_

## Stale

_(none)_

---

## Recommended next validations

1. Offline/CDN check → A-0004  
2. Locate mlx-optiq Lab repo → A-0005  
3. Product decisions for A-0006 / A-0007 / A-0008 (blocks real architecture work)

---

```json
{
  "mode": "assumptions",
  "total": 12,
  "unvalidated": 6,
  "validated": 6,
  "invalidated": 0,
  "stale": 0,
  "high_risk_unvalidated": ["A-0006", "A-0007"],
  "storage": ".governance/assumptions.json",
  "confidence": "high"
}
```
