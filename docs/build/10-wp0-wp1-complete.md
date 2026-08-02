# WP-0 + WP-1 complete — critical gap mitigations

**Date:** 2026-08-02  
**Tests:** `57 passed` (`pytest tests/lab -v` in `mlx-optiq-dev`)  
**Shell path:** Flask incremental (mitigation plan option A)

---

## WP-0 — Demo honesty

| Action | Status |
|--------|--------|
| Sample-data banner on prototype | Done |
| Send / Export / Promote disabled or labeled | Done |
| Prototype `README.md` | Done |
| Root + dev README map | Done |

---

## WP-1 — Machine & Models

| Deliverable | Module / route | Gap |
|-------------|----------------|-----|
| Fit Engine | `optiq/lab/fit_engine.py` | G1 |
| Fit predict API | `POST /api/fit/predict` | G1 |
| Fit calibrate | `POST /api/fit/calibrate` | G1 |
| Fit-gated load | `POST /api/models/load` → 409 if blocked | G1, G7 |
| Models page | `GET /models` | G7, C1 |
| Machine state | `optiq/lab/machine.py`, `GET /api/machine` | strip |
| Machine strip UI | `_base.html` (real RAM + port probes) | strip |
| Server page points to Models | `settings_server.html` banner | G7 |

### Fit verdicts

`comfortable` | `degraded` | `will_not_fit` | `hard_fail`  
Hard-fail prior: ctx ≥ 65536 and kv_bits ≤ 6 (MTLResource).  
KV/activation are **estimates** (listed in `estimate_notes`). No Capability Scores.

### Load path

Primary: **Models → Load with Fit** (sidebar Primary).  
Settings → Server remains for advanced MTP/sampler; banner redirects to Models.

---

## Gaps still open (not this tranche)

| Gap | Needs |
|-----|--------|
| G2 Measurement / eval GUI | WP-3 |
| G3 Full Work/Models/Runs IA | WP-2 SPA/shell |
| G4/G8 Full run-health UI in chat | WP-2 |
| G6 Global Runs page | WP-2 |
| G10 Chat table stakes | WP-2 |
| G11 Crown jewels | WP-3/4 |
| G12 Remote auth | WP-4 |

Phase 0 spine (events, bus, dual-write, provenance plumbing) remains in place.

---

## Verify

```bash
cd mlx-optiq-dev
pytest tests/lab -v
# New: test_fit_engine, test_machine, test_models_fit_api
```
