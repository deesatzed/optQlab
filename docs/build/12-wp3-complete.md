# WP-3 complete — Measurement (thesis)

**Date:** 2026-08-02  
**Commit target:** after push  
**Tests:** lab suite including `test_wp3_eval.py`

---

## Delivered

| Item | Detail |
|------|--------|
| **3A Eval as Run** | `eval` job kind via job bus; CLI `optiq eval --output-json` for suite tasks; results in `evals` table |
| **3B BYO eval** | Import JSONL prompt sets; job generates with mlx_lm + real string match scoring |
| **3C Promote gate** | `POST /api/eval/promote` — blocks when candidate capability_score &lt; baseline |
| **3D Knob consequences** | `POST /api/fit/consequences` + Models UI + Eval page table (ΔGB estimates labeled) |
| **3E Crown-jewel homes** | Documented on Eval page; Fit/KV on Models; MTP on Server; Eval primary for scores |

### Routes

| Path | Role |
|------|------|
| `/eval` | Measurement UI |
| `POST /api/eval/run` | Submit eval job |
| `GET/POST /api/eval/sets` | BYO sets |
| `GET /api/eval/results` | Stored scores |
| `POST /api/eval/compare` | Diff two evals |
| `POST /api/eval/promote` | Regression gate |
| `POST /api/fit/consequences` | Knob preview |

### Modules

- `optiq/lab/eval_service.py`
- `optiq/lab/eval_job.py`
- `optiq/lab/routes/eval_routes.py`
- `optiq/lab/templates/eval.html`
- CLI: `gsm8k` now writes `--output-json` (was missing)

---

## Gap status

| Gap | Status |
|-----|--------|
| **G2** Measurement severed from use | 🟢 GUI path for eval + promote |
| T1 / T3 BYO + promote | 🟢 |
| G11 crown jewels | 🟡 GUI homes listed; not every CLI flag |

---

## Honest notes

- Suite jobs invoke **real** `optiq eval` (can be long / need a model on disk).
- BYO jobs load the model with **mlx_lm** and score real generations — not simulated scores.
- Unit tests cover scoring, store, promote, APIs **without** requiring a multi-GB model in CI.
- Capability Scores in the UI come only from stored eval rows or CLI JSON — never invented by Fit.

## Verify

```bash
cd mlx-optiq-dev && pytest tests/lab -v
# UI: open Eval → import BYO set or run gsm8k-50 against a local path → Runs → promote check
```
