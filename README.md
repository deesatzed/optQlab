# OptiQ Lab (`optQlab`)

Workspace for the OptiQ Lab redesign: interactive UI prototype, product specs, gap mitigation plans, and an editable **Phase 0 spine** worktree bootstrapped from `mlx-optiq` 0.4.7.

## Layout

| Path | Contents |
|------|----------|
| `OptiQ Lab interactive prototype/` | High-fidelity IA prototype (Work / Models / Runs + Machine strip). **Sample data — not live Lab.** |
| `mlx-optiq-dev/` | Editable Python package (schema v2, events, job bus, dual-write chats, provenance APIs). |
| `docs/plans/` | Phase 0 plan + **gap mitigation plan** |
| `docs/build/` | Discovery, handoff, gap board |
| `.governance/` | Assumption registry |

## Phase 0 spine tests

```bash
cd mlx-optiq-dev
pip install -e ".[dev]"
pytest tests/lab -v
```

Requires conda/env with mlx-optiq dependencies (see `mlx-optiq-dev/pyproject.toml`).

## Important docs

- Gap mitigation (primary roadmap): `docs/plans/2026-08-02-gap-mitigation-plan.md`
- Gap board: `docs/build/08-gap-board.md`
- Non-technical gaps: `docs/build/09-gaps-nontechnical.md`
- UX redesign thesis: `OptiQ Lab interactive prototype/uploads/optiq-lab-ux-redesign.md`

## Status

- **Phase 0 (spine):** complete in `mlx-optiq-dev` (lab tests green).
- **User-facing redesign (Fit, Machine strip, Eval GUI, etc.):** planned — not shipped as main Lab UI.
- Do not treat prototype Capability Scores as measured results.
