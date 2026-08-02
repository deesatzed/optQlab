# mlx-optiq-dev

Editable worktree for **OptiQ Lab Phase 0 spine**, bootstrapped from the pip-installed
`mlx-optiq==0.4.7` package (site-packages).

This is **not** the upstream release tree. It is a local development checkout so Phase 0
schema, jobs bus, and provenance changes can be made without modifying the conda env
site-packages install.

## Setup

```bash
# Use the py313 conda env
/Users/o2satz/miniforge3/envs/py313/bin/pip install -e ".[dev]"
```

Verify the import resolves to this worktree:

```bash
/Users/o2satz/miniforge3/envs/py313/bin/python -c "import optiq; print(optiq.__file__)"
# expected: .../mlx-optiq-dev/optiq/__init__.py
```

## Test

```bash
# From mlx-optiq-dev root
/Users/o2satz/miniforge3/envs/py313/bin/python -m pytest tests/ -v
```

Tests use a real temp `OPTIQ_HOME` (see `tests/lab/conftest.py`). No mocks.

## Critical-gap mitigations (WP-0 / WP-1)

| Feature | Route / module |
|---------|----------------|
| Fit Engine | `optiq/lab/fit_engine.py`, `POST /api/fit/predict`, `POST /api/fit/calibrate` |
| Fit-gated load | `POST /api/models/load` (409 when blocked unless `force`) |
| Models UI | `/models` |
| Machine strip | `_base.html` + `GET /api/machine` (real RAM + TCP probes) |

```bash
pytest tests/lab/test_fit_engine.py tests/lab/test_machine.py tests/lab/test_models_fit_api.py -v
```

## Constraints

- Do not edit the site-packages copy under miniforge3
- No mock data, placeholders, or cached fake responses
- Never invent Capability Scores in Fit
- Commit work in this repo only
