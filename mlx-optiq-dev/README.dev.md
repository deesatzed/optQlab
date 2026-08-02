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

## Constraints

- Do not edit the site-packages copy under miniforge3
- No mock data, placeholders, or cached fake responses
- Commit work in this repo only
