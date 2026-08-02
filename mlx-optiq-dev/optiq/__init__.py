"""OptiQ: Optimizing compiler for PyTorch → MLX with data-driven mixed-precision quantization."""

try:  # single source of truth: pyproject's version, not a constant that drifts
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("mlx-optiq")
except Exception:  # running from a source tree with no installed dist
    __version__ = "0.4.2.dev0"

# Register OptiQ-shipped model types that mlx-lm doesn't yet have upstream
# (qwen3_5_text). Import-time side effect so any downstream caller that
# does ``import optiq`` (directly or transitively via the CLI) gets the
# additional model types available for ``mlx_lm.load``.
try:
    from optiq.mlx_lm_patches import _register as _optiq_mlx_lm_register
    _optiq_mlx_lm_register.register()
except Exception:
    # Never block package import on patch registration — if mlx-lm's
    # internals change, surface cleanly at use-time instead.
    pass
