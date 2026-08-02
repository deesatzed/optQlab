"""Hidden-state correctors for native MTP experiments."""

from .diagonal_affine import (
    DiagonalAffineCorrector,
    NoOpCorrector,
    blend_with_identity,
    deterministic_train_mask,
    ensure_nonempty_split,
    fit_c0_stat,
    fit_c1_diagonal,
    load_corrector,
)
from .low_rank import LowRankResidualCorrector, fit_c2_low_rank


def load_runtime_corrector(path, *, blend: float | None = None):
    """Load a corrector artifact and optionally damp it toward no-op."""

    if path is None:
        return None
    corrector = load_corrector(path)
    if blend is None:
        return corrector
    if isinstance(corrector, LowRankResidualCorrector):
        return LowRankResidualCorrector(
            mean=corrector.mean,
            in_proj=corrector.in_proj,
            out_proj=(corrector.out_proj * float(blend)).astype("float32"),
            rank=corrector.rank,
            kind=f"{corrector.kind}_blend",
            hidden_variant=corrector.hidden_variant,
            metadata={**corrector.metadata, "runtime_blend": float(blend)},
        )
    if isinstance(corrector, DiagonalAffineCorrector):
        return blend_with_identity(corrector, float(blend), kind=f"{corrector.kind}_blend")
    raise TypeError(f"unsupported corrector type: {type(corrector).__name__}")

__all__ = [
    "DiagonalAffineCorrector",
    "NoOpCorrector",
    "blend_with_identity",
    "deterministic_train_mask",
    "ensure_nonempty_split",
    "fit_c0_stat",
    "fit_c1_diagonal",
    "load_corrector",
    "load_runtime_corrector",
    "LowRankResidualCorrector",
    "fit_c2_low_rank",
]
