"""Custom VerifyCore kernel experiments."""

from .verify_mlp_fused import (
    gate_up_swiglu_qmv4_activation,
    gate_up_swiglu_qmv4_activation_rowwise,
    gate_up_swiglu_qmv4_activation_split,
    is_gate_up_swiglu_qmv4_eligible,
    is_small_m_qmm4_eligible,
    small_m_qmm4_matmul,
)
from .fused_norm import (
    fused_add_rmsnorm,
    fused_gdn_norm_gate,
    is_fused_add_rmsnorm_eligible,
    is_fused_gdn_norm_gate_eligible,
)
from .native_gdn_tail import (
    is_native_gdn_tail_eligible,
    native_gdn_norm_gate_out_qmv8,
)

__all__ = [
    "fused_add_rmsnorm",
    "fused_gdn_norm_gate",
    "gate_up_swiglu_qmv4_activation",
    "gate_up_swiglu_qmv4_activation_rowwise",
    "gate_up_swiglu_qmv4_activation_split",
    "is_fused_add_rmsnorm_eligible",
    "is_fused_gdn_norm_gate_eligible",
    "is_gate_up_swiglu_qmv4_eligible",
    "is_native_gdn_tail_eligible",
    "native_gdn_norm_gate_out_qmv8",
    "is_small_m_qmm4_eligible",
    "small_m_qmm4_matmul",
]
