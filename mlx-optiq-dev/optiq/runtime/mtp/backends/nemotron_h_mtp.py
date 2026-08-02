"""Nemotron-H native MTP backend facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import DraftTokens, ModelState, MTPBackend, VerifyOutput
from optiq.runtime.mtp.profiles import DEFAULT_PROFILE_NAME


class NemotronHMTPBackend(MTPBackend):
    arch_id = "nemotron-h-mtp"

    def load(self, model_path: Path) -> ModelState:
        from optiq.runtime.mtp.mtp_patch import MTPContract
        from optiq.runtime.mtp.runtime import load

        runtime = load(model_path, mtp=True, contract=MTPContract())
        return ModelState(
            model_path=Path(model_path),
            runtime=runtime,
            metadata={"arch_id": self.arch_id, "contract_gated": True},
        )

    def verify(self, state: ModelState, draft_tokens: DraftTokens, hidden: Any) -> VerifyOutput:
        raise NotImplementedError("NemotronHMTPBackend.verify is wired through generation.py")

    def propose(self, state: ModelState, hidden: Any) -> DraftTokens:
        raise NotImplementedError("NemotronHMTPBackend.propose is wired through generation.py")

    def recommended_profile(self) -> str:
        return DEFAULT_PROFILE_NAME

    def health(self) -> dict[str, Any]:
        return {
            "arch_id": self.arch_id,
            "runtime_path": "mtplx.runtime + mtplx.nemotron_h_mtp_patch + mtplx.generation",
            "support_level": "experimental-native-contract-gated",
            "contract_required": True,
            "supported_model_types": ["nemotron_h", "nemotron_h_puzzle"],
            "limits": {"num_nextn_predict_layers": 1, "mtp_pattern_chars": ["*", "E"]},
            "references": [
                "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/nemotron_h_mtp.py",
                "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/nemotron_h.py",
            ],
        }
