"""GLM-4 MoE native MTP backend facade.

The speculative sampler remains shared through ``generation.py``.  This facade
marks GLM-4 MoE and GLM-4 MoE Lite as executable only behind the same verified
runtime-contract gate used for the DeepSeek-family backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import DraftTokens, ModelState, MTPBackend, VerifyOutput
from optiq.runtime.mtp.profiles import DEFAULT_PROFILE_NAME


class GLMMTPBackend(MTPBackend):
    arch_id = "glm4-moe-mtp"

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
        raise NotImplementedError("GLMMTPBackend.verify is wired through generation.py")

    def propose(self, state: ModelState, hidden: Any) -> DraftTokens:
        raise NotImplementedError("GLMMTPBackend.propose is wired through generation.py")

    def recommended_profile(self) -> str:
        return DEFAULT_PROFILE_NAME

    def health(self) -> dict[str, Any]:
        return {
            "arch_id": self.arch_id,
            "runtime_path": "mtplx.runtime + mtplx.glm_mtp_patch + mtplx.generation",
            "support_level": "experimental-native-contract-gated",
            "contract_required": True,
            "supported_model_types": ["glm4_moe", "glm4_moe_lite"],
            "references": [
                "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/glm4_moe_mtp.py",
                "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/glm4_moe_lite_mtp.py",
                "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/glm4_moe.py",
                "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/glm4_moe_lite.py",
            ],
        }
