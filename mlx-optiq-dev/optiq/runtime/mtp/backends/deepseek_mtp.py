"""DeepSeek V3/V3.2 native MTP backend facade.

The speculative sampler is still shared through ``generation.py``.  This facade
keeps the architecture registry honest: DeepSeek now has a concrete runtime
loader, but it remains verified-contract gated until real checkpoints pass the
same exactness and long-run gates as Qwen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import DraftTokens, ModelState, MTPBackend, VerifyOutput
from optiq.runtime.mtp.profiles import DEFAULT_PROFILE_NAME


class DeepSeekMTPBackend(MTPBackend):
    arch_id = "deepseek-v3-mtp"

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
        raise NotImplementedError("DeepSeekMTPBackend.verify is wired through generation.py")

    def propose(self, state: ModelState, hidden: Any) -> DraftTokens:
        raise NotImplementedError("DeepSeekMTPBackend.propose is wired through generation.py")

    def recommended_profile(self) -> str:
        return DEFAULT_PROFILE_NAME

    def health(self) -> dict[str, Any]:
        return {
            "arch_id": self.arch_id,
            "runtime_path": "mtplx.runtime + mtplx.deepseek_mtp_patch + mtplx.generation",
            "support_level": "experimental-native-contract-gated",
            "contract_required": True,
            "supported_model_types": ["deepseek_v3", "deepseek_v32", "glm_moe_dsa"],
            "references": [
                "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/deepseek_mtp.py",
                "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/deepseek_v3.py",
                "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/deepseek_v32.py",
            ],
        }

