"""Backend protocol for MTPLX architecture-specific native MTP runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DraftTokens:
    token_ids: tuple[int, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VerifyOutput:
    accepted_token_ids: tuple[int, ...]
    corrected_token_id: int | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelState:
    model_path: Path
    runtime: Any
    metadata: dict[str, Any]


class MTPBackend(ABC):
    """Architecture-specific MTP propose/verify adapter.

    The shared speculative sampler lives outside the backend.  Backends own the
    model-specific details: loading, draft proposal, target verification, and
    health reporting for kernel/profile requirements.
    """

    arch_id: str

    @abstractmethod
    def load(self, model_path: Path) -> ModelState:
        raise NotImplementedError

    @abstractmethod
    def verify(self, state: ModelState, draft_tokens: DraftTokens, hidden: Any) -> VerifyOutput:
        raise NotImplementedError

    @abstractmethod
    def propose(self, state: ModelState, hidden: Any) -> DraftTokens:
        raise NotImplementedError

    @abstractmethod
    def recommended_profile(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> dict[str, Any]:
        raise NotImplementedError
