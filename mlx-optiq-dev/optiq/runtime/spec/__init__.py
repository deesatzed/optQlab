"""Generic speculative-decoding runtime for OptiQ.

Separate from the existing ``optiq/runtime/mtp/`` which is the mature
Qwen-MTP-specific implementation. This module hosts the generic loop
(draft K tokens, target verifies K+1, accept prefix, commit) plus
per-architecture drafter adapters that share the loop.

First adapter: Gemma-4 ``-assistant`` drafters (4-layer Q-only model that
consumes typed K/V from the target). See
https://github.com/ollama/ollama/pull/15980 for the algorithmic reference
we ported (MIT-licensed, attribution preserved).

Future adapters could fold the Qwen MTP head onto this loop, but the
existing ``optiq.runtime.mtp`` keeps working untouched until that
migration is validated separately.
"""
from __future__ import annotations

from .drafters import GemmaAssistantDrafter
from .runtime import spec_generate, SpecConfig, SpecEvent

__all__ = [
    "GemmaAssistantDrafter",
    "spec_generate",
    "SpecConfig",
    "SpecEvent",
]
