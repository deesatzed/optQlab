"""High-level MTPLX runtime loading primitives."""

from __future__ import annotations

import inspect as py_inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import inspect_model, load_config, resolve_model_dir
from .mtp_adapters import install_saved_mtp_lora_adapter, mtp_adapter_depth
from .mtp_patch import MTPContract, inject_mtp_support, validate_mtp_support


@dataclass
class MTPLXRuntime:
    model: Any
    tokenizer: Any
    model_path: Path
    mtp_enabled: bool
    contract: MTPContract
    mtp_adapter_path: Path | None = None
    mtp_adapter_metadata: dict[str, Any] | None = None
    diagnostic_counters: dict[str, int] = field(default_factory=dict)
    _forward_ar_supports_emit_logits: bool | None = field(default=None, init=False, repr=False)
    _forward_ar_supports_logits_keep: bool | None = field(default=None, init=False, repr=False)

    def _count(self, key: str, amount: int = 1) -> None:
        self.diagnostic_counters[key] = int(self.diagnostic_counters.get(key, 0)) + int(amount)

    @staticmethod
    def _sequence_len(input_ids: Any) -> int:
        shape = getattr(input_ids, "shape", ())
        if len(shape) >= 2:
            return int(shape[1])
        if shape:
            return int(shape[0])
        return 1

    def _forward_ar_capabilities(self) -> tuple[bool, bool]:
        if (
            self._forward_ar_supports_emit_logits is None
            or self._forward_ar_supports_logits_keep is None
        ):
            try:
                params = py_inspect.signature(self.model.__call__).parameters
            except Exception:
                params = {}
            accepts_kwargs = any(
                param.kind == py_inspect.Parameter.VAR_KEYWORD
                for param in params.values()
            )
            patched_kwargs = bool(self.mtp_enabled and accepts_kwargs)
            self._forward_ar_supports_emit_logits = (
                "emit_logits" in params or patched_kwargs
            )
            self._forward_ar_supports_logits_keep = (
                "logits_keep" in params or patched_kwargs
            )
        return (
            bool(self._forward_ar_supports_emit_logits),
            bool(self._forward_ar_supports_logits_keep),
        )

    def forward_ar(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        self._count("forward_ar_hidden_calls" if return_hidden else "forward_ar_plain_calls")
        if not self.mtp_enabled and return_hidden:
            raise RuntimeError("return_hidden requires an MTP-patched runtime")
        kwargs = {}
        if hidden_variant is not None:
            kwargs["hidden_variant"] = hidden_variant
        supports_emit_logits, supports_logits_keep = self._forward_ar_capabilities()
        if supports_emit_logits:
            kwargs["emit_logits"] = bool(emit_logits)
        elif not emit_logits:
            self._count("forward_ar_emit_logits_unsupported")
        if logits_keep is not None and supports_logits_keep:
            kwargs["logits_keep"] = int(logits_keep)
        elif logits_keep is not None:
            self._count("forward_ar_logits_keep_unsupported")
        sequence_len = self._sequence_len(input_ids)
        if bool(emit_logits) or not supports_emit_logits:
            if logits_keep is not None and supports_logits_keep:
                emitted = min(sequence_len, max(1, int(logits_keep)))
            else:
                emitted = sequence_len
            self._count("logits_tokens_emitted", emitted)
            if emitted == 1:
                self._count("final_logits_tokens_emitted", 1)
            else:
                self._count("full_logits_tokens_emitted", emitted)
        if not return_hidden and hidden_variant is None and not kwargs:
            return self.model(input_ids, cache=cache)
        return self.model(
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            **kwargs,
        )

    def forward_ar_capture(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        capture_backend: str | None = None,
    ):
        from .gdn_capture import forward_with_gdn_capture

        return forward_with_gdn_capture(
            self.model,
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            capture_backend=capture_backend,
        )

    def draft_mtp(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache=None,
        concat_order: str | None = None,
        return_hidden: bool = False,
        mtp_hidden_variant: str = "post_norm",
        mtp_depth: int | None = None,
        position_offset: int | None = None,
    ):
        if not self.mtp_enabled:
            raise RuntimeError("MTP is not enabled for this runtime")
        self._count("draft_mtp_calls")
        with mtp_adapter_depth(self.model, mtp_depth):
            kwargs = {
                "mtp_cache": mtp_cache,
                "concat_order": concat_order,
                "return_hidden": return_hidden,
                "mtp_hidden_variant": mtp_hidden_variant,
                "position_offset": position_offset,
            }
            try:
                params = py_inspect.signature(self.model.mtp_forward).parameters
            except Exception:
                params = {}
            if "mtp_depth" in params:
                kwargs["mtp_depth"] = mtp_depth
            return self.model.mtp_forward(hidden_states, next_token_ids, **kwargs)

    def update_mtp_cache(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache=None,
        concat_order: str | None = None,
        position_offset: int | None = None,
    ):
        if not self.mtp_enabled:
            raise RuntimeError("MTP is not enabled for this runtime")
        self._count("update_mtp_cache_calls")
        update = getattr(self.model, "mtp_update_cache", None)
        if update is not None:
            kwargs = {
                "mtp_cache": mtp_cache,
                "concat_order": concat_order,
                "position_offset": position_offset,
            }
            try:
                params = py_inspect.signature(update).parameters
            except Exception:
                params = {}
            if "mtp_depth" in params:
                kwargs["mtp_depth"] = None
            return update(hidden_states, next_token_ids, **kwargs)
        _logits, hidden = self.model.mtp_forward(
            hidden_states,
            next_token_ids,
            mtp_cache=mtp_cache,
            concat_order=concat_order,
            return_hidden=True,
            mtp_hidden_variant="post_norm",
            position_offset=position_offset,
        )
        return hidden

    def make_cache(self):
        inner = getattr(self.model, "language_model", self.model)
        cache = inner.make_cache()
        from .cache_state import (
            configure_owned_recurrent_state_cache,
            configure_tail_owned_attention_kv_cache,
        )

        configure_owned_recurrent_state_cache(cache)
        configure_tail_owned_attention_kv_cache(cache)
        return cache

    def make_mtp_cache(self):
        if not self.mtp_enabled:
            raise RuntimeError("MTP is not enabled for this runtime")
        self._count("make_mtp_cache_calls")
        cache = self.model.make_mtp_cache()
        from .cache_state import configure_mtp_attention_kv_cache

        configure_mtp_attention_kv_cache(cache)
        return cache


def attach(
    model,
    tokenizer,
    model_path: Path | str,
    *,
    mtp: bool = True,
    contract: MTPContract | None = None,
    mtp_adapter: Path | str | None = None,
) -> MTPLXRuntime:
    """Attach MTP support to an ALREADY-LOADED ``(model, tokenizer)`` pair.

    Same final state as ``load()`` but skips the ``mlx_lm.load`` call —
    used when the model is already in memory (e.g. mlx_lm.server has
    already loaded it). Avoids the +N GB duplicate-model footprint.
    Modifies ``model`` in place: injects MTP head, configures
    attention-split + native MLP. Caller MUST NOT use ``model`` for
    non-MTP inference afterward unless that's the only path (currently
    fine in ``optiq serve --mtp`` because all generation is routed
    through OptiqEngine).
    """
    path = resolve_model_dir(model_path)
    config = load_config(path)
    contract = (contract or MTPContract()).with_config_defaults(config)
    mtp_enabled = False
    if mtp:
        from .deepseek_mtp_patch import inject_deepseek_mtp_support, is_deepseek_mtp_config
        from .glm_mtp_patch import inject_glm_mtp_support, is_glm_mtp_config
        from .mimo_mtp_patch import inject_mimo_mtp_support, is_mimo_mtp_config
        from .nemotron_h_mtp_patch import inject_nemotron_h_mtp_support, is_nemotron_h_mtp_config

        if is_nemotron_h_mtp_config(config):
            mtp_enabled = inject_nemotron_h_mtp_support(model, path, config, contract)
        elif is_mimo_mtp_config(config):
            mtp_enabled = inject_mimo_mtp_support(model, path, config, contract)
        elif is_glm_mtp_config(config):
            mtp_enabled = inject_glm_mtp_support(model, path, config, contract)
        elif is_deepseek_mtp_config(config):
            mtp_enabled = inject_deepseek_mtp_support(model, path, config, contract)
        else:
            mtp_enabled = inject_mtp_support(model, path, config, contract)
        if not mtp_enabled or not validate_mtp_support(model):
            raise RuntimeError(f"MTP injection failed for {path}")
    from .attention_split import configure_split_full_attention
    from .native_mlp import configure_native_mlp

    configure_split_full_attention(model)
    configure_native_mlp(model)
    adapter_path = Path(mtp_adapter) if mtp_adapter is not None else None
    adapter_metadata = None
    if adapter_path is not None:
        if not mtp_enabled:
            raise RuntimeError("MTP adapter requires mtp=True")
        adapter_metadata = install_saved_mtp_lora_adapter(model, adapter_path)
    return MTPLXRuntime(
        model, tokenizer, path, mtp_enabled, contract,
        mtp_adapter_path=adapter_path,
        mtp_adapter_metadata=adapter_metadata,
    )


def load(
    model_path: Path | str,
    *,
    mtp: bool = True,
    contract: MTPContract | None = None,
    mtp_adapter: Path | str | None = None,
) -> MTPLXRuntime:
    """Load an MLX model and optionally inject native MTP support.

    Prefer ``attach()`` when the model is already loaded (e.g. by
    ``mlx_lm.server``) — that path skips the second ``mlx_lm.load``
    and avoids the +N GB duplicate-model footprint in unified memory.
    """
    from mlx_lm.utils import load as mlx_lm_load
    model, tokenizer = mlx_lm_load(str(Path(model_path)))
    return attach(
        model, tokenizer, model_path,
        mtp=mtp, contract=contract, mtp_adapter=mtp_adapter,
    )


def inspect(path: Path | str):
    return inspect_model(path)
