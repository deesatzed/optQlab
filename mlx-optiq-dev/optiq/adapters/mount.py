"""Mounted LoRA — reversible, per-request adapter switching.

mlx-lm's stock ``load_adapters`` *merges* adapter weights into the base
model: it replaces each targeted linear with a ``LoRALinear`` that folds
the LoRA residual into the same module, and the only way to get back to
the unadapted base is to reload the whole model. That rules out
per-request adapter switching during serving.

This module provides the alternative: **mounted** LoRA layers that keep
adapter weights in their own dict, gated by a ContextVar the server can
flip per request.

Key classes:

  * ``MountedLoRALinear`` — drop-in replacement for ``nn.Linear`` /
    ``nn.QuantizedLinear`` that holds a dict of ``{adapter_id: (A, B,
    scale)}``. Forward adds the adapter's residual if one is active;
    otherwise behaves exactly like the base linear.

  * ``AdapterActivation`` — context manager that sets the active adapter
    id for the duration of its ``with`` block. Uses a ``ContextVar``
    so concurrent requests in different asyncio tasks / threads don't
    interfere.

  * ``mount_adapter_on_model`` — utility that takes a model and an
    AdapterInfo, loads the adapter's weights off disk, and registers
    them on every MountedLoRALinear in the model.

  * ``prepare_model_for_mounted_lora`` — walks a model and replaces
    every ``q_proj``/``v_proj`` (configurable via target suffixes)
    with a ``MountedLoRALinear``. Idempotent.
"""

from __future__ import annotations

import contextvars
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten


# ---------------------------------------------------------------------------
# Per-request active-adapter state
# ---------------------------------------------------------------------------
_active_adapter: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_optiq_active_adapter", default=None
)


def set_active_adapter(adapter_id: Optional[str]) -> contextvars.Token:
    """Set the active adapter id for the current context.

    Returns a Token that should be passed to ``reset_active_adapter`` when
    the request is done. Prefer the ``AdapterActivation`` context manager
    which handles the reset automatically.
    """
    return _active_adapter.set(adapter_id)


def reset_active_adapter(token: contextvars.Token) -> None:
    _active_adapter.reset(token)


def get_active_adapter() -> Optional[str]:
    """Return the active adapter id seen by ``MountedLoRALinear.__call__``.

    Resolution order:
      1. The cross-thread serve-time pin (``set_serve_adapter`` /
         ``with serve_adapter_active(...)``). Used by ``optiq serve`` and
         the Labs server where the HTTP handler thread and mlx_lm.server's
         dedicated ``_generation_thread`` are different OS threads; a
         ``ContextVar`` set in one is invisible to the other.
      2. The per-context ``ContextVar`` (this module's original primitive,
         preserved for in-process / asyncio callers that don't have a
         threading split).
    """
    pin = _serve_active_adapter
    if pin is not None:
        return pin
    return _active_adapter.get()


# ---------------------------------------------------------------------------
# Cross-thread serve-time pin
# ---------------------------------------------------------------------------
# ContextVar is per-thread. mlx_lm.server runs each chat completion through
# a *dedicated* generation thread (ResponseGenerator._generation_thread),
# distinct from the per-request HTTP handler thread that we patch in
# ``optiq.serve.install_multi_adapter``. Setting a ContextVar in the
# handler thread is invisible to the generation thread, so the adapter
# silently no-ops at forward time. The pin below sidesteps that: it is
# a plain module-level variable, gated by a re-entrant lock so the patch
# can serialize adapter-selecting requests around the original do_POST
# (which blocks until generation completes).
_serve_active_adapter: Optional[str] = None
_serve_pin_lock = threading.RLock()


def set_serve_adapter(adapter_id: Optional[str]) -> None:
    """Pin the active adapter for cross-thread serving. Call with ``None``
    to clear. Callers should hold ``_serve_pin_lock`` for the duration of
    the request to keep concurrent requests from racing on the pin."""
    global _serve_active_adapter
    _serve_active_adapter = adapter_id


class serve_adapter_active:
    """Context manager wrapper around ``set_serve_adapter`` that also takes
    out ``_serve_pin_lock`` so the pin is exclusively held for the duration
    of one HTTP request. Used by ``optiq.serve``'s patched ``do_POST``."""

    def __init__(self, adapter_id: Optional[str]):
        self.adapter_id = adapter_id
        self._held = False

    def __enter__(self) -> "serve_adapter_active":
        if self.adapter_id is not None:
            _serve_pin_lock.acquire()
            self._held = True
            set_serve_adapter(self.adapter_id)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._held:
            try:
                set_serve_adapter(None)
            finally:
                _serve_pin_lock.release()
                self._held = False


class AdapterActivation:
    """Context manager: ``with AdapterActivation('my-agent'): generate(...)``.

    Activation is scoped to the current asyncio task / thread. Concurrent
    requests with different adapter ids won't step on each other.
    """

    def __init__(self, adapter_id: Optional[str]):
        self.adapter_id = adapter_id
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> "AdapterActivation":
        self._token = _active_adapter.set(self.adapter_id)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _active_adapter.reset(self._token)
            self._token = None


# ---------------------------------------------------------------------------
# MountedLoRALinear
# ---------------------------------------------------------------------------
@dataclass
class _LoRAWeights:
    """Per-adapter LoRA parameters for a single linear.

    Shape convention matches mlx-lm's ``LoRALinear``:
      * ``A``: (in_features, rank)   — also called ``lora_a``
      * ``B``: (rank, out_features)  — also called ``lora_b``
    Forward applied as ``(x @ A) @ B * scale``, no transposes needed.
    """
    A: mx.array
    B: mx.array
    scale: float
    rank: int


class MountedLoRALinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` / ``nn.QuantizedLinear`` that
    supports mounting multiple LoRA adapters and gating which one is
    active per-request via a ContextVar.

    Attributes:
        base: the original linear module (nn.Linear or nn.QuantizedLinear).
              Weights are frozen as far as LoRA is concerned.
        adapters: ``{adapter_id: _LoRAWeights}`` mounted on this layer.
    """

    def __init__(self, base: nn.Module, in_features: int, out_features: int):
        super().__init__()
        self.base = base
        self.in_features = in_features
        self.out_features = out_features
        # mlx.nn doesn't follow plain-dict attributes into parameters so the
        # adapter weights here are NOT part of the module's trainable
        # parameter set — they're bolted on at runtime.
        self.adapters: Dict[str, _LoRAWeights] = {}

    def add_adapter(self, adapter_id: str, A: mx.array, B: mx.array,
                    scale: float, rank: int) -> None:
        # mlx-lm convention: A is (in_features, rank), B is (rank, out_features)
        if A.shape != (self.in_features, rank):
            raise ValueError(
                f"A shape {A.shape} incompatible; expected "
                f"({self.in_features}, {rank})"
            )
        if B.shape != (rank, self.out_features):
            raise ValueError(
                f"B shape {B.shape} incompatible; expected "
                f"({rank}, {self.out_features})"
            )
        self.adapters[adapter_id] = _LoRAWeights(A=A, B=B, scale=scale, rank=rank)

    def remove_adapter(self, adapter_id: str) -> bool:
        return self.adapters.pop(adapter_id, None) is not None

    def has_adapter(self, adapter_id: str) -> bool:
        return adapter_id in self.adapters

    def __call__(self, x: mx.array) -> mx.array:
        out = self.base(x)
        aid = get_active_adapter()
        if aid is not None:
            # Stacking syntax: "sft+dpo" or "sft,dpo" applies multiple
            # registered adapters simultaneously (their LoRA residuals
            # sum). Plain "sft" is a single-adapter request as before.
            # The "+" form is friendlier than "," for HTTP query strings
            # and request bodies; both accepted.
            if "+" in aid or "," in aid:
                sep = "+" if "+" in aid else ","
                ids = [s.strip() for s in aid.split(sep) if s.strip()]
            else:
                ids = [aid]
            for single_id in ids:
                w = self.adapters.get(single_id)
                if w is None:
                    continue
                # LoRA residual: (x @ A) @ B * scale — mlx-lm convention
                lora_out = (x @ w.A) @ w.B
                out = out + lora_out * w.scale
        return out


# ---------------------------------------------------------------------------
# Model walking / mounting
# ---------------------------------------------------------------------------
def _match_target_suffix(name: str, suffixes: Tuple[str, ...]) -> bool:
    tail = name.split(".")[-1]
    return tail in suffixes


def _iter_transformer_blocks(model: nn.Module):
    """Yield ``(layer_idx, block)`` pairs for the text transformer.

    Handles pure-LLM (``model.model.layers``) and VLM-wrapped
    (``model.language_model.model.layers``) layouts.
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        for i, b in enumerate(inner.layers):
            yield i, b
        return
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            for i, b in enumerate(inner.layers):
                yield i, b
            return
    raise RuntimeError(
        f"could not find transformer blocks on {type(model).__name__}"
    )


def _infer_linear_shape(m: nn.Module) -> Optional[Tuple[int, int]]:
    """Return (in_features, out_features) for an nn.Linear or
    nn.QuantizedLinear. None for other types."""
    if isinstance(m, nn.QuantizedLinear):
        # Quantized linear stores weight as a packed uint32. Its
        # out_features and in_features are accessible as attributes.
        out_f = m.weight.shape[0]
        # packed last dim — multiply by (32 / bits)
        packed_last = m.weight.shape[1]
        in_f = packed_last * (32 // m.bits)
        return in_f, out_f
    if isinstance(m, nn.Linear):
        return m.weight.shape[1], m.weight.shape[0]
    return None


def prepare_model_for_mounted_lora(
    model: nn.Module,
    target_suffixes: Tuple[str, ...] = (
        # All 7 Unsloth target modules. Matches the OptiQ LoRA trainer's
        # default (``optiq.lora.config.UNSLOTH_TARGET_MODULES``) so the
        # mounted-LoRA path uses every layer the adapter was trained on,
        # not just attention's q/v slice. Modules the adapter doesn't have
        # weights for are silently skipped per-layer.
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
) -> int:
    """Walk the model and replace each matching linear with a
    MountedLoRALinear wrapping the original.

    Idempotent: skips modules already wrapped.

    Returns the number of linears wrapped. Use this ONCE at server
    startup; subsequent ``mount_adapter_on_model`` calls only register
    per-adapter weights without re-walking.
    """
    wrapped = 0
    for layer_idx, block in _iter_transformer_blocks(model):
        replacements = []
        for name, m in block.named_modules():
            if not _match_target_suffix(name, target_suffixes):
                continue
            if isinstance(m, MountedLoRALinear):
                continue
            shape = _infer_linear_shape(m)
            if shape is None:
                continue
            in_f, out_f = shape
            wrapper = MountedLoRALinear(base=m, in_features=in_f, out_features=out_f)
            replacements.append((name, wrapper))
            wrapped += 1
        if replacements:
            block.update_modules(tree_unflatten(replacements))
    return wrapped


def _load_adapter_weights(adapter_dir: Path) -> Dict[str, mx.array]:
    """Read the adapter safetensors. Supports both mlx-lm save name
    ``adapters.safetensors`` and PEFT name ``adapter_model.safetensors``."""
    for name in ("adapters.safetensors", "adapter_model.safetensors"):
        p = adapter_dir / name
        if p.exists():
            return mx.load(str(p))
    raise FileNotFoundError(
        f"no adapter weights at {adapter_dir}/adapters.safetensors "
        f"or {adapter_dir}/adapter_model.safetensors"
    )


def _load_adapter_config(adapter_dir: Path) -> dict:
    for name in ("optiq_lora_config.json", "adapter_config.json"):
        p = adapter_dir / name
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"no adapter_config.json in {adapter_dir}")


def mount_adapter_on_model(
    model: nn.Module,
    adapter_id: str,
    adapter_dir: Path,
    target_suffixes: Tuple[str, ...] = (
        # All 7 Unsloth target modules. Matches the OptiQ LoRA trainer's
        # default (``optiq.lora.config.UNSLOTH_TARGET_MODULES``) so the
        # mounted-LoRA path uses every layer the adapter was trained on,
        # not just attention's q/v slice. Modules the adapter doesn't have
        # weights for are silently skipped per-layer.
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
) -> int:
    """Load adapter weights from ``adapter_dir`` and register them on
    every MountedLoRALinear in the model under the key ``adapter_id``.

    Expects the model was already prepared via
    ``prepare_model_for_mounted_lora``. If it wasn't, this function
    prepares it lazily.

    Returns the number of layers to which the adapter was mounted.
    """
    prepare_model_for_mounted_lora(model, target_suffixes=target_suffixes)

    weights = _load_adapter_weights(Path(adapter_dir))
    config = _load_adapter_config(Path(adapter_dir))

    # Alpha / scale: mlx-lm's adapter_config writes ``lora_parameters.scale``;
    # PEFT writes ``lora_alpha`` and ``r``, where alpha = rank * scale.
    scale = None
    if "lora_parameters" in config and isinstance(config["lora_parameters"], dict):
        scale = float(config["lora_parameters"].get("scale", 20.0))
    if scale is None:
        alpha = float(config.get("lora_alpha", 16))
        rank_fallback = int(config.get("r", 8))
        scale = alpha / rank_fallback if rank_fallback else 1.0

    mounted = 0
    for layer_idx, block in _iter_transformer_blocks(model):
        for name, m in block.named_modules():
            if not isinstance(m, MountedLoRALinear):
                continue
            # Locate this layer's LoRA weights in the adapter safetensors.
            # mlx-lm writes keys like
            # "model.layers.{i}.self_attn.q_proj.lora_a" etc.
            # Our trainer also emits that exact format (PEFT-compatible).
            A_key, B_key = _find_weight_pair(weights, layer_idx, name)
            if A_key is None or B_key is None:
                continue
            A = weights[A_key]
            B = weights[B_key]
            # mlx-lm convention: A is (in_features, rank), B is (rank, out_features)
            rank = A.shape[1]
            m.add_adapter(adapter_id, A=A, B=B, scale=scale, rank=rank)
            mounted += 1
    return mounted


def _find_weight_pair(weights: Dict[str, mx.array],
                       layer_idx: int,
                       module_suffix: str) -> Tuple[Optional[str], Optional[str]]:
    """Probe the likely key layouts for (A, B) of a layer's LoRA weights."""
    probe_prefixes = [
        # pure LLM
        f"model.layers.{layer_idx}.{module_suffix}",
        # VLM-wrapped
        f"language_model.model.layers.{layer_idx}.{module_suffix}",
    ]
    A_candidates = ("lora_a", "lora_A", "lora_A.weight")
    B_candidates = ("lora_b", "lora_B", "lora_B.weight")

    for prefix in probe_prefixes:
        for a_suf in A_candidates:
            A_key = f"{prefix}.{a_suf}"
            if A_key in weights:
                for b_suf in B_candidates:
                    B_key = f"{prefix}.{b_suf}"
                    if B_key in weights:
                        return A_key, B_key
    return None, None


def unmount_adapter_from_model(model: nn.Module, adapter_id: str) -> int:
    """Remove ``adapter_id`` from every MountedLoRALinear. Returns the
    number of layers from which it was removed. Does NOT unwrap the
    MountedLoRALinear back into a plain linear — the mount stays in
    place for other adapters (or future re-mount of this one)."""
    removed = 0
    for _, block in _iter_transformer_blocks(model):
        for _, m in block.named_modules():
            if isinstance(m, MountedLoRALinear):
                if m.remove_adapter(adapter_id):
                    removed += 1
    return removed


def list_mounted_adapters(model: nn.Module) -> Dict[str, int]:
    """Return ``{adapter_id: n_layers}`` — how many layers each adapter
    is currently mounted on."""
    counts: Dict[str, int] = {}
    for _, block in _iter_transformer_blocks(model):
        for _, m in block.named_modules():
            if isinstance(m, MountedLoRALinear):
                for aid in m.adapters:
                    counts[aid] = counts.get(aid, 0) + 1
    return counts
