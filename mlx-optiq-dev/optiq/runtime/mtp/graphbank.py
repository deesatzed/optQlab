"""Speculative decode graph-bank scaffolding for MLX.

The first useful job of this module is to make graph-capture eligibility
explicit.  The current Qwen3.6 MLX cache keeps full-attention positions as
Python integers, so a safe compiled decode graph cannot replay across decode
steps until those offsets become tensor inputs/outputs.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import mlx.core as mx

from .gdn_capture import resolve_gdn_capture_backend


@dataclass
class GraphBankStats:
    calls: int = 0
    compiled_calls: int = 0
    fallback_calls: int = 0
    promoted_cache_entries: int = 0
    warmed_lengths: list[int] = field(default_factory=list)
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    compile_errors: dict[str, int] = field(default_factory=dict)
    promotion_failures: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpecDecodeGraphBank:
    """Fixed-length verify dispatcher with safe fallback instrumentation.

    `mx.compile` can capture array trees, but the stock MLX Qwen3.6 cache also
    stores decode offsets as Python integers.  Replaying a compiled closure that
    captured those integers would use stale RoPE/mask positions, so the safe
    backend refuses to compile until explicit tensor cache state lands.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        max_verify_len: int = 6,
        allow_python_cache_capture: bool = False,
        promote_tensor_offsets: bool = True,
        capture_backend: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.max_verify_len = max_verify_len
        self.allow_python_cache_capture = allow_python_cache_capture
        self.promote_tensor_offsets = promote_tensor_offsets
        self.capture_backend = resolve_gdn_capture_backend(capture_backend)
        self._capture_accepts_backend = _accepts_capture_backend(runtime)
        self.stats = GraphBankStats()
        self._compiled: dict[tuple[str, int, tuple[int, ...]], Any] = {}

    def forward_ar(self, input_ids, *, cache=None, return_hidden: bool = True):
        return self._forward(
            "forward",
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
        )

    def forward_ar_capture(self, input_ids, *, cache=None, return_hidden: bool = True):
        return self._forward(
            "capture",
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
        )

    def _forward(self, kind: str, input_ids, *, cache=None, return_hidden: bool = True):
        started = time.perf_counter()
        self.stats.calls += 1
        length = _decode_length(input_ids)
        reason = self._fallback_reason(length, cache)
        if reason is not None:
            return self._fallback(
                kind,
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                reason=reason,
                started=started,
            )

        try:
            key = (kind, length, _cache_container_signature(cache))
            fn = self._compiled.get(key)
            if fn is None:
                if kind == "capture":
                    fn = self._compile_capture_length(
                        length,
                        cache=cache,
                        return_hidden=return_hidden,
                    )
                else:
                    fn = self._compile_length(
                        length,
                        cache=cache,
                        return_hidden=return_hidden,
                    )
                self._compiled[key] = fn
            result = fn(input_ids)
            self.stats.compiled_calls += 1
            self.stats.elapsed_s += time.perf_counter() - started
            return result
        except Exception as exc:  # pragma: no cover - exercised by real MLX cache probes
            key = type(exc).__name__
            self.stats.compile_errors[key] = self.stats.compile_errors.get(key, 0) + 1
            return self._fallback(
                kind,
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                reason=f"compile_error:{key}",
                started=started,
            )

    def warm(
        self,
        lengths: range | list[int] | tuple[int, ...],
        *,
        cache_factory,
        token_factory,
    ) -> None:
        """Warm eligible shapes using caller-provided disposable cache/tokens."""
        for length in lengths:
            if length < 1 or length > self.max_verify_len:
                continue
            cache = cache_factory()
            tokens = token_factory(length)
            self.forward_ar(tokens, cache=cache, return_hidden=True)
            if length not in self.stats.warmed_lengths:
                self.stats.warmed_lengths.append(length)

    def to_dict(self) -> dict[str, Any]:
        data = self.stats.to_dict()
        data["max_verify_len"] = self.max_verify_len
        data["allow_python_cache_capture"] = self.allow_python_cache_capture
        data["promote_tensor_offsets"] = self.promote_tensor_offsets
        data["capture_backend"] = self.capture_backend
        data["compiled_lengths"] = sorted({length for _, length, _ in self._compiled})
        data["compiled_paths"] = [
            f"{kind}:{length}"
            for kind, length in sorted({(kind, length) for kind, length, _ in self._compiled})
        ]
        data["compiled_entry_count"] = len(self._compiled)
        return data

    def reset(self) -> None:
        """Drop compiled closures after cache container identity changes."""
        self._compiled.clear()

    def _fallback_reason(self, length: int, cache: Any) -> str | None:
        if length < 1:
            return "invalid_length"
        if length > self.max_verify_len:
            return "length_outside_graphbank"
        if cache is None:
            return None
        if self.allow_python_cache_capture:
            return None
        if self.promote_tensor_offsets:
            promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=length)
            self.stats.promoted_cache_entries += promoted
            for reason, count in failures.items():
                self.stats.promotion_failures[reason] = (
                    self.stats.promotion_failures.get(reason, 0) + count
                )
        if cache_has_python_offsets(cache):
            return "python_cache_offsets"
        return None

    def _fallback(
        self,
        kind: str,
        input_ids,
        *,
        cache,
        return_hidden: bool,
        reason: str,
        started: float,
    ):
        self.stats.fallback_calls += 1
        self.stats.fallback_reasons[reason] = self.stats.fallback_reasons.get(reason, 0) + 1
        if kind == "capture":
            result = self._runtime_forward_ar_capture(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
            )
        else:
            result = self.runtime.forward_ar(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
            )
        self.stats.elapsed_s += time.perf_counter() - started
        return result

    def _compile_length(self, length: int, *, cache: Any, return_hidden: bool):
        def verify_fn(input_ids):
            if _decode_length(input_ids) != length:
                raise ValueError("compiled verify length mismatch")
            return self.runtime.forward_ar(input_ids, cache=cache, return_hidden=return_hidden)

        return mx.compile(
            verify_fn,
            inputs=cache_array_tree(cache),
            outputs=cache_array_tree(cache),
        )

    def _compile_capture_length(self, length: int, *, cache: Any, return_hidden: bool):
        def verify_fn(input_ids):
            if _decode_length(input_ids) != length:
                raise ValueError("compiled verify length mismatch")
            return self._runtime_forward_ar_capture(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
            )

        return mx.compile(
            verify_fn,
            inputs=cache_array_tree(cache),
            outputs=cache_array_tree(cache),
        )

    def _runtime_forward_ar_capture(self, input_ids, *, cache=None, return_hidden: bool = True):
        if self._capture_accepts_backend:
            return self.runtime.forward_ar_capture(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                capture_backend=self.capture_backend,
            )
        return self.runtime.forward_ar_capture(
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
        )


def _decode_length(input_ids: Any) -> int:
    shape = getattr(input_ids, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("input_ids must have shape [batch, tokens]")
    return int(shape[1])


def _cache_container_signature(cache: Any) -> tuple[int, ...]:
    if cache is None:
        return ()
    signature: list[int] = [id(cache)]
    for entry in cache:
        signature.append(id(entry))
        if entry is None:
            continue
        if hasattr(entry, "compile_state"):
            state = getattr(entry, "compile_state")
            if isinstance(state, list):
                signature.extend(id(item) for item in state)
            continue
        if hasattr(entry, "cache"):
            signature.append(id(getattr(entry, "cache")))
            continue
        state = getattr(entry, "state", None)
        if isinstance(state, list):
            signature.append(id(state))
    return tuple(signature)


def _accepts_capture_backend(runtime: Any) -> bool:
    import inspect

    try:
        signature = inspect.signature(runtime.forward_ar_capture)
    except (AttributeError, TypeError, ValueError):
        return False
    return "capture_backend" in signature.parameters


def cache_has_python_offsets(cache: Any) -> bool:
    for entry in cache or []:
        if entry is None:
            continue
        offset = getattr(entry, "offset", None)
        if isinstance(offset, int):
            return True
        idx = getattr(entry, "_idx", None)
        if isinstance(idx, int):
            return True
    return False


class TensorOffsetKVCache:
    """Full-attention KV cache adapter with array-backed mutable offset.

    Stock `KVCache.offset` is a Python integer.  In a compiled verify graph that
    integer is graph-constant state, so RoPE and mask positions can silently go
    stale.  This adapter keeps the existing key/value buffers, stores the offset
    in `cache[2]`, and mutates the three-array state through operations visible
    to `mx.compile(inputs=..., outputs=...)`.
    """

    def __init__(
        self,
        keys: mx.array,
        values: mx.array,
        offset: int | mx.array,
        *,
        step: int = 256,
    ) -> None:
        offset_array = (
            offset
            if isinstance(offset, mx.array)
            else mx.array(offset, dtype=mx.int32)
        )
        self.cache = [keys, values, offset_array]
        self.rollback_state = [None, None, None]
        self.step = step

    @classmethod
    def from_kv_cache(cls, entry: Any, *, reserve_tokens: int) -> "TensorOffsetKVCache":
        cache = cls(
            entry.keys,
            entry.values,
            entry.offset,
            step=getattr(entry, "step", 256),
        )
        cache.ensure_capacity(int(entry.offset) + reserve_tokens)
        return cache

    @property
    def keys(self):
        return self.cache[0]

    @keys.setter
    def keys(self, value):
        self.cache[0] = value

    @property
    def values(self):
        return self.cache[1]

    @values.setter
    def values(self, value):
        self.cache[1] = value

    @property
    def offset(self):
        return self.cache[2]

    @offset.setter
    def offset(self, value):
        self.cache[2] = (
            value
            if isinstance(value, mx.array)
            else mx.array(value, dtype=mx.int32)
        )

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, value):
        self.cache = value

    @property
    def compile_state(self):
        return [self.cache, self.rollback_state]

    def ensure_capacity(self, needed: int) -> None:
        if self.keys is None or self.values is None:
            return
        capacity = int(self.keys.shape[2])
        if needed <= capacity:
            return
        new_capacity = ((needed + self.step - 1) // self.step) * self.step
        extra = new_capacity - capacity
        k_shape = (*self.keys.shape[:2], extra, self.keys.shape[3])
        v_shape = (*self.values.shape[:2], extra, self.values.shape[3])
        self.keys = mx.concatenate(
            [self.keys, mx.zeros(k_shape, dtype=self.keys.dtype)],
            axis=2,
        )
        self.values = mx.concatenate(
            [self.values, mx.zeros(v_shape, dtype=self.values.dtype)],
            axis=2,
        )

    def update_and_fetch(self, keys, values):
        steps = int(keys.shape[2])
        self.rollback_state[0] = self.cache[2]
        self.rollback_state[1] = mx.slice(
            self.cache[0],
            self.cache[2],
            axes=(2,),
            slice_size=keys.shape,
        )
        self.rollback_state[2] = mx.slice(
            self.cache[1],
            self.cache[2],
            axes=(2,),
            slice_size=values.shape,
        )
        self.cache[0] = mx.slice_update(
            self.cache[0],
            keys,
            self.cache[2],
            axes=(2,),
        )
        self.cache[1] = mx.slice_update(
            self.cache[1],
            values,
            self.cache[2],
            axes=(2,),
        )
        self.cache[2] = self.cache[2] + steps
        return self.cache[0], self.cache[1]

    def make_mask(self, N: int, window_size=None, return_array: bool = False):
        del return_array
        if self.keys is None:
            return None
        capacity = int(self.keys.shape[2])
        rinds = mx.arange(capacity)
        linds = self.cache[2] + mx.arange(N)
        mask = linds[:, None] >= rinds[None, :]
        if window_size is not None:
            mask = mask & (linds[:, None] < rinds[None, :] + window_size)
        return mask

    def size(self):
        value = self.cache[2]
        mx.eval(value)
        return int(value.item())

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = int(n)
        if (
            self.rollback_state[0] is not None
            and self.rollback_state[1] is not None
            and self.rollback_state[2] is not None
            and int(self.rollback_state[1].shape[2]) == n
        ):
            self.cache[0] = mx.slice_update(
                self.cache[0],
                self.rollback_state[1],
                self.rollback_state[0],
                axes=(2,),
            )
            self.cache[1] = mx.slice_update(
                self.cache[1],
                self.rollback_state[2],
                self.rollback_state[0],
                axes=(2,),
            )
            self.cache[2] = self.rollback_state[0]
        else:
            self.cache[2] = mx.maximum(
                self.cache[2] - n,
                mx.array(0, dtype=self.cache[2].dtype),
            )
        return n

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes + self.cache[2].nbytes


def promote_kv_cache_offsets(
    cache: Any,
    *,
    reserve_tokens: int,
) -> tuple[int, dict[str, int]]:
    """Replace stock full-attention KV caches with tensor-offset adapters."""
    promoted = 0
    failures: dict[str, int] = {}
    if cache is None:
        return promoted, failures
    for idx, entry in enumerate(cache):
        if entry is None:
            continue
        if isinstance(entry, TensorOffsetKVCache):
            entry.ensure_capacity(entry.size() + reserve_tokens)
            continue
        if _env_enabled("MTPLX_GRAPHBANK_PRESERVE_PAGED_KV"):
            try:
                from .cache_state import (
                    TensorOffsetVllmMetalPagedKVCache,
                    VllmMetalPagedKVCache,
                )
            except Exception:  # pragma: no cover - import guard for minimal test envs
                TensorOffsetVllmMetalPagedKVCache = None
                VllmMetalPagedKVCache = None
            if (
                VllmMetalPagedKVCache is not None
                and isinstance(entry, VllmMetalPagedKVCache)
            ):
                if entry.key_cache is None or entry.value_cache is None:
                    failures["empty_paged_kv_cache"] = (
                        failures.get("empty_paged_kv_cache", 0) + 1
                    )
                    continue
                cache[idx] = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(entry)
                promoted += 1
                continue
        offset = getattr(entry, "offset", None)
        if not isinstance(offset, int):
            continue
        if getattr(entry, "_idx", None) is not None:
            failures["rotating_or_indexed_cache"] = (
                failures.get("rotating_or_indexed_cache", 0) + 1
            )
            continue
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        if keys is None or values is None:
            failures["empty_kv_cache"] = failures.get("empty_kv_cache", 0) + 1
            continue
        if (
            len(getattr(keys, "shape", ())) != 4
            or len(getattr(values, "shape", ())) != 4
        ):
            failures["unsupported_kv_shape"] = failures.get("unsupported_kv_shape", 0) + 1
            continue
        cache[idx] = TensorOffsetKVCache.from_kv_cache(
            entry,
            reserve_tokens=reserve_tokens,
        )
        promoted += 1
    return promoted, failures


def _env_enabled(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def cache_array_tree(cache: Any) -> list[Any]:
    """Return the arrays a compiled closure can legally capture."""
    tree: list[Any] = []
    for entry in cache or []:
        if entry is None:
            tree.append(None)
            continue
        if hasattr(entry, "compile_state"):
            tree.append(getattr(entry, "compile_state"))
            continue
        if hasattr(entry, "cache"):
            tree.append(getattr(entry, "cache"))
            continue
        leaves = []
        for name in ("keys", "values", "left_padding", "lengths", "_lengths"):
            if hasattr(entry, name):
                leaves.append(getattr(entry, name))
        if not leaves and hasattr(entry, "state"):
            leaves.append(entry.state)
        tree.append(leaves)
    return tree
