"""Conservative cache snapshot helpers for correctness gates."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
import time
from typing import Any

from .attention_context import current_attention_phase

SUPPORTED_DETACH_MODES = {
    "eval_only",
    "contiguous_eval",
    "selected_slice_contiguous_eval",
    "metal_copy_leaf",
}


@dataclass(frozen=True)
class CacheSnapshot:
    states: tuple[Any, ...]
    meta_states: tuple[Any, ...]


def _normalize_detach_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("-", "_")
    if normalized not in SUPPORTED_DETACH_MODES:
        raise ValueError(
            "detach mode must be one of "
            f"{sorted(SUPPORTED_DETACH_MODES)}; got {mode!r}"
        )
    return normalized


class TailOwnedKVCache:
    """KV cache that owner-copies the newly produced attention tail.

    Stock MLX KV cache stores each new K/V slice directly into a persistent
    cache. This diagnostic keeps the same logical cache contract but cuts the
    tail tensor's lazy lineage before insertion, which is far cheaper than
    periodically copying the whole historical KV buffer.
    """

    def __init__(
        self,
        *,
        mode: str = "contiguous_eval",
        step: int = 256,
        keys: Any | None = None,
        values: Any | None = None,
        offset: int = 0,
    ) -> None:
        self.keys = keys
        self.values = values
        self.offset = int(offset)
        self.step = int(step)
        self.mode = _normalize_detach_mode(mode)
        self.tail_owner_updates = 0
        self.tail_owner_arrays = 0
        self.tail_owner_bytes = 0
        self.tail_owner_time_s = 0.0

    @classmethod
    def from_cache(cls, entry: Any, *, mode: str, step: int | None = None) -> "TailOwnedKVCache":
        return cls(
            mode=mode,
            step=int(step or getattr(entry, "step", 256)),
            keys=getattr(entry, "keys", None),
            values=getattr(entry, "values", None),
            offset=int(getattr(entry, "offset", 0)),
        )

    def _own_tail(self, keys: Any, values: Any) -> tuple[Any, Any]:
        started = time.perf_counter()
        owned_keys = detach_array_leaf(keys, mode=self.mode)
        owned_values = detach_array_leaf(values, mode=self.mode)
        self.tail_owner_time_s += time.perf_counter() - started
        self.tail_owner_updates += 1
        self.tail_owner_arrays += 2
        self.tail_owner_bytes += int(owned_keys.nbytes) + int(owned_values.nbytes)
        return owned_keys, owned_values

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        keys, values = self._own_tail(keys, values)
        prev = self.offset
        steps = int(keys.shape[2])
        if self.keys is None or (prev + steps) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + steps - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += steps
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def size(self) -> int:
        return int(self.offset)

    @property
    def state(self):
        if self.keys is None or self.values is None:
            return self.keys, self.values
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
        )

    @state.setter
    def state(self, value) -> None:
        self.keys, self.values = value
        self.offset = 0 if self.keys is None else int(self.keys.shape[2])

    @property
    def meta_state(self) -> tuple[str, ...]:
        return (str(self.step), str(self.offset), self.mode)

    @meta_state.setter
    def meta_state(self, value) -> None:
        if not value:
            return
        self.step = int(value[0])
        self.offset = int(value[1])
        if len(value) > 2:
            self.mode = _normalize_detach_mode(str(value[2]))

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(int(self.offset), int(n))
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self) -> bool:
        return self.keys is None

    @property
    def nbytes(self) -> int:
        if self.keys is None or self.values is None:
            return 0
        return int(self.keys.nbytes) + int(self.values.nbytes)

    def tail_owner_stats(self) -> dict[str, int | float | str]:
        return {
            "mode": self.mode,
            "updates": int(self.tail_owner_updates),
            "arrays": int(self.tail_owner_arrays),
            "bytes": int(self.tail_owner_bytes),
            "time_s": float(self.tail_owner_time_s),
        }


class BlockOwnedKVCache(TailOwnedKVCache):
    """Full-attention KV cache with independent physical token blocks."""

    def __init__(
        self,
        *,
        mode: str = "contiguous_eval",
        block_size: int = 1024,
        keys: Any | None = None,
        values: Any | None = None,
        offset: int = 0,
    ) -> None:
        self.block_size = int(block_size)
        self.key_blocks: list[Any] = []
        self.value_blocks: list[Any] = []
        self._pending_keys = None
        self._pending_values = None
        self._tail_shape: tuple[int, int, int] | None = None
        self._tail_dtypes: tuple[Any, Any] | None = None
        super().__init__(mode=mode, step=block_size, offset=0)
        if keys is not None and values is not None:
            self._load_contiguous_state(keys, values, int(offset))

    @classmethod
    def from_cache(
        cls,
        entry: Any,
        *,
        mode: str,
        block_size: int | None = None,
    ) -> "BlockOwnedKVCache":
        return cls(
            mode=mode,
            block_size=int(block_size or getattr(entry, "step", 1024) or 1024),
            keys=getattr(entry, "keys", None),
            values=getattr(entry, "values", None),
            offset=int(getattr(entry, "offset", 0)),
        )

    def _record_shape(self, keys: Any, values: Any) -> None:
        self._tail_shape = (
            int(keys.shape[0]),
            int(keys.shape[1]),
            int(keys.shape[3]),
        )
        self._tail_dtypes = (keys.dtype, values.dtype)

    def _new_blocks_like(self, keys: Any, values: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        B, n_kv_heads, _, k_head_dim = keys.shape
        v_head_dim = int(values.shape[3])
        key_block = mx.zeros(
            (B, n_kv_heads, self.block_size, k_head_dim),
            dtype=keys.dtype,
        )
        value_block = mx.zeros(
            (B, n_kv_heads, self.block_size, v_head_dim),
            dtype=values.dtype,
        )
        mx.eval(key_block, value_block)
        return key_block, value_block

    def _ensure_capacity_for(self, absolute_pos: int, keys: Any, values: Any) -> None:
        needed_blocks = (int(absolute_pos) // self.block_size) + 1
        while len(self.key_blocks) < needed_blocks:
            key_block, value_block = self._new_blocks_like(keys, values)
            self.key_blocks.append(key_block)
            self.value_blocks.append(value_block)

    def _finalize_block(self, block_index: int) -> None:
        import mlx.core as mx

        key_block = mx.contiguous(self.key_blocks[block_index])
        value_block = mx.contiguous(self.value_blocks[block_index])
        mx.eval(key_block, value_block)
        self.key_blocks[block_index] = key_block
        self.value_blocks[block_index] = value_block

    def _load_contiguous_state(self, keys: Any, values: Any, offset: int) -> None:
        self.key_blocks = []
        self.value_blocks = []
        self._pending_keys = None
        self._pending_values = None
        self.offset = 0
        total = int(offset)
        if total <= 0:
            return
        self._record_shape(keys, values)
        cursor = 0
        while cursor < total:
            take = min(self.block_size, total - cursor)
            key_block, value_block = self._new_blocks_like(keys, values)
            key_tail = keys[..., cursor : cursor + take, :]
            value_tail = values[..., cursor : cursor + take, :]
            key_tail, value_tail = self._own_tail(key_tail, value_tail)
            key_block[..., :take, :] = key_tail
            value_block[..., :take, :] = value_tail
            self.key_blocks.append(key_block)
            self.value_blocks.append(value_block)
            if take == self.block_size:
                self._finalize_block(len(self.key_blocks) - 1)
            cursor += take
        self.offset = total

    @property
    def keys(self):
        return self._active_arrays()[0]

    @keys.setter
    def keys(self, value) -> None:
        if value is None:
            self.key_blocks = []
            self.value_blocks = []
            self.offset = 0
            self._pending_keys = None
            return
        self._pending_keys = value
        if self._pending_values is not None:
            self._load_contiguous_state(
                self._pending_keys,
                self._pending_values,
                int(value.shape[2]),
            )

    @property
    def values(self):
        return self._active_arrays()[1]

    @values.setter
    def values(self, value) -> None:
        if value is None:
            self._pending_values = None
            return
        self._pending_values = value
        if self._pending_keys is not None:
            self._load_contiguous_state(
                self._pending_keys,
                self._pending_values,
                int(value.shape[2]),
            )

    def _active_arrays(self) -> tuple[Any | None, Any | None]:
        import mlx.core as mx

        if self.offset <= 0 or not self.key_blocks:
            return None, None
        full_blocks = self.offset // self.block_size
        partial = self.offset % self.block_size
        key_parts = list(self.key_blocks[:full_blocks])
        value_parts = list(self.value_blocks[:full_blocks])
        if partial:
            key_parts.append(self.key_blocks[full_blocks][..., :partial, :])
            value_parts.append(self.value_blocks[full_blocks][..., :partial, :])
        if len(key_parts) == 1:
            return key_parts[0], value_parts[0]
        return mx.concatenate(key_parts, axis=2), mx.concatenate(value_parts, axis=2)

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        self.update_without_fetch(keys, values)
        active_keys, active_values = self._active_arrays()
        return active_keys, active_values

    def update_without_fetch(self, keys: Any, values: Any) -> None:
        self._record_shape(keys, values)
        steps = int(keys.shape[2])
        cursor = 0
        while cursor < steps:
            absolute_pos = self.offset
            block_index = absolute_pos // self.block_size
            in_block = absolute_pos % self.block_size
            take = min(steps - cursor, self.block_size - in_block)
            self._ensure_capacity_for(absolute_pos, keys, values)
            key_tail = keys[..., cursor : cursor + take, :]
            value_tail = values[..., cursor : cursor + take, :]
            key_tail, value_tail = self._own_tail(key_tail, value_tail)
            self.key_blocks[block_index][..., in_block : in_block + take, :] = key_tail
            self.value_blocks[block_index][..., in_block : in_block + take, :] = value_tail
            self.offset += take
            cursor += take
            if in_block + take == self.block_size:
                self._finalize_block(block_index)

    def active_block_slices(self) -> list[tuple[int, Any, Any]]:
        """Return active physical KV block slices as ``(start, keys, values)``."""
        blocks = []
        if self.offset <= 0 or not self.key_blocks:
            return blocks
        full_blocks = self.offset // self.block_size
        partial = self.offset % self.block_size
        for block_index in range(full_blocks):
            blocks.append(
                (
                    block_index * self.block_size,
                    self.key_blocks[block_index],
                    self.value_blocks[block_index],
                )
            )
        if partial:
            blocks.append(
                (
                    full_blocks * self.block_size,
                    self.key_blocks[full_blocks][..., :partial, :],
                    self.value_blocks[full_blocks][..., :partial, :],
                )
            )
        return blocks

    @property
    def state(self):
        return self._active_arrays()

    @state.setter
    def state(self, value) -> None:
        keys, values = value
        self.key_blocks = []
        self.value_blocks = []
        self.offset = 0
        if keys is not None and values is not None:
            self._load_contiguous_state(keys, values, int(keys.shape[2]))

    @property
    def meta_state(self) -> tuple[str, ...]:
        return (str(self.block_size), str(self.offset), self.mode)

    @meta_state.setter
    def meta_state(self, value) -> None:
        if not value:
            return
        self.block_size = int(value[0])
        self.step = self.block_size
        self.offset = int(value[1])
        if len(value) > 2:
            self.mode = _normalize_detach_mode(str(value[2]))

    def empty(self) -> bool:
        return not self.key_blocks or self.offset <= 0

    @property
    def nbytes(self) -> int:
        total = 0
        for key_block, value_block in zip(self.key_blocks, self.value_blocks):
            total += int(key_block.nbytes) + int(value_block.nbytes)
        return total


def _vllm_metal_reference_path() -> Path:
    override = os.environ.get("MTPLX_VLLM_METAL_REPO")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[1] / "REFERENCES:TOOLS" / "vllm-metal"


def _paged_attention_impl_from_env() -> str:
    return (
        os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "")
        .strip()
        .lower()
        .replace("-", "_")
    )


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _dynamic_paged_num_blocks(*, block_size: int, configured_blocks: int) -> int:
    if not _env_truthy("MTPLX_DYNAMIC_PAGED_KV"):
        return int(configured_blocks)
    min_blocks = max(
        int(configured_blocks),
        _env_int("MTPLX_DYNAMIC_PAGED_KV_MIN_BLOCKS", int(configured_blocks)),
    )
    request_tokens = max(0, _env_int("MTPLX_DYNAMIC_PAGED_KV_TOKENS", 0))
    previous_high_water = max(
        0,
        _env_int("MTPLX_DYNAMIC_PAGED_KV_PREVIOUS_HIGH_WATER", 0),
    )
    session_tokens = int((previous_high_water * 3 + 1) // 2)
    margin = max(0, _env_int("MTPLX_DYNAMIC_PAGED_KV_MARGIN", 128))
    needed = max(request_tokens, session_tokens) + margin
    if needed <= margin:
        return min_blocks
    required_blocks = (needed + int(block_size) - 1) // int(block_size)
    return max(min_blocks, required_blocks)


def _paged_attention_requires_external_ops(*, turboquant_config: Any | None = None) -> bool:
    if turboquant_config is not None:
        return True
    impl = _paged_attention_impl_from_env()
    if impl in {"fast_sdpa_gather", "sdpa_gather", "exact_gather"}:
        return False
    if impl == "sdpa_2pass_paged":
        return False
    if impl == "mlx_vector_paged":
        return _env_truthy("MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN")
    return True


def _load_vllm_metal_ops():
    errors: list[str] = []
    override = os.environ.get("MTPLX_VLLM_METAL_REPO", "").strip()
    if override:
        repo = Path(override).expanduser()
        if not repo.exists():
            raise RuntimeError(f"MTPLX_VLLM_METAL_REPO does not exist: {repo}")
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        for name in ("vllm_metal.metal", "vllm_metal"):
            sys.modules.pop(name, None)
        try:
            from vllm_metal.metal import get_ops

            return get_ops()
        except Exception as exc:
            raise RuntimeError(
                "failed to load vllm-metal ops from MTPLX_VLLM_METAL_REPO="
                f"{repo}: {exc}"
            ) from exc

    try:
        metal = importlib.import_module("vllm_metal.metal")
        return metal.get_ops()
    except Exception as exc:
        errors.append(f"vendored vllm_metal.metal: {exc}")

    repo = _vllm_metal_reference_path()
    if repo.exists():
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        for name in ("vllm_metal.metal", "vllm_metal"):
            sys.modules.pop(name, None)
        try:
            from vllm_metal.metal import get_ops

            return get_ops()
        except Exception as exc:
            errors.append(f"reference checkout {repo}: {exc}")
    else:
        errors.append(f"reference checkout missing: {repo}")

    raise RuntimeError(
        "vllm-metal paged-attention ops are unavailable; "
        + "; ".join(errors)
        + ". Install MTPLX with its Darwin/arm64 dependencies or set "
        "MTPLX_VLLM_METAL_REPO to a working vllm-metal checkout."
    )


# Module-level latch so we only warn once per process when the optional
# vllm-metal external ops can't load. Subsequent calls stay silent.
_VLLM_METAL_OPS_UNAVAILABLE_WARNED = False


def _warn_vllm_metal_ops_unavailable(exc: BaseException, *, context: str) -> None:
    """Emit a single, plain-stderr warning when external ops are missing.

    We deliberately avoid the ``warnings`` module here because MTPLX runs under
    request-scoped warning filters that can swallow the message; a direct
    ``print`` to stderr is what the operator actually sees in the server log.
    """

    global _VLLM_METAL_OPS_UNAVAILABLE_WARNED
    if _VLLM_METAL_OPS_UNAVAILABLE_WARNED:
        return
    _VLLM_METAL_OPS_UNAVAILABLE_WARNED = True
    print(
        f"mtplx: vllm-metal external ops unavailable ({context}): {exc}; "
        "falling back to packaged paged-attention path. Install nanobind and "
        "build vllm_metal/paged_ops.cpp, or set MTPLX_VLLM_METAL_REPO, to "
        "re-enable the external ops.",
        file=sys.stderr,
        flush=True,
    )


def _load_vllm_metal_ops_optional(*, context: str):
    """Load vllm-metal ops if available; return ``None`` (and warn once) if not.

    Use this on the graceful-fallback paths where the caller has a valid
    in-tree alternative.  Use ``_load_vllm_metal_ops`` directly only for
    diagnostic / explicitly-required paths.
    """

    try:
        return _load_vllm_metal_ops()
    except RuntimeError as exc:
        _warn_vllm_metal_ops_unavailable(exc, context=context)
        return None


class VllmMetalPagedKVCache:
    """Preallocated full-attention KV pages backed by vLLM-Metal primitives.

    This is an opt-in diagnostic cache for Step 11B/11C.  It stores accepted K/V
    in physical token pages shaped like vLLM-Metal's paged attention kernel:
    ``[num_blocks, block_size, num_kv_heads, head_dim]``.  Logical positions are
    currently contiguous from zero, so the block table is trivial while still
    exercising the native paged read path.
    """

    def __init__(
        self,
        *,
        block_size: int = 16,
        num_blocks: int = 1024,
        keys: Any | None = None,
        values: Any | None = None,
        offset: int = 0,
        turboquant_config: Any | None = None,
    ) -> None:
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        self.offset = 0
        self.key_cache = None
        self.value_cache = None
        self.key_scale_cache = None
        self.value_scale_cache = None
        self.key_zero_cache = None
        self.turboquant_config = turboquant_config
        self.turboquant = turboquant_config is not None
        self._shape: tuple[int, int, int] | None = None
        self._dtypes: tuple[Any, Any] | None = None
        self.update_calls = 0
        self.paged_attention_calls = 0
        self.partitioned_attention_calls = 0
        self.turboquant_attention_calls = 0
        self.active_array_calls = 0
        self.dense_fallback_calls = 0
        self.dense_fallback_calls_by_phase: dict[str, int] = {}
        self.paged_attention_bailouts_by_phase_reason: dict[str, int] = {}
        self.paged_attention_last_bailout: dict[str, int | str] = {}
        self.paged_attention_large_q_path = ""
        self.large_q_split_sdpa_fallback_calls = 0
        self.large_q_split_sdpa_fallback_calls_by_phase: dict[str, int] = {}
        self.partitioned_paged_calls = 0
        self.partitioned_paged_calls_by_phase: dict[str, int] = {}
        self.grow_events = 0
        self.cache_write_time_s = 0.0
        self.attention_time_s = 0.0
        if keys is not None and values is not None:
            self._load_contiguous_state(keys, values, int(offset))

    @classmethod
    def from_cache(
        cls,
        entry: Any,
        *,
        block_size: int = 16,
        num_blocks: int = 1024,
        turboquant_config: Any | None = None,
    ) -> "VllmMetalPagedKVCache":
        return cls(
            block_size=block_size,
            num_blocks=num_blocks,
            keys=getattr(entry, "keys", None),
            values=getattr(entry, "values", None),
            offset=int(getattr(entry, "offset", 0)),
            turboquant_config=turboquant_config,
        )

    @property
    def capacity(self) -> int:
        return int(self.block_size) * int(self.num_blocks)

    def _grow_to_capacity(self, required_tokens: int) -> bool:
        if not _env_truthy("MTPLX_DYNAMIC_PAGED_KV"):
            return False
        required_blocks = (int(required_tokens) + self.block_size - 1) // self.block_size
        grown_blocks = max(
            required_blocks,
            int((self.num_blocks * 3 + 1) // 2),
            int(self.num_blocks) + 1,
        )
        if grown_blocks <= self.num_blocks:
            return True
        if self.key_cache is None or self.value_cache is None:
            self.num_blocks = int(grown_blocks)
            self.grow_events += 1
            return True

        import mlx.core as mx

        extra_blocks = int(grown_blocks) - int(self.num_blocks)
        key_extra = mx.zeros(
            (extra_blocks, *self.key_cache.shape[1:]),
            dtype=self.key_cache.dtype,
        )
        value_extra = mx.zeros(
            (extra_blocks, *self.value_cache.shape[1:]),
            dtype=self.value_cache.dtype,
        )
        grown_arrays = [key_extra, value_extra]
        self.key_cache = mx.concatenate([self.key_cache, key_extra], axis=0)
        self.value_cache = mx.concatenate([self.value_cache, value_extra], axis=0)
        grown_arrays.extend([self.key_cache, self.value_cache])
        if self.key_scale_cache is not None:
            extra = mx.zeros(
                (extra_blocks, *self.key_scale_cache.shape[1:]),
                dtype=self.key_scale_cache.dtype,
            )
            self.key_scale_cache = mx.concatenate([self.key_scale_cache, extra], axis=0)
            grown_arrays.extend([extra, self.key_scale_cache])
        if self.value_scale_cache is not None:
            extra = mx.zeros(
                (extra_blocks, *self.value_scale_cache.shape[1:]),
                dtype=self.value_scale_cache.dtype,
            )
            self.value_scale_cache = mx.concatenate([self.value_scale_cache, extra], axis=0)
            grown_arrays.extend([extra, self.value_scale_cache])
        if self.key_zero_cache is not None:
            extra = mx.zeros(
                (extra_blocks, *self.key_zero_cache.shape[1:]),
                dtype=self.key_zero_cache.dtype,
            )
            self.key_zero_cache = mx.concatenate([self.key_zero_cache, extra], axis=0)
            grown_arrays.extend([extra, self.key_zero_cache])
        self.num_blocks = int(grown_blocks)
        self.grow_events += 1
        mx.eval(*grown_arrays)
        return True

    def _ensure_allocated(self, keys: Any, values: Any) -> None:
        import mlx.core as mx

        if int(keys.shape[0]) != 1:
            raise ValueError("VllmMetalPagedKVCache currently supports batch size 1")
        shape = (int(keys.shape[1]), int(keys.shape[3]), int(values.shape[3]))
        dtypes = (keys.dtype, values.dtype)
        if self.key_cache is not None:
            if self._shape != shape or self._dtypes != dtypes:
                raise ValueError(
                    "paged KV cache shape/dtype changed: "
                    f"had {self._shape}/{self._dtypes}, got {shape}/{dtypes}"
                )
            return
        n_kv_heads, k_head_dim, v_head_dim = shape
        # Defense-in-depth: if a TurboQuant cache reaches first allocation but
        # the external vllm-metal ops can't load, gracefully degrade to the
        # plain paged layout instead of crashing the in-flight request. The
        # install-time path already drops turboquant_config when ops are
        # missing, but downstream callers (e.g. snapshot restore) may still
        # construct a TurboQuant cache without going through that gate.
        if self.turboquant and _load_vllm_metal_ops_optional(
            context="TurboQuant cache allocation"
        ) is None:
            self.turboquant = False
            self.turboquant_config = None
        if self.turboquant:
            from .turboquant import (
                CENTROIDS_3BIT,
                SCALE_GROUP_SIZE,
                packed_dim,
                validate_head_dim,
            )

            validate_head_dim(k_head_dim)
            validate_head_dim(v_head_dim)
            cfg = self.turboquant_config
            if int(cfg.value_bits) != 3:
                raise ValueError(
                    "MTPLX TurboQuant currently supports v_quant='q3_0' for "
                    "the Metal paged diagnostic path"
                )
            key_dtype = mx.int8 if cfg.key_dtype_name == "int8" else mx.uint8
            self.key_cache = mx.zeros(
                (
                    self.num_blocks,
                    self.block_size,
                    n_kv_heads,
                    packed_dim(k_head_dim, int(cfg.key_bits)),
                ),
                dtype=key_dtype,
            )
            self.value_cache = mx.zeros(
                (
                    self.num_blocks,
                    self.block_size,
                    n_kv_heads,
                    packed_dim(v_head_dim, int(cfg.value_bits)),
                ),
                dtype=mx.uint8,
            )
            scale_shape = (
                self.num_blocks,
                self.block_size,
                n_kv_heads,
                k_head_dim // SCALE_GROUP_SIZE,
            )
            self.key_scale_cache = mx.zeros(scale_shape, dtype=mx.float16)
            self.value_scale_cache = mx.zeros(scale_shape, dtype=mx.float16)
            self.key_zero_cache = mx.zeros(scale_shape, dtype=mx.float16)
            self._turboquant_v_centroids = mx.array(CENTROIDS_3BIT, dtype=mx.float32)
            mx.eval(
                self.key_cache,
                self.value_cache,
                self.key_scale_cache,
                self.value_scale_cache,
                self.key_zero_cache,
                self._turboquant_v_centroids,
            )
        else:
            self.key_cache = mx.zeros(
                (self.num_blocks, self.block_size, n_kv_heads, k_head_dim),
                dtype=keys.dtype,
            )
            self.value_cache = mx.zeros(
                (self.num_blocks, self.block_size, n_kv_heads, v_head_dim),
                dtype=values.dtype,
            )
            mx.eval(self.key_cache, self.value_cache)
        self._shape = shape
        self._dtypes = dtypes

    def _write_tail(self, keys: Any, values: Any) -> None:
        import mlx.core as mx

        self._ensure_allocated(keys, values)
        steps = int(keys.shape[2])
        if self.offset + steps > self.capacity:
            required = self.offset + steps
            if not self._grow_to_capacity(required):
                raise ValueError(
                    f"paged KV cache capacity exceeded: {required} > {self.capacity}"
                )
        started = time.perf_counter()
        slot_mapping = mx.arange(self.offset, self.offset + steps, dtype=mx.int64)
        k_3d = mx.contiguous(keys[0].transpose(1, 0, 2))
        v_3d = mx.contiguous(values[0].transpose(1, 0, 2))
        if self.turboquant:
            if (
                self.key_scale_cache is None
                or self.value_scale_cache is None
                or self.key_zero_cache is None
            ):
                raise RuntimeError("TurboQuant scale caches were not allocated")
            cfg = self.turboquant_config
            ops = _load_vllm_metal_ops_optional(context="TurboQuant tq_encode")
            if ops is None or not hasattr(ops, "tq_encode"):
                # Graceful fallback: external ops are missing or stripped down.
                # Drop the TurboQuant snapshot path for this cache and reroute
                # to the plain paged layout. The cache is still empty for this
                # write (no prior tq_encode could have succeeded without ops),
                # so it is safe to re-allocate.
                if ops is not None and not hasattr(ops, "tq_encode"):
                    _warn_vllm_metal_ops_unavailable(
                        RuntimeError(
                            "local vLLM-Metal ops do not expose tq_encode"
                        ),
                        context="TurboQuant tq_encode",
                    )
                self.turboquant = False
                self.turboquant_config = None
                self.key_cache = None
                self.value_cache = None
                self.key_scale_cache = None
                self.value_scale_cache = None
                self.key_zero_cache = None
                self._shape = None
                self._dtypes = None
                self._ensure_allocated(keys, values)
                flat_k = self.key_cache.reshape(
                    -1, int(keys.shape[1]), int(keys.shape[3])
                )
                flat_v = self.value_cache.reshape(
                    -1, int(values.shape[1]), int(values.shape[3])
                )
                flat_k[slot_mapping] = k_3d
                flat_v[slot_mapping] = v_3d
                self.key_cache = flat_k.reshape(self.key_cache.shape)
                self.value_cache = flat_v.reshape(self.value_cache.shape)
                self.offset += steps
                self.update_calls += 1
                self.cache_write_time_s += time.perf_counter() - started
                return
            (
                self.key_cache,
                self.value_cache,
                self.key_scale_cache,
                self.value_scale_cache,
                self.key_zero_cache,
            ) = ops.tq_encode(
                k_3d,
                v_3d,
                self.key_cache,
                self.value_cache,
                self.key_scale_cache,
                self.value_scale_cache,
                self.key_zero_cache,
                slot_mapping,
                self._turboquant_v_centroids,
                int(cfg.value_bits),
                int(cfg.key_bits),
                bool(cfg.key_signed),
            )
        else:
            flat_k = self.key_cache.reshape(-1, int(keys.shape[1]), int(keys.shape[3]))
            flat_v = self.value_cache.reshape(-1, int(values.shape[1]), int(values.shape[3]))
            flat_k[slot_mapping] = k_3d
            flat_v[slot_mapping] = v_3d
            self.key_cache = flat_k.reshape(self.key_cache.shape)
            self.value_cache = flat_v.reshape(self.value_cache.shape)
        self.offset += steps
        self.update_calls += 1
        self.cache_write_time_s += time.perf_counter() - started

    def _load_contiguous_state(self, keys: Any, values: Any, offset: int) -> None:
        self.key_cache = None
        self.value_cache = None
        self.key_scale_cache = None
        self.value_scale_cache = None
        self.key_zero_cache = None
        self._shape = None
        self._dtypes = None
        self.offset = 0
        total = min(int(offset), int(keys.shape[2]))
        if total <= 0:
            return
        self._write_tail(keys[..., :total, :], values[..., :total, :])

    def _partition_threshold(self) -> int:
        return _env_int("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", 2048)

    def _partitioned_attention_enabled(self) -> bool:
        return _env_truthy("MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN")

    @staticmethod
    def _safe_2pass_paged_q_len(*, query_heads: int, kv_heads: int) -> int:
        """Max q_len that keeps the packaged paged-tail Metal threadgroup legal."""

        query_heads = max(1, int(query_heads))
        kv_heads = max(1, int(kv_heads))
        if query_heads % kv_heads:
            return 0
        gqa_factor = max(1, query_heads // kv_heads)
        return max(1, 1024 // max(1, 32 * gqa_factor))

    def _long_context_dense_fallback_forbidden(self) -> bool:
        if _env_truthy("MTPLX_ALLOW_LONG_CONTEXT_DENSE_FALLBACK"):
            return False
        if _env_truthy("MTPLX_ALLOW_PAGED_ACTIVE_ARRAY_SNAPSHOT"):
            return False
        # Only the QA assertion env var should turn the dense-fallback path
        # into a hard error. MTPLX_SUSTAINED_PREFILL is a *product* profile
        # flag (set by `--profile sustained`) and must not abort production
        # requests when the partitioned-paged kernel returns None - the
        # caller in attention_split.py already handles None by routing to
        # scaled_dot_product_attention with cache.state, which is correct.
        asserted = _env_truthy("MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS")
        return asserted and int(self.offset) >= self._partition_threshold()

    def long_context_dense_fallback_forbidden(self) -> bool:
        return self._long_context_dense_fallback_forbidden()

    def _record_paged_bailout(
        self,
        reason: str,
        *,
        impl: str = "",
        offset: int | None = None,
        q_len: int | None = None,
        max_q_len: int | None = None,
        sliding_window: int | None = None,
        partitioned_enabled: bool | None = None,
        partition_threshold: int | None = None,
    ) -> None:
        phase = current_attention_phase()
        normalized_reason = (reason or "unknown").strip().lower() or "unknown"
        key = f"{phase}:{normalized_reason}"
        self.paged_attention_bailouts_by_phase_reason[key] = (
            int(self.paged_attention_bailouts_by_phase_reason.get(key, 0)) + 1
        )
        self.paged_attention_last_bailout = {
            "phase": phase,
            "reason": normalized_reason,
            "impl": str(impl or ""),
            "offset": int(self.offset if offset is None else offset),
            "q_len": int(q_len or 0),
            "max_q_len": int(max_q_len or 0),
            "block_size": int(self.block_size),
            "sliding_window": int(-1 if sliding_window is None else sliding_window),
            "partitioned_enabled": int(
                self._partitioned_attention_enabled()
                if partitioned_enabled is None
                else bool(partitioned_enabled)
            ),
            "partition_threshold": int(
                self._partition_threshold()
                if partition_threshold is None
                else partition_threshold
            ),
        }
        if _env_truthy("MTPLX_PAGED_ATTENTION_TRACE"):
            print(
                "mtplx_paged_attention_bailout "
                + " ".join(f"{k}={v}" for k, v in self.paged_attention_last_bailout.items()),
                file=sys.stderr,
            )

    def record_dense_fallback(self) -> None:
        self.dense_fallback_calls += 1
        phase = current_attention_phase()
        self.dense_fallback_calls_by_phase[phase] = (
            int(self.dense_fallback_calls_by_phase.get(phase, 0)) + 1
        )

    def _paged_range(self, start: int, end: int) -> tuple[Any, Any]:
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("paged KV cache is not allocated")
        start = max(0, int(start))
        end = min(int(end), int(self.offset))
        if end <= start:
            raise ValueError(f"invalid paged KV range: {start}:{end}")
        flat_k = self.key_cache.reshape(
            -1,
            int(self.key_cache.shape[2]),
            int(self.key_cache.shape[3]),
        )[start:end]
        flat_v = self.value_cache.reshape(
            -1,
            int(self.value_cache.shape[2]),
            int(self.value_cache.shape[3]),
        )[start:end]
        return flat_k.transpose(1, 0, 2)[None, ...], flat_v.transpose(1, 0, 2)[None, ...]

    def _large_q_split_sdpa_fallback(
        self,
        queries: Any,
        *,
        scale: float,
        sliding_window: int,
        mask: Any | None,
    ):
        import mlx.core as mx

        if self.turboquant:
            self._record_paged_bailout(
                "turboquant_unsupported",
                impl="large_q_split_sdpa",
                offset=int(self.offset),
                q_len=int(queries.shape[2]),
                sliding_window=int(sliding_window),
            )
            return None
        if mask is not None and mask != "causal":
            self._record_paged_bailout(
                "unsupported_mask",
                impl="large_q_split_sdpa",
                offset=int(self.offset),
                q_len=int(queries.shape[2]),
                sliding_window=int(sliding_window),
            )
            return None

        q_len = int(queries.shape[2])
        if _env_truthy("MTPLX_ASSERT_NO_LARGE_Q_SPLIT_FALLBACK"):
            raise RuntimeError(
                "large-q split SDPA fallback was invoked while "
                "MTPLX_ASSERT_NO_LARGE_Q_SPLIT_FALLBACK=1 "
                f"phase={current_attention_phase()} offset={int(self.offset)} "
                f"q_len={q_len} threshold={self._partition_threshold()}"
            )
        cached_prefix_len = max(0, int(self.offset) - q_len)
        query_heads = int(queries.shape[1])
        q_chunk_size = max(
            1,
            _env_int("MTPLX_VLLM_METAL_PAGED_LARGE_Q_CHUNK_SIZE", 2048),
        )
        kv_chunk_size = max(
            1,
            _env_int("MTPLX_VLLM_METAL_PAGED_LARGE_Q_KV_CHUNK_SIZE", 1024),
        )
        key_start = 0
        if int(sliding_window) > 0:
            key_start = max(0, int(self.offset) - int(sliding_window))
        outputs: list[Any] = []
        very_negative = mx.array(-1.0e30, dtype=mx.float32)
        eps = mx.array(1.0e-20, dtype=mx.float32)

        for q_start in range(0, q_len, q_chunk_size):
            q_end = min(q_len, q_start + q_chunk_size)
            q = queries[:, :, q_start:q_end, :].astype(mx.float32)
            q_positions = cached_prefix_len + mx.arange(q_start, q_end)
            max_key_for_chunk = int(cached_prefix_len + q_end)
            running_max = mx.full(
                (int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), 1),
                very_negative,
                dtype=mx.float32,
            )
            running_denom = mx.zeros_like(running_max)
            running_acc = mx.zeros(
                (
                    int(q.shape[0]),
                    int(q.shape[1]),
                    int(q.shape[2]),
                    int(self.value_cache.shape[3]),
                ),
                dtype=mx.float32,
            )
            for k_start in range(key_start, min(int(self.offset), max_key_for_chunk), kv_chunk_size):
                k_end = min(int(self.offset), max_key_for_chunk, k_start + kv_chunk_size)
                if k_end <= k_start:
                    continue
                keys, values = self._paged_range(k_start, k_end)
                kv_heads = int(keys.shape[1])
                if kv_heads != query_heads and query_heads % kv_heads:
                    return None
                k = keys.astype(mx.float32)
                v = values.astype(mx.float32)
                repeat = query_heads // kv_heads if kv_heads != query_heads else 1
                if repeat > 1:
                    q_for_scores = q.reshape(
                        int(q.shape[0]),
                        kv_heads,
                        repeat,
                        int(q.shape[2]),
                        int(q.shape[3]),
                    )
                    k_for_scores = k[:, :, None, :, :]
                    scores = mx.matmul(
                        q_for_scores,
                        k_for_scores.transpose(0, 1, 2, 4, 3),
                    ).reshape(
                        int(q.shape[0]),
                        query_heads,
                        int(q.shape[2]),
                        int(k.shape[2]),
                    ) * float(scale)
                else:
                    scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * float(scale)
                if mask == "causal":
                    key_positions = mx.arange(k_start, k_end)
                    allowed = q_positions[:, None] >= key_positions[None, :]
                    valid = mx.any(allowed, axis=-1, keepdims=True)
                    scores = mx.where(allowed[None, None, :, :], scores, very_negative)
                else:
                    valid = mx.ones(scores.shape[:-1] + (1,), dtype=mx.bool_)
                local_max = mx.max(scores, axis=-1, keepdims=True)
                local_max = mx.where(valid, local_max, very_negative)
                weights = mx.where(valid, mx.exp(scores - local_max), 0.0)
                local_denom = mx.sum(weights, axis=-1, keepdims=True)
                if repeat > 1:
                    local_acc = mx.matmul(
                        weights.reshape(
                            int(q.shape[0]),
                            kv_heads,
                            repeat,
                            int(q.shape[2]),
                            int(k.shape[2]),
                        ),
                        v[:, :, None, :, :],
                    ).reshape(
                        int(q.shape[0]),
                        query_heads,
                        int(q.shape[2]),
                        int(v.shape[3]),
                    )
                else:
                    local_acc = mx.matmul(weights, v)
                new_max = mx.maximum(running_max, local_max)
                old_scale = mx.exp(running_max - new_max)
                new_scale = mx.exp(local_max - new_max)
                new_scale = mx.where(valid, new_scale, 0.0)
                running_acc = running_acc * old_scale + local_acc * new_scale
                running_denom = running_denom * old_scale + local_denom * new_scale
                running_max = new_max
            outputs.append(running_acc / mx.maximum(running_denom, eps))

        if not outputs:
            return None
        self.large_q_split_sdpa_fallback_calls += 1
        phase = current_attention_phase()
        self.large_q_split_sdpa_fallback_calls_by_phase[phase] = (
            int(self.large_q_split_sdpa_fallback_calls_by_phase.get(phase, 0)) + 1
        )
        self.paged_attention_large_q_path = "large_q_split_sdpa_fallback"
        if _env_truthy("MTPLX_PREFILL_ROUTE_TRACE"):
            print(
                "mtplx_prefill_route "
                f"path=large_q_split_sdpa_fallback phase={phase} "
                f"offset={int(self.offset)} q_len={q_len} "
                f"q_chunk={q_chunk_size} kv_chunk={kv_chunk_size}",
                file=sys.stderr,
            )
        return mx.concatenate(outputs, axis=2).astype(queries.dtype)

    def _active_arrays(self) -> tuple[Any | None, Any | None]:
        self.active_array_calls += 1
        if (
            _env_truthy("MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS")
            and self._long_context_dense_fallback_forbidden()
        ):
            raise RuntimeError(
                "Paged KV cache attempted to materialize active K/V arrays in "
                f"the long-context path phase={current_attention_phase()} "
                f"offset={int(self.offset)} threshold={self._partition_threshold()}"
            )
        if self.key_cache is None or self.value_cache is None or self.offset <= 0:
            return None, None
        if self.turboquant:
            return None, None
        flat_k = self.key_cache.reshape(
            -1,
            int(self.key_cache.shape[2]),
            int(self.key_cache.shape[3]),
        )[: self.offset]
        flat_v = self.value_cache.reshape(
            -1,
            int(self.value_cache.shape[2]),
            int(self.value_cache.shape[3]),
        )[: self.offset]
        return flat_k.transpose(1, 0, 2)[None, ...], flat_v.transpose(1, 0, 2)[None, ...]

    @property
    def keys(self):
        return self._active_arrays()[0]

    @keys.setter
    def keys(self, value) -> None:
        if value is None:
            self.key_cache = None
            self.value_cache = None
            self.key_scale_cache = None
            self.value_scale_cache = None
            self.key_zero_cache = None
            self.offset = 0
            return
        if self.value_cache is not None:
            self._load_contiguous_state(value, self.values, int(value.shape[2]))

    @property
    def values(self):
        return self._active_arrays()[1]

    @values.setter
    def values(self, value) -> None:
        if value is None:
            self.key_cache = None
            self.value_cache = None
            self.key_scale_cache = None
            self.value_scale_cache = None
            self.key_zero_cache = None
            self.offset = 0
            return
        if self.key_cache is not None:
            self._load_contiguous_state(self.keys, value, int(value.shape[2]))

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        self.update_without_fetch(keys, values)
        return self._active_arrays()

    def update_without_fetch(self, keys: Any, values: Any) -> None:
        self._write_tail(keys, values)

    def size(self) -> int:
        return int(self.offset)

    @property
    def state(self):
        return self._active_arrays()

    @state.setter
    def state(self, value) -> None:
        keys, values = value
        self.key_cache = None
        self.value_cache = None
        self.key_scale_cache = None
        self.value_scale_cache = None
        self.key_zero_cache = None
        self.offset = 0
        self._shape = None
        self._dtypes = None
        if keys is not None and values is not None:
            self._load_contiguous_state(keys, values, int(keys.shape[2]))

    @property
    def meta_state(self) -> tuple[str, ...]:
        return (str(self.block_size), str(self.num_blocks), str(self.offset))

    @meta_state.setter
    def meta_state(self, value) -> None:
        if not value:
            return
        self.block_size = int(value[0])
        self.num_blocks = int(value[1])
        self.offset = int(value[2])

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(int(self.offset), int(n))
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self) -> bool:
        return self.key_cache is None or self.offset <= 0

    @property
    def nbytes(self) -> int:
        if self.key_cache is None or self.value_cache is None:
            return 0
        total = int(self.key_cache.nbytes) + int(self.value_cache.nbytes)
        for extra in (
            self.key_scale_cache,
            self.value_scale_cache,
            self.key_zero_cache,
        ):
            if extra is not None:
                total += int(extra.nbytes)
        return total

    def _effective_sliding_window(self, requested: int) -> int:
        raw = os.environ.get("MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW")
        if raw is None or not raw.strip():
            return int(requested)
        return int(raw)

    def _active_attention_arrays(self, sliding_window: int) -> tuple[Any | None, Any | None]:
        keys, values = self._active_arrays()
        if keys is None or values is None or int(sliding_window) <= 0:
            return keys, values
        take = min(int(sliding_window), int(keys.shape[2]))
        return keys[..., -take:, :], values[..., -take:, :]

    def paged_attention(
        self,
        queries: Any,
        *,
        scale: float,
        sliding_window: int = -1,
        mask: Any | None = None,
        impl_override: str | None = None,
    ):
        import mlx.core as mx

        started = time.perf_counter()
        q_len = int(queries.shape[2]) if hasattr(queries, "shape") and len(queries.shape) >= 3 else 0
        sliding_window = self._effective_sliding_window(sliding_window)
        impl_source = (
            impl_override
            if impl_override is not None
            else os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "")
        )
        impl = impl_source.strip().lower().replace("-", "_")
        max_q_len = int(os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "16") or "16")
        partitioned_enabled = self._partitioned_attention_enabled()
        partition_threshold = self._partition_threshold()

        def bailout(reason: str) -> None:
            self._record_paged_bailout(
                reason,
                impl=impl,
                offset=int(self.offset),
                q_len=q_len,
                max_q_len=max_q_len,
                sliding_window=int(sliding_window),
                partitioned_enabled=partitioned_enabled,
                partition_threshold=partition_threshold,
            )
            return None

        def run_partitioned_paged(*, force_fp32_paged: bool = False):
            if self.turboquant:
                return bailout("turboquant_unsupported")
            if self.key_cache is None or self.value_cache is None:
                return bailout("empty_cache")
            kernel_queries = queries.astype(mx.float32) if force_fp32_paged else queries
            kernel_key_cache = (
                self.key_cache.astype(mx.float32)
                if force_fp32_paged
                else self.key_cache
            )
            kernel_value_cache = (
                self.value_cache.astype(mx.float32)
                if force_fp32_paged
                else self.value_cache
            )
            q_3d = mx.contiguous(kernel_queries[0].transpose(1, 0, 2))
            used_blocks = (int(self.offset) + int(self.block_size) - 1) // int(self.block_size)
            if used_blocks <= 0:
                return bailout("blocks_invalid")
            block_tables = mx.arange(used_blocks, dtype=mx.int32)[None, :]
            seq_lens = mx.array([int(self.offset)], dtype=mx.int32)
            cu_seqlens_q = mx.array([0, q_len], dtype=mx.int32)
            partition_size = int(
                os.environ.get("MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE") or "512"
            )
            max_num_partitions = max(
                1,
                (int(self.offset) + partition_size - 1) // partition_size,
            )
            try:
                ops = _load_vllm_metal_ops()
            except RuntimeError:
                split_out = self._large_q_split_sdpa_fallback(
                    queries,
                    scale=scale,
                    sliding_window=int(sliding_window),
                    mask=mask,
                )
                if split_out is not None:
                    self.paged_attention_calls += 1
                    self.attention_time_s += time.perf_counter() - started
                    return split_out
                return bailout("partitioned_unavailable")
            if _env_truthy("MTPLX_VLLM_METAL_PAGED_USE_PRIMITIVE") and hasattr(
                ops,
                "paged_attention_partitioned_primitive",
            ):
                out = mx.array(0)
                ops.paged_attention_partitioned_primitive(
                    q_3d,
                    kernel_key_cache,
                    kernel_value_cache,
                    int(kernel_key_cache.shape[2]),
                    float(scale),
                    0.0,
                    block_tables,
                    seq_lens,
                    cu_seqlens_q,
                    int(self.block_size),
                    int(self.offset),
                    int(sliding_window),
                    out,
                )
                self.paged_attention_calls += 1
                self.partitioned_attention_calls += 1
                self.partitioned_paged_calls += 1
                phase = current_attention_phase()
                self.partitioned_paged_calls_by_phase[phase] = (
                    int(self.partitioned_paged_calls_by_phase.get(phase, 0)) + 1
                )
                self.attention_time_s += time.perf_counter() - started
                self.paged_attention_large_q_path = "partitioned_paged"
                if _env_truthy("MTPLX_PREFILL_ROUTE_TRACE"):
                    print(
                        "mtplx_prefill_route "
                        f"path=partitioned_paged phase={phase} "
                        f"offset={int(self.offset)} q_len={q_len} "
                        f"partition_size=primitive",
                        file=sys.stderr,
                    )
                out = out.transpose(1, 0, 2)[None, ...]
                return out.astype(queries.dtype) if force_fp32_paged else out
            if not hasattr(ops, "paged_attention_v2_online_partitioned"):
                split_out = self._large_q_split_sdpa_fallback(
                    queries,
                    scale=scale,
                    sliding_window=int(sliding_window),
                    mask=mask,
                )
                if split_out is not None:
                    self.paged_attention_calls += 1
                    self.attention_time_s += time.perf_counter() - started
                    return split_out
                return bailout("partitioned_unavailable")
            out = mx.zeros(q_3d.shape, dtype=q_3d.dtype)
            exp_sums = mx.zeros(
                (q_len, int(q_3d.shape[1]), max_num_partitions),
                dtype=mx.float32,
            )
            max_logits = mx.zeros(
                (q_len, int(q_3d.shape[1]), max_num_partitions),
                dtype=mx.float32,
            )
            tmp_out = mx.zeros(
                (
                    q_len,
                    int(q_3d.shape[1]),
                    max_num_partitions,
                    int(q_3d.shape[2]),
                ),
                dtype=q_3d.dtype,
            )
            mx.eval(
                out,
                q_3d,
                self.key_cache,
                self.value_cache,
                block_tables,
                seq_lens,
                cu_seqlens_q,
                exp_sums,
                max_logits,
                tmp_out,
            )
            ops.paged_attention_v2_online_partitioned(
                out,
                q_3d,
                kernel_key_cache,
                kernel_value_cache,
                int(kernel_key_cache.shape[2]),
                float(scale),
                0.0,
                block_tables,
                seq_lens,
                cu_seqlens_q,
                int(self.block_size),
                int(self.offset),
                int(sliding_window),
                exp_sums,
                max_logits,
                tmp_out,
            )
            mx.synchronize()
            self.paged_attention_calls += 1
            self.partitioned_attention_calls += 1
            self.partitioned_paged_calls += 1
            phase = current_attention_phase()
            self.partitioned_paged_calls_by_phase[phase] = (
                int(self.partitioned_paged_calls_by_phase.get(phase, 0)) + 1
            )
            self.attention_time_s += time.perf_counter() - started
            self.paged_attention_large_q_path = "partitioned_paged"
            if _env_truthy("MTPLX_PREFILL_ROUTE_TRACE"):
                print(
                    "mtplx_prefill_route "
                    f"path=partitioned_paged phase={phase} "
                    f"offset={int(self.offset)} q_len={q_len} "
                    f"partition_size={partition_size} partitions={max_num_partitions}",
                    file=sys.stderr,
                )
            out = out.transpose(1, 0, 2)[None, ...]
            return out.astype(queries.dtype) if force_fp32_paged else out

        if self.key_cache is None or self.value_cache is None or self.offset <= 0:
            return bailout("empty_cache")
        if int(queries.shape[0]) != 1:
            return bailout("batch_not_1")
        if q_len <= 0:
            return bailout("q_len_invalid")
        if self.turboquant and impl in {"fast_sdpa_gather", "sdpa_gather", "exact_gather"}:
            raise ValueError("TurboQuant cannot use exact-gather attention")
        if not self.turboquant and impl in {"fast_sdpa_gather", "sdpa_gather", "exact_gather"}:
            from mlx_lm.models.base import scaled_dot_product_attention

            keys, values = self._active_attention_arrays(sliding_window)
            if keys is None or values is None:
                return bailout("empty_cache")
            out = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=None,
                scale=scale,
                mask=mask,
            )
            self.paged_attention_calls += 1
            self.attention_time_s += time.perf_counter() - started
            return out
        if not self.turboquant and impl in {"sdpa_2pass_paged", "mlx_vector_paged"}:
            from .kernels.sdpa_2pass_paged import sdpa_2pass_paged_tail

            two_pass_threshold = int(
                os.environ.get(
                    "MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD",
                    "1024",
                )
                or "1024"
            )
            if int(self.offset) < two_pass_threshold:
                from mlx_lm.models.base import scaled_dot_product_attention

                keys, values = self._active_attention_arrays(sliding_window)
                if keys is None or values is None:
                    return None
                out = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=None,
                    scale=scale,
                    mask=mask,
                )
                self.paged_attention_calls += 1
                self.attention_time_s += time.perf_counter() - started
                return out
            safe_tail_q_len = self._safe_2pass_paged_q_len(
                query_heads=int(queries.shape[1]),
                kv_heads=int(self.key_cache.shape[2]),
            )
            effective_max_q_len = min(max_q_len, safe_tail_q_len)
            if q_len > effective_max_q_len:
                if partitioned_enabled and int(self.offset) >= partition_threshold:
                    self.paged_attention_large_q_path = "partitioned_paged"
                    return run_partitioned_paged(force_fp32_paged=False)
                return bailout("q_len_gt_max")
            out = sdpa_2pass_paged_tail(
                queries=queries,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                offset=int(self.offset),
                block_size=int(self.block_size),
                scale=float(scale),
                mask=mask,
                max_q_len=effective_max_q_len,
                sliding_window=int(sliding_window),
            )
            if out is not None:
                self.paged_attention_calls += 1
                self.attention_time_s += time.perf_counter() - started
                return out
            return bailout("kernel_unavailable")
        force_fp32_paged = impl in {"fp32_paged", "paged_fp32"}
        kernel_queries = queries.astype(mx.float32) if force_fp32_paged else queries
        kernel_key_cache = (
            self.key_cache.astype(mx.float32) if force_fp32_paged else self.key_cache
        )
        kernel_value_cache = (
            self.value_cache.astype(mx.float32) if force_fp32_paged else self.value_cache
        )
        q_3d = mx.contiguous(kernel_queries[0].transpose(1, 0, 2))
        used_blocks = (self.offset + self.block_size - 1) // self.block_size
        block_tables = mx.arange(used_blocks, dtype=mx.int32)[None, :]
        seq_lens = mx.array([self.offset], dtype=mx.int32)
        cu_seqlens_q = mx.array([0, q_len], dtype=mx.int32)
        if self.turboquant:
            if (
                self.key_scale_cache is None
                or self.value_scale_cache is None
                or self.key_zero_cache is None
            ):
                return bailout("turboquant_unsupported")
            tq_ops = _load_vllm_metal_ops_optional(
                context="TurboQuant paged_attention_primitive"
            )
            if tq_ops is None or not hasattr(tq_ops, "paged_attention_primitive"):
                return bailout("turboquant_unsupported")
            cfg = self.turboquant_config
            out = mx.array(0)
            tq_ops.paged_attention_primitive(
                q_3d,
                self.key_cache,
                self.value_cache,
                int(self.key_cache.shape[2]),
                float(scale),
                0.0,
                block_tables,
                seq_lens,
                cu_seqlens_q,
                int(self.block_size),
                int(self.offset),
                int(sliding_window),
                out,
                key_scale_cache=self.key_scale_cache,
                value_scale_cache=self.value_scale_cache,
                key_zero_cache=self.key_zero_cache,
                v_centroids=self._turboquant_v_centroids,
                use_turboquant=True,
                quant_type=str(cfg.key_quant),
                v_bits=int(cfg.value_bits),
            )
            self.paged_attention_calls += 1
            self.turboquant_attention_calls += 1
            self.attention_time_s += time.perf_counter() - started
            return out.transpose(1, 0, 2)[None, ...]
        if partitioned_enabled and int(self.offset) >= partition_threshold:
            return run_partitioned_paged(force_fp32_paged=force_fp32_paged)
        out = mx.array(0)
        # The bare vllm_metal default path needs external ops. If they aren't
        # available (no nanobind, no JIT-built paged_ops.so, no
        # MTPLX_VLLM_METAL_REPO), fall through to the in-tree split-sdpa
        # fallback used by the partitioned/turboquant paths so the request
        # does not crash mid-stream. Operators who want the optimal kernel
        # install nanobind + run vllm_metal/metal/build.py.
        ops = _load_vllm_metal_ops_optional(
            context="vllm_metal default paged_attention"
        )
        if ops is None:
            split_out = self._large_q_split_sdpa_fallback(
                queries,
                scale=scale,
                sliding_window=int(sliding_window),
                mask=mask,
            )
            if split_out is not None:
                self.paged_attention_calls += 1
                self.attention_time_s += time.perf_counter() - started
                return split_out
            return bailout("vllm_metal_ops_unavailable")
        ops.paged_attention_primitive(
            q_3d,
            kernel_key_cache,
            kernel_value_cache,
            int(kernel_key_cache.shape[2]),
            float(scale),
            0.0,
            block_tables,
            seq_lens,
            cu_seqlens_q,
            int(self.block_size),
            int(self.offset),
            int(sliding_window),
            out,
        )
        self.paged_attention_calls += 1
        self.attention_time_s += time.perf_counter() - started
        out = out.transpose(1, 0, 2)[None, ...]
        return out.astype(queries.dtype) if force_fp32_paged else out

    def paged_stats(self) -> dict[str, Any]:
        return {
            "mode": "vllm_metal_paged_turboquant" if self.turboquant else "vllm_metal_paged",
            "block_size": int(self.block_size),
            "num_blocks": int(self.num_blocks),
            "capacity": int(self.capacity),
            "offset": int(self.offset),
            "updates": int(self.update_calls),
            "paged_attention_calls": int(self.paged_attention_calls),
            "partitioned_attention_calls": int(self.partitioned_attention_calls),
            "turboquant_attention_calls": int(self.turboquant_attention_calls),
            "active_array_calls": int(self.active_array_calls),
            "dense_fallback_calls": int(self.dense_fallback_calls),
            "prefill_dense_fallback_calls": int(
                self.dense_fallback_calls_by_phase.get("prefill", 0)
            ),
            "decode_dense_fallback_calls": int(
                self.dense_fallback_calls_by_phase.get("decode_verify", 0)
            ),
            "ar_dense_fallback_calls": int(
                self.dense_fallback_calls_by_phase.get("ar_decode", 0)
            ),
            "postcommit_dense_fallback_calls": int(
                self.dense_fallback_calls_by_phase.get("postcommit", 0)
            ),
            "paged_attention_bailouts_by_phase_reason": dict(
                self.paged_attention_bailouts_by_phase_reason
            ),
            "paged_attention_large_q_path": str(self.paged_attention_large_q_path),
            "large_q_split_sdpa_fallback_calls": int(
                self.large_q_split_sdpa_fallback_calls
            ),
            "large_q_split_sdpa_fallback_calls_by_phase": dict(
                self.large_q_split_sdpa_fallback_calls_by_phase
            ),
            "prefill_large_q_split_sdpa_fallback_calls": int(
                self.large_q_split_sdpa_fallback_calls_by_phase.get("prefill", 0)
            ),
            "decode_large_q_split_sdpa_fallback_calls": int(
                self.large_q_split_sdpa_fallback_calls_by_phase.get("decode_verify", 0)
            ),
            "partitioned_paged_calls": int(self.partitioned_paged_calls),
            "partitioned_paged_calls_by_phase": dict(
                self.partitioned_paged_calls_by_phase
            ),
            "prefill_partitioned_paged_calls": int(
                self.partitioned_paged_calls_by_phase.get("prefill", 0)
            ),
            "decode_partitioned_paged_calls": int(
                self.partitioned_paged_calls_by_phase.get("decode_verify", 0)
            ),
            "grow_events": int(self.grow_events),
            "turboquant": int(bool(self.turboquant)),
            "turboquant_k_quant": (
                str(self.turboquant_config.key_quant) if self.turboquant else ""
            ),
            "turboquant_v_quant": (
                str(self.turboquant_config.value_quant) if self.turboquant else ""
            ),
            "sliding_window": int(
                os.environ.get("MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW") or "-1"
            ),
            "bytes": int(self.nbytes),
            "cache_write_time_s": float(self.cache_write_time_s),
            "attention_time_s": float(self.attention_time_s),
        }


class TensorOffsetVllmMetalPagedKVCache:
    """GraphBank-safe paged KV cache with an array-backed offset.

    ``VllmMetalPagedKVCache`` stores the decode offset as a Python integer,
    which is unsafe for ``mx.compile`` replay.  This adapter preserves the
    physical page buffers but makes the offset and rollback window part of the
    compiled array state.
    """

    def __init__(
        self,
        *,
        key_cache: Any,
        value_cache: Any,
        offset: int | Any,
        block_size: int,
        num_blocks: int,
    ) -> None:
        import mlx.core as mx

        self.cache = [
            key_cache,
            value_cache,
            offset if isinstance(offset, mx.array) else mx.array(offset, dtype=mx.int32),
        ]
        self.rollback_state = [None, None, None]
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        self.update_calls = 0
        self.paged_attention_calls = 0
        self.cache_write_time_s = 0.0
        self.attention_time_s = 0.0

    @classmethod
    def from_paged_cache(cls, entry: VllmMetalPagedKVCache) -> "TensorOffsetVllmMetalPagedKVCache":
        if entry.key_cache is None or entry.value_cache is None:
            raise ValueError("cannot promote empty paged KV cache")
        return cls(
            key_cache=entry.key_cache,
            value_cache=entry.value_cache,
            offset=int(entry.offset),
            block_size=int(entry.block_size),
            num_blocks=int(entry.num_blocks),
        )

    @property
    def key_cache(self):
        return self.cache[0]

    @key_cache.setter
    def key_cache(self, value) -> None:
        self.cache[0] = value

    @property
    def value_cache(self):
        return self.cache[1]

    @value_cache.setter
    def value_cache(self, value) -> None:
        self.cache[1] = value

    @property
    def offset(self):
        return self.cache[2]

    @offset.setter
    def offset(self, value) -> None:
        import mlx.core as mx

        self.cache[2] = value if isinstance(value, mx.array) else mx.array(value, dtype=mx.int32)

    @property
    def capacity(self) -> int:
        return int(self.block_size) * int(self.num_blocks)

    @property
    def compile_state(self):
        return [self.cache, self.rollback_state]

    def _flat_key_cache(self):
        return self.cache[0].reshape(-1, int(self.cache[0].shape[2]), int(self.cache[0].shape[3]))

    def _flat_value_cache(self):
        return self.cache[1].reshape(-1, int(self.cache[1].shape[2]), int(self.cache[1].shape[3]))

    def update_without_fetch(self, keys: Any, values: Any) -> None:
        import mlx.core as mx

        steps = int(keys.shape[2])
        started = time.perf_counter()
        k_3d = mx.contiguous(keys[0].transpose(1, 0, 2))
        v_3d = mx.contiguous(values[0].transpose(1, 0, 2))
        flat_k = self._flat_key_cache()
        flat_v = self._flat_value_cache()
        self.rollback_state[0] = self.cache[2]
        self.rollback_state[1] = mx.slice(
            flat_k,
            self.cache[2],
            axes=(0,),
            slice_size=k_3d.shape,
        )
        self.rollback_state[2] = mx.slice(
            flat_v,
            self.cache[2],
            axes=(0,),
            slice_size=v_3d.shape,
        )
        flat_k = mx.slice_update(flat_k, k_3d, self.cache[2], axes=(0,))
        flat_v = mx.slice_update(flat_v, v_3d, self.cache[2], axes=(0,))
        self.cache[0] = flat_k.reshape(self.cache[0].shape)
        self.cache[1] = flat_v.reshape(self.cache[1].shape)
        self.cache[2] = self.cache[2] + steps
        self.update_calls += 1
        self.cache_write_time_s += time.perf_counter() - started

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        self.update_without_fetch(keys, values)
        return self.state

    def make_mask(self, N: int, window_size=None, return_array: bool = False):
        import mlx.core as mx

        del return_array
        rinds = mx.arange(self.capacity)
        linds = self.cache[2] + mx.arange(N)
        mask = linds[:, None] >= rinds[None, :]
        if window_size is not None:
            mask = mask & (linds[:, None] < rinds[None, :] + window_size)
        return mask

    def paged_attention(
        self,
        queries: Any,
        *,
        scale: float,
        sliding_window: int = -1,
        mask: Any | None = None,
        impl_override: str | None = None,
    ):
        del impl_override
        if int(sliding_window) > 0:
            return None
        if int(queries.shape[0]) != 1:
            return None
        static_max_offset = self._static_attention_max_offset()
        started = time.perf_counter()
        from .kernels.sdpa_2pass_paged import sdpa_2pass_paged_tail_dynamic_offset

        out = sdpa_2pass_paged_tail_dynamic_offset(
            queries=queries,
            key_cache=self.cache[0],
            value_cache=self.cache[1],
            offset=self.cache[2],
            block_size=int(self.block_size),
            scale=float(scale),
            mask=mask,
            max_q_len=int(os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "16") or "16"),
            max_offset=static_max_offset,
        )
        if out is not None:
            self.paged_attention_calls += 1
            self.attention_time_s += time.perf_counter() - started
        return out

    def _static_attention_max_offset(self) -> int | None:
        raw = os.environ.get("MTPLX_GRAPHBANK_PAGED_STATIC_MAX_OFFSET")
        if raw is None or not raw.strip():
            return None
        return int(raw)

    @property
    def state(self):
        flat_k = self._flat_key_cache()
        flat_v = self._flat_value_cache()
        keys = flat_k.transpose(1, 0, 2)[None, ...]
        values = flat_v.transpose(1, 0, 2)[None, ...]
        return keys, values

    @state.setter
    def state(self, value) -> None:
        keys, values = value
        self.key_cache = None
        self.value_cache = None
        self.offset = 0
        if keys is None or values is None:
            return
        paged = VllmMetalPagedKVCache(
            block_size=int(self.block_size),
            num_blocks=int(self.num_blocks),
        )
        paged.update_without_fetch(keys, values)
        self.cache = [
            paged.key_cache,
            paged.value_cache,
            self.offset,
        ]

    def size(self) -> int:
        import mlx.core as mx

        mx.eval(self.cache[2])
        return int(self.cache[2].item())

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        import mlx.core as mx

        n = int(n)
        if (
            self.rollback_state[0] is not None
            and self.rollback_state[1] is not None
            and self.rollback_state[2] is not None
            and int(self.rollback_state[1].shape[0]) == n
        ):
            flat_k = self._flat_key_cache()
            flat_v = self._flat_value_cache()
            flat_k = mx.slice_update(
                flat_k,
                self.rollback_state[1],
                self.rollback_state[0],
                axes=(0,),
            )
            flat_v = mx.slice_update(
                flat_v,
                self.rollback_state[2],
                self.rollback_state[0],
                axes=(0,),
            )
            self.cache[0] = flat_k.reshape(self.cache[0].shape)
            self.cache[1] = flat_v.reshape(self.cache[1].shape)
            self.cache[2] = self.rollback_state[0]
        else:
            self.cache[2] = mx.maximum(
                self.cache[2] - n,
                mx.array(0, dtype=self.cache[2].dtype),
            )
        return n

    def empty(self) -> bool:
        return self.key_cache is None or self.value_cache is None

    @property
    def nbytes(self) -> int:
        if self.key_cache is None or self.value_cache is None:
            return 0
        return int(self.key_cache.nbytes) + int(self.value_cache.nbytes) + int(self.cache[2].nbytes)

    def paged_stats(self) -> dict[str, int | float | str]:
        return {
            "mode": "tensor_offset_vllm_metal_paged",
            "block_size": int(self.block_size),
            "num_blocks": int(self.num_blocks),
            "capacity": int(self.capacity),
            "offset": int(self.size()),
            "static_max_offset": int(self._static_attention_max_offset() or self.capacity),
            "updates": int(self.update_calls),
            "paged_attention_calls": int(self.paged_attention_calls),
            "bytes": int(self.nbytes),
            "cache_write_time_s": float(self.cache_write_time_s),
            "attention_time_s": float(self.attention_time_s),
        }


class OwnedRecurrentStateCache:
    """Fixed-shape recurrent cache with persistent owned state buffers.

    Qwen3Next GDN layers keep only two recurrent leaves: the causal-conv tail
    and the GDN matrix state.  Stock ``ArraysCache`` replaces those leaves with
    whatever expression produced the newest state.  This diagnostic instead
    keeps stable buffers and copies each accepted state into them, forcing the
    official cache entry to be owned data at the commit boundary.
    """

    def __init__(
        self,
        size: int = 2,
        *,
        mode: str = "persistent_eval",
        initial: list[Any] | tuple[Any, ...] | None = None,
        left_padding: Any | None = None,
        lengths: Any | None = None,
    ) -> None:
        self.cache = [None] * int(size)
        self.mode = str(mode).strip().lower().replace("-", "_") or "persistent_eval"
        if self.mode not in {"persistent_eval"}:
            raise ValueError("owned recurrent state mode must be 'persistent_eval'")
        self._owned_buffers = [None] * int(size)
        self.left_padding = left_padding
        self.lengths = lengths
        self.owner_updates = 0
        self.owner_arrays = 0
        self.owner_allocations = 0
        self.owner_inplace_updates = 0
        self.owner_bytes = 0
        self.owner_time_s = 0.0
        if initial is not None:
            self.replace_state(initial)

    @classmethod
    def from_cache(
        cls,
        entry: Any,
        *,
        mode: str = "persistent_eval",
    ) -> "OwnedRecurrentStateCache":
        return cls(
            size=len(getattr(entry, "cache", getattr(entry, "state", [None, None]))),
            mode=mode,
            initial=list(getattr(entry, "state", [None, None])),
            left_padding=getattr(entry, "left_padding", None),
            lengths=getattr(entry, "lengths", None),
        )

    def __getitem__(self, idx: int) -> Any:
        return self.cache[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        # Model forward writes speculative state through ``cache[i] = ...``.
        # Keep those cheap; commit/restore paths call ``replace_state`` to force
        # the owned-copy boundary only for authoritative state.
        self.cache[idx] = value

    @property
    def state(self) -> list[Any]:
        return self.cache

    @state.setter
    def state(self, value: list[Any] | tuple[Any, ...] | None) -> None:
        self.replace_state(value)

    def replace_state(self, value: list[Any] | tuple[Any, ...] | None) -> None:
        if value is None:
            self.cache = [None] * len(self.cache)
            self._owned_buffers = [None] * len(self._owned_buffers)
            return
        for idx, item in enumerate(value):
            if idx >= len(self.cache):
                break
            self.cache[idx] = self._own_value(idx, item)
        for idx in range(len(value), len(self.cache)):
            self.cache[idx] = None

    @property
    def meta_state(self) -> tuple[str, str]:
        return ("owned_recurrent_state", self.mode)

    @meta_state.setter
    def meta_state(self, value: Any) -> None:
        if not value:
            return
        if isinstance(value, (list, tuple)) and len(value) > 1:
            mode = str(value[1]).strip().lower().replace("-", "_")
            if mode in {"persistent_eval"}:
                self.mode = mode

    @property
    def batch_size(self) -> int:
        for item in self.cache:
            if item is not None:
                return int(item.shape[0])
        if self.left_padding is not None:
            return int(self.left_padding.size)
        if self.lengths is not None:
            return int(self.lengths.size)
        return 1

    def _own_value(self, idx: int, value: Any) -> Any:
        import mlx.core as mx

        if value is None or not isinstance(value, mx.array):
            return value
        started = time.perf_counter()
        existing = self._owned_buffers[idx]
        if existing is value:
            mx.eval(existing)
            self.owner_updates += 1
            self.owner_arrays += 1
            self.owner_bytes += int(existing.nbytes)
            self.owner_time_s += time.perf_counter() - started
            return existing
        reusable = (
            isinstance(existing, mx.array)
            and tuple(existing.shape) == tuple(value.shape)
            and existing.dtype == value.dtype
        )
        if reusable:
            target = existing
            self.owner_inplace_updates += 1
        else:
            target = mx.zeros(value.shape, dtype=value.dtype)
            self.owner_allocations += 1
        full_slice = tuple(slice(None) for _ in range(len(value.shape)))
        target[full_slice] = value
        mx.eval(target)
        self._owned_buffers[idx] = target
        self.owner_updates += 1
        self.owner_arrays += 1
        self.owner_bytes += int(target.nbytes)
        self.owner_time_s += time.perf_counter() - started
        return target

    def filter(self, batch_indices: Any) -> None:
        self.replace_state(
            [item[batch_indices] if item is not None else None for item in self.cache]
        )
        if self.left_padding is not None:
            self.left_padding = self.left_padding[batch_indices]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]

    def extend(self, other: Any) -> None:
        import mlx.core as mx

        a_batch = self.batch_size
        b_batch = other.batch_size

        def cat(a: Any, b: Any) -> Any:
            shape = dtype = None
            if a is not None:
                shape = a.shape
                dtype = a.dtype
            if b is not None:
                shape = b.shape
                dtype = b.dtype
            if shape is None:
                return None
            if a is None:
                a = mx.zeros((a_batch,) + shape[1:], dtype=dtype)
            if b is None:
                b = mx.zeros((b_batch,) + shape[1:], dtype=dtype)
            return mx.concatenate([a, b])

        self.replace_state([cat(c, o) for c, o in zip(self.cache, other.cache)])
        self.left_padding = cat(self.left_padding, getattr(other, "left_padding", None))
        self.lengths = cat(self.lengths, getattr(other, "lengths", None))

    def extract(self, idx: int) -> "OwnedRecurrentStateCache":
        return OwnedRecurrentStateCache(
            len(self.cache),
            mode=self.mode,
            initial=[item[idx : idx + 1] if item is not None else None for item in self.cache],
            left_padding=(
                self.left_padding[idx : idx + 1]
                if self.left_padding is not None
                else None
            ),
            lengths=self.lengths[idx : idx + 1] if self.lengths is not None else None,
        )

    def prepare(self, lengths=None, **kwargs) -> None:
        import mlx.core as mx

        if lengths is not None:
            self.lengths = mx.array(lengths)

    def finalize(self) -> None:
        self.lengths = None
        self.left_padding = None

    def advance(self, N: int) -> None:
        if self.lengths is not None:
            self.lengths -= N
        if self.left_padding is not None:
            self.left_padding -= N

    def make_mask(self, N: int):
        import mlx.core as mx

        if self.left_padding is not None:
            pos = mx.arange(N)
            return pos >= self.left_padding[:, None]
        if self.lengths is not None:
            pos = mx.arange(N)
            return pos < self.lengths[:, None]
        return None

    def is_trimmable(self) -> bool:
        return False

    def empty(self) -> bool:
        return all(item is None for item in self.cache)

    @property
    def nbytes(self) -> int:
        return sum(int(item.nbytes) for item in self.cache if item is not None)

    def owner_stats(self) -> dict[str, int | float | str]:
        return {
            "mode": self.mode,
            "updates": int(self.owner_updates),
            "arrays": int(self.owner_arrays),
            "allocations": int(self.owner_allocations),
            "inplace_updates": int(self.owner_inplace_updates),
            "bytes": int(self.owner_bytes),
            "time_s": float(self.owner_time_s),
        }


def replace_recurrent_cache_state(entry: Any, state: list[Any] | tuple[Any, ...]) -> None:
    if hasattr(entry, "replace_state"):
        entry.replace_state(state)
        return
    if hasattr(entry, "__setitem__") and len(state) >= 2:
        entry[0] = state[0]
        entry[1] = state[1]
        return
    current = getattr(entry, "state", None)
    if isinstance(current, list) and len(current) == len(state):
        current[:] = list(state)
    else:
        entry.state = list(state)


def install_owned_recurrent_state_cache(
    cache: list[Any],
    *,
    mode: str = "persistent_eval",
) -> dict[str, int | str]:
    """Replace stock fixed recurrent caches with persistent owner caches."""
    normalized = str(mode).strip().lower().replace("-", "_") or "persistent_eval"
    stats: dict[str, int | str] = {
        "enabled": 1,
        "mode": normalized,
        "entries": 0,
        "skipped": 0,
    }
    for idx, entry in enumerate(cache or []):
        if entry is None or _is_trimmable(entry):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if isinstance(entry, OwnedRecurrentStateCache):
            entry.mode = normalized
            stats["entries"] = int(stats["entries"]) + 1
            continue
        state = getattr(entry, "state", None)
        if not isinstance(state, list) or len(state) != 2:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if hasattr(entry, "keys") or hasattr(entry, "values"):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        cache[idx] = OwnedRecurrentStateCache.from_cache(entry, mode=normalized)
        stats["entries"] = int(stats["entries"]) + 1
    return stats


def configure_owned_recurrent_state_cache(cache: list[Any]) -> dict[str, int | str]:
    raw = os.environ.get("MTPLX_OWNED_RECURRENT_STATE") or ""
    normalized = raw.strip().lower().replace("-", "_")
    if normalized not in {"1", "true", "yes", "on", "persistent", "persistent_eval"}:
        return {"enabled": 0, "entries": 0, "skipped": 0, "mode": "disabled"}
    mode = os.environ.get("MTPLX_OWNED_RECURRENT_STATE_MODE") or "persistent_eval"
    return install_owned_recurrent_state_cache(cache, mode=mode)


def owned_recurrent_state_stats(cache: list[Any] | None) -> dict[str, int | float | str]:
    aggregate: dict[str, int | float | str] = {
        "enabled": 0,
        "entries": 0,
        "updates": 0,
        "arrays": 0,
        "allocations": 0,
        "inplace_updates": 0,
        "bytes": 0,
        "time_s": 0.0,
        "mode": "disabled",
    }
    for entry in cache or []:
        if not isinstance(entry, OwnedRecurrentStateCache):
            continue
        stats = entry.owner_stats()
        aggregate["enabled"] = 1
        aggregate["entries"] = int(aggregate["entries"]) + 1
        aggregate["updates"] = int(aggregate["updates"]) + int(stats["updates"])
        aggregate["arrays"] = int(aggregate["arrays"]) + int(stats["arrays"])
        aggregate["allocations"] = int(aggregate["allocations"]) + int(stats["allocations"])
        aggregate["inplace_updates"] = int(aggregate["inplace_updates"]) + int(
            stats["inplace_updates"]
        )
        aggregate["bytes"] = int(aggregate["bytes"]) + int(stats["bytes"])
        aggregate["time_s"] = float(aggregate["time_s"]) + float(stats["time_s"])
        aggregate["mode"] = str(stats["mode"])
    return aggregate


def install_tail_owned_attention_kv_cache(
    cache: list[Any],
    *,
    mode: str = "contiguous_eval",
    step: int | None = None,
) -> dict[str, int | str]:
    """Replace stock full-attention KV caches with tail-owner caches."""
    normalized = _normalize_detach_mode(mode)
    stats: dict[str, int | str] = {
        "enabled": 1,
        "mode": normalized,
        "entries": 0,
        "skipped": 0,
    }
    for idx, entry in enumerate(cache or []):
        if entry is None:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if isinstance(entry, TailOwnedKVCache):
            entry.mode = normalized
            stats["entries"] = int(stats["entries"]) + 1
            continue
        if not _is_trimmable(entry):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if getattr(entry, "_idx", None) is not None:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if not hasattr(entry, "keys") or not hasattr(entry, "values"):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        cache[idx] = TailOwnedKVCache.from_cache(
            entry,
            mode=normalized,
            step=step,
        )
        stats["entries"] = int(stats["entries"]) + 1
    return stats


def install_block_owned_attention_kv_cache(
    cache: list[Any],
    *,
    mode: str = "contiguous_eval",
    block_size: int = 1024,
) -> dict[str, int | str]:
    """Replace stock full-attention KV caches with block-owner caches."""
    normalized = _normalize_detach_mode(mode)
    stats: dict[str, int | str] = {
        "enabled": 1,
        "mode": normalized,
        "entries": 0,
        "skipped": 0,
        "block_size": int(block_size),
    }
    for idx, entry in enumerate(cache or []):
        if entry is None:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if isinstance(entry, BlockOwnedKVCache):
            entry.mode = normalized
            entry.block_size = int(block_size)
            entry.step = int(block_size)
            stats["entries"] = int(stats["entries"]) + 1
            continue
        if isinstance(entry, TailOwnedKVCache):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if not _is_trimmable(entry):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if getattr(entry, "_idx", None) is not None:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if not hasattr(entry, "keys") or not hasattr(entry, "values"):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        cache[idx] = BlockOwnedKVCache.from_cache(
            entry,
            mode=normalized,
            block_size=block_size,
        )
        stats["entries"] = int(stats["entries"]) + 1
    return stats


def install_vllm_metal_paged_attention_kv_cache(
    cache: list[Any],
    *,
    block_size: int = 16,
    num_blocks: int = 1024,
    turboquant_config: Any | None = None,
) -> dict[str, int | str]:
    """Replace stock full-attention KV caches with vLLM-Metal paged caches."""
    stats: dict[str, int | str] = {
        "enabled": 1,
        "mode": "vllm_metal_paged_turboquant" if turboquant_config else "vllm_metal_paged",
        "entries": 0,
        "skipped": 0,
        "block_size": int(block_size),
        "num_blocks": int(num_blocks),
        "turboquant": int(bool(turboquant_config)),
        "attention_impl": _paged_attention_impl_from_env() or "vllm_metal",
    }
    external_ops_required = _paged_attention_requires_external_ops(
        turboquant_config=turboquant_config,
    )
    stats["external_ops_required"] = int(external_ops_required)
    if turboquant_config is not None:
        stats["turboquant_k_quant"] = str(turboquant_config.key_quant)
        stats["turboquant_v_quant"] = str(turboquant_config.value_quant)
    # Validate the optional dependency once at install time only for paths that
    # actually dispatch into the external vLLM-Metal ops. The packaged
    # mlx_vector_paged and sdpa_2pass_paged paths are in-tree and must survive a
    # clean product checkout without REFERENCES:TOOLS.
    #
    # When the install is for a TurboQuant snapshot but external ops can't be
    # loaded (no nanobind, no JIT-built paged_ops.so, no MTPLX_VLLM_METAL_REPO),
    # we MUST NOT raise: that would crash any in-flight request that triggered
    # the install. Instead, downgrade gracefully to the plain paged-cache
    # layout. The cache stays functional via the in-tree mlx_vector_paged /
    # sdpa_2pass_paged kernels, just without the TurboQuant compression.
    if external_ops_required:
        try:
            _load_vllm_metal_ops()
        except RuntimeError as exc:
            if turboquant_config is not None:
                _warn_vllm_metal_ops_unavailable(
                    exc, context="TurboQuant install"
                )
                turboquant_config = None
                stats["mode"] = "vllm_metal_paged"
                stats["turboquant"] = 0
                stats["external_ops_required"] = int(
                    _paged_attention_requires_external_ops(
                        turboquant_config=None,
                    )
                )
                stats["turboquant_disabled_reason"] = "vllm_metal_ops_unavailable"
                stats.pop("turboquant_k_quant", None)
                stats.pop("turboquant_v_quant", None)
            else:
                raise
    for idx, entry in enumerate(cache or []):
        if entry is None:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if isinstance(entry, VllmMetalPagedKVCache):
            entry.block_size = int(block_size)
            entry.num_blocks = int(num_blocks)
            entry.turboquant_config = turboquant_config
            entry.turboquant = turboquant_config is not None
            stats["entries"] = int(stats["entries"]) + 1
            continue
        if isinstance(entry, TailOwnedKVCache):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if not _is_trimmable(entry):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if getattr(entry, "_idx", None) is not None:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if not hasattr(entry, "keys") or not hasattr(entry, "values"):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        cache[idx] = VllmMetalPagedKVCache.from_cache(
            entry,
            block_size=block_size,
            num_blocks=num_blocks,
            turboquant_config=turboquant_config,
        )
        stats["entries"] = int(stats["entries"]) + 1
    return stats


def configure_tail_owned_attention_kv_cache(cache: list[Any]) -> dict[str, int | str]:
    paged_raw = os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN") or ""
    if paged_raw.strip().lower() in {"1", "true", "yes", "on"}:
        from .turboquant import config_from_env

        block_size = int(os.environ.get("MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE") or "16")
        configured_blocks = int(os.environ.get("MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS") or "1024")
        num_blocks = _dynamic_paged_num_blocks(
            block_size=block_size,
            configured_blocks=configured_blocks,
        )
        return install_vllm_metal_paged_attention_kv_cache(
            cache,
            block_size=block_size,
            num_blocks=num_blocks,
            turboquant_config=config_from_env(),
        )
    raw = os.environ.get("MTPLX_OWNED_ATTN_KV") or ""
    normalized = raw.strip().lower().replace("-", "_")
    if normalized not in {
        "1",
        "true",
        "yes",
        "on",
        "tail",
        "tail_owned",
        "block",
        "block_owned",
    }:
        return {"enabled": 0, "entries": 0, "skipped": 0, "mode": "disabled"}
    mode = os.environ.get("MTPLX_OWNED_ATTN_KV_MODE") or "contiguous_eval"
    if normalized in {"block", "block_owned"}:
        block_raw = (
            os.environ.get("MTPLX_OWNED_ATTN_KV_BLOCK_SIZE")
            or os.environ.get("MTPLX_OWNED_ATTN_KV_STEP")
            or "1024"
        )
        return install_block_owned_attention_kv_cache(
            cache,
            mode=mode,
            block_size=int(block_raw),
        )
    step_raw = os.environ.get("MTPLX_OWNED_ATTN_KV_STEP")
    step = int(step_raw) if step_raw else None
    return install_tail_owned_attention_kv_cache(cache, mode=mode, step=step)


def configure_mtp_attention_kv_cache(cache: list[Any]) -> dict[str, int | str]:
    """Optionally put the native MTP layer on the vLLM-Metal paged KV path.

    Trunk paged attention is controlled by ``MTPLX_VLLM_METAL_PAGED_ATTN``.
    The MTP layer is kept behind a separate flag because it changes the draft
    proposal path and therefore needs its own speed and parity evidence.
    """

    raw = os.environ.get("MTPLX_VLLM_METAL_PAGED_MTP_ATTN") or ""
    if raw.strip().lower() not in {"1", "true", "yes", "on"}:
        return {"enabled": 0, "entries": 0, "skipped": 0, "mode": "disabled"}
    block_size = int(
        os.environ.get("MTPLX_VLLM_METAL_PAGED_MTP_BLOCK_SIZE")
        or os.environ.get("MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE")
        or "16"
    )
    num_blocks = int(
        os.environ.get("MTPLX_VLLM_METAL_PAGED_MTP_NUM_BLOCKS")
        or os.environ.get("MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS")
        or "1024"
    )
    num_blocks = _dynamic_paged_num_blocks(
        block_size=block_size,
        configured_blocks=num_blocks,
    )
    stats = install_vllm_metal_paged_attention_kv_cache(
        cache,
        block_size=block_size,
        num_blocks=num_blocks,
        turboquant_config=None,
    )
    stats["mode"] = "vllm_metal_paged_mtp"
    return stats


def tail_owned_attention_kv_stats(cache: list[Any] | None) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "enabled": 0,
        "entries": 0,
        "updates": 0,
        "arrays": 0,
        "bytes": 0,
        "time_s": 0.0,
        "mode": "disabled",
    }
    for entry in cache or []:
        if isinstance(entry, VllmMetalPagedKVCache):
            stats = entry.paged_stats()
            aggregate["enabled"] = 1
            aggregate["entries"] = int(aggregate["entries"]) + 1
            aggregate["updates"] = int(aggregate["updates"]) + int(stats["updates"])
            aggregate["arrays"] = int(aggregate["arrays"]) + int(
                stats["paged_attention_calls"]
            )
            aggregate["bytes"] = int(aggregate["bytes"]) + int(stats["bytes"])
            aggregate["time_s"] = float(aggregate["time_s"]) + float(
                stats["cache_write_time_s"]
            ) + float(stats["attention_time_s"])
            aggregate["mode"] = str(stats["mode"])
            aggregate["block_size"] = int(stats["block_size"])
            aggregate["num_blocks"] = int(stats["num_blocks"])
            aggregate["capacity"] = int(stats["capacity"])
            aggregate["partitioned_attention_calls"] = int(
                aggregate.get("partitioned_attention_calls", 0)
            ) + int(stats.get("partitioned_attention_calls", 0))
            aggregate["turboquant_attention_calls"] = int(
                aggregate.get("turboquant_attention_calls", 0)
            ) + int(stats.get("turboquant_attention_calls", 0))
            aggregate["active_array_calls"] = int(
                aggregate.get("active_array_calls", 0)
            ) + int(stats.get("active_array_calls", 0))
            aggregate["dense_fallback_calls"] = int(
                aggregate.get("dense_fallback_calls", 0)
            ) + int(stats.get("dense_fallback_calls", 0))
            for key in (
                "prefill_dense_fallback_calls",
                "decode_dense_fallback_calls",
                "ar_dense_fallback_calls",
                "postcommit_dense_fallback_calls",
                "large_q_split_sdpa_fallback_calls",
                "prefill_large_q_split_sdpa_fallback_calls",
                "decode_large_q_split_sdpa_fallback_calls",
                "partitioned_paged_calls",
                "prefill_partitioned_paged_calls",
                "decode_partitioned_paged_calls",
            ):
                aggregate[key] = int(aggregate.get(key, 0)) + int(stats.get(key, 0))
            for dict_key in (
                "large_q_split_sdpa_fallback_calls_by_phase",
                "partitioned_paged_calls_by_phase",
            ):
                phase_counts = stats.get(dict_key) or {}
                if isinstance(phase_counts, dict):
                    merged = dict(aggregate.get(dict_key) or {})
                    for phase, count in phase_counts.items():
                        merged[str(phase)] = int(merged.get(str(phase), 0)) + int(count)
                    aggregate[dict_key] = merged
            bailouts = stats.get("paged_attention_bailouts_by_phase_reason") or {}
            if isinstance(bailouts, dict):
                merged = dict(aggregate.get("paged_attention_bailouts_by_phase_reason") or {})
                for reason_key, count in bailouts.items():
                    merged[str(reason_key)] = int(merged.get(str(reason_key), 0)) + int(count)
                aggregate["paged_attention_bailouts_by_phase_reason"] = merged
            large_q_path = str(stats.get("paged_attention_large_q_path") or "")
            if large_q_path:
                aggregate["paged_attention_large_q_path"] = large_q_path
            aggregate["grow_events"] = int(
                aggregate.get("grow_events", 0)
            ) + int(stats.get("grow_events", 0))
            aggregate["turboquant"] = int(
                aggregate.get("turboquant", 0)
            ) or int(stats.get("turboquant", 0))
            if stats.get("turboquant_k_quant"):
                aggregate["turboquant_k_quant"] = str(stats["turboquant_k_quant"])
            if stats.get("turboquant_v_quant"):
                aggregate["turboquant_v_quant"] = str(stats["turboquant_v_quant"])
            continue
        if not isinstance(entry, TailOwnedKVCache):
            continue
        stats = entry.tail_owner_stats()
        aggregate["enabled"] = 1
        aggregate["entries"] = int(aggregate["entries"]) + 1
        aggregate["updates"] = int(aggregate["updates"]) + int(stats["updates"])
        aggregate["arrays"] = int(aggregate["arrays"]) + int(stats["arrays"])
        aggregate["bytes"] = int(aggregate["bytes"]) + int(stats["bytes"])
        aggregate["time_s"] = float(aggregate["time_s"]) + float(stats["time_s"])
        aggregate["mode"] = str(stats["mode"])
    return aggregate


def _clone_tree(value: Any) -> Any:
    import mlx.core as mx

    if value is None:
        return None
    if isinstance(value, mx.array):
        # `mx.array(existing_array)` can preserve storage identity for mutable
        # cache buffers. Force a new array expression so later KV writes cannot
        # mutate the saved snapshot behind our back.
        return value + mx.zeros((), dtype=value.dtype)
    if isinstance(value, tuple):
        return tuple(_clone_tree(v) for v in value)
    if isinstance(value, list):
        return [_clone_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_tree(v) for k, v in value.items()}
    return value


def snapshot_cache(cache: list[Any]) -> CacheSnapshot:
    return CacheSnapshot(
        states=tuple(_clone_tree(getattr(c, "state", None)) for c in cache),
        meta_states=tuple(_clone_tree(getattr(c, "meta_state", None)) for c in cache),
    )


def snapshot_untrimmable_cache(cache: list[Any]) -> CacheSnapshot:
    """Snapshot only recurrent/non-trimmable cache state.

    Attention KV caches can roll back by trimming their offset. GDN recurrent
    caches cannot, so those are the states we copy before speculative verify.
    """
    states = []
    meta_states = []
    for entry in cache:
        if _is_trimmable(entry):
            states.append(None)
            meta_states.append(None)
        else:
            states.append(_clone_tree(getattr(entry, "state", None)))
            meta_states.append(_clone_tree(getattr(entry, "meta_state", None)))
    return CacheSnapshot(states=tuple(states), meta_states=tuple(meta_states))


def restore_cache(cache: list[Any], snapshot: CacheSnapshot) -> None:
    for entry, state, meta_state in zip(cache, snapshot.states, snapshot.meta_states):
        if state is not None:
            _restore_state_preserving_container(entry, state)
        if meta_state is not None:
            entry.meta_state = _clone_tree(meta_state)


def _restore_state_preserving_container(entry: Any, state: Any) -> None:
    cloned = _clone_tree(state)
    if hasattr(entry, "replace_state"):
        entry.replace_state(cloned)
        return
    current = getattr(entry, "state", None)
    if isinstance(current, list) and isinstance(cloned, list) and len(current) == len(cloned):
        current[:] = cloned
        return
    entry.state = cloned


def rollback_after_verify(cache: list[Any], snapshot: CacheSnapshot, verified_tokens: int) -> None:
    """Undo a speculative target verify pass."""
    for entry in cache:
        if _is_trimmable(entry) and hasattr(entry, "trim"):
            entry.trim(verified_tokens)
    restore_cache(cache, snapshot)


def detach_array_leaf(value: Any, *, mode: str) -> Any:
    """Return an evaluated cache leaf according to the configured detach mode."""
    import mlx.core as mx

    if not isinstance(value, mx.array):
        return value
    normalized = _normalize_detach_mode(mode)
    if normalized == "eval_only":
        mx.eval(value)
        return value
    if normalized == "metal_copy_leaf":
        from .kernels.copy_leaf import metal_copy_leaf

        return metal_copy_leaf(value)
    leaf = mx.contiguous(value)
    mx.eval(leaf)
    return leaf


def detach_recurrent_cache_state(
    cache: list[Any],
    *,
    components: set[str],
    mode: str,
) -> dict[str, int]:
    """Detach official recurrent cache state in-place.

    This is intentionally limited to non-trimmable recurrent entries. Attention
    KV caches remain on their normal trim/update path until attribution says the
    tail needs its own owner-copy implementation.
    """
    import mlx.core as mx

    requested = {item.strip().lower().replace("-", "_") for item in components if item}
    supported = {"conv", "gdn"}
    requested &= supported
    stats = {"entries": 0, "arrays": 0, "bytes": 0}
    if not requested:
        return stats

    for entry in cache:
        if _is_trimmable(entry):
            continue
        state = getattr(entry, "state", None)
        if not isinstance(state, (list, tuple)) or len(state) < 2:
            continue
        mutable = list(state)
        changed = False
        for index, component in ((0, "conv"), (1, "gdn")):
            if component not in requested:
                continue
            value = mutable[index]
            if not isinstance(value, mx.array):
                continue
            detached = detach_array_leaf(value, mode=mode)
            mutable[index] = detached
            changed = True
            stats["arrays"] += 1
            stats["bytes"] += int(detached.nbytes)
        if not changed:
            continue
        stats["entries"] += 1
        if hasattr(entry, "replace_state"):
            entry.replace_state(mutable)
        elif isinstance(state, list):
            state[:] = mutable
        else:
            entry.state = tuple(mutable)
    return stats


def detach_attention_cache_state(
    cache: list[Any],
    *,
    mode: str,
) -> dict[str, int]:
    """Evaluate or owner-copy attention KV cache arrays in-place."""
    import mlx.core as mx

    stats = {"entries": 0, "arrays": 0, "bytes": 0}
    normalized = _normalize_detach_mode(mode)
    for entry in cache:
        if not _is_trimmable(entry):
            continue
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        if not isinstance(keys, mx.array) or not isinstance(values, mx.array):
            continue
        if normalized == "eval_only":
            mx.eval(keys, values)
            detached_keys, detached_values = keys, values
        else:
            detached_keys = detach_array_leaf(keys, mode=normalized)
            detached_values = detach_array_leaf(values, mode=normalized)
            entry.keys = detached_keys
            entry.values = detached_values
        stats["entries"] += 1
        stats["arrays"] += 2
        stats["bytes"] += int(detached_keys.nbytes) + int(detached_values.nbytes)
    return stats


def detach_cache_state(
    cache: list[Any],
    *,
    components: set[str],
    mode: str,
) -> dict[str, int]:
    """Detach requested cache groups and combine accounting stats."""
    requested = {item.strip().lower().replace("-", "_") for item in components if item}
    stats = detach_recurrent_cache_state(
        cache,
        components=requested & {"gdn", "conv"},
        mode=mode,
    )
    if "attn" in requested or "attn_tail" in requested:
        attn_stats = detach_attention_cache_state(cache, mode=mode)
        for key, value in attn_stats.items():
            stats[key] = int(stats.get(key, 0)) + int(value)
    return stats


def _is_trimmable(entry: Any) -> bool:
    try:
        return bool(entry.is_trimmable())
    except Exception:
        return False
