"""Warm-prefix state reuse for MTPLX target prefill.

SessionBank is deliberately conservative in this first version: it stores
exact token-prefix entries in memory, restores cloned cache state into a fresh
runtime cache, then forwards only the suffix tokens. The benchmark gate compares
the warm result against a cold full prefill before any generation path uses it.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import mlx.core as mx
import numpy as np

from .cache_state import CacheSnapshot, _clone_tree, restore_cache, snapshot_cache
from .runtime import MTPLXRuntime

GIB = 1024**3
DEFAULT_MAX_ENTRIES = 8
DEFAULT_MAX_BYTES = 24 * GIB
DEFAULT_PER_SESSION_MAX_BYTES = 8 * GIB
DEFAULT_IDLE_TTL_S = 60 * 60


class CacheMissReason(str, Enum):
    NEW_SESSION = "new_session"
    PREFIX_DIVERGENCE_AT_TOKEN = "prefix_divergence_at_token"
    MODEL_MISMATCH = "model_mismatch"
    TEMPLATE_MISMATCH = "template_mismatch"
    POLICY_MISMATCH = "policy_mismatch"
    EVICTED = "evicted"
    BACKGROUND_BYPASS = "background_bypass"
    SESSION_BUSY = "session_busy"
    SNAPSHOT_DESYNC = "snapshot_desync"
    NO_SNAPSHOT_COVERAGE = "no_snapshot_coverage"


def token_prefix_hash(token_ids: list[int] | tuple[int, ...]) -> str:
    h = hashlib.sha256()
    for token in token_ids:
        h.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()


def common_prefix_len(left: list[int] | tuple[int, ...], right: list[int] | tuple[int, ...]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    return limit


# Policies that share the committed-mtp-cache representation. An entry stored
# under any of these policies can be safely reused for a lookup that requests
# any other policy in this set, because the cache snapshot shape is identical
# (``last_window`` is just a runtime trim of the same committed cache).
_COMMITTED_CACHE_POLICIES = frozenset({"committed", "last_window"})


def _mtp_history_policy_compatible(
    entry_policy: str | None, lookup_policy: str | None
) -> bool:
    """Return True if a bank entry stored under ``entry_policy`` may be reused
    for a lookup that resolved to ``lookup_policy``.

    Equality is always compatible. Beyond that, ``committed`` and
    ``last_window`` are treated as interchangeable because both rely on the
    same committed mtp-history cache shape; the only difference between them
    is a runtime trim that is applied during prefill, which is moot once the
    cache is being restored from a stored snapshot.
    """
    if entry_policy == lookup_policy:
        return True
    if entry_policy is None or lookup_policy is None:
        return False
    return (
        entry_policy in _COMMITTED_CACHE_POLICIES
        and lookup_policy in _COMMITTED_CACHE_POLICIES
    )


def _tree_nbytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, CacheSnapshot):
        return _tree_nbytes(value.states) + _tree_nbytes(value.meta_states)
    if isinstance(value, mx.array):
        return int(value.nbytes)
    if isinstance(value, (list, tuple)):
        return sum(_tree_nbytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_tree_nbytes(item) for item in value.values())
    return 0


def _snapshot_nbytes(snapshot: CacheSnapshot) -> int:
    return _tree_nbytes(snapshot.states) + _tree_nbytes(snapshot.meta_states)


@dataclass
class SessionBankEntry:
    token_ids: tuple[int, ...]
    token_hash: str
    model_path: str
    mtp_enabled: bool
    hidden_variant: str | None
    cache_snapshot: CacheSnapshot
    logits: Any
    hidden: Any | None
    cache_ref: list[Any] | None = None
    created_at_s: float = field(default_factory=time.time)
    last_access_s: float = field(default_factory=time.time)
    hits: int = 0
    nbytes: int = 0
    session_id: str | None = None
    template_hash: str | None = None
    mtp_history_policy: str | None = None
    draft_head_identity: str | None = None
    policy_fingerprint: str | None = None
    mtp_history_snapshot: Any | None = None
    snapshot_epoch: int = 0
    mtp_snapshot_epoch: int | None = None
    eviction_reason: str | None = None

    @property
    def prefix_len(self) -> int:
        return len(self.token_ids)


@dataclass
class SessionBankRestore:
    entry: SessionBankEntry
    cache: list[Any]
    logits: Any
    hidden: Any | None
    restored_nbytes: int
    restore_mode: str = "clone"
    cache_miss_reason: str | None = None
    mtp_history_snapshot: Any | None = None
    mtp_history_cache: list[Any] | None = None


class SessionBank:
    """In-memory exact prefix table for warm target prefill."""

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        per_session_max_bytes: int = DEFAULT_PER_SESSION_MAX_BYTES,
        idle_ttl_s: float = DEFAULT_IDLE_TTL_S,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        if per_session_max_bytes < 1:
            raise ValueError("per_session_max_bytes must be >= 1")
        if idle_ttl_s <= 0:
            raise ValueError("idle_ttl_s must be > 0")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.per_session_max_bytes = int(per_session_max_bytes)
        self.idle_ttl_s = float(idle_ttl_s)
        self._entries: dict[tuple[int, ...], SessionBankEntry] = {}
        self.last_miss_reason: str | None = None
        self.last_put_nbytes: int = 0
        self.last_put_skipped_oversized_snapshot: bool = False
        self.eviction_log: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def total_nbytes(self) -> int:
        return sum(entry.nbytes for entry in self._entries.values())

    def put(
        self,
        *,
        runtime: MTPLXRuntime,
        token_ids: list[int] | tuple[int, ...],
        cache: list[Any],
        logits: Any,
        hidden: Any | None,
        hidden_variant: str | None = None,
        keep_live_ref: bool = False,
        session_id: str | None = None,
        template_hash: str | None = None,
        mtp_history_policy: str | None = None,
        draft_head_identity: str | None = None,
        policy_fingerprint: str | None = None,
        mtp_history_snapshot: Any | None = None,
        snapshot_epoch: int = 0,
        mtp_snapshot_epoch: int | None = None,
        nbytes_override: int | None = None,
    ) -> SessionBankEntry | None:
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("cannot store an empty prefix")
        if mtp_snapshot_epoch is not None and int(mtp_snapshot_epoch) != int(snapshot_epoch):
            raise ValueError("trunk and MTP snapshots must share the same commit boundary")
        self.last_put_nbytes = 0
        self.last_put_skipped_oversized_snapshot = False
        if nbytes_override is not None and int(nbytes_override) > self.per_session_max_bytes:
            self.last_put_nbytes = int(nbytes_override)
            self.last_put_skipped_oversized_snapshot = True
            self.eviction_log.append(
                {
                    "reason": "skipped_oversized_snapshot",
                    "session_id": session_id,
                    "prefix_len": len(tokens),
                    "token_hash": token_prefix_hash(tokens),
                    "nbytes": int(nbytes_override),
                    "budget": int(self.per_session_max_bytes),
                }
            )
            return None
        try:
            snapshot = snapshot_cache(cache)
        except RuntimeError as exc:
            if "materialize active K/V arrays" not in str(exc):
                raise
            self.last_put_skipped_oversized_snapshot = True
            self.eviction_log.append(
                {
                    "reason": "skipped_dense_materializing_snapshot",
                    "session_id": session_id,
                    "prefix_len": len(tokens),
                    "token_hash": token_prefix_hash(tokens),
                    "nbytes": 0,
                    "budget": int(self.per_session_max_bytes),
                    "error": str(exc),
                }
            )
            return None
        computed_nbytes = (
            _snapshot_nbytes(snapshot)
            + _tree_nbytes(logits)
            + _tree_nbytes(hidden)
            + _tree_nbytes(mtp_history_snapshot)
        )
        entry_nbytes = int(nbytes_override if nbytes_override is not None else computed_nbytes)
        self.last_put_nbytes = int(entry_nbytes)
        if entry_nbytes > self.per_session_max_bytes:
            self.last_put_skipped_oversized_snapshot = True
            self.eviction_log.append(
                {
                    "reason": "skipped_oversized_snapshot",
                    "session_id": session_id,
                    "prefix_len": len(tokens),
                    "token_hash": token_prefix_hash(tokens),
                    "nbytes": int(entry_nbytes),
                    "budget": int(self.per_session_max_bytes),
                }
            )
            return None
        entry = SessionBankEntry(
            token_ids=tokens,
            token_hash=token_prefix_hash(tokens),
            model_path=str(runtime.model_path),
            mtp_enabled=bool(runtime.mtp_enabled),
            hidden_variant=hidden_variant,
            cache_snapshot=snapshot,
            logits=_clone_tree(logits),
            hidden=_clone_tree(hidden),
            cache_ref=cache if keep_live_ref else None,
            nbytes=int(entry_nbytes),
            session_id=session_id,
            template_hash=template_hash,
            mtp_history_policy=mtp_history_policy,
            draft_head_identity=draft_head_identity,
            policy_fingerprint=policy_fingerprint,
            mtp_history_snapshot=_clone_tree(mtp_history_snapshot),
            snapshot_epoch=int(snapshot_epoch),
            mtp_snapshot_epoch=(
                int(mtp_snapshot_epoch)
                if mtp_snapshot_epoch is not None
                else (int(snapshot_epoch) if mtp_history_snapshot is not None else None)
            ),
        )
        self._entries[tokens] = entry
        self._evict_if_needed(protected_tokens=tokens)
        return entry

    def longest_prefix(self, token_ids: list[int] | tuple[int, ...]) -> SessionBankEntry | None:
        tokens = tuple(int(token) for token in token_ids)
        best: SessionBankEntry | None = None
        for prefix, entry in self._entries.items():
            if len(prefix) > len(tokens):
                continue
            if tokens[: len(prefix)] != prefix:
                continue
            if best is None or len(prefix) > len(best.token_ids):
                best = entry
        return best

    def near_prefix_candidates(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        max_token_gap: int = 8,
        min_matched_tokens: int = 64,
    ) -> list[tuple[SessionBankEntry, int]]:
        """Return entries whose divergence is only at the prompt boundary.

        This is for tokenizer-boundary drift in tool-call transcripts. It does
        not accept arbitrary shared system prompts: the common prefix must reach
        within ``max_token_gap`` tokens of the stored entry's end.
        """
        tokens = tuple(int(token) for token in token_ids)
        gap_limit = max(0, int(max_token_gap))
        min_match = max(1, int(min_matched_tokens))
        matches: list[tuple[SessionBankEntry, int]] = []
        self._purge_expired()
        for entry in self._entries.values():
            prefix = entry.token_ids
            if not prefix:
                continue
            matched = common_prefix_len(tokens, prefix)
            gap = len(prefix) - matched
            required_match = min(min_match, max(1, len(prefix) - gap_limit))
            if gap < 0 or gap > gap_limit or matched < required_match:
                continue
            matches.append((entry, matched))
        matches.sort(key=lambda item: (item[1], item[0].prefix_len), reverse=True)
        return matches

    def restore(
        self,
        runtime: MTPLXRuntime,
        token_ids: list[int] | tuple[int, ...],
        *,
        mode: str = "clone",
        hidden_variant: str | None = None,
        template_hash: str | None = None,
        mtp_history_policy: str | None = None,
        draft_head_identity: str | None = None,
        policy_fingerprint: str | None = None,
    ) -> SessionBankRestore | None:
        mode = str(mode).replace("-", "_")
        if mode == "reference_lease":
            mode = "reference"
        if mode not in {"clone", "reference"}:
            raise ValueError("mode must be 'clone', 'reference', or 'reference_lease'")
        self.last_miss_reason = None
        self._purge_expired()
        entry = self.longest_prefix(token_ids)
        if entry is None:
            self.last_miss_reason = (
                CacheMissReason.PREFIX_DIVERGENCE_AT_TOKEN.value
                if self._entries
                else CacheMissReason.NEW_SESSION.value
            )
            return None
        if entry.model_path != str(runtime.model_path):
            self.last_miss_reason = CacheMissReason.MODEL_MISMATCH.value
            return None
        if hidden_variant is not None and entry.hidden_variant != hidden_variant:
            self.last_miss_reason = CacheMissReason.POLICY_MISMATCH.value
            return None
        if template_hash is not None and entry.template_hash != template_hash:
            self.last_miss_reason = CacheMissReason.TEMPLATE_MISMATCH.value
            return None
        if mtp_history_policy is not None and not _mtp_history_policy_compatible(
            entry.mtp_history_policy, mtp_history_policy
        ):
            self.last_miss_reason = CacheMissReason.POLICY_MISMATCH.value
            return None
        if draft_head_identity is not None and entry.draft_head_identity != draft_head_identity:
            self.last_miss_reason = CacheMissReason.POLICY_MISMATCH.value
            return None
        if policy_fingerprint is not None and entry.policy_fingerprint != policy_fingerprint:
            self.last_miss_reason = CacheMissReason.POLICY_MISMATCH.value
            return None
        if (
            entry.mtp_snapshot_epoch is not None
            and int(entry.mtp_snapshot_epoch) != int(entry.snapshot_epoch)
        ):
            self.last_miss_reason = CacheMissReason.SNAPSHOT_DESYNC.value
            return None
        actual_restore_mode = "clone"
        if mode == "reference" and entry.cache_ref is not None:
            cache = entry.cache_ref
            entry.cache_ref = None
            actual_restore_mode = "reference_lease"
        else:
            cache = runtime.make_cache()
            restore_cache(cache, entry.cache_snapshot)
        mtp_history_cache = None
        if entry.mtp_history_snapshot is not None:
            mtp_history_cache = runtime.make_mtp_cache()
            restore_cache(mtp_history_cache, entry.mtp_history_snapshot)
        entry.hits += 1
        entry.last_access_s = time.time()
        return SessionBankRestore(
            entry=entry,
            cache=cache,
            logits=_clone_tree(entry.logits),
            hidden=_clone_tree(entry.hidden),
            restored_nbytes=entry.nbytes,
            restore_mode=actual_restore_mode,
            mtp_history_snapshot=_clone_tree(entry.mtp_history_snapshot),
            mtp_history_cache=mtp_history_cache,
        )

    def clear(self, *, session_id: str | None = None) -> int:
        if session_id is None:
            count = len(self._entries)
            self._entries.clear()
            return count
        victims = [
            tokens
            for tokens, entry in self._entries.items()
            if entry.session_id == session_id
        ]
        for tokens in victims:
            self._entries.pop(tokens, None)
        return len(victims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "per_session_max_bytes": self.per_session_max_bytes,
            "idle_ttl_s": self.idle_ttl_s,
            "entries": len(self._entries),
            "total_nbytes": self.total_nbytes,
            "last_miss_reason": self.last_miss_reason,
            "prefixes": [
                {
                    "session_id": entry.session_id,
                    "prefix_len": entry.prefix_len,
                    "token_hash": entry.token_hash,
                    "model_path": entry.model_path,
                    "mtp_enabled": entry.mtp_enabled,
                    "hidden_variant": entry.hidden_variant,
                    "template_hash": entry.template_hash,
                    "mtp_history_policy": entry.mtp_history_policy,
                    "draft_head_identity": entry.draft_head_identity,
                    "policy_fingerprint": entry.policy_fingerprint,
                    "hits": entry.hits,
                    "nbytes": entry.nbytes,
                    "created_at_s": entry.created_at_s,
                    "last_access_s": entry.last_access_s,
                    "has_live_ref": entry.cache_ref is not None,
                    "snapshot_epoch": entry.snapshot_epoch,
                    "mtp_snapshot_epoch": entry.mtp_snapshot_epoch,
                }
                for entry in sorted(self._entries.values(), key=lambda item: item.prefix_len)
            ],
            "eviction_log": list(self.eviction_log[-16:]),
        }

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            entry
            for entry in self._entries.values()
            if now - float(entry.last_access_s) > self.idle_ttl_s
        ]
        for entry in expired:
            self._evict_entry(entry, reason=CacheMissReason.EVICTED.value)

    def _session_nbytes(self, session_id: str | None) -> int:
        return sum(
            entry.nbytes
            for entry in self._entries.values()
            if entry.session_id == session_id
        )

    def _evict_if_needed(self, *, protected_tokens: tuple[int, ...] | None = None) -> None:
        while True:
            if not self._entries:
                return
            session_over_budget = {
                entry.session_id
                for entry in self._entries.values()
                if self._session_nbytes(entry.session_id) > self.per_session_max_bytes
            }
            reason: str | None = None
            candidates = list(self._entries.values())
            if len(self._entries) > self.max_entries:
                reason = CacheMissReason.EVICTED.value
            elif self.total_nbytes > self.max_bytes:
                reason = CacheMissReason.EVICTED.value
            elif session_over_budget:
                reason = CacheMissReason.EVICTED.value
                candidates = [
                    entry
                    for entry in candidates
                    if entry.session_id in session_over_budget
                ]
            else:
                return

            unprotected = [
                entry
                for entry in candidates
                if protected_tokens is None or entry.token_ids != protected_tokens
            ]
            if unprotected:
                candidates = unprotected
            elif len(candidates) == 1:
                entry = candidates[0]
                if (
                    entry.nbytes > self.per_session_max_bytes
                    or entry.nbytes > self.max_bytes
                ):
                    self._evict_entry(entry, reason=reason or CacheMissReason.EVICTED.value)
                    continue
                return
            victim = min(
                candidates,
                key=lambda entry: (entry.last_access_s, -entry.nbytes, entry.created_at_s),
            )
            self._evict_entry(victim, reason=reason)

    def _evict_entry(self, entry: SessionBankEntry, *, reason: str) -> None:
        entry.eviction_reason = reason
        self._entries.pop(entry.token_ids, None)
        self.eviction_log.append(
            {
                "reason": reason,
                "session_id": entry.session_id,
                "prefix_len": entry.prefix_len,
                "token_hash": entry.token_hash,
                "nbytes": entry.nbytes,
                "last_access_s": entry.last_access_s,
            }
        )


def prefill_target(
    runtime: MTPLXRuntime,
    token_ids: list[int],
    *,
    return_hidden: bool = True,
) -> tuple[list[Any], Any, Any | None, float]:
    """Prefill using the same all-but-last/last-token split as generation."""
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    cache = runtime.make_cache()
    elapsed = 0.0
    if len(token_ids) > 1:
        started = time.perf_counter()
        prefill = runtime.forward_ar(
            mx.array([token_ids[:-1]]),
            cache=cache,
            return_hidden=False,
        )
        mx.eval(prefill)
        elapsed += time.perf_counter() - started

    started = time.perf_counter()
    result = runtime.forward_ar(
        mx.array([[token_ids[-1]]]),
        cache=cache,
        return_hidden=return_hidden,
    )
    if return_hidden:
        logits, hidden_seq = result
        mx.eval(logits, hidden_seq)
        hidden = hidden_seq[:, -1:, :]
    else:
        logits = result
        hidden = None
        mx.eval(logits)
    elapsed += time.perf_counter() - started
    return cache, logits[:, -1, :], hidden, elapsed


def prefill_target_with_session_bank(
    runtime: MTPLXRuntime,
    token_ids: list[int],
    bank: SessionBank,
    *,
    return_hidden: bool = True,
    restore_mode: str = "clone",
) -> tuple[list[Any], Any, Any | None, float, dict[str, Any]]:
    started_total = time.perf_counter()
    restored = bank.restore(runtime, token_ids, mode=restore_mode)
    if restored is None:
        cache, logits, hidden, elapsed = prefill_target(
            runtime,
            token_ids,
            return_hidden=return_hidden,
        )
        return cache, logits, hidden, elapsed, {
            "hit": False,
            "prefix_len": 0,
            "suffix_len": len(token_ids),
        }

    suffix = list(token_ids[restored.entry.prefix_len :])
    if not suffix:
        elapsed = time.perf_counter() - started_total
        return restored.cache, restored.logits, restored.hidden, elapsed, {
            "hit": True,
            "prefix_len": restored.entry.prefix_len,
            "suffix_len": 0,
            "restored_nbytes": restored.restored_nbytes,
            "restore_included_s": elapsed,
            "restore_mode": restore_mode,
        }

    elapsed_suffix = 0.0
    if len(suffix) > 1:
        started = time.perf_counter()
        prefill = runtime.forward_ar(
            mx.array([suffix[:-1]]),
            cache=restored.cache,
            return_hidden=False,
        )
        mx.eval(prefill)
        elapsed_suffix += time.perf_counter() - started

    started = time.perf_counter()
    result = runtime.forward_ar(
        mx.array([[suffix[-1]]]),
        cache=restored.cache,
        return_hidden=return_hidden,
    )
    if return_hidden:
        logits, hidden_seq = result
        mx.eval(logits, hidden_seq)
        hidden = hidden_seq[:, -1:, :]
    else:
        logits = result
        hidden = None
        mx.eval(logits)
    elapsed_suffix += time.perf_counter() - started
    elapsed_total = time.perf_counter() - started_total
    return restored.cache, logits[:, -1, :], hidden, elapsed_total, {
        "hit": True,
        "prefix_len": restored.entry.prefix_len,
        "suffix_len": len(suffix),
        "restored_nbytes": restored.restored_nbytes,
        "suffix_forward_s": elapsed_suffix,
        "restore_and_suffix_s": elapsed_total,
        "restore_mode": restore_mode,
    }


def max_abs_diff(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    diff = mx.abs(left.astype(mx.float32) - right.astype(mx.float32))
    mx.eval(diff)
    return float(np.max(np.asarray(diff)))
