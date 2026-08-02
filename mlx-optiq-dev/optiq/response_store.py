"""In-memory store for OpenAI Responses ``previous_response_id`` resumption.

The Responses API supports follow-up requests that pass
``previous_response_id`` instead of resending the full conversation. The
server is expected to look up the prior response's input + output and
prepend them to the new request's input.

This is a process-local, TTL-bounded dict; it does not survive restarts.
For the OptiQ local-serve use case (one developer machine, one workload
at a time) that's sufficient — a sturdier store would mean Redis, which
is out of scope.

Eviction is lazy: every ``put`` and ``get`` walks the index and drops
expired entries. A capped LRU keeps memory bounded even if the user
never lets entries expire (32 MiB of stored items max by default).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Optional


DEFAULT_TTL_SECONDS = 60 * 60  # 1 hour — matches OpenAI's default
DEFAULT_MAX_BYTES = 32 * 1024 * 1024  # 32 MiB


class ResponseStore:
    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        # OrderedDict gives O(1) move-to-end for LRU semantics.
        self._items: "OrderedDict[str, dict]" = OrderedDict()
        self._size_bytes = 0

    @staticmethod
    def _estimate_size(entry: dict) -> int:
        # ``input`` and ``output`` are the heavy fields; serialise once for
        # both byte accounting and a cheap deep-copy on retrieval.
        import json
        return len(json.dumps(entry["input"])) + len(json.dumps(entry["output"]))

    def put(
        self,
        response_id: str,
        *,
        input_items: list,
        output_items: list,
        instructions: Optional[str] = None,
    ) -> None:
        entry = {
            "input": list(input_items),
            "output": list(output_items),
            "instructions": instructions,
            "created_at": time.time(),
        }
        entry_size = self._estimate_size(entry)
        with self._lock:
            self._evict_expired_locked()
            if response_id in self._items:
                self._size_bytes -= self._items[response_id].get("_size", 0)
            entry["_size"] = entry_size
            self._items[response_id] = entry
            self._items.move_to_end(response_id)
            self._size_bytes += entry_size
            self._evict_lru_locked()

    def get(self, response_id: str) -> Optional[dict]:
        with self._lock:
            self._evict_expired_locked()
            entry = self._items.get(response_id)
            if entry is None:
                return None
            self._items.move_to_end(response_id)
            return {
                "input": list(entry["input"]),
                "output": list(entry["output"]),
                "instructions": entry.get("instructions"),
            }

    def _evict_expired_locked(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        # Iterate in insertion order; expired entries are oldest first.
        expired_ids = []
        for rid, entry in self._items.items():
            if entry["created_at"] < cutoff:
                expired_ids.append(rid)
            else:
                break
        for rid in expired_ids:
            self._size_bytes -= self._items[rid].get("_size", 0)
            del self._items[rid]

    def _evict_lru_locked(self) -> None:
        while self._size_bytes > self.max_bytes and self._items:
            rid, entry = self._items.popitem(last=False)
            self._size_bytes -= entry.get("_size", 0)


_STORE: Optional[ResponseStore] = None


def get_store() -> ResponseStore:
    """Lazily build the process-global store."""
    global _STORE
    if _STORE is None:
        _STORE = ResponseStore()
    return _STORE
