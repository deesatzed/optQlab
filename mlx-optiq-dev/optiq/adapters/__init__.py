"""OptiQ runtime adapter management.

Serving LoRA adapters on top of OptiQ base models. This package provides:

  * ``AdapterRegistry`` — loads / tracks / unloads adapters at runtime
  * ``resolve_adapter_source`` — HF repo_id or local path → local dir on disk
  * Per-request adapter selection via a ``X-OptiQ-Adapter`` HTTP header
    (plumbing lives in ``optiq/serve.py``)

Memory model: the OptiQ base model loads once (~5 GB for a 9B-4bit).
Each adapter adds ~20–150 MB depending on rank / layers adapted. An LRU
cache caps total adapter memory; adapters evicted when over threshold are
re-fetched on demand.
"""

from .registry import AdapterRegistry, AdapterInfo
from .resolver import resolve_adapter_source

__all__ = ["AdapterRegistry", "AdapterInfo", "resolve_adapter_source"]
