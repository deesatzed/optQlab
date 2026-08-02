"""SSD expert streaming for OptiQ MoE quants.

Large MoE quants (e.g. Qwen3.6-35B-A3B at 21 GB) OOM on a 24 GB Mac: the
fused expert weight tensor for every layer is held resident, even though
only top-k of N experts fire per token. This module streams the active
experts' weights off SSD on demand instead.

Design (validated empirically by a spike before this shipped):
  * Lazy mmap slicing of the fused tensor does NOT bound memory (MLX
    materializes the whole operand on a fancy-index), so we slice at the
    byte level with safetensors.get_slice, which reads only the requested
    expert rows off disk.
  * The expert *weights* (uint32, ~95% of the model) are streamed. The
    scales/biases (bf16, ~1 GB total) stay resident: they are small, and
    bf16 row-slicing is not supported by the installed safetensors.
  * Per token we read only the unique active experts, remap the router
    indices into that compact set, and run mx.gather_qmm over it. Output is
    bit-identical to the resident QuantizedSwitchLinear.

Active-expert counts are bucketed to a small fixed set of sizes so the
gather_qmm operand has a constant shape; without this MLX recompiles the
kernel on every call and prefill is ~40x slower.

Performance levers (Qwen3.6-35B-A3B, 24 GB M4):
  * Concurrent ``pread`` (read the active experts in parallel off a shared fd,
    queue depth 24) is the big win: decode 3.3 -> 8.8 tok/s, prefill 0.8 ->
    26.5 tok/s.
  * Speculative prefetch (warm the previous token's experts in the background
    during compute) and the LRU expert cache (``cache_experts``) are both OFF
    by default: when the model fits the OS page cache they add redundant work
    and slightly regress decode. They only help when the model is far larger
    than RAM. Enable prefetch via OPTIQ_STREAM_PREFETCH=1.
"""

from __future__ import annotations

import glob
import json
import os
import struct
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open


# safetensors dtype -> (numpy read dtype, optional mlx bitcast dtype). bf16 has
# no numpy equivalent, so we read the raw u16 bytes and bitcast to bfloat16.
_ST_DTYPE = {
    "F64": (np.float64, None), "F32": (np.float32, None), "F16": (np.float16, None),
    "BF16": (np.uint16, mx.bfloat16),
    "I64": (np.int64, None), "I32": (np.int32, None),
    "I16": (np.int16, None), "I8": (np.int8, None),
    "U32": (np.uint32, None), "U16": (np.uint16, None), "U8": (np.uint8, None),
    "BOOL": (np.bool_, None),
}


class _ShardTensorReader:
    """Reads whole tensors from a safetensors shard by raw byte range (no mmap).

    Loading the resident params via MLX's mmap would pin every shard as a Metal
    buffer (~22 GB) since the params are scattered across all shards. Reading
    just each tensor's bytes keeps the load footprint to the resident size."""

    _headers: dict[str, tuple] = {}

    @classmethod
    def _header(cls, shard_path: str):
        h = cls._headers.get(shard_path)
        if h is None:
            with open(shard_path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            cls._headers[shard_path] = h = (hdr, 8 + n)
        return h

    @classmethod
    def read(cls, shard_path: str, key: str) -> mx.array:
        hdr, data_start = cls._header(shard_path)
        meta = hdr[key]
        s, e = meta["data_offsets"]
        with open(shard_path, "rb") as f:
            f.seek(data_start + s)
            raw = f.read(e - s)
        np_dt, view_dt = _ST_DTYPE[meta["dtype"]]
        arr = mx.array(np.frombuffer(raw, dtype=np_dt).copy())
        if view_dt is not None:
            arr = arr.view(view_dt)
        return arr.reshape(meta["shape"])


class _ShardWeightReader:
    """Reads expert weight rows (uint32) from a safetensors shard by byte range
    with ``os.pread`` (thread-safe, positional), so the active experts can be
    read concurrently. Also supports background prefetch (``warm``) to overlap
    SSD I/O with GPU compute."""

    _fds: dict[str, int] = {}              # one shared O_RDONLY fd per shard
    _pool = None                            # shared read/prefetch thread pool

    @classmethod
    def _thread_pool(cls):
        if cls._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            # Queue depth tuned for NVMe; matches the SwiftLM-style concurrency.
            cls._pool = ThreadPoolExecutor(max_workers=24,
                                           thread_name_prefix="optiq-expert")
        return cls._pool

    def __init__(self, shard_path: str, key: str, cache_experts: int = 0):
        hdr, data_start = _ShardTensorReader._header(shard_path)
        meta = hdr[key]                     # [num_experts, out, in_packed], U32
        self._out, self._inp = meta["shape"][1], meta["shape"][2]
        self._stride = self._out * self._inp * 4          # bytes per expert
        self._base = data_start + meta["data_offsets"][0]
        fd = _ShardWeightReader._fds.get(shard_path)
        if fd is None:
            fd = os.open(shard_path, os.O_RDONLY)
            _ShardWeightReader._fds[shard_path] = fd
        self._fd = fd

    def _read_one(self, i: int) -> np.ndarray:
        return np.frombuffer(
            os.pread(self._fd, self._stride, self._base + i * self._stride),
            dtype=np.uint32)

    def read(self, ids: list[int]) -> mx.array:
        """Compact [len(ids), out, in_packed] uint32 array, experts read in
        parallel off SSD."""
        if len(ids) > 1:
            rows = list(self._thread_pool().map(self._read_one, ids))
        else:
            rows = [self._read_one(ids[0])]
        return mx.array(np.stack(rows).reshape(len(ids), self._out, self._inp))

    def warm(self, ids) -> None:
        """Fire-and-forget background reads to warm the OS page cache for the
        given experts (used by speculative prefetch)."""
        pool = self._thread_pool()
        base, stride, fd = self._base, self._stride, self._fd
        for i in ids:
            pool.submit(os.pread, fd, stride, base + int(i) * stride)


# Speculative-prefetch registry. Every streaming projection registers here in
# creation order. On the first projection call of a decode token, we kick off
# background reads of the experts each projection used on the PREVIOUS token
# (routing is stable token-to-token, ~70% hit), overlapping SSD I/O with the
# current token's GPU compute.
_STREAM_MODULES: list = []
_CALL_COUNTER = [0]
# Speculative prefetch is OFF by default. Measured (Qwen3.6-35B-A3B, 24 GB Mac):
# it slightly REGRESSES decode (8.81 -> 8.14 tok/s) because the model mostly
# fits the OS page cache, so the prefetch reads are redundant and contend for
# the read pool. It only pays off when the model is far larger than RAM (truly
# cold experts); enable then via OPTIQ_STREAM_PREFETCH=1.
_PREFETCH = [os.environ.get("OPTIQ_STREAM_PREFETCH", "0") == "1"]


def _maybe_prefetch(is_decode: bool) -> None:
    n = len(_STREAM_MODULES)
    if not n:
        return
    c = _CALL_COUNTER[0]
    _CALL_COUNTER[0] = c + 1
    # First projection call of a decode token -> prefetch last token's experts.
    if _PREFETCH[0] and is_decode and c % n == 0:
        for mod in _STREAM_MODULES:
            ids = mod._last_uids
            if ids is not None:
                mod._reader.warm(ids)


class StreamingQuantizedSwitchLinear(nn.Module):
    """Drop-in for mlx_lm's QuantizedSwitchLinear that streams expert weights
    from SSD. scales/biases are resident; only the active experts' weights are
    read per call. Output matches the resident op bit-for-bit."""

    def __init__(
        self,
        shard_path: str,
        weight_key: str,
        scales: mx.array,
        biases: mx.array | None,
        group_size: int,
        bits: int,
        mode: str = "affine",
        bias: mx.array | None = None,
        cache_experts: int = 0,
    ):
        super().__init__()
        self.scales = scales
        if biases is not None:
            self.biases = biases
        if bias is not None:
            self.bias = bias
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._reader = _ShardWeightReader(shard_path, weight_key, cache_experts)
        self._num_experts = scales.shape[0]
        self._last_uids = None
        _STREAM_MODULES.append(self)
        self.freeze()

    @property
    def num_experts(self) -> int:
        return self._num_experts

    def _bucket(self, k: int) -> int:
        """Round the active-expert count up to a fixed bucket so the compact
        operand has one of a few shapes (not a new one every call). MLX keys
        kernel compilation on shape, so this turns ~120 recompiles per prefill
        into a handful."""
        for b in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
            if b >= k:
                return min(b, self._num_experts)
        return self._num_experts

    def __call__(self, x, indices, sorted_indices=False):
        shape = indices.shape
        _maybe_prefetch(is_decode=(shape[0] == 1))
        flat = np.array(indices, copy=False).reshape(-1).astype(np.int64)
        uids, inverse = np.unique(flat, return_inverse=True)
        k = len(uids)
        kp = self._bucket(k)
        self._last_uids = uids

        weight = self._reader.read(uids.tolist())          # streamed [k, ...]
        uids_mx = mx.array(uids.astype(np.uint32))
        scales = self["scales"][uids_mx]                   # resident gather [k, ...]
        biases = self["biases"][uids_mx] if "biases" in self else None

        # Pad the operand up to the bucket size by repeating the last expert.
        # Padded rows are never referenced by rhs_indices, so output is
        # unchanged; only the shape is made constant.
        if kp > k:
            rep = kp - k
            weight = mx.concatenate([weight, mx.repeat(weight[-1:], rep, axis=0)], axis=0)
            scales = mx.concatenate([scales, mx.repeat(scales[-1:], rep, axis=0)], axis=0)
            if biases is not None:
                biases = mx.concatenate([biases, mx.repeat(biases[-1:], rep, axis=0)], axis=0)

        rhs = mx.array(inverse.reshape(shape).astype(np.uint32))
        out = mx.gather_qmm(
            x, weight, scales, biases,
            rhs_indices=rhs, transpose=True,
            group_size=self.group_size, bits=self.bits, mode=self.mode,
            sorted_indices=False,
        )
        if "bias" in self:
            out = out + mx.expand_dims(self["bias"][indices], -2)
        return out


# ---------------------------------------------------------------------------
# Model loading: build a MoE quant with streaming expert weights
# ---------------------------------------------------------------------------


def _resolve_snapshot(model_path: str) -> Path:
    """Local snapshot dir for a repo id (or the path itself if local)."""
    if os.path.isdir(model_path):
        return Path(model_path)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_path))


def _walk(root, dotted: str):
    """Resolve a dotted module path (with list indices) to (parent, leaf)."""
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, parts[-1]


# Routed-expert weight tensors live under one of these path segments depending
# on the arch's fused-expert module: ``.switch_mlp.`` (Qwen3 / Nemotron
# SwitchMLP) or ``.mlp.experts.`` (Gemma / gpt-oss SwitchGLU). Both are a single
# fused ``QuantizedSwitchLinear`` per projection, which is what we stream.
_EXPERT_SEGMENTS = (".switch_mlp.", ".mlp.experts.")


def _is_expert_weight(key: str) -> bool:
    """True for a fused routed-expert weight tensor (any supported arch)."""
    return key.endswith(".weight") and any(seg in key for seg in _EXPERT_SEGMENTS)


def load_streaming(model_path: str, cache_experts: int = 0, verbose: bool = True):
    """Load an OptiQ MoE quant with its expert weights streamed from SSD.

    Every ``switch_mlp.{gate,up,down}_proj`` (a QuantizedSwitchLinear) is
    replaced with a StreamingQuantizedSwitchLinear: scales/biases stay
    resident, the uint32 expert weights are read on demand. The big fused
    weight tensors are never materialized, so a quant that would OOM resident
    (e.g. Qwen3.6-35B-A3B, 21 GB on a 24 GB Mac) fits.

    Returns ``(model, tokenizer)``.
    """
    from mlx_lm.utils import load_model, load_tokenizer

    # Teach mlx-lm to build any custom OptiQ decoder this artifact needs before
    # load_model runs. For mistral4 the base repo's ``mistral3`` wrapper otherwise
    # falls back to ``llama`` and silently drops every one of the 128 experts.
    # install() is idempotent and arch-gated (routes on text_config.model_type),
    # so it's a no-op for every other family.
    from optiq.models import mistral4, llada2
    mistral4.install()
    llada2.install()

    # Cap MLX's buffer cache for the duration of the streaming load + run. The
    # resident-param read below touches hundreds of tensors; without a cap MLX
    # retains their freed buffers and the transient cache (~2x the resident
    # footprint) OOM-kills a 100B-class load on a 36 GB Mac — even though the
    # steady resident set is small (e.g. ~10.7 GB for Qwen3.5-122B-A10B at
    # 2/4-bit, with 35 GB of experts streamed). Decode here is I/O-bound on the
    # expert reads, so a small cache costs little throughput. This is in
    # load_streaming so every caller — optiq serve, the Lab — gets it.
    mx.set_cache_limit(512 * 1024 * 1024)

    snap = _resolve_snapshot(model_path)
    config = json.loads((snap / "config.json").read_text())
    quant = config.get("quantization", {})
    top_bits = quant.get("bits", 4)
    top_gs = quant.get("group_size", 64)
    mode = quant.get("mode", "affine")

    weight_map = json.loads(
        (snap / "model.safetensors.index.json").read_text())["weight_map"]

    # Build the model lazily so nothing is materialized yet. fast_quantized_load
    # keeps mlx-lm from constructing random float weights and quantizing them
    # before load_weights overwrites the lot — a transient that peaks near the
    # full model size (53 GiB for the 122B) and is the real reason a big quant
    # crawls or gets OS-killed on load.
    from optiq.runtime.fast_load import fast_quantized_load

    with fast_quantized_load():
        model, _ = load_model(snap, lazy=True, strict=False)

    expert_paths = sorted({
        k[: -len(".weight")]
        for k in weight_map
        if _is_expert_weight(k)
    })

    n = 0
    for path in expert_paths:
        # Strip a leading "model." that mlx-lm sometimes drops from the tree.
        tree_path = path[len("model."):] if path.startswith("model.") else path
        try:
            parent, leaf = _walk(model, tree_path)
            mod = getattr(parent, leaf)
        except (AttributeError, IndexError, KeyError):
            continue
        if not isinstance(mod, nn.Module) or "scales" not in mod:
            continue

        ov = quant.get(path, {})
        bits = ov.get("bits", top_bits) if isinstance(ov, dict) else top_bits
        gs = ov.get("group_size", top_gs) if isinstance(ov, dict) else top_gs

        # Keep scales/biases lazy here — materializing them now would mmap
        # their shards. The shard-by-shard copy below is the only place that
        # touches shards, so it can release each before mapping the next.
        scales = mod["scales"]
        biases = mod["biases"] if "biases" in mod else None
        bias = mod["bias"] if "bias" in mod else None

        weight_key = path + ".weight"
        shard = str(snap / weight_map[weight_key])
        streaming = StreamingQuantizedSwitchLinear(
            shard, weight_key, scales=scales, biases=biases,
            group_size=gs, bits=bits, mode=mode, bias=bias,
            cache_experts=cache_experts,
        )
        setattr(parent, leaf, streaming)
        n += 1

    # Load the resident params by raw byte range (no mmap). MLX's mmap would
    # pin every shard as a Metal buffer (~22 GB) since the resident params are
    # scattered across all shards; reading just each tensor's bytes keeps the
    # load footprint to the resident size (~4.5 GB), so a server doesn't OOM on
    # load. Keys absent from the index (e.g. tied weights) are eval'd as-is.
    from mlx.utils import tree_flatten, tree_unflatten
    mx.reset_peak_memory()
    fresh, batch = [], []
    for k, v in tree_flatten(model.parameters()):
        shard = weight_map.get(k)
        a = _ShardTensorReader.read(str(snap / shard), k) if shard else (v + 0)
        fresh.append((k, a))
        batch.append(a)
        # Eval in small batches so each Metal command buffer stays small and the
        # per-tensor numpy read buffers free immediately (a single big eval can
        # exceed a command buffer's working-memory budget and OOM).
        if len(batch) >= 8:
            mx.eval(batch)
            mx.clear_cache()
            batch = []
    if batch:
        mx.eval(batch)
    model.update(tree_unflatten(fresh))
    del fresh
    mx.clear_cache()
    if verbose:
        print(f"[moe_stream] swapped {n} expert projections; "
              f"resident {mx.get_active_memory()/1e9:.2f} GB "
              f"(load peak {mx.get_peak_memory()/1e9:.2f} GB)", flush=True)

    tokenizer = load_tokenizer(snap)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Detection helpers (used by optiq serve to decide when to stream)
# ---------------------------------------------------------------------------


def is_streamable_moe(model_path: str) -> bool:
    """True if the model is a sharded MoE quant with fused routed-expert weight
    tensors (``.switch_mlp.`` or ``.mlp.experts.``) that load_streaming can
    stream."""
    try:
        snap = _resolve_snapshot(model_path)
        idx = snap / "model.safetensors.index.json"
        if not idx.exists():
            return False
        wm = json.loads(idx.read_text())["weight_map"]
        return any(_is_expert_weight(k) for k in wm)
    except Exception:
        return False


def model_disk_bytes(model_path: str) -> int:
    """Total bytes of the model's safetensors shards."""
    try:
        snap = _resolve_snapshot(model_path)
        return sum(os.path.getsize(f) for f in glob.glob(str(snap / "*.safetensors")))
    except Exception:
        return 0


def should_stream(model_path: str, headroom: float = 0.70) -> bool:
    """Auto-detect: stream a MoE quant whose on-disk weights are too large to
    sit resident in ``headroom`` of total RAM (where MLX would OOM at compute)."""
    if not is_streamable_moe(model_path):
        return False
    try:
        import psutil
        total = psutil.virtual_memory().total
    except Exception:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:
            return False
    return model_disk_bytes(model_path) > headroom * total
