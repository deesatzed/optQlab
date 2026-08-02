"""Pipeline-parallel inference across OptiQ cluster nodes.

Each rank loads ONLY its slice of the transformer layers, so a quant too big for
any single Mac's RAM (e.g. Qwen3.5-122B-A10B-OptiQ-2bit, ~46 GB) runs *resident*
split across two Macs instead of SSD-streaming on one.

The ring is exposed to mlx_lm as ONE model. ``DistributedModel`` (rank 0)
implements the standard ``model(inputs, cache)`` forward by driving the ring:
it embeds + runs its layers, passes activations down the line, and the last rank
returns the logits. So mlx_lm's own ``stream_generate`` runs on rank 0 and
handles sampling, structured output (logits processors), streaming and stop
tokens — the cluster serves through the *exact same generation code path* as a
single Mac. The other ranks run ``run_worker`` (a bare forward-participation
loop). ``optiq.cluster.serve`` wires this behind a thin OpenAI HTTP layer.

Memory win: ``load(lazy=True)`` never materializes weights until eval'd, so each
rank evals only its own layer slice (+ embed/norm/head) — resident RAM ~ shard.
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx


def _dissect(model):
    """Return (core, head_fn) for a Qwen3.5-family / standard decoder model.
    core has .embed_tokens, .layers, .norm; head projects hidden -> logits."""
    inner = getattr(model, "language_model", model)
    core = getattr(inner, "model", inner)
    if hasattr(inner, "lm_head") and inner.lm_head is not None:
        head = inner.lm_head
    else:  # tied embedding
        head = core.embed_tokens.as_linear
    return core, head


def _shard_bounds(n_layers: int, size: int, weights=None) -> list[int]:
    """Contiguous layer boundaries [0, b1, ..., n]. With ``weights`` (per rank,
    e.g. usable RAM) the split is proportional so a Mac with more RAM holds more
    layers — keeps a 46 GB model off the 24 GB Mac's swap."""
    if not weights:
        weights = [1.0] * size
    total = sum(weights) or 1.0
    cum, acc = [0.0], 0.0
    for w in weights:
        acc += w
        cum.append(acc)
    return [round(c / total * n_layers) for c in cum]


def _self_ram_bytes() -> float:
    """Total physical RAM (only a fallback — see ``_self_available_bytes``)."""
    import subprocess
    try:
        return float(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                    capture_output=True, text=True,
                                    timeout=4).stdout.strip())
    except Exception:
        return 16e9


def parse_available_bytes(vm_stat_out: str, total_bytes: float) -> float | None:
    """RAM a Mac can actually give us, from ``vm_stat`` output.

    ``total - (wired + anonymous + compressed)`` — the same quantity Activity
    Monitor calls available. Counting only free+inactive+purgeable+speculative
    undercounts, because *active* file-backed pages (page cache) are evictable
    too: a Mac idling with 8 GiB of cached files is not using 8 GiB.
    """
    page, vals = 16384, {}
    for line in vm_stat_out.splitlines():
        if "page size of" in line:
            for tok in line.split():
                if tok.isdigit():
                    page = int(tok)
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().rstrip(".")
        if v.isdigit():
            vals[k.strip().strip('"')] = int(v)
    wired = vals.get("Pages wired down", 0)
    anon = vals.get("Anonymous pages", 0)
    comp = vals.get("Pages occupied by compressor", 0)
    if not (wired or anon):
        return None
    used = (wired + anon + comp) * page
    return max(0.0, float(total_bytes) - used)


def _self_available_bytes() -> float:
    """RAM this Mac can actually give us *right now*.

    Sharding on total RAM is wrong: the Mac you're sitting at is running a
    browser, an editor and the Lab, so a big share of its memory is already
    spoken for. Weighting by installed RAM hands the *busiest* machine the
    biggest shard, which then swaps and can take the machine down.
    """
    import subprocess
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=4).stdout
        got = parse_available_bytes(out, _self_ram_bytes())
        if got is not None:
            return got
    except Exception:
        pass
    return _self_ram_bytes() * 0.5  # conservative fallback


def _gpu_wired_limit_bytes() -> float:
    """How much memory Metal will keep resident on the GPU.

    This, not system free memory, is what decides whether a shard *runs*. macOS
    caps GPU-wired memory at ``iogpu.wired_limit_mb`` (0 = default, ~75% of RAM).
    Push a shard past it and Metal can no longer hold the weights: throughput
    collapses ~50x while nothing swaps and free memory still looks fine.

    Measured on a 24 GiB M4 (18 GiB limit), Qwen3.5-122B-A10B-2bit::

        holding  7.3 GiB (isolated)    1.21 ms/layer
        holding 17.3 GiB (under limit) 7.62 tok/s end-to-end
        holding 18.5 GiB (over limit) ~65   ms/layer   <- collapse
    """
    import subprocess
    try:
        mb = int(subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"],
                                capture_output=True, text=True,
                                timeout=4).stdout.strip() or 0)
        if mb > 0:
            return float(mb) * 1024 ** 2
    except Exception:
        pass
    return _self_ram_bytes() * 0.75          # macOS default


def _gpu_reserve_bytes() -> float:
    """GPU memory to keep below the wired limit for activations, the KV cache and
    the MoE expert gather. Override with ``OPTIQ_CLUSTER_GPU_RESERVE_GB``."""
    import os
    try:
        return float(os.environ.get("OPTIQ_CLUSTER_GPU_RESERVE_GB", "0.75")) * 1024 ** 3
    except Exception:
        return 0.75 * 1024 ** 3


def _bounds_by_bytes(layer_bytes, layer_caps) -> list[int]:
    """Contiguous layer boundaries chosen from the *actual* size of each layer.

    Splitting by layer COUNT assumes every layer weighs the same. They do not
    (a MoE's later blocks are heavier), so a count-based split hands one rank
    more bytes than its capacity and the load is refused after the fact. Walk
    the real byte sizes and give each rank a share proportional to its capacity.
    """
    n, size = len(layer_bytes), len(layer_caps)
    total_b = float(sum(layer_bytes)) or 1.0
    total_c = float(sum(layer_caps)) or 1.0
    bounds, acc, i = [0], 0.0, 0
    for r in range(size - 1):
        target = total_b * (sum(layer_caps[: r + 1]) / total_c)
        while i < n and acc + layer_bytes[i] / 2.0 < target:
            acc += layer_bytes[i]
            i += 1
        i = min(max(i, bounds[-1] + 1), n - (size - 1 - r))   # >=1 layer each
        bounds.append(i)
    bounds.append(n)
    return bounds


def _headroom_bytes(rank: int = 0) -> float:
    """Memory a rank must keep free to *run* the forward: activations, KV cache,
    MLX's buffer-reuse pool, MoE expert dequant.

    Calibrated against a measured 2-node run of Qwen3.5-122B-A10B-2bit (42.8 GiB
    resident across a 36 + 24 GiB pair), which decoded fine with ~3 GiB spare on
    the smaller node. A larger reserve refuses configurations that demonstrably
    work; a smaller one lets weights exceed free RAM and the machine swaps to
    death. Override with ``OPTIQ_CLUSTER_HEADROOM_GB``; a comma-separated list
    sets it per rank (``"1.2,0.2"`` keeps the machine you work on comfortable and
    runs a dedicated node lean, so the lean node takes more layers)."""
    import os
    raw = os.environ.get("OPTIQ_CLUSTER_HEADROOM_GB", "2")
    try:
        parts = [float(x) for x in str(raw).split(",") if x.strip()]
        val = parts[rank] if rank < len(parts) else parts[-1]
        return val * 1024 ** 3
    except Exception:
        return 2.0 * 1024 ** 3


def _param_bytes(mod) -> int:
    try:
        return int(sum(int(p.nbytes) for _, p in _flat(mod.parameters())))
    except Exception:
        return 0


GIB = 1024 ** 3


def _resolve_model_dir(model_path: str) -> str:
    """Local dir for a repo id / path, without loading any weights."""
    import os
    if os.path.isdir(model_path):
        return model_path
    try:
        from mlx_lm.utils import get_model_path
        p = get_model_path(model_path)
        return str(p[0] if isinstance(p, tuple) else p)
    except Exception:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_path)


def _disk_weight_bytes(model_dir: str) -> int:
    """Total weight bytes straight off disk — no mmap, no allocation. Lets us
    refuse an oversized model *before* ``mlx_lm.load`` touches memory. On a Mac
    whose swap is already full, that load is what gets OS-killed."""
    import glob
    import os
    return int(sum(os.path.getsize(f)
                   for f in glob.glob(os.path.join(model_dir, "*.safetensors"))))


def _config_vocab(model_dir: str) -> int:
    """vocab_size from config.json, so the last rank need not embed a probe token
    (and therefore need not materialize embed_tokens at all)."""
    import json
    import os
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
    except Exception:
        return 0
    for src in (cfg, cfg.get("text_config") or {}, cfg.get("language_config") or {}):
        if isinstance(src, dict) and "vocab_size" in src:
            return int(src["vocab_size"])
    return 0


def _embed_head_bytes(model_dir: str) -> tuple[float, float]:
    """(embed_bytes, head_bytes) estimated from config: packed weight + per-group
    fp16 scales/biases. Rank 0 pays for the embedding, the last rank for the head;
    without this the preflight under-estimates both ends of the ring."""
    import json
    import os
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
    except Exception:
        return 0.0, 0.0
    tc = cfg.get("text_config") or {}
    vocab = cfg.get("vocab_size") or tc.get("vocab_size") or 0
    hidden = cfg.get("hidden_size") or tc.get("hidden_size") or 0
    q = cfg.get("quantization") or {}
    bits, group = q.get("bits", 4), q.get("group_size", 64)
    if not (vocab and hidden):
        return 0.0, 0.0
    n = vocab * hidden
    emb = n * bits / 8 + 2 * (n / max(group, 1)) * 2
    tied = bool(cfg.get("tie_word_embeddings") or tc.get("tie_word_embeddings"))
    return float(emb), 0.0 if tied else float(emb)


def _config_layers(model_dir: str) -> int:
    import json
    import os
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
    except Exception:
        return 0
    for src in (cfg, cfg.get("text_config") or {}, cfg.get("language_config") or {}):
        for k in ("num_hidden_layers", "n_layers", "num_layers"):
            if isinstance(src, dict) and k in src:
                return int(src[k])
    return 0


def _fit_error(model_path, shards, caps, avails, room, size) -> str:
    # `room` is rank 0's reserve; OPTIQ_CLUSTER_HEADROOM_GB may set one per rank,
    # so each row has to report its own or the arithmetic reads as broken.
    over = [i for i in range(size) if shards[i] > caps[i]]
    rows = "\n".join(
        f"  rank {i}: needs {shards[i]/GIB:5.1f} GiB | can spare {caps[i]/GIB:5.1f} GiB "
        f"(free {avails[i]/GIB:.1f} GiB - {_headroom_bytes(i)/GIB:.1f} GiB to run)"
        f"{'   <-- does not fit' if i in over else ''}"
        for i in range(size))
    return (
        f"\n\n{model_path} does not fit the cluster's *currently free* memory.\n"
        f"{rows}\n\n"
        f"Weights {sum(shards)/GIB:.1f} GiB vs {sum(caps)/GIB:.1f} GiB spare across "
        f"{size} node(s).\n"
        f"Fix: free memory on the node(s) above (quit apps; a Mac whose swap is "
        f"full needs a reboot), use a smaller quant, add a Mac, or lower the "
        f"run-reserve with OPTIQ_CLUSTER_HEADROOM_GB=<gb> (risks swapping).\n")


def _flat(tree):
    from mlx.utils import tree_flatten
    return tree_flatten(tree)


class _ShardContext:
    """Loads this rank's layer slice and runs it as one stage of the ring."""

    def __init__(self, model_path: str, verbose: bool = True):
        import optiq  # noqa: F401 — registers custom arches (qwen3_5, etc.)
        from mlx_lm import load

        self.model_path = model_path
        self.world = mx.distributed.init()
        self.rank, self.size = self.world.rank(), self.world.size()
        self.is_first, self.is_last = self.rank == 0, self.rank == self.size - 1

        t0 = time.time()

        # Split by what each Mac can ACTUALLY spare right now, minus the memory
        # needed to *run* the forward. Weighting by installed RAM hands the
        # machine you're working on the biggest shard, which then swaps and can
        # take it down. Gathered so every rank computes identical bounds.
        avail = _self_available_bytes()
        room = _headroom_bytes(self.rank)
        gpu_cap = _gpu_wired_limit_bytes() - _gpu_reserve_bytes()
        # A shard must fit BOTH: system RAM (or the Mac swaps) and Metal's
        # wired limit (or the GPU spills and throughput collapses).
        w = max(5e8, min(avail - room, gpu_cap))
        wa = mx.distributed.all_gather(mx.array([w], dtype=mx.float32))
        av = mx.distributed.all_gather(mx.array([avail], dtype=mx.float32))
        mx.eval(wa, av)
        caps = [float(x) for x in wa.tolist()]
        avails = [float(x) for x in av.tolist()]
        self.capacity, self.avail = caps[self.rank], avail

        # EARLY PREFLIGHT — size the model straight off disk, before mlx_lm.load
        # mmaps anything. On a Mac whose swap is already exhausted, that load is
        # what gets silently OS-killed, so a check placed after it never runs.
        mdir = _resolve_model_dir(model_path)
        total_b, n_cfg = _disk_weight_bytes(mdir), _config_layers(mdir)
        layer_caps = caps
        if total_b and n_cfg:
            emb_b, head_b = _embed_head_bytes(mdir)
            layers_b = max(0.0, total_b - emb_b - head_b)
            # Rank 0 also carries the embedding, the last rank the norm + head.
            # Split LAYERS over what is left after those fixed costs, otherwise
            # the ends of the ring get their full proportional share of layers
            # *plus* a table they alone must hold, and overshoot their capacity.
            fixed = [0.0] * self.size
            fixed[0] += emb_b + head_b
            layer_caps = [max(1e6, caps[i] - fixed[i]) for i in range(self.size)]
            eb = _shard_bounds(n_cfg, self.size, layer_caps)
            est = [layers_b * (eb[i + 1] - eb[i]) / n_cfg + fixed[i]
                   for i in range(self.size)]
            if any(est[i] > caps[i] for i in range(self.size)):
                raise RuntimeError(
                    _fit_error(model_path, est, caps, avails, room, self.size))

        # See optiq.runtime.fast_load: mlx-lm otherwise builds random float
        # weights and quantizes them before load_weights throws them away, a
        # transient every rank pays in full (53 GiB for the 122B).
        from optiq.runtime.fast_load import fast_quantized_load
        with fast_quantized_load():
            model, self.tokenizer = load(model_path, lazy=True)
        self.core, self.head = _dissect(model)
        inner = getattr(model, "language_model", model)
        n = len(self.core.layers)

        # Exact split from the real per-layer byte sizes (metadata only —
        # nothing is materialized yet), honouring each rank's fixed cost.
        lb = [float(_param_bytes(l)) for l in self.core.layers]
        fixed = [0.0] * self.size
        fixed[0] += _param_bytes(self.core.embed_tokens)
        fixed[0] += _param_bytes(self.core.norm)
        if hasattr(inner, "lm_head") and inner.lm_head is not None:
            fixed[0] += _param_bytes(inner.lm_head)
        lcaps = [max(1e6, caps[i] - fixed[i]) for i in range(self.size)]
        b = _bounds_by_bytes(lb, lcaps)
        self.lo, self.hi = b[self.rank], b[self.rank + 1]
        self.my_layers = self.core.layers[self.lo:self.hi]
        self.capacity, self.avail = caps[self.rank], avail

        self.cache_src = next(
            (o for o in (model, getattr(model, "language_model", None), self.core)
             if o is not None and hasattr(o, "make_cache")), None)
        self.dtype = self.core.norm.weight.dtype
        self.d_model = self.core.norm.weight.shape[0]

        # PREFLIGHT: does this rank's shard actually fit in the memory it can
        # spare? Weights are still lazy here — nothing is resident yet — so we
        # can refuse cleanly. Skipping this is what swap-kills a Mac: mx.eval()
        # below is the point of no return.
        # Only the weights this rank will actually materialize (see below):
        # rank 0 embeds, the last rank norms + projects to logits. A rank that
        # does neither must not pay for them.
        lm_head = getattr(inner, "lm_head", None)
        tied = lm_head is None
        shard_b = sum(_param_bytes(l) for l in self.my_layers)
        if self.is_first:
            shard_b += _param_bytes(self.core.embed_tokens)
            shard_b += _param_bytes(self.core.norm)
            if not tied:
                shard_b += _param_bytes(lm_head)
        sb = mx.distributed.all_gather(mx.array([float(shard_b)], dtype=mx.float32))
        mx.eval(sb)
        shards = [float(x) for x in sb.tolist()]
        if any(shards[i] > caps[i] for i in range(self.size)):
            raise RuntimeError(
                _fit_error(model_path, shards, caps, avails, room, self.size))

        # Materialize ONLY what this rank runs. Everything else stays lazy/mmapped
        # and never becomes resident. Rank 0 embeds AND projects to logits: the
        # last rank sends back the final hidden state (d_model floats) instead of
        # logits (vocab floats), so the head runs on the fastest Mac and ~80x less
        # data crosses the wire. Middle/last ranks hold only their layers.
        # Materialize INCREMENTALLY. One mx.eval over the whole shard builds a
        # single huge graph and holds every intermediate at once, so the load
        # peak far exceeds the resident size and the machine swaps (a 43 GiB
        # model took 103s to load off a warm SSD — it was swapping). Evaluating
        # a layer at a time and dropping MLX's buffer-reuse pool between layers
        # keeps the peak at roughly one layer above the shard.
        def _ev(mod):
            ps = [p for _, p in _flat(mod.parameters())]
            if ps:
                mx.eval(ps)

        if self.is_first:
            _ev(self.core.embed_tokens)
            _ev(self.core.norm)
            if not tied:
                _ev(lm_head)
        for i, layer in enumerate(self.my_layers):
            _ev(layer)
            mx.clear_cache()          # release the reuse pool before the next
            if verbose and self.is_first and (i + 1) % 8 == 0:
                print(f"[rank {self.rank}] materialized {i + 1}/{len(self.my_layers)} "
                      f"layers | resident {mx.get_active_memory() / GIB:.1f} GiB",
                      flush=True)
        mx.clear_cache()

        # Logits width, so rank 0 knows the shape to receive. Read from
        # config.json — probing the head would force the last rank to embed a
        # token, materializing an embedding table it otherwise never touches.
        self.vocab = _config_vocab(mdir)
        if not self.vocab:  # fall back: last rank probes, broadcast the width
            v = int(self.head(self.core.norm(
                self.core.embed_tokens(mx.array([[0]])))).shape[-1]) if self.is_last else 0
            va = mx.distributed.all_sum(mx.array([v], dtype=mx.int32))
            mx.eval(va)
            self.vocab = int(va[0])

        self.load_s = time.time() - t0
        self.resident_gb = mx.get_active_memory() / GIB
        self.n_layers = n
        if verbose:
            print(f"[rank {self.rank}/{self.size}] layers {self.lo}-{self.hi - 1} "
                  f"({len(self.my_layers)}/{n}) | loaded {self.load_s:.1f}s | "
                  f"resident {self.resident_gb:.1f} GiB | "
                  f"cap {self.capacity / GIB:.1f} GiB "
                  f"(free {avail / GIB:.1f}, GPU limit "
                  f"{_gpu_wired_limit_bytes() / GIB:.1f})",
                  flush=True)

    def make_caches(self):
        """Fresh per-request caches for THIS rank's layers (correct type per
        layer for hybrid arches: ArraysCache for linear, KVCache for full-attn)."""
        if self.cache_src is not None:
            return self.cache_src.make_cache()[self.lo:self.hi]
        from mlx_lm.models.cache import KVCache
        return [KVCache() for _ in self.my_layers]

    def run_shard(self, h, caches):
        """Run this rank's layers on hidden state ``h``, building the masks from
        ``caches`` (a fa/attention mask + an ssm mask for hybrid arches)."""
        from mlx_lm.models.base import create_attention_mask
        try:
            from mlx_lm.models.base import create_ssm_mask
        except ImportError:
            create_ssm_mask = None
        fa_ref = next((c for l, c in zip(self.my_layers, caches)
                       if not getattr(l, "is_linear", False)), None)
        ssm_ref = next((c for l, c in zip(self.my_layers, caches)
                        if getattr(l, "is_linear", False)), None)
        fa = create_attention_mask(h, fa_ref) if fa_ref is not None else None
        ssm = (create_ssm_mask(h, ssm_ref)
               if (create_ssm_mask is not None and ssm_ref is not None) else None)
        for layer, c in zip(self.my_layers, caches):
            m = ssm if getattr(layer, "is_linear", False) else fa
            h = layer(h, m, c)
        return h


def _bcast(ctx, flag=0, reset=0, seq=0):
    """all-sum trick: rank 0 holds (flag, reset, seq_len), others hold zeros ->
    everyone gets rank 0's values. Keeps the workers in lockstep with rank 0's
    per-forward calls. flag=0 tells the workers to shut down."""
    hdr = (mx.array([flag, reset, seq], dtype=mx.int32) if ctx.is_first
           else mx.zeros((3,), dtype=mx.int32))
    hdr = mx.distributed.all_sum(hdr)
    mx.eval(hdr)
    return int(hdr[0]), int(hdr[1]), int(hdr[2])


def _eval_caches(caches) -> None:
    """Materialize the KV / recurrent caches after a forward.

    Defensive only. ``mx.eval`` on the layer output already forces the cache
    chain (measured: 18 layers of the 122B cost a flat 18.5 ms/step over 64
    sequential steps with no cache eval at all, and 18.7 ms with it). Kept
    because rank 0 hands mlx_lm a ``_ProxyCache`` whose ``state`` is empty, so
    mlx_lm's own ``mx.eval([c.state for c in prompt_cache])`` never touches the
    real caches -- and it costs ~0.2 ms to not have to reason about that again.
    """
    st = []
    for c in caches:
        cs = getattr(c, "state", None)
        if cs is None:
            continue
        st.extend(cs if isinstance(cs, (list, tuple)) else [cs])
    if st:
        mx.eval(st)


class _ProxyCache:
    """Stand-in cache mlx_lm holds on rank 0; the real KV lives on the workers.
    generate_step only needs ``.state`` (eval'd during prefill) and ``.offset``
    (used only with KV quant, which the cluster doesn't enable)."""

    def __init__(self):
        self.offset = 0

    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        pass


class DistributedModel:
    """Makes the ring look like a single mlx_lm model on rank 0.

    ``model(inputs, cache)`` broadcasts a forward to the workers, runs rank 0's
    layers, passes activations down the ring, and returns the last rank's logits.
    A new ``cache`` object (mlx_lm makes one per request) triggers a ring-wide
    cache reset. mlx_lm's own generate loop then samples on rank 0.
    """

    def __init__(self, ctx: _ShardContext):
        self.ctx = ctx
        self._cur_cache = object()   # sentinel so the first call resets
        self._caches = None

    def make_cache(self):
        return [_ProxyCache()]

    def __call__(self, inputs, cache=None, input_embeddings=None):
        ctx = self.ctx
        reset = cache is not self._cur_cache
        self._cur_cache = cache
        emb = input_embeddings if input_embeddings is not None else None
        seq = int((emb if emb is not None else inputs).shape[1])
        _bcast(ctx, 1, int(reset), seq)          # wake workers for this forward
        if reset or self._caches is None:
            self._caches = ctx.make_caches()
        h = emb if emb is not None else ctx.core.embed_tokens(inputs)
        h = ctx.run_shard(h, self._caches)
        _eval_caches(self._caches)
        if ctx.size == 1:
            return ctx.head(ctx.core.norm(h))
        mx.eval(mx.distributed.send(h, 1))
        # The last rank returns the final hidden state; rank 0 (the fastest Mac,
        # and the one holding norm+head) projects it to logits.
        last = mx.distributed.recv((1, 1, ctx.d_model), ctx.dtype, src=ctx.size - 1)
        mx.eval(last)
        logits = ctx.head(ctx.core.norm(last))
        mx.eval(logits)
        return logits


def run_worker(ctx: _ShardContext):
    """Ranks 1..N-1: one forward per rank-0 model call; own persistent caches."""
    caches = None
    while True:
        flag, reset, seq = _bcast(ctx)
        if flag == 0:
            break
        if reset or caches is None:
            caches = ctx.make_caches()
        h = mx.distributed.recv((1, seq, ctx.d_model), ctx.dtype, src=ctx.rank - 1)
        mx.eval(h)
        h = ctx.run_shard(h, caches)
        _eval_caches(caches)
        if not ctx.is_last:
            mx.eval(mx.distributed.send(h, ctx.rank + 1))
        else:
            mx.eval(mx.distributed.send(h[:, -1:, :], 0))


def pipeline_generate(model_path: str, prompt: str, max_tokens: int = 100,
                      verbose: bool = True) -> dict:
    """One-shot generation via mlx_lm's stream_generate through the ring."""
    ctx = _ShardContext(model_path, verbose=verbose)
    if not ctx.is_first:
        run_worker(ctx)
        return {"rank": ctx.rank}

    from mlx_lm import stream_generate
    model = DistributedModel(ctx)
    t0, text, n = time.time(), "", 0
    for resp in stream_generate(model, ctx.tokenizer, prompt, max_tokens=max_tokens):
        text += resp.text
        n += 1
    _bcast(ctx, 0)  # shut the workers down
    dt = time.time() - t0
    if verbose:
        print(f"\n[rank 0] {n} tokens in {dt:.1f}s = {n / max(dt, 1e-9):.2f} tok/s",
              flush=True)
        print(f"[rank 0] output: {text!r}", flush=True)
    return {"text": text, "tokens": n, "tok_per_s": n / max(dt, 1e-9),
            "resident_gb": ctx.resident_gb}


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 \
        else "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    pipeline_generate(model_path, prompt, max_tokens)
