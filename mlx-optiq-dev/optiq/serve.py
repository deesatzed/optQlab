"""OptiQ Serve — OpenAI-compatible HTTP server with OptiQ-aware KV cache.

Wraps ``mlx_lm.server`` and injects KV-cache quantization into the generation
loop. Two modes:

1. **Uniform** (``--kv-bits 4/8``) — every layer uses the same bit-width via
   mlx-lm's ``QuantizedKVCache``. Equivalent to ``mlx_lm.server``'s own
   quantized KV path (which upstream doesn't expose as a flag).

2. **Mixed-precision** (``--kv-config path/to/kv_config.json``) — per-layer
   bit-widths from ``optiq kv-cache`` sensitivity analysis. Protects KV-
   sensitive layers (e.g. layer 0, which is ~56x more sensitive than
   average) at higher precision while aggressively quantizing the rest.
   This is the on-brand OptiQ serving path — only we know which layers
   to preserve.

CLI:
    optiq serve --kv-bits 4                     # uniform quant
    optiq serve --kv-config kv_config.json      # mixed-precision
    optiq serve                                 # fp16 (no quant)

Any flag accepted by ``mlx_lm.server`` is forwarded as-is.
"""

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class _LayerQuantConfig:
    layer_idx: int
    bits: int
    group_size: int


def _load_kv_config(path: str) -> list[_LayerQuantConfig]:
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"kv_config must be a JSON list, got {type(raw).__name__}")
    out = []
    for entry in raw:
        out.append(_LayerQuantConfig(
            layer_idx=int(entry["layer_idx"]),
            bits=int(entry["bits"]),
            group_size=int(entry.get("group_size", 64)),
        ))
    return out


def install_quantized_kv(
    kv_bits: Optional[int],
    kv_group_size: int,
    quantized_kv_start: int,
) -> None:
    """Enable uniform KV-cache quantization.

    Patches ``mlx_lm.server.stream_generate`` to pass kv-quant args to
    mlx-lm's generation loop. Must be called before ``mlx_lm.server.main()``.

    Installs ``patch_rotating_to_quantized`` first. install_mixed_kv (the
    --kv-config path) already did; this one did not, so `optiq serve --kv-bits 4
    --no-fused-kv` on a sliding-window model hit stock mlx-lm's
    ``RotatingKVCache.to_quantized`` -> NotImplementedError. CLAUDE.md states the
    patch as universal ("auto-installed by optiq serve"); the code held it for one
    flag.
    """
    from .runtime.kv import patch_rotating_to_quantized
    patch_rotating_to_quantized()
    if kv_bits is None:
        return

    # Without one of these the flag is a no-op: the server's batch path does
    # not quantize the cache on its own. Prefer teaching it to (keeps
    # batching); fall back to disabling batching.
    from .runtime.kv.batch import install_batch_kv_quant
    if not install_batch_kv_quant(default=(kv_bits, kv_group_size)):
        force_sequential_for_kv_quant("--kv-bits")

    import mlx_lm.server as server_mod
    from mlx_lm.generate import stream_generate as original_stream_generate

    def patched_stream_generate(*args, **kwargs):
        kwargs.setdefault("kv_bits", kv_bits)
        kwargs.setdefault("kv_group_size", kv_group_size)
        kwargs.setdefault("quantized_kv_start", quantized_kv_start)
        yield from original_stream_generate(*args, **kwargs)

    server_mod.stream_generate = patched_stream_generate


def force_sequential_for_kv_quant(reason: str = "KV quantization") -> bool:
    """Route generation through mlx-lm's sequential path so KV quant applies.

    mlx-lm's server prefers ``BatchGenerator``, and ``BatchGenerator`` never
    quantizes the KV cache: it does not call ``maybe_quantize_kv_cache``, does
    not mention ``kv_bits``, and ``BatchKVCache`` has no ``to_quantized`` at
    all. Both of OptiQ's KV hooks (this module's ``stream_generate`` /
    ``maybe_quantize_kv_cache`` patches) sit on the *sequential* path, which
    the server only takes when ``_is_batchable`` is False.

    So with a batchable model, ``optiq serve --kv-config`` / ``--kv-bits``
    loaded the config, logged the summary, and then served an ordinary fp16
    cache -- a silent no-op. Confirmed at runtime, not by reading: a probe on
    the served 26B with ``--kv-config`` reported ``cache=BatchKVCache
    bits=None``.

    Forcing the sequential path costs cross-request batching, which a local
    single-user setup was not using anyway, and is strictly better than
    honoring the flag in name only. Callers that would rather have batching
    should not pass a KV-quant flag.
    """
    try:
        import mlx_lm.server as server_mod
    except Exception:
        return False
    if getattr(server_mod, "_optiq_forced_sequential", False):
        return True
    # ResponseGenerator owns _is_batchable today; scan for the owner rather
    # than hardcoding, so an upstream move degrades to a no-op we can see
    # (the return value) instead of patching the wrong class.
    target = None
    for obj in vars(server_mod).values():
        if isinstance(obj, type) and "_is_batchable" in vars(obj):
            target = obj
            break
    if target is None:
        return False

    def _is_batchable(self, args):
        return False

    target._is_batchable = _is_batchable
    server_mod._optiq_forced_sequential = True
    server_mod._optiq_sequential_reason = reason
    return True


def install_mixed_kv(
    kv_configs: list[_LayerQuantConfig],
    quantized_kv_start: int,
) -> None:
    """Enable per-layer mixed-precision KV-cache quantization.

    Two patches:
      1. Replace ``mlx_lm.generate.maybe_quantize_kv_cache`` with a version
         that looks up per-layer bits/group_size from ``kv_configs``.
      2. Inject a non-None ``kv_bits`` into ``stream_generate`` so the
         quantize-cache hook actually runs (upstream skips when kv_bits is
         None; our patched function ignores the bits arg and uses the map).

    Must be called before ``mlx_lm.server.main()``. Idempotent.
    """
    if not kv_configs:
        return

    bit_map = {c.layer_idx: (c.bits, c.group_size) for c in kv_configs}

    # Prefer keeping batching. The batch path has no quantization of its own,
    # so optiq installs cache classes that quantize on write and can merge;
    # only if that hook point is gone do we fall back to disabling batching,
    # which honors the flag at the cost of concurrent requests.
    from .runtime.kv.batch import install_batch_kv_quant
    if not install_batch_kv_quant(bit_map):
        force_sequential_for_kv_quant("--kv-config")

    import gc
    import mlx.core as mx
    # NOT ``import mlx_lm.generate as gen_mod``: mlx_lm/__init__.py re-exports
    # the ``generate`` *function*, which shadows the submodule of the same
    # name, so that form binds the function and the assignment below would set
    # an attribute on a function object instead of patching the module. Silent
    # -- the hook simply never installs. (mlx_lm.convert is shadowed the same
    # way; mlx_lm.server and mlx_lm.utils are not.)
    import sys as _sys

    import mlx_lm.generate  # noqa: F401  — ensure it is in sys.modules
    gen_mod = _sys.modules["mlx_lm.generate"]
    from mlx_lm.models.cache import QuantizedKVCache, RotatingKVCache
    from .runtime.kv.rotating import quantize_cache_layer

    # Install our RotatingKVCache.to_quantized override so sliding-window
    # layers can be quantized too (Gemma 3/4, Cohere R2, OLMo 3, Phi-3/4,
    # EXAONE 4, Ministral 3, etc.). Idempotent — safe to call repeatedly.
    from optiq.runtime.kv import (
        RotatingQuantizedKVCache,
        patch_rotating_to_quantized,
    )
    patch_rotating_to_quantized()

    _converted: set = set()
    _reported: list = []

    def patched_maybe_quantize(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
        # Per-layer mixed-precision conversion with streaming behavior:
        # quantize one layer's K and V at a time, force eval, drop fp16
        # references, clear the buffer pool. Same shape of fix as
        # optiq.runtime.streaming_kv_quant but with per-layer bit lookups.
        for idx, c in enumerate(prompt_cache):
            if idx not in bit_map:
                continue
            if not hasattr(c, "to_quantized"):
                continue
            if c.offset < quantized_kv_start:
                continue
            if c.keys is None:
                continue
            if isinstance(c, (QuantizedKVCache, RotatingQuantizedKVCache)):
                continue  # already quantized
            bits, group_size = bit_map[idx]

            new_c = quantize_cache_layer(c, bits=bits, group_size=group_size)
            if new_c is c:
                continue

            prompt_cache[idx] = new_c
            _converted.add(idx)
            del c, new_c
            gc.collect()
            mx.clear_cache()

        # Report once, on the first request that actually converts something.
        # This feature was inert for two releases precisely because there was
        # no positive signal: the startup banner announced the bit-widths from
        # the config file, which says nothing about whether the cache was ever
        # touched. Announce what happened, not what was requested.
        if _converted and not _reported:
            _reported.append(True)
            skipped = sorted(set(bit_map) - _converted)
            import logging
            logging.info(
                "[optiq.serve] KV cache quantized: %d/%d configured layers "
                "converted%s", len(_converted), len(bit_map),
                f" ({len(skipped)} not eligible: {skipped[:8]})" if skipped else "")

    gen_mod.maybe_quantize_kv_cache = patched_maybe_quantize

    import mlx_lm.server as server_mod
    from mlx_lm.generate import stream_generate as original_stream_generate

    # Inject a non-None kv_bits (any value — our patch ignores it) so the
    # upstream early-return on kv_bits is None doesn't skip our hook.
    trigger_bits = max(c.bits for c in kv_configs)

    def patched_stream_generate(*args, **kwargs):
        kwargs.setdefault("kv_bits", trigger_bits)
        kwargs.setdefault("kv_group_size", 64)
        kwargs.setdefault("quantized_kv_start", quantized_kv_start)
        yield from original_stream_generate(*args, **kwargs)

    server_mod.stream_generate = patched_stream_generate


def summarize_kv_config(kv_configs: list[_LayerQuantConfig]) -> str:
    """One-line summary for startup logging."""
    if not kv_configs:
        return "fp16"
    bits_counts: dict[int, int] = {}
    for c in kv_configs:
        bits_counts[c.bits] = bits_counts.get(c.bits, 0) + 1
    parts = [f"{n}@{b}-bit" for b, n in sorted(bits_counts.items())]
    avg = sum(c.bits for c in kv_configs) / len(kv_configs)
    return f"{len(kv_configs)} layers: {', '.join(parts)} (avg {avg:.2f} bits)"


# ---------------------------------------------------------------------------
# Logprobs shim, shared by every OptiQ stream_generate replacement.
#
# mlx-lm's ``_serve_single`` unconditionally evaluates
# ``gen.logprobs[gen.token].item()`` on each yielded ``GenerationResponse``.
# OptiQ's fast streaming loops (MTP, separate-drafter spec, vision) don't
# surface a per-token logprob vector, so yielding ``logprobs=None`` crashes
# the server with ``TypeError: 'NoneType' object is not subscriptable`` on
# any request (tool-calling requests always hit this path). This stand-in
# returns 0.0 for any index so the access is safe. The ``logprobs`` /
# ``top_logprobs`` fields in the response are then placeholders; the default
# serving path never requests real logprobs.
# ---------------------------------------------------------------------------


class _NullLogprobs:
    def __getitem__(self, idx):
        import mlx.core as mx

        return mx.array(0.0)


_NULL_LOGPROBS = _NullLogprobs()


# ---------------------------------------------------------------------------
# MTP speculative decoding (off by default; opt-in via ``optiq serve --mtp``).
# ---------------------------------------------------------------------------

_MTP_ENGINE = None


def install_mtp_speculation(model_path: str, depth: int = 2) -> None:
    """Route generation through ``OptiqEngine`` so the model's in-checkpoint
    MTP head drives speculative decoding.

    Patches ``mlx_lm.server.stream_generate`` to yield tokens produced by
    our engine's ``generate_stream`` (true per-token streaming — no
    synthetic burst). The engine is loaded lazily on first request and
    shared across all subsequent requests.

    Per-token ``GenerationResponse`` tracks ``from_draft`` (True when a
    token was sampled by the MTP head and accepted by verify, the
    speculation win), and the final event carries ``finish_reason='stop'``.
    """
    if depth <= 0:
        return

    import mlx_lm.server as server_mod
    from mlx_lm.generate import GenerationResponse

    def _get_engine(model=None, tokenizer=None):
        global _MTP_ENGINE
        if _MTP_ENGINE is None:
            from optiq.runtime.engine import OptiqEngine
            if model is not None and tokenizer is not None:
                # Reuse mlx_lm.server's already-loaded model — injects MTP
                # in place, no second copy of the weights in unified memory.
                print(f"[optiq.serve] attaching MTP engine to loaded model "
                      f"({model_path})...")
                _MTP_ENGINE = OptiqEngine.from_loaded(model, tokenizer, model_path)
            else:
                # Standalone path (no preloaded model handed in) — falls
                # back to the duplicated-load behavior. Callers should
                # prefer the patched stream_generate path below.
                print(f"[optiq.serve] loading MTP engine from {model_path}...")
                _MTP_ENGINE = OptiqEngine(model_path)
            if not _MTP_ENGINE.has_mtp:
                raise RuntimeError(
                    f"--mtp requested but {model_path!r} has no MTP head. "
                    f"Re-convert with v0.1.0+ optiq convert (which preserves "
                    f"mtp.* tensors)."
                )
            print(f"[optiq.serve] MTP engine ready (depth={depth}).")
        return _MTP_ENGINE

    def patched_stream_generate(model, tokenizer, prompt, max_tokens=256, **kwargs):
        engine = _get_engine(model=model, tokenizer=tokenizer)
        # mlx-lm passes prompt as either str, list[int], or mx.array.
        if isinstance(prompt, str):
            prompt_text = prompt
        else:
            ids = prompt.tolist() if hasattr(prompt, "tolist") else list(prompt)
            # mlx-lm already applied the chat template; decode preserves
            # special tokens so the model sees the same prompt back.
            prompt_text = tokenizer.decode(ids, skip_special_tokens=False)

        temperature = float(kwargs.get("temp", kwargs.get("temperature", 0.0)))
        # Forward truncation params so the MTP draft + verify both use
        # the same effective distribution as the production sampler.
        # Without this, rejection sampling sees a diffuse temp-only
        # distribution → near-zero acceptance vs the truncated one a
        # user actually decodes from.
        top_p = float(kwargs.get("top_p", 0.0))
        top_k = int(kwargs.get("top_k", 0) or 0)
        min_p = float(kwargs.get("min_p", 0.0))

        for ev in engine.generate_stream(
            prompt=prompt_text,
            max_tokens=max_tokens,
            depth=depth,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            chat_template=False,  # already applied upstream
        ):
            yield GenerationResponse(
                text=ev["text"],
                token=ev["token"],
                logprobs=_NULL_LOGPROBS,
                from_draft=ev["from_draft"],
                prompt_tokens=ev["prompt_tokens"],
                prompt_tps=ev["prompt_tokens"] / max(ev["prefill_time_s"], 1e-6),
                generation_tokens=ev["generation_index"] + 1,
                generation_tps=ev["decode_tps"],
                peak_memory=0.0,
                finish_reason="stop" if ev["done"] else None,
            )

    server_mod.stream_generate = patched_stream_generate


def install_eos_terminated_tool_calls() -> None:
    """Recover tool calls from families that end a call with EOS, not a marker.

    mlx-lm's server tracks reasoning/tool/normal with a state machine over the
    generated tokens, and flushes the accumulated tool text with::

        if prev_state == "tool" and tool_text:
            tool_calls.append(tool_text)

    Qwen-style formats reach that check because ``</tool_call>`` transitions the
    machine back to ``normal``. Mistral and Devstral have **no** tool-call end
    marker (``tokenizer.tool_call_end`` is ``""``) — the call is terminated by
    EOS, which the machine treats as a stop and reports as state ``None``. So
    ``prev_state`` is ``None`` at the end, the flush is skipped, and a complete,
    parseable tool call is silently dropped: the client gets a message with
    neither ``content`` nor ``tool_calls``, ``finish_reason: "stop"``, and a
    non-zero completion-token count. Agents then hang waiting for a call that
    was generated and thrown away.

    Reproduced on ``Devstral-Small-2-24B``: the model emits
    ``[TOOL_CALLS]read_file[ARGS]{"path": "AGENTS.md"}</s>`` and
    ``tokenizer.tool_parser`` parses it correctly — only the plumbing loses it.

    The fix points the ``tool``-state EOS edge at ``normal`` instead of ``None``,
    so the existing flush runs. Scoped to tokenizers with an empty end marker,
    so families that already work are untouched. Generation still stops: the
    server takes ``finish_reason`` from ``stream_generate`` regardless, and then
    promotes it to ``tool_calls``.
    """
    import mlx_lm.server as server_mod

    RG = getattr(server_mod, "ResponseGenerator", None)
    if RG is None or getattr(RG, "_optiq_eos_tool_fix", False):
        return

    _orig_make_sm = RG._make_state_machine

    def patched_make_state_machine(
        self, model_key, tokenizer, stop_words, initial_state="normal",
    ):
        sm, sequences = _orig_make_sm(
            self, model_key, tokenizer, stop_words, initial_state)
        if not getattr(tokenizer, "has_tool_calling", False):
            return sm, sequences
        if getattr(tokenizer, "tool_call_end_tokens", None):
            return sm, sequences          # has a real end marker; already fine
        states = getattr(sm, "_states", None)
        if not isinstance(states, dict) or "tool" not in states:
            return sm, sequences
        trie, dst = states["tool"]
        if any(d is None for d in dst):
            # Idempotent: a cached machine has already been rewritten.
            states["tool"] = (
                trie, tuple("normal" if d is None else d for d in dst),
            )
        return sm, sequences

    RG._make_state_machine = patched_make_state_machine
    RG._optiq_eos_tool_fix = True


def install_tool_argument_normalization() -> None:
    """Make tool-call arguments a mapping by the time the chat template runs.

    Gemma-4's canonical template (Google, 2026-07-09) raises on
    ``tool_calls[].function.arguments`` that is neither a mapping nor ``None``.
    mlx-lm converts the OpenAI wire format's JSON string for us, but leaves two
    holes: ``json.loads`` throws ``TypeError`` on a client-sent object, and its
    ``if args:`` guard skips ``""``, so a no-argument tool call reaches the
    template as a string. See ``optiq.runtime.tool_args``.

    Pre-normalizing to a JSON object string (rather than straight to a mapping)
    keeps mlx-lm's own conversion the one that runs, so this composes with any
    future change there. The post-pass is belt-and-braces for the case where
    mlx-lm stops converting at all.
    """
    import mlx_lm.server as server_mod

    from .runtime.tool_args import (
        coerce_arguments_to_json_string, normalize_tool_arguments,
    )

    if getattr(server_mod, "_optiq_tool_args_normalized", False):
        return

    from .runtime.message_shape import (
        fold_user_after_tool, normalize_role_alternation,
    )

    current = server_mod.process_message_content

    def patched_process_message_content(messages):
        coerce_arguments_to_json_string(messages)
        result = current(messages)
        normalize_tool_arguments(messages)
        # Strict-alternation templates (Mistral, Devstral) reject two same-role
        # turns in a row, and a user turn straight after tool results. Both are
        # shapes agent harnesses produce and permissive templates tolerate.
        # Rewrite in place so mlx-lm's caller sees the result.
        shaped = normalize_role_alternation(fold_user_after_tool(messages))
        if len(shaped) != len(messages):
            messages[:] = shaped
        return result

    server_mod.process_message_content = patched_process_message_content
    server_mod._optiq_tool_args_normalized = True


# ---------------------------------------------------------------------------
# Vision (multimodal) serving.
#
# mlx-lm's server is text-only: ``process_message_content`` raises on any
# non-text content part, and the chat-template path drops images. OptiQ's
# vision support lives in ``OptiqEngine`` (vendored vision tower + sidecar +
# merge). This installer makes the server image-aware:
#
#   1. ``ResponseGenerator`` is patched at four points: ``generate`` deep-copies
#      the original messages (image parts intact) onto the per-request
#      ``generation_args``, ``_tokenize`` reduces ``content`` to text so the
#      stock template path is undisturbed, ``_is_batchable`` forces image
#      requests off the batch path, and ``_serve_single`` hands the images to
#      ``vlm_stream_generate`` through a global for the duration of one call.
#   2. ``stream_generate`` is wrapped: when a request carried images, route it
#      through ``OptiqEngine.generate_stream(messages=...)`` (vendored encode +
#      merge + mlx-lm decode); otherwise delegate to whatever stream_generate
#      was already installed (stock, MTP, or KV-quant).
#
# Single-user local serving is the design point (Lab + ``optiq serve``); the
# pending-images handoff is a process global, fine for serialized local use.
# ---------------------------------------------------------------------------

_VLM_ENGINE = None
_PENDING_VLM: dict = {}


def _message_has_image(messages) -> bool:
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                    "image_url", "image", "input_image"
                ):
                    return True
    return False


def install_model_field_policy(allow_switch: bool = False) -> None:
    """Decide what a request's ``model`` field means for ``optiq serve``.

    ``optiq serve --model X`` hosts exactly ONE model, but the OpenAI/Anthropic
    wire protocol requires a ``model`` field on every request. Stock mlx-lm reads
    that field as "which repo to load", so any value that isn't the served path
    (a basename, a friendly alias, or Claude Code's default ``claude-…``) makes
    it try to DOWNLOAD that id from the Hub — a 404, or a multi-second hang.

    Default (single-model): the ``model`` field is a **label** — every request is
    served by ``--model X``, whatever the client sends. This fixes the basename /
    alias / wrong-default 404 and matches how the Lab already serves (one model
    per process, switched by restarting the server, not by this field).

    ``allow_switch=True``: keep stock mlx-lm's per-request model switching (the
    field selects among cached models; unknown ids still load/download). Opt in
    only when you actually want one running server to hot-swap between quants.

    Adapter/draft fields are untouched either way, so per-request LoRA still
    works. Idempotent."""
    if allow_switch:
        return
    import mlx_lm.server as server_mod

    MP = server_mod.ModelProvider
    if getattr(MP, "_optiq_model_pin", False):
        return
    orig_load = MP.load

    def patched_load(self, model_path, adapter_path=None, draft_model_path=None):
        served = self._model_map.get("default_model")
        if served is not None and model_path not in ("default_model", served):
            model_path = "default_model"  # single-model: any label -> served
        return orig_load(self, model_path, adapter_path, draft_model_path)

    MP.load = patched_load
    MP._optiq_model_pin = True


def install_vision_serving(model_path: str) -> None:
    """Enable image+text requests on ``optiq serve`` / Lab for a model that
    ships an ``optiq_vision`` sidecar. No-op-safe to call for text-only models
    (the image path only triggers when a request actually carries an image).

    Robustness note: images are tagged onto the per-request ``args`` object in
    ``_tokenize`` (so they travel with the request through mlx-lm's queue, never
    a shared slot), image requests are forced off the batch path, and the global
    handoff to ``stream_generate`` is set+consumed entirely inside the
    synchronous ``_serve_single`` call on the single generation thread — so
    concurrent requests can't cross images."""
    import copy

    import mlx_lm.server as server_mod
    from mlx_lm.generate import GenerationResponse

    current_stream_generate = server_mod.stream_generate
    RG = getattr(server_mod, "ResponseGenerator", None)

    def _get_engine(model=None, tokenizer=None):
        global _VLM_ENGINE
        if _MTP_ENGINE is not None:
            return _MTP_ENGINE  # share the MTP-attached engine if present
        if _VLM_ENGINE is None:
            from optiq.runtime.engine import OptiqEngine
            if model is not None and tokenizer is not None:
                _VLM_ENGINE = OptiqEngine.from_loaded(model, tokenizer, model_path)
            else:
                _VLM_ENGINE = OptiqEngine(model_path)
            print(f"[optiq.serve] vision engine ready ({model_path}).")
        return _VLM_ENGINE

    def vlm_stream_generate(model, tokenizer, prompt, max_tokens=256, **kwargs):
        pending = _PENDING_VLM.get("messages")
        if not pending:
            yield from current_stream_generate(
                model, tokenizer, prompt, max_tokens, **kwargs
            )
            return
        engine = _get_engine(model=model, tokenizer=tokenizer)
        temperature = float(kwargs.get("temp", kwargs.get("temperature", 0.0)))
        for ev in engine.generate_stream(
            prompt="",
            messages=pending,  # OptiqEngine's frontend handles image_url parts
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=float(kwargs.get("top_p", 0.0)),
            top_k=int(kwargs.get("top_k", 0) or 0),
            min_p=float(kwargs.get("min_p", 0.0)),
        ):
            yield GenerationResponse(
                text=ev["text"],
                token=ev["token"],
                logprobs=_NULL_LOGPROBS,
                from_draft=ev.get("from_draft", False),
                prompt_tokens=ev["prompt_tokens"],
                prompt_tps=ev["prompt_tokens"] / max(ev["prefill_time_s"], 1e-6),
                generation_tokens=ev["generation_index"] + 1,
                generation_tps=ev["decode_tps"],
                peak_memory=0.0,
                finish_reason="stop" if ev["done"] else None,
            )

    server_mod.stream_generate = vlm_stream_generate

    if RG is None:
        return

    # 1) Capture images per-request at submission time — BEFORE the request is
    #    queued (and thus before ``_is_batchable`` runs). Tag the original
    #    messages onto the per-request ``generation_args`` so they travel with
    #    the request, never a shared slot.
    _orig_generate = RG.generate

    def _patched_generate(self, request, generation_args, progress_callback=None):
        msgs = getattr(request, "messages", None)
        if msgs and _message_has_image(msgs):
            generation_args._optiq_images = copy.deepcopy(msgs)
        return _orig_generate(self, request, generation_args, progress_callback)

    RG.generate = _patched_generate

    # 2) Strip image parts before mlx-lm's text-only template path runs (so
    #    ``process_message_content`` / ``apply_chat_template`` are undisturbed).
    _orig_tokenize = RG._tokenize

    def _patched_tokenize(self, tokenizer, request, args):
        if getattr(args, "_optiq_images", None):
            for message in getattr(request, "messages", []) or []:
                content = message.get("content")
                if isinstance(content, list):
                    message["content"] = "".join(
                        frag.get("text", "")
                        for frag in content
                        if isinstance(frag, dict) and frag.get("type") == "text"
                    )
        return _orig_tokenize(self, tokenizer, request, args)

    RG._tokenize = _patched_tokenize

    # 3) Force image requests off the batch path (batch bypasses stream_generate).
    _orig_is_batchable = RG._is_batchable

    def _patched_is_batchable(self, args):
        if getattr(args, "_optiq_images", None):
            return False
        return _orig_is_batchable(self, args)

    RG._is_batchable = _patched_is_batchable

    # 4) Hand the images to ``vlm_stream_generate`` race-free: set + clear the
    #    global entirely inside the synchronous ``_serve_single`` call (single
    #    generation thread, no other request interleaves).
    _orig_serve_single = RG._serve_single

    def _patched_serve_single(self, request):
        _rqueue, _req, args = request
        imgs = getattr(args, "_optiq_images", None)
        if imgs:
            _PENDING_VLM["messages"] = imgs
        try:
            return _orig_serve_single(self, request)
        finally:
            _PENDING_VLM.pop("messages", None)

    RG._serve_single = _patched_serve_single


# ---------------------------------------------------------------------------
# Gemma-4 ``-assistant`` drafter speculative decoding.
#
# Sister-path to ``install_mtp_speculation`` for the model family that ships
# a separate small drafter instead of a bundled MTP head. Same outer loop
# pattern: lazy-load the drafter on first request, patch
# ``mlx_lm.server.stream_generate`` to route through ``spec_generate``.
# ---------------------------------------------------------------------------

_SPEC_DRAFTER = None


def install_assistant_drafter(target_model_path: str, drafter_id: str) -> None:
    """Route generation through ``optiq.runtime.spec.spec_generate`` so a
    separate ``-assistant`` drafter (Gemma-4 family) drives speculative
    decoding against the already-loaded target.

    Patches ``mlx_lm.server.stream_generate`` to yield tokens produced by
    the spec loop. The drafter is loaded lazily on first request and shared
    across all subsequent requests. γ=1 greedy only — the spec runtime
    enforces this and raises if a caller asks for γ>1 or temperature>0.

    Per-token ``GenerationResponse`` carries ``from_draft=True`` when a
    token came from an accepted draft.
    """
    import mlx_lm.server as server_mod
    from mlx_lm.generate import GenerationResponse

    def _get_drafter():
        global _SPEC_DRAFTER
        if _SPEC_DRAFTER is None:
            from optiq.runtime.spec import GemmaAssistantDrafter
            print(f"[optiq.serve] loading -assistant drafter {drafter_id}...")
            _SPEC_DRAFTER = GemmaAssistantDrafter.from_pretrained(drafter_id)
            print(f"[optiq.serve] drafter ready (target={target_model_path}).")
        return _SPEC_DRAFTER

    def patched_stream_generate(model, tokenizer, prompt, max_tokens=256, **kwargs):
        from optiq.runtime.spec import spec_generate, SpecConfig
        import time

        drafter = _get_drafter()

        if isinstance(prompt, str):
            prompt_text = prompt
        else:
            ids = prompt.tolist() if hasattr(prompt, "tolist") else list(prompt)
            prompt_text = tokenizer.decode(ids, skip_special_tokens=False)

        cfg = SpecConfig(
            gamma=1,
            max_tokens=max_tokens,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            accept_temp=0.0,
        )

        t0 = time.time()
        n_tokens = 0
        for ev in spec_generate(model, drafter, tokenizer, prompt_text, cfg):
            if ev.kind == "token":
                n_tokens += 1
                elapsed = max(time.time() - t0, 1e-6)
                yield GenerationResponse(
                    text=ev.text,
                    token=ev.token_id,
                    logprobs=_NULL_LOGPROBS,
                    from_draft=ev.from_draft,
                    prompt_tokens=0,
                    prompt_tps=0.0,
                    generation_tokens=n_tokens,
                    generation_tps=n_tokens / elapsed,
                    peak_memory=0.0,
                    finish_reason=None,
                )
            elif ev.kind == "done":
                elapsed = max(time.time() - t0, 1e-6)
                yield GenerationResponse(
                    text="",
                    token=0,
                    logprobs=_NULL_LOGPROBS,
                    from_draft=False,
                    prompt_tokens=0,
                    prompt_tps=0.0,
                    generation_tokens=n_tokens,
                    generation_tps=n_tokens / elapsed,
                    peak_memory=0.0,
                    finish_reason="stop",
                )

    server_mod.stream_generate = patched_stream_generate


# ---------------------------------------------------------------------------
# Multi-adapter mounted-LoRA serving
# ---------------------------------------------------------------------------
# A request that sets ``"adapters": "<name>"`` in its body picks one of the
# pre-mounted adapters for the duration of that request. Switching is
# instant: the model isn't reloaded; only the ContextVar that
# ``MountedLoRALinear`` reads in its forward pass flips.
#
# Compared to mlx-lm's stock ``adapter_path`` (one adapter at boot, swap by
# full model reload), this path keeps N adapters resident on a single base
# at ~30 MB/adapter and switches per-request in microseconds.


_MOUNTED_ADAPTERS: list[tuple[str, str]] = []  # [(name, resolved_path), ...]


def install_multi_adapter(adapter_specs: list[tuple[str, str]]) -> None:
    """Pre-register multiple LoRA adapters for per-request switching.

    Each spec is a ``(name, adapter_dir)`` pair. After ``mlx_lm.server``
    loads the base model, every adapter in ``adapter_specs`` is mounted
    via OptiQ's ``MountedLoRALinear``. Requests select an adapter by
    setting ``"adapters": "<name>"`` in the body; omitting that field
    runs the base model with no adapter active.

    Also adds a ``GET /v1/adapters`` endpoint that lists registered
    names + per-layer mount counts for inspection.

    Patches:
      * ``ModelProvider._load``  — mount adapters after base model load.
      * ``APIHandler.do_POST``   — set/reset the active-adapter ContextVar
        around each request based on the ``adapters`` body field.
      * ``APIHandler.do_GET``    — serve ``/v1/adapters``.

    Idempotent: calling twice replaces the registered specs without
    re-patching the server.
    """
    import mlx_lm.server as server_mod

    global _MOUNTED_ADAPTERS
    _MOUNTED_ADAPTERS = list(adapter_specs)

    # ---- 1) Patch ModelProvider._load to mount adapters after base load ----
    if not getattr(server_mod, "_optiq_multi_adapter_load_patched", False):
        orig_load = server_mod.ModelProvider._load

        def patched_load(self, model_path, adapter_path=None, draft_model_path=None):
            # Always load the base model WITHOUT the legacy single-adapter
            # path — we mount adapters separately so we keep one base
            # resident and gate by ContextVar.
            result = orig_load(self, model_path, adapter_path=None,
                               draft_model_path=draft_model_path)
            from .adapters.mount import (
                prepare_model_for_mounted_lora,
                mount_adapter_on_model,
                list_mounted_adapters,
            )
            try:
                prepare_model_for_mounted_lora(self.model)
                for name, path in _MOUNTED_ADAPTERS:
                    n = mount_adapter_on_model(self.model, name, _PathFromStr(path))
                    print(f"  [optiq.adapters] mounted {name!r} on {n} layers "
                          f"(from {path})")
                mounted = list_mounted_adapters(self.model)
                if mounted:
                    summary = ", ".join(f"{k}({v})" for k, v in mounted.items())
                    print(f"  [optiq.adapters] active mounts: {summary}")
            except Exception as exc:
                print(f"  [optiq.adapters] mounting skipped: {exc}")
            return result

        server_mod.ModelProvider._load = patched_load
        server_mod._optiq_multi_adapter_load_patched = True

    # ---- 2) Patch APIHandler so requests can pick an adapter ---------------
    if not getattr(server_mod, "_optiq_multi_adapter_handler_patched", False):
        orig_do_POST = server_mod.APIHandler.do_POST
        orig_do_GET = getattr(server_mod.APIHandler, "do_GET", None)
        from .adapters.mount import (
            serve_adapter_active, list_mounted_adapters,
        )

        def patched_do_POST(self):
            # Peek at the adapter selector before delegating. We parse the
            # body twice (once here, once in the original handler) but it's
            # small JSON so the cost is trivial.
            adapter_name = _peek_adapter_field(self)
            # Sentinel values that mean "base, no adapter active". Lets a
            # caller opt out of the single-adapter auto-activation below
            # without unmounting. Useful for A/B comparing adapter vs
            # base from the same served process (e.g. when building DPO
            # preference data from with-adapter vs no-adapter samples).
            _BASE_SENTINELS = {"", "none", "base", "off"}
            if (isinstance(adapter_name, str)
                    and adapter_name.lower() in _BASE_SENTINELS):
                adapter_name = None
                _explicit_base = True
            else:
                _explicit_base = False
            # Single-adapter mode: if exactly one adapter is registered
            # and the client didn't request a specific one (or explicitly
            # opt out via a sentinel), default to the only available
            # adapter. This preserves the classic ``optiq serve --adapter
            # X`` UX where the adapter applies to every request without
            # the client setting a body field.
            if (adapter_name is None
                    and not _explicit_base
                    and len(_MOUNTED_ADAPTERS) == 1):
                adapter_name = _MOUNTED_ADAPTERS[0][0]
            if adapter_name:
                # Stacking syntax "sft+dpo" or "sft,dpo" applies
                # multiple registered adapters simultaneously. Validate
                # that every listed name is registered before letting
                # the request through to the generation thread.
                if "+" in adapter_name or "," in adapter_name:
                    sep = "+" if "+" in adapter_name else ","
                    parts = [s.strip() for s in adapter_name.split(sep)
                             if s.strip()]
                else:
                    parts = [adapter_name]
                registered = {n for n, _ in _MOUNTED_ADAPTERS}
                missing = [p for p in parts if p not in registered]
                if missing:
                    self._set_completion_headers(400)
                    self.end_headers()
                    self.wfile.write(
                        ('{"error": "unknown adapter(s): ' +
                         ", ".join(missing) + '. Registered: ' +
                         ", ".join(sorted(registered)) +
                         '. Use \\"base\\" (or omit the field) to '
                         'disable activation, or \\"a+b\\" to stack '
                         'multiple."}'
                        ).encode()
                    )
                    return
            # ``serve_adapter_active`` sets a module-level pin (with a lock
            # for serialization) that ``MountedLoRALinear.__call__`` reads.
            # Necessary because mlx_lm.server runs the model forward in a
            # dedicated generation thread distinct from this handler
            # thread; a per-context ContextVar set here would be invisible
            # to that thread. Passing ``None`` makes the context manager
            # a no-op so base-model requests don't take the lock.
            with serve_adapter_active(adapter_name):
                return orig_do_POST(self)

        def patched_do_GET(self):
            if self.path == "/v1/adapters":
                provider = self.response_generator.model_provider
                model = getattr(provider, "model", None)
                mounted: dict[str, int] = {}
                if model is not None:
                    try:
                        mounted = list_mounted_adapters(model)
                    except Exception:
                        mounted = {}
                payload = {
                    "object": "list",
                    "data": [
                        {
                            "id": name,
                            "path": path,
                            "mounted_layers": mounted.get(name, 0),
                        }
                        for name, path in _MOUNTED_ADAPTERS
                    ],
                }
                self._set_completion_headers(200)
                self.end_headers()
                import json as _json
                self.wfile.write(_json.dumps(payload).encode())
                return
            if orig_do_GET is not None:
                return orig_do_GET(self)
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

        server_mod.APIHandler.do_POST = patched_do_POST
        server_mod.APIHandler.do_GET = patched_do_GET
        server_mod._optiq_multi_adapter_handler_patched = True


def _peek_adapter_field(handler) -> str | None:
    """Read the ``adapters`` field from the request body without consuming
    the rfile stream (the original do_POST needs to re-read the body).

    We can't actually peek without buffering, so we stash the bytes on the
    handler and replace ``rfile.read`` to return them. This is a bit ugly
    but keeps the change additive: the original handler is unaware.
    """
    import io
    import json as _json

    length = handler.headers.get("Content-Length")
    if not length:
        return None
    try:
        length = int(length)
    except Exception:
        return None
    raw = handler.rfile.read(length)
    # Replace rfile so the original handler sees the same bytes.
    handler.rfile = io.BufferedReader(io.BytesIO(raw))
    # Rewrite Content-Length unchanged; BaseHTTPRequestHandler reads it from
    # headers, not the stream, so no adjustment needed.
    try:
        body = _json.loads(raw.decode())
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    name = body.get("adapters") or body.get("adapter")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _PathFromStr(p):
    from pathlib import Path
    return Path(p)


# ---------------------------------------------------------------------------
# DiffusionGemma serving
# ---------------------------------------------------------------------------


def install_diffusion_serving() -> None:
    """Make ``optiq serve`` work for DiffusionGemma.

    ``mlx_lm.server`` is autoregressive: it loads via ``mlx_lm.load`` (which
    can't load ``diffusion_gemma``) and streams one token at a time. This patches
    the server's ``load`` and ``stream_generate`` names so a ``diffusion_gemma``
    checkpoint loads through OptiQ's vendored decoder and generates with the
    masked-diffusion loop (confidence-threshold sampler — the fast default).

    The diffusion canvas commits many tokens per denoising step, which doesn't
    map onto the server's per-token protocol, so we run the decode to completion
    and replay the result token-by-token as ``GenerationResponse`` objects (the
    server's detokenizer rebuilds the text from ``.token``). Idempotent; non-
    diffusion models fall through unchanged.
    """
    import mlx.core as mx
    import mlx_lm.server as server_mod
    from mlx_lm.generate import GenerationResponse

    if getattr(server_mod, "_optiq_diffusion_served", False):
        return

    from .vlm.diffusion_gemma.generate import DEFAULT_SAMPLER
    from .vlm.diffusion_gemma.generate import load as diffusion_load
    from .vlm.diffusion_gemma.loader import load_config
    from .vlm._mlxvlm.generate.diffusion import stream_diffusion_generate

    orig_load = server_mod.load
    orig_stream = server_mod.stream_generate
    _zero_logprobs = {}  # cache one vocab-sized zeros array per model

    def load(model_path, *args, **kwargs):
        model_type = None
        try:
            model_type = load_config(model_path).get("model_type")
        except Exception:
            pass
        if model_type == "diffusion_gemma":
            model, tokenizer = diffusion_load(model_path)
            model._optiq_diffusion = True
            return model, tokenizer
        if model_type == "llada2_moe":
            # LLaDA2-MoE: masked-diffusion decode via the vendored decoder.
            from .models import llada2
            from mlx_lm import load as mlx_load
            llada2.install()
            model, tokenizer = mlx_load(model_path)
            model._optiq_diffusion = True
            model._optiq_llada2 = True
            return model, tokenizer
        return orig_load(model_path, *args, **kwargs)

    def stream_generate(model, tokenizer, prompt, *args, **kwargs):
        if not getattr(model, "_optiq_diffusion", False):
            yield from orig_stream(model, tokenizer, prompt, *args, **kwargs)
            return

        if getattr(model, "_optiq_llada2", False):
            # LLaDA2 block-diffusion decode runs to completion (the canvas commits
            # many tokens per step), then we replay the ids token-by-token so the
            # server's detokenizer rebuilds the text, same as the gemma path.
            from .models import llada2
            ids = prompt if hasattr(prompt, "ndim") else mx.array(list(prompt), dtype=mx.int32)
            prompt_list = [int(t) for t in (ids.tolist() if hasattr(ids, "tolist") else prompt)]
            temperature = float(kwargs.get("temp", kwargs.get("temperature", 0.0)) or 0.0)
            out_ids = llada2.diffusion_generate(
                model, prompt_list, gen_length=kwargs.get("max_tokens", 512),
                temperature=temperature)
            vocab = model.args.vocab_size
            if vocab not in _zero_logprobs:
                _zero_logprobs[vocab] = mx.zeros((vocab,), dtype=mx.float32)
            zlp = _zero_logprobs[vocab]
            prev, n = "", len(out_ids)
            for i, tok in enumerate(out_ids):
                cur = tokenizer.decode(out_ids[:i + 1])
                text, prev = cur[len(prev):], cur
                yield GenerationResponse(
                    text=text, token=int(tok), logprobs=zlp, from_draft=False,
                    prompt_tokens=int(ids.size), prompt_tps=0.0,
                    generation_tokens=i + 1, generation_tps=0.0, peak_memory=0.0,
                    finish_reason=("stop" if i == n - 1 else None),
                )
            return

        max_tokens = kwargs.get("max_tokens", 512)
        ids = prompt if hasattr(prompt, "ndim") else mx.array(list(prompt), dtype=mx.int32)
        input_ids = ids[None] if ids.ndim == 1 else ids

        vocab = model.config.text_config.vocab_size
        if vocab not in _zero_logprobs:
            _zero_logprobs[vocab] = mx.zeros((vocab,), dtype=mx.float32)
        zlp = _zero_logprobs[vocab]

        # Run the masked-diffusion decode to completion, collecting committed
        # tokens + their detokenized text segments. The canvas starts from random
        # noise and occasionally collapses to an empty result — retry on empty so
        # serve never returns a blank response.
        temperature = float(kwargs.get("temp", kwargs.get("temperature", 0.0)) or 0.0)
        chunks = []  # (token, text)
        for _attempt in range(3):
            chunks = []
            for r in stream_diffusion_generate(
                model, tokenizer, tokenizer, input_ids, None, None,
                max_tokens=max_tokens, skip_special_token_ids=set(),
                temperature=temperature, diffusion_sampler=DEFAULT_SAMPLER,
            ):
                if getattr(r, "is_draft", False):
                    continue
                tok = r.token if r.token is not None else 0
                chunks.append((int(tok), r.text))
            if any(text.strip() for _, text in chunks):
                break

        n = len(chunks)
        for i, (tok, text) in enumerate(chunks):
            yield GenerationResponse(
                text=text, token=tok, logprobs=zlp, from_draft=False,
                prompt_tokens=int(input_ids.size), prompt_tps=0.0,
                generation_tokens=i + 1, generation_tps=0.0, peak_memory=0.0,
                finish_reason=("stop" if i == n - 1 else None),
            )

    server_mod.load = load
    server_mod.stream_generate = stream_generate
    server_mod._optiq_diffusion_served = True


# ---------------------------------------------------------------------------
# Dhara-AR tri-mode serving
# ---------------------------------------------------------------------------


def install_dhara_serving(mode: str = "selfspec") -> None:
    """Serve Dhara-AR in a non-AR decode mode (``selfspec`` or ``diffusion``).

    Dhara's AR mode already serves through stock ``mlx_lm.server`` (it's a
    registered AR model). This patches ``stream_generate`` so a ``dhara_ar``
    model decodes with the chosen mode instead — ``selfspec`` is what
    ``optiq serve --mtp`` selects (the model self-drafts a block and verifies it
    with its own AR forward, AR-quality). The decode runs to completion and is
    replayed token-by-token as ``GenerationResponse`` (the server detokenizes
    from ``.token``). Idempotent; non-dhara models fall through.
    """
    import mlx.core as mx
    import numpy as np
    import mlx_lm.server as server_mod
    from mlx_lm.generate import GenerationResponse

    if getattr(server_mod, "_optiq_dhara_served", False):
        return
    from .mlx_lm_patches.dhara_decode import diffusion_ids, selfspec_ids

    orig_load = server_mod.load
    orig_stream = server_mod.stream_generate
    _zlp = {}

    def load(path, *args, **kwargs):
        model, tokenizer = orig_load(path, *args, **kwargs)
        if getattr(model, "model_type", "") == "dhara_ar":
            model._dhara_mode = mode
        return model, tokenizer

    def stream_generate(model, tokenizer, prompt, *args, **kwargs):
        if getattr(model, "_dhara_mode", None) is None:
            yield from orig_stream(model, tokenizer, prompt, *args, **kwargs)
            return
        max_tokens = kwargs.get("max_tokens", 256)
        ids = list(np.asarray(prompt).reshape(-1).astype(np.int64)) \
            if hasattr(prompt, "__len__") or hasattr(prompt, "shape") else [int(prompt)]
        fn = selfspec_ids if model._dhara_mode == "selfspec" else diffusion_ids
        out_ids = [int(t) for t in fn(model, tokenizer, ids, max_new=max_tokens)]

        V = model.args.vocab_size
        if V not in _zlp:
            _zlp[V] = mx.zeros((V,), dtype=mx.float32)
        zlp = _zlp[V]
        n = len(out_ids)
        for i, tok in enumerate(out_ids):
            yield GenerationResponse(
                text=tokenizer.decode([tok]), token=tok, logprobs=zlp,
                from_draft=False, prompt_tokens=len(ids), prompt_tps=0.0,
                generation_tokens=i + 1, generation_tps=0.0, peak_memory=0.0,
                finish_reason=("stop" if i == n - 1 else None),
            )

    server_mod.load = load
    server_mod.stream_generate = stream_generate
    server_mod._optiq_dhara_served = True


# ---------------------------------------------------------------------------
# SSD expert streaming for large MoE quants (optiq.runtime.moe_stream)
# ---------------------------------------------------------------------------

_STREAM_EXPERTS_CACHE = 0
_PRELOADED_STREAMING = None  # (model_path, model, tokenizer)


def install_streaming_experts(mode: str = "auto", cache_experts: int = 0,
                              model_path: str | None = None) -> None:
    """Route MoE quant loading through ``optiq.runtime.moe_stream`` so a model
    whose fused expert tensors would OOM resident streams the active experts
    from SSD instead.

    ``mode``:
      * ``"on"``   — always stream a streamable MoE quant
      * ``"auto"`` — stream only when the model is too big to fit resident
      * ``"off"``  — no-op

    Patches ``ModelProvider._load``: when streaming applies to the model being
    loaded (and no adapter/draft model is requested, which streaming does not
    support), build it with ``load_streaming`` and populate
    ``self.model``/``self.tokenizer``; otherwise fall through to the original
    loader. Idempotent.
    """
    if mode == "off":
        return
    import mlx_lm.server as server_mod

    global _STREAM_EXPERTS_CACHE, _PRELOADED_STREAMING
    _STREAM_EXPERTS_CACHE = cache_experts
    server_mod._optiq_stream_experts_mode = mode

    from .runtime.moe_stream import (
        is_streamable_moe, load_streaming, should_stream,
    )

    def _wants(mp: str) -> bool:
        return ((mode == "on" and is_streamable_moe(mp))
                or (mode == "auto" and should_stream(mp)))

    # Pre-load the streaming model on THIS (main) thread. mlx-lm's lazy model
    # construction OOMs for a large MoE quant when run inside mlx_lm.server's
    # request handler (works fine on the main thread), so we build the streamed
    # model here and hand the ready model to the handler.
    if model_path and _wants(model_path):
        try:
            model, tok = load_streaming(model_path, cache_experts=cache_experts)
            _PRELOADED_STREAMING = (model_path, model, tok)
            print(f"[optiq.serve] SSD expert streaming: pre-loaded {model_path}",
                  flush=True)
        except Exception as exc:
            print(f"[optiq.serve] streaming pre-load failed ({exc}); "
                  f"the server will try at first request", flush=True)

    if getattr(server_mod, "_optiq_stream_experts_patched", False):
        return
    orig_load = server_mod.ModelProvider._load

    def patched_load(self, model_path, adapter_path=None, draft_model_path=None):
        pre = _PRELOADED_STREAMING
        use_pre = (pre and pre[0] == model_path
                   and adapter_path is None and draft_model_path is None)
        if use_pre or (_wants(model_path)
                       and adapter_path is None and draft_model_path is None):
            try:
                if use_pre:
                    model, tokenizer = pre[1], pre[2]
                else:
                    model, tokenizer = load_streaming(
                        model_path, cache_experts=_STREAM_EXPERTS_CACHE)
                self.model_key = (model_path, adapter_path, draft_model_path)
                self.model = model
                self.tokenizer = tokenizer
                self.draft_model = None
                if (self.cli_args.use_default_chat_template
                        and tokenizer.chat_template is None):
                    tokenizer.chat_template = tokenizer.default_chat_template
                return
            except Exception as exc:  # never block serving on the fast path
                print(f"  [optiq.serve] expert streaming failed ({exc}); "
                      f"falling back to resident load", flush=True)
        return orig_load(self, model_path, adapter_path=adapter_path,
                         draft_model_path=draft_model_path)

    server_mod.ModelProvider._load = patched_load
    server_mod._optiq_stream_experts_patched = True
