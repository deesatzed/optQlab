"""OptiQ unified inference engine.

Replaces MTPLX's slow Python wrapper loop with a custom generation loop
built on mlx-lm primitives, while keeping MTPLX's MTP module + helper
methods for the draft step. Single engine for both regimes:

  * **AR**  — fast prefill + decode via ``lm.model(...)`` (bypasses the
              ``_MTPLXTextModel.__call__`` slow path).
  * **MTP** — speculative cycle: draft K tokens with the attached MTP head,
              verify with main model in parallel, greedy-accept prefix.

Compatibility with OptiQ KV-quant: ``make_prompt_cache`` produces the
model's preferred cache class; we expose ``--kv-bits`` to convert
``KVCache`` instances to ``QuantizedKVCache`` for long-context decode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import mlx.core as mx


@dataclass
class GenStats:
    prompt_tokens: int
    generated_tokens: int = 0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    drafts_attempted: int = 0
    drafts_accepted: int = 0
    speculation_cycles: int = 0
    mode: str = "AR"
    depth: int = 0
    text: str = ""

    @property
    def decode_tok_s(self) -> float:
        return self.generated_tokens / self.decode_time_s if self.decode_time_s > 0 else 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.drafts_accepted / self.drafts_attempted if self.drafts_attempted else 0.0

    def summary(self) -> str:
        base = (
            f"[optiq.engine] mode={self.mode} depth={self.depth} "
            f"tokens={self.generated_tokens} tok_s={self.decode_tok_s:.2f} "
            f"prefill={self.prefill_time_s:.2f}s"
        )
        if self.mode == "MTP":
            base += (
                f" cycles={self.speculation_cycles} "
                f"accept={self.acceptance_rate:.1%}"
            )
        return base


def _maybe_quantize_kv(cache: list, kv_bits: Optional[int], kv_group_size: int):
    """Convert any ``KVCache`` entry to ``QuantizedKVCache`` in-place.

    ``MambaCache`` / linear-attn caches are passed through untouched
    (they don't expose ``to_quantized``).
    """
    if kv_bits is None:
        return
    for i, c in enumerate(cache):
        if hasattr(c, "to_quantized"):
            cache[i] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


def _logits_to_token(logits: "mx.array", temp: float) -> int:
    """Greedy at temp=0, simple multinomial otherwise."""
    if temp == 0.0:
        return int(mx.argmax(logits, axis=-1).item())
    probs = mx.softmax(logits / temp, axis=-1)
    return int(mx.random.categorical(mx.log(probs)).item())


def _truncated_logits(
    logits: "mx.array",
    temp: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> "mx.array":
    """Apply temperature scaling and truncation (top_k / top_p / min_p).
    Returns scaled+masked logits where excluded tokens have -inf. Caller
    softmaxes to get the truncated distribution. Mirrors mlx_lm's
    sample_utils so OptiQ's spec-decoding distributions match what
    mlx_lm.server uses for real serving."""
    out = logits / max(temp, 1e-6)
    if top_k and top_k > 0:
        from mlx_lm.sample_utils import apply_top_k
        out = apply_top_k(out, int(top_k))
    if top_p and 0.0 < top_p < 1.0:
        from mlx_lm.sample_utils import apply_top_p
        out = apply_top_p(out, float(top_p))
    if min_p and min_p > 0.0:
        from mlx_lm.sample_utils import apply_min_p
        out = apply_min_p(out, float(min_p))
    return out


def _sample_with_dist(
    logits: "mx.array",
    temp: float,
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
):
    """Sample a token and return ``(token, distribution_or_None)``.

    For temp>0, applies temperature + top_p/top_k/min_p truncation, then
    returns the full distribution alongside the sample so verify can do
    rejection sampling on the SAME truncated distribution. For temp=0,
    returns ``(argmax, None)`` (caller should use greedy verification).
    """
    if temp == 0.0:
        return int(mx.argmax(logits, axis=-1).item()), None
    masked = _truncated_logits(logits, temp, top_p, top_k, min_p)
    probs = mx.softmax(masked, axis=-1)
    sampled = int(mx.random.categorical(mx.log(probs + 1e-30)).item())
    return sampled, probs


def _verify_speculative(
    draft_token: int,
    draft_probs,  # mx.array (vocab,) | None
    main_logits: "mx.array",
    temp: float,
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
) -> tuple[bool, int]:
    """Speculative decoding verify step. Optimized for minimal
    CPU/GPU sync points — see ``_rejection_verify_array`` below.

    At ``temp == 0`` (or ``draft_probs is None``): greedy match — accept
    iff main's argmax equals the draft. One sync point per call.

    At ``temp > 0``: rejection sampling (Leviathan / Chen) on the
    TRUNCATED distribution. All math stays on-device; we extract the
    accept bool + committed token together in one ``mx.eval``. Down
    from four ``.item()`` syncs per call in the v0 implementation.

    Returns ``(accepted, token_to_commit)``.
    """
    if temp == 0.0 or draft_probs is None:
        v_tok = int(mx.argmax(main_logits, axis=-1).item())
        if v_tok == draft_token:
            return True, draft_token
        return False, v_tok

    accepted_arr, chosen_arr = _rejection_verify_array(
        draft_token, draft_probs, main_logits, temp, top_p, top_k, min_p,
    )
    # Single sync — fetches the bool decision and the token id together.
    mx.eval(accepted_arr, chosen_arr)
    accepted = bool(accepted_arr.item())
    chosen = int(chosen_arr.item())
    return accepted, chosen


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _rejection_kernel(
    draft_token_arr: "mx.array",
    draft_probs: "mx.array",
    p_main: "mx.array",
):
    """Fused on-device rejection sampling kernel.

    Inputs are all mx.arrays — captures the random state so the compiled
    graph is reproducible. Operates on a pre-truncated p_main; truncation
    happens in the caller because top-k/top-p use shape-dependent kernels
    that don't play well with mx.compile (already compiled inside mlx_lm
    via their own `apply_top_k` / `apply_top_p`).

    Returns ``(accepted_arr, chosen_arr)`` both as scalars.
    """
    q_dt = draft_probs[draft_token_arr]
    p_dt = p_main[draft_token_arr]
    accept_prob = mx.minimum(1.0, p_dt / mx.maximum(q_dt, 1e-12))
    u = mx.random.uniform(shape=())
    accepted = u < accept_prob

    diff = mx.maximum(p_main - draft_probs, 0.0)
    norm = diff.sum()
    safe = norm > 1e-9
    residual = mx.where(safe, diff / mx.maximum(norm, 1e-12), p_main)
    replacement = mx.random.categorical(mx.log(residual + 1e-30))

    chosen = mx.where(accepted, draft_token_arr, replacement)
    return accepted, chosen


def _rejection_verify_array(
    draft_token: int,
    draft_probs: "mx.array",
    main_logits: "mx.array",
    temp: float,
    top_p: float,
    top_k: int,
    min_p: float,
):
    """Pure-array rejection sampling. Returns ``(accepted_arr, chosen_arr)``,
    both mx.array scalars. Caller does ONE ``mx.eval`` + two ``.item()``
    to extract values — instead of four interleaved ``.item()`` syncs.

    The trick: compute accept_prob, sample u, sample the residual
    replacement candidate, and select between draft_token vs replacement
    via ``mx.where`` — all on device. Materializing both values together
    is one round-trip instead of four.
    """
    masked = _truncated_logits(main_logits, temp, top_p, top_k, min_p)
    p_main = mx.softmax(masked, axis=-1)
    # Wrap scalar token id in an mx.array so the kernel can use it via
    # array indexing. dtype matches what mx.random.categorical returns
    # so the final mx.where doesn't have to upcast.
    dt_arr = mx.array(draft_token, dtype=mx.uint32)
    return _rejection_kernel(dt_arr, draft_probs, p_main)


class OptiqEngine:
    """One inference engine, two modes (AR / MTP), both fast.

    Load once, generate many. Reuses model state across calls; users
    can stream multiple generations through the same loaded engine.
    """

    def __init__(self, model_path: str):
        # Use the vendored MTPLX loader to attach the MTP head; we
        # then sidestep the slow ``__call__`` by talking to ``lm.model``
        # directly. Wrapper's helper methods (``_mtp_core``,
        # ``_mtp_full_attention_layer``) remain available.
        from optiq.runtime.mtp.runtime import load as mtplx_load
        try:
            rt = mtplx_load(model_path)
        except RuntimeError:
            # Not every model has an MTP head -- plenty of bases ship none at
            # all. Fall back to the plain runtime so the engine still serves AR
            # and the multimodal path, exactly as ``from_loaded`` already does.
            # Without this, OptiqEngine("<any model with no MTP head>") died on
            # "MTP injection failed", which took image inference down with it.
            # depth>0 stays guarded by ``has_mtp``.
            rt = mtplx_load(model_path, mtp=False)
        self._init_from_runtime(rt, model_path)

    @classmethod
    def from_loaded(cls, model, tokenizer, model_path: str) -> "OptiqEngine":
        """Build an engine using an already-loaded ``(model, tokenizer)``
        pair (e.g. from ``mlx_lm.load`` already done by mlx_lm.server).

        Avoids the +N GB duplicate-model footprint that ``__init__``'s
        full ``mtplx_load`` path incurs. The MTP head is still injected
        into ``model`` in place, but only one copy lives in unified
        memory. Use via ``install_mtp_speculation`` so the engine and
        the serving runtime share the same backing tensors.
        """
        from optiq.runtime.mtp.runtime import attach
        inst = cls.__new__(cls)
        try:
            rt = attach(model, tokenizer, model_path)
        except RuntimeError:
            # No MTP head on this artifact — attach the plain runtime so the
            # engine still serves AR (and the multimodal path). MTP-only
            # features (depth>0) stay unavailable, guarded by ``has_mtp``.
            rt = attach(model, tokenizer, model_path, mtp=False)
        inst._init_from_runtime(rt, model_path)
        return inst

    def _init_from_runtime(self, rt, model_path: str) -> None:
        self._rt = rt
        self._model = self._rt.model
        self._lm = self._model.language_model
        self._inner = self._lm.model  # fast forward path (no wrapper)
        # Logits projection: untied lm_head OR tied embedding-as-linear.
        # Small Qwen3.5 (e.g. 0.8B) ties word embeddings to lm_head and
        # does NOT carry a standalone lm_head attribute; bigger models
        # (9B+) untie them and expose lm_head directly.
        if hasattr(self._lm, "lm_head"):
            self._lm_head = self._lm.lm_head
        else:
            # Tied case — lm_head shares the embedding matrix.
            embed = self._inner.embed_tokens
            self._lm_head = lambda h, _embed=embed: _embed.as_linear(h)
        self._tokenizer = self._rt.tokenizer
        self._has_mtp = getattr(self._lm, "mtp", None) is not None
        self._model_path = model_path
        # Vision/audio front-end (lazy): present iff the artifact ships an
        # ``optiq_vision.safetensors`` sidecar and a frontend is registered for
        # the arch. Loaded on first multimodal request to keep text-only startup
        # cheap.
        self._vision_fe = None
        self._vision_checked = False
        # Telemetry: the most recent ``GenStats`` from ``generate_stream`` so
        # callers can inspect literature-standard acceptance rate
        # (drafts_accepted / drafts_attempted) after a streaming run.
        self._last_stats: "GenStats | None" = None

    @property
    def has_mtp(self) -> bool:
        return self._has_mtp

    @property
    def tokenizer(self):
        return self._tokenizer

    # ------------------------------------------------------------------
    # Forward primitives — these are the ONLY model calls used in the loop.
    # ------------------------------------------------------------------

    def _forward(self, ids: "mx.array", cache, *, input_embeddings=None,
                per_layer_inputs=None):
        """Fast main-model forward — stock mlx-lm path.
        Used for prefill and AR decode (no captures needed).
        MTP verify uses ``forward_with_gdn_capture`` directly for the
        rollback machinery.

        For multimodal prefill, pass ``input_embeddings`` (already merged with
        the vision/audio soft tokens) and, for gemma-4, ``per_layer_inputs``;
        ``ids`` is then ignored by the inner model."""
        if input_embeddings is not None:
            kw = {"input_embeddings": input_embeddings}
            if per_layer_inputs is not None:
                kw["per_layer_inputs"] = per_layer_inputs
            hidden = self._inner(None, cache=cache, **kw)
        else:
            hidden = self._inner(ids, cache=cache)
        logits = self._lm_head(hidden)
        return hidden, logits

    # ------------------------------------------------------------------
    # Multimodal prefill
    # ------------------------------------------------------------------

    def _local_model_dir(self) -> "str | None":
        """The on-disk directory for this model, whatever it was named with.

        ``model_path`` is whatever the caller passed. Every documented way of
        loading an OptiQ VLM passes a Hub repo id -- ``optiq serve --model
        mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit`` is what the model cards
        tell you to run -- and a repo id is not a directory. The vision gate used
        to test ``os.path.isdir(model_path)`` directly, so for every Hub-loaded
        model it was False and the front-end was switched off before the sidecar
        was ever looked for. The model then reported "no vision sidecar" while
        the sidecar sat in the snapshot, downloaded and unread. Reported on the
        26B by a user on 0.3.2; we never saw it because every vision test we ran
        pointed at a local ``optiq_output/`` directory.

        The weights are already downloaded by the time an engine exists, so this
        only resolves the cache path. It never fetches.
        """
        from ..sidecar_layout import local_model_dir

        return local_model_dir(self._model_path)

    def _no_vision_reason(self) -> str:
        """Why this model cannot take an image. Say which of the three it is.

        The old message asserted "no vision sidecar" for every cause, including
        the case where the sidecar was present and the loader had simply been
        handed a repo id. A user on the 26B reported exactly that: the file is
        right there, and OptiQ says it is not.
        """
        import json
        import os

        from optiq.vlm import get_frontend, has_vision_sidecar

        model_dir = self._local_model_dir()
        if model_dir is None:
            return (f"cannot take images: no local snapshot for "
                    f"{self._model_path!r}. Download the model first.")
        if not has_vision_sidecar(model_dir):
            return (f"cannot take images: this model ships no vision sidecar "
                    f"(looked for optiq/optiq_vision.safetensors and "
                    f"optiq_vision.safetensors under {model_dir}).")
        err = getattr(self, "_vision_error", None)
        if err:
            return (f"cannot take images: this model HAS a vision sidecar but its "
                    f"front-end failed to load -- {err}")
        try:
            mt = json.load(open(os.path.join(model_dir, "config.json"))).get("model_type", "?")
        except Exception:
            mt = "?"
        if get_frontend(mt) is None:
            return (f"cannot take images: this model has a vision sidecar, but "
                    f"OptiQ has no vision front-end registered for model_type "
                    f"{mt!r}.")
        return "cannot take images: vision front-end unavailable."

    def _load_vision_frontend(self):
        """Lazily resolve + load the vision front-end for this artifact.
        Returns the frontend instance, or ``None`` if the model is text-only."""
        if self._vision_checked:
            return self._vision_fe
        self._vision_checked = True

        import json
        import os

        from optiq.vlm import get_frontend, has_vision_sidecar

        model_dir = self._local_model_dir()
        if model_dir is None or not has_vision_sidecar(model_dir):
            return None
        try:
            cfg = json.load(open(os.path.join(model_dir, "config.json")))
            factory = get_frontend(cfg.get("model_type", ""))
            if factory is None:
                return None
            self._vision_fe = factory(model_dir, self._lm)
        except Exception as e:                                   # noqa: BLE001
            # A model that HAS a sidecar but fails to build its front-end is a
            # bug, not a text-only model. Swallowing it here is what let the
            # path above stay broken -- every failure looked like "text-only".
            self._vision_fe = None
            self._vision_error = f"{type(e).__name__}: {e}"
        return self._vision_fe

    def _vlm_prefill_inputs(self, prompt: str, images: list, *,
                            messages: Optional[list] = None,
                            enable_thinking: bool = False):
        """Build (prompt_ids, merged_embeddings, per_layer_inputs) for an
        image+text turn. Raises if the model has no vision front-end.

        Pass ``messages`` (full chat list with image content parts) to preserve
        history + multiple images; otherwise a single user turn is built from
        ``prompt`` + ``images``."""
        fe = self._load_vision_frontend()
        if fe is None:
            raise RuntimeError(self._no_vision_reason())
        if messages is None:
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
        inp = fe.preprocess(messages, tokenizer=self._tokenizer,
                            enable_thinking=enable_thinking)
        merged, extra = fe.merged_embeddings(inp)
        return inp["input_ids"], merged, extra.get("per_layer_inputs")

    def _mtp_draft(
        self,
        hidden_last: "mx.array",
        next_token_id: int,
        mtp_cache: list,
    ):
        """One MTP draft step. Returns (logits, new_hidden).

        Reuses ``_MTPLXTextModel._mtp_core`` — the wrapper class is still
        on the model, only its slow ``__call__`` is unused.
        """
        next_ids = mx.array([[next_token_id]])
        return self._lm._mtp_core(
            hidden_last,
            next_ids,
            mtp_cache=mtp_cache,
            emit_logits=True,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        depth: int = 0,
        temperature: float = 0.0,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        chat_template: bool = True,
        images: Optional[list] = None,
        messages: Optional[list] = None,
    ) -> GenStats:
        """Generate continuation. depth=0 means AR; depth>=1 enables MTP.

        Greedy at temp=0 — guarantees byte-identical output regardless
        of depth. With depth>=1, MTP drafts K tokens per cycle and verifies
        in parallel. All-or-nothing acceptance (v1): if any draft mismatches,
        accept the verified prefix + 1 corrected token, discard the rest.
        """
        if depth > 0 and not self._has_mtp:
            raise RuntimeError(
                f"depth={depth} but model has no MTP head; "
                f"load a model produced by `optiq mtp-patch`."
            )

        tok = self._tokenizer
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self._model)

        if images or messages:
            ids_arr, merged, pli = self._vlm_prefill_inputs(
                prompt, images or [], messages=messages)
            prompt_ids = ids_arr[0].tolist()
            t0 = time.perf_counter()
            hidden, logits = self._forward(
                ids_arr, cache=cache, input_embeddings=merged, per_layer_inputs=pli,
            )
            mx.eval(logits)
            prefill_time = time.perf_counter() - t0
        else:
            # Render via chat template if available (matches mlx-lm UX).
            rendered = prompt
            if chat_template:
                try:
                    if getattr(tok, "chat_template", None):
                        rendered = tok.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                except Exception:
                    pass
            prompt_ids = tok.encode(rendered)
            t0 = time.perf_counter()
            hidden, logits = self._forward(mx.array([prompt_ids]), cache=cache)
            mx.eval(logits)
            prefill_time = time.perf_counter() - t0

        # Convert to QuantizedKVCache for decode if requested
        if kv_bits is not None:
            _maybe_quantize_kv(cache, kv_bits, kv_group_size)

        stats = GenStats(
            prompt_tokens=len(prompt_ids),
            prefill_time_s=prefill_time,
            mode="MTP" if depth > 0 else "AR",
            depth=depth,
        )

        # Initial token from main model's final logits
        last_token = _logits_to_token(logits[0, -1], temperature)
        last_hidden = hidden[:, -1:, :]  # keep batch + last position
        generated_ids: list[int] = [last_token]
        eos_ids = self._eos_ids()

        # Decode loop ---------------------------------------------------
        t0 = time.perf_counter()
        try:
            if depth == 0:
                self._ar_loop(
                    cache, last_token, generated_ids, max_tokens, temperature, eos_ids,
                )
            else:
                self._mtp_loop(
                    cache, last_hidden, last_token, generated_ids,
                    max_tokens, depth, temperature, eos_ids, stats,
                )
        finally:
            stats.decode_time_s = time.perf_counter() - t0
            stats.generated_tokens = len(generated_ids)
            stats.text = tok.decode(generated_ids)
        return stats

    # ------------------------------------------------------------------
    # Streaming generation — yields tokens as they are produced.
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        depth: int = 0,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        min_p: float = 0.0,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        chat_template: bool = True,
        images: Optional[list] = None,
        messages: Optional[list] = None,
    ):
        """Yields ``StreamEvent`` per token (see below) as they are produced.

        ``StreamEvent`` is a dict with keys:
          - ``token`` (int): the generated token id
          - ``text`` (str): the detokenized piece (may be empty for some BPE pairs)
          - ``from_draft`` (bool): True if this token was sampled by the MTP
            head and accepted by verify (the speculation win); False for AR
            tokens, the verified replacement on partial-accept, or the bonus
            token at the end of an all-accept cycle.
          - ``prompt_tokens`` (int): prompt length (constant across the stream)
          - ``generation_index`` (int): 0-based index of this token in the output
          - ``prefill_time_s`` (float): prefill wall time (constant)
          - ``decode_tps`` (float): running decode tokens/sec (updates each step)
          - ``done`` (bool): True only on the final event
        """
        if depth > 0 and not self._has_mtp:
            raise RuntimeError(
                f"depth={depth} but model has no MTP head. "
                f"Convert with v0.1.0+ ``optiq convert`` to preserve mtp.* tensors."
            )

        tok = self._tokenizer
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self._model)

        if images or messages:
            ids_arr, merged, pli = self._vlm_prefill_inputs(
                prompt, images or [], messages=messages)
            prompt_ids = ids_arr[0].tolist()
            t0 = time.perf_counter()
            hidden, logits = self._forward(
                ids_arr, cache=cache, input_embeddings=merged, per_layer_inputs=pli,
            )
            mx.eval(logits)
            prefill_time = time.perf_counter() - t0
        else:
            rendered = prompt
            if chat_template:
                try:
                    if getattr(tok, "chat_template", None):
                        rendered = tok.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                except Exception:
                    pass
            prompt_ids = tok.encode(rendered)
            t0 = time.perf_counter()
            hidden, logits = self._forward(mx.array([prompt_ids]), cache=cache)
            mx.eval(logits)
            prefill_time = time.perf_counter() - t0

        if kv_bits is not None:
            _maybe_quantize_kv(cache, kv_bits, kv_group_size)

        eos_ids = self._eos_ids()
        prompt_tokens = len(prompt_ids)

        # First token from prefill logits.
        first_token = _logits_to_token(logits[0, -1], temperature)
        first_hidden = hidden[:, -1:, :]

        decode_start = time.perf_counter()
        n_generated = 0

        def _make_event(tid: int, from_draft: bool, done: bool = False) -> dict:
            elapsed = time.perf_counter() - decode_start
            return {
                "token": int(tid),
                "text": tok.decode([int(tid)], skip_special_tokens=True),
                "from_draft": bool(from_draft),
                "prompt_tokens": prompt_tokens,
                "generation_index": n_generated,
                "prefill_time_s": prefill_time,
                "decode_tps": (n_generated + 1) / max(elapsed, 1e-6),
                "done": done,
            }

        # Buffer-one-ahead pattern so we can mark the final event done=True.
        pending = (first_token, False)
        n_generated = 1
        # Build the per-mode iterator for the rest of the tokens.
        if depth == 0:
            it = self._ar_loop_iter(
                cache, first_token, max_tokens - 1, temperature, eos_ids,
            )
        else:
            stats = GenStats(
                prompt_tokens=prompt_tokens, prefill_time_s=prefill_time,
                mode="MTP", depth=depth,
            )
            # Persist for post-stream telemetry inspection (e.g. benchmarks
            # that want the literature drafts_accepted / drafts_attempted).
            self._last_stats = stats
            it = self._mtp_loop_iter(
                cache, first_hidden, first_token, max_tokens - 1,
                depth, temperature, eos_ids, stats,
                top_p=top_p, top_k=top_k, min_p=min_p,
            )

        for nxt_tok, nxt_from_draft in it:
            # Emit the previous event (we now know there's more coming).
            tok_id, from_draft = pending
            yield _make_event(tok_id, from_draft, done=False)
            pending = (nxt_tok, nxt_from_draft)
            n_generated += 1
        # Emit the final buffered event with done=True.
        tok_id, from_draft = pending
        yield _make_event(tok_id, from_draft, done=True)

    # ------------------------------------------------------------------
    # AR mode — generator version (yields per token).  ``_ar_loop`` is a
    # thin wrapper that drains the iterator into ``gen_ids`` so the
    # existing ``generate()`` API still works unchanged.
    # ------------------------------------------------------------------

    def _ar_loop_iter(self, cache, start_token: int, remaining: int,
                      temp: float, eos_ids: set):
        """Yields ``(token_id, from_draft=False)`` per generated token."""
        cur = start_token
        n = 0
        while n < remaining:
            inp = mx.array([[cur]])
            _, logits = self._forward(inp, cache=cache)
            mx.eval(logits)
            t = _logits_to_token(logits[0, -1], temp)
            yield (t, False)
            n += 1
            if t in eos_ids:
                return
            cur = t

    def _ar_loop(self, cache, last_token, gen_ids, max_tokens, temp, eos_ids):
        remaining = max_tokens - len(gen_ids)
        for tok, _ in self._ar_loop_iter(cache, last_token, remaining, temp, eos_ids):
            gen_ids.append(tok)

    # ------------------------------------------------------------------
    # MTP speculation mode — draft K, verify K, greedy-accept prefix.
    # Generator version (``_mtp_loop_iter``) yields tokens as they are
    # committed; ``_mtp_loop`` drains into ``gen_ids`` for the existing
    # generate() API.
    # ------------------------------------------------------------------

    def _mtp_loop(
        self,
        cache,
        last_hidden: "mx.array",
        last_token: int,
        gen_ids: list,
        max_tokens: int,
        depth: int,
        temp: float,
        eos_ids: set,
        stats: GenStats,
    ):
        remaining = max_tokens - len(gen_ids)
        for tok, _ in self._mtp_loop_iter(
            cache, last_hidden, last_token, remaining, depth, temp, eos_ids, stats,
        ):
            gen_ids.append(tok)

    def _mtp_loop_iter(
        self,
        cache,
        last_hidden: "mx.array",
        last_token: int,
        remaining: int,
        depth: int,
        temp: float,
        eos_ids: set,
        stats: GenStats,
        *,
        top_p: float = 0.0,
        top_k: int = 0,
        min_p: float = 0.0,
    ):
        """Yields ``(token_id, from_draft)`` per committed token.

        ``from_draft=True`` when the token was sampled by the MTP head and
        accepted by verify (the speculation win); ``False`` for the
        verified replacement or bonus token from the main model.

        On Apple Silicon K is fixed for the whole call. We tested a
        HuggingFace-style dynamic-depth adapter (+1/-1 around full-accept)
        and it lost 4-17% vs fixed K=1 across 4B/9B/27B × greedy/unsloth-
        sampler. Reason: on MLX/Metal K=2 verify costs ~2x K=1 (vs ~1.1x
        on CUDA where HF's heuristic lives), so K=2's best-case
        tokens/cost (1.5) is below K=1's best case (2.0). Even an oracle
        adapter can't break that ratio without a custom K-token verify
        kernel. We kept depth=1 as the optimal default and removed the
        dynamic path.
        """
        # Vendored MTPLX helpers — properly handle GDN (linear-attn)
        # recurrent state across speculation rollback.
        from optiq.runtime.mtp.cache_state import (
            snapshot_untrimmable_cache,
            rollback_after_verify,
        )
        from optiq.runtime.mtp.gdn_capture import (
            forward_with_gdn_capture,
            commit_captured_prefix,
        )

        n = 0  # tokens yielded so far
        while n < remaining:
            cycle_K = depth
            # ---- DRAFT ----
            # MTP head's own KV cache for draft sequence; resets each
            # cycle because we draft from scratch off committed history.
            # Also capture each draft's full distribution so verify can
            # do proper rejection sampling at temp>0 (Leviathan et al).
            # At temp==0 we skip the distributions (greedy path is
            # deterministic argmax compare and doesn't need them).
            draft_tokens: list[int] = []
            draft_dists: list = []  # list of mx.array (vocab,) | None
            mtp_cache: list = [None]
            cur_hidden = last_hidden
            cur_token = last_token
            for k in range(cycle_K):
                d_logits, cur_hidden = self._mtp_draft(cur_hidden, cur_token, mtp_cache)
                mx.eval(d_logits)
                t, q = _sample_with_dist(
                    d_logits[0, -1], temp,
                    top_p=top_p, top_k=top_k, min_p=min_p,
                )
                draft_tokens.append(t)
                draft_dists.append(q)
                cur_token = t
            stats.drafts_attempted += cycle_K

            # ---- SNAPSHOT non-trimmable (GDN/linear-attn) state ----
            before_verify = snapshot_untrimmable_cache(cache)

            # ---- VERIFY via GDN-capturing forward ----
            verify_ids = [last_token] + draft_tokens
            v_logits, v_hidden, captures = forward_with_gdn_capture(
                self._model,
                mx.array([verify_ids]),
                cache=cache,
                return_hidden=True,
            )
            mx.eval(v_logits, v_hidden)
            stats.speculation_cycles += 1

            # Acceptance: greedy at temp=0, rejection sampling otherwise.
            # Find longest accepted prefix; on first reject, record the
            # corrective replacement token from _verify_speculative.
            n_accept = 0
            verified_replacement = None
            for i, dt in enumerate(draft_tokens):
                accepted, tok = _verify_speculative(
                    dt, draft_dists[i], v_logits[0, i], temp,
                    top_p=top_p, top_k=top_k, min_p=min_p,
                )
                if accepted:
                    n_accept += 1
                else:
                    verified_replacement = tok
                    break

            if n_accept == cycle_K:
                # All K accepted + bonus from main's K-th prediction.
                bonus = _logits_to_token(v_logits[0, cycle_K], temp)
                # Yield K drafts (from_draft=True) then bonus (False).
                for d in draft_tokens:
                    yield (d, True)
                    n += 1
                    if n >= remaining or d in eos_ids:
                        return
                yield (bonus, False)
                n += 1
                stats.drafts_accepted += n_accept
                last_hidden = v_hidden[:, cycle_K:cycle_K + 1, :]
                last_token = bonus
                if n >= remaining or bonus in eos_ids:
                    return
            else:
                # Partial accept — fast path: commit_captured_prefix
                # replays GDN state. Slow fallback: full rollback + rerun.
                committed_drafts = draft_tokens[:n_accept]
                keep_tokens = n_accept + 1
                verified_tokens = cycle_K + 1
                n_drop = verified_tokens - keep_tokens

                committed = commit_captured_prefix(
                    cache, captures,
                    keep_tokens=keep_tokens,
                    verified_tokens=verified_tokens,
                )
                if committed:
                    for c in cache:
                        if hasattr(c, "trim"):
                            c.trim(n_drop)
                    last_hidden = v_hidden[:, n_accept:n_accept + 1, :]
                else:
                    rollback_after_verify(
                        cache, before_verify, verified_tokens=verified_tokens,
                    )
                    rerun_ids = [last_token] + committed_drafts
                    _, r_hidden, _ = forward_with_gdn_capture(
                        self._model,
                        mx.array([rerun_ids]),
                        cache=cache,
                        return_hidden=True,
                    )
                    mx.eval(r_hidden)
                    last_hidden = r_hidden[:, n_accept:n_accept + 1, :]
                last_token = verified_replacement
                stats.drafts_accepted += n_accept

                # Yield accepted drafts (True) then the corrective (False).
                for d in committed_drafts:
                    yield (d, True)
                    n += 1
                    if n >= remaining or d in eos_ids:
                        return
                yield (verified_replacement, False)
                n += 1
                if n >= remaining or verified_replacement in eos_ids:
                    return

    # ------------------------------------------------------------------
    # Cache snapshot / restore — hybrid-aware.
    #
    # KVCache / ConcatenateKVCache / QuantizedKVCache: ``offset`` is the
    # only mutable bookkeeping; their underlying arrays are append-only
    # (we'd just overwrite slots on next forward).  Restoring ``offset``
    # is sufficient.
    #
    # ArraysCache (Qwen3.5's linear-attn / GatedDeltaNet): the recurrent
    # state lives in ``self.cache`` (a list of arrays).  Forward calls
    # mutate the entries; restoring requires saving the list of array
    # references *before* the forward.  Arrays are immutable in MLX so
    # holding refs is a true snapshot.
    # ------------------------------------------------------------------

    def _snapshot_cache(self, cache: list) -> list:
        snaps = []
        for c in cache:
            if hasattr(c, "offset"):
                # Trimmable cache — offset is all we need.
                snaps.append(("trimmable", int(c.offset)))
            else:
                # ArraysCache-style — snapshot the cache list + lengths.
                state = list(c.cache) if hasattr(c, "cache") else None
                lengths = getattr(c, "lengths", None)
                snaps.append(("arrays", state, lengths))
        return snaps

    def _restore_cache(self, cache: list, snaps: list) -> None:
        for c, snap in zip(cache, snaps):
            if snap[0] == "trimmable":
                c.offset = snap[1]
            elif snap[0] == "arrays":
                _, state, lengths = snap
                if state is not None and hasattr(c, "cache"):
                    c.cache = list(state)
                if lengths is not None and hasattr(c, "lengths"):
                    c.lengths = lengths

    def _truncate_cache(self, cache: list, n_drop: int) -> None:
        """Drop last ``n_drop`` tokens — only used when ALL drafts accept
        AND cache is uniformly trimmable. Hybrid models go through the
        snapshot/restore path instead."""
        for c in cache:
            if hasattr(c, "offset"):
                c.offset = max(0, c.offset - n_drop)

    def _eos_ids(self) -> set:
        tok = self._tokenizer
        out = set()
        for candidate in (
            getattr(tok, "eos_token_id", None),
            getattr(tok, "eot_token_id", None),
        ):
            if candidate is None:
                continue
            if isinstance(candidate, (list, tuple, set)):
                out.update(int(c) for c in candidate)
            else:
                out.add(int(candidate))
        return out


def generate(
    model_path: str,
    prompt: str,
    max_tokens: int = 256,
    depth: int = 0,
    temperature: float = 0.0,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    chat_template: bool = True,
    verbose: bool = True,
) -> GenStats:
    """Convenience one-shot helper. Loads the model and runs one generation."""
    engine = OptiqEngine(model_path)
    stats = engine.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        depth=depth,
        temperature=temperature,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        chat_template=chat_template,
    )
    if verbose:
        print(stats.summary())
    return stats
