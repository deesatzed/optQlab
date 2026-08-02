"""Structured / JSON-constrained generation for ``optiq serve``.

Adds OpenAI ``response_format`` (``json_object`` / ``json_schema``) and the
vLLM-style ``guided_json`` / ``guided_regex`` / ``guided_choice`` extensions to
the served OpenAI-compatible API by masking the model's logits to only the
tokens that keep the output valid for the requested schema/regex.

Deliberately uses **lm-format-enforcer**, not xgrammar: lm-format-enforcer is
pure-Python (``pydantic`` + ``interegular``) with **no PyTorch dependency**, so
OptiQ stays MLX-native. It hands us the set of allowed next-token ids per step
and we apply the mask to MLX logits directly.

Wired into ``mlx_lm.server`` with a few small monkeypatches (see ``install``):

* ``ModelProvider.load`` — stash the active tokenizer (generation runs on a
  single background thread, so a module global is safe).
* ``APIHandler.handle_completion`` — parse the request body into a spec and
  carry it on a ContextVar (the handler thread that has ``self.body``).
* ``ResponseGenerator.generate`` — read the ContextVar and attach the spec to
  the per-request ``GenerationArguments`` (the object that crosses the work
  queue to the generation thread), and disable thinking for reasoning models.
* ``_make_logits_processors(args)`` — if the request carried a spec, append the
  constraining processor built from the just-loaded tokenizer.
"""

from __future__ import annotations

import json
import re

import mlx.core as mx
import numpy as np

# Cache the (expensive) per-tokenizer vocab table, keyed by tokenizer identity.
_TOKDATA_CACHE: dict = {}
_installed = False


def _hf(tokenizer):
    """Unwrap mlx-lm's TokenizerWrapper to the underlying HF tokenizer."""
    return getattr(tokenizer, "_tokenizer", tokenizer)


def _tokenizer_data(tokenizer):
    key = id(tokenizer)
    td = _TOKDATA_CACHE.get(key)
    if td is not None:
        return td
    from lmformatenforcer import TokenEnforcerTokenizerData

    hf = _hf(tokenizer)
    vocab_size = len(hf)
    token_0 = hf.encode("0")[-1]
    special = set(getattr(hf, "all_special_ids", []) or [])
    regular = []
    for tid in range(vocab_size):
        if tid in special:
            continue
        after0 = hf.decode([token_0, tid])[1:]
        plain = hf.decode([tid])
        regular.append((tid, after0, len(after0) > len(plain)))
    eos = hf.eos_token_id
    td = TokenEnforcerTokenizerData(
        regular, lambda toks: hf.decode(toks).rstrip("�"), eos, False, vocab_size
    )
    _TOKDATA_CACHE[key] = td
    return td


def _tool_choice_forces_call(tool_choice) -> bool:
    """True when ``tool_choice`` obliges the model to emit a tool call.

    OpenAI semantics: ``"required"`` (any tool) or ``{"type":"function", ...}``
    (a specific tool) both *mandate* a call; ``"auto"``/``"none"``/absent don't.
    """
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() == "required"
    if isinstance(tool_choice, dict):
        return str(tool_choice.get("type") or "").lower() == "function"
    return False


def _forced_tool_call_spec(body):
    """When ``tool_choice`` requires a call, build a JSON spec that constrains
    generation to a single ``{"name": <a valid tool>, "arguments": {...}}``
    object. The tool-healer then promotes that bare JSON to OpenAI
    ``tool_calls``. Returns ``None`` when no call is forced or no tools exist.

    mlx-lm's server ignores ``tool_choice`` entirely, so without this a
    ``"required"`` request can be answered with plain prose (an API-contract
    violation). Constraining the logits guarantees a well-formed call.
    """
    tools = body.get("tools")
    if not _tool_choice_forces_call(body.get("tool_choice")):
        return None
    if not isinstance(tools, list) or not tools:
        return None
    names = [
        t["function"]["name"]
        for t in tools
        if isinstance(t, dict)
        and isinstance(t.get("function"), dict)
        and t["function"].get("name")
    ]
    tc = body.get("tool_choice")
    if isinstance(tc, dict):  # a specific function was named — narrow to it
        fn = (tc.get("function") or {}).get("name")
        if fn:
            names = [fn]
    if not names:
        return None
    schema = {
        "type": "object",
        "properties": {
            "name": {"enum": names},
            "arguments": {"type": "object"},
        },
        "required": ["name", "arguments"],
    }
    return ("force_tool", schema)


def parse_spec(body):
    """Extract a structured-output spec ``(kind, value)`` from a request body.

    Returns ``None`` when the request asks for free-form text.
    """
    if not isinstance(body, dict):
        return None
    rf = body.get("response_format")
    if isinstance(rf, dict):
        t = rf.get("type")
        if t == "json_object":
            return ("json", None)  # any syntactically valid JSON
        if t == "json_schema":
            js = rf.get("json_schema") or {}
            schema = js.get("schema") if isinstance(js, dict) else None
            return ("json", schema)
    gj = body.get("guided_json")
    if gj is not None:
        return ("json", json.loads(gj) if isinstance(gj, str) else gj)
    if body.get("guided_regex"):
        return ("regex", body["guided_regex"])
    if body.get("guided_choice"):
        return ("choice", list(body["guided_choice"]))
    # tool_choice="required" / a specific function: force a well-formed call.
    return _forced_tool_call_spec(body)


def _make_parser(spec):
    from lmformatenforcer import JsonSchemaParser, RegexParser

    kind, val = spec
    if kind == "json":
        return JsonSchemaParser(val)  # None => any valid JSON
    if kind == "regex":
        return RegexParser(val)
    if kind == "choice":
        return RegexParser("|".join(re.escape(str(c)) for c in val))
    raise ValueError(f"unknown structured spec kind: {kind}")


class _Processor:
    """mlx-lm logits processor that masks to grammar-valid next tokens.

    mlx-lm passes a *bounded* token-context window (sized for repetition
    penalty), not the full sequence, so we can't slice the generated prefix out
    of it. Instead we track the generated tokens ourselves: the processor is
    called once per step, just before the next token is sampled, and the last
    entry of the context is the token produced on the previous step. We skip the
    first call (nothing generated yet) and append the latest token thereafter.
    """

    def __init__(self, enforcer):
        self.enf = enforcer
        self.gen: list[int] = []
        self._started = False

    def __call__(self, tokens, logits):
        if self._started and int(tokens.size) > 0:
            self.gen.append(int(tokens[-1]))
        self._started = True
        allowed = self.enf.get_allowed_tokens(self.gen).allowed_tokens
        # Size the mask to the model's actual logits width (which may be padded
        # beyond the tokenizer vocab); allowed ids index into it, pad stays -inf.
        v = logits.shape[-1]
        mask = np.full(v, -1e9, dtype=np.float32)
        if allowed:
            idx = np.asarray([a for a in allowed if 0 <= a < v], dtype=np.int64)
            if idx.size:
                mask[idx] = 0.0
        m = mx.array(mask)
        return logits + (m[None, :] if logits.ndim == 2 else m)


class _ForceToolOpen:
    """Force the first generated token to the model's ``<tool_call>`` opener.

    Used for ``tool_choice="required"``: mlx-lm ignores ``tool_choice``, so a
    model can answer in prose instead of calling. Forcing the native tool-call
    opening token commits it to a call, and because the output is then the
    model's *native* format (``<tool_call>{...}</tool_call>``), mlx-lm's
    **streaming** tool parser recognizes it — unlike a bare-JSON constraint,
    which only the non-streaming healer can recover. After the opener the model
    completes a well-formed call on its own (it is trained to).
    """

    def __init__(self, open_token_id: int):
        self.open_id = int(open_token_id)
        self._first = True

    def __call__(self, tokens, logits):
        if not self._first:
            return logits
        self._first = False
        v = logits.shape[-1]
        mask = np.full(v, -1e9, dtype=np.float32)
        if 0 <= self.open_id < v:
            mask[self.open_id] = 0.0
        m = mx.array(mask)
        return logits + (m[None, :] if logits.ndim == 2 else m)


def _tool_call_open_id(tokenizer):
    """Return the single-token id for the model's ``<tool_call>`` opener, or
    None if the model doesn't use that convention (then we fall back to the
    bare-JSON constraint + non-streaming healer)."""
    hf = _hf(tokenizer)
    for tag in ("<tool_call>",):
        ids = hf.encode(tag, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


def make_processor(tokenizer, spec):
    from lmformatenforcer import TokenEnforcer

    if spec[0] == "force_tool":
        open_id = _tool_call_open_id(tokenizer)
        if open_id is not None:
            return _ForceToolOpen(open_id)
        # Fallback: constrain to a bare tool-call JSON; the non-streaming
        # healer promotes it to tool_calls (streaming clients on such models
        # won't get enforcement, but no crash).
        return _Processor(TokenEnforcer(_tokenizer_data(tokenizer), _make_parser(("json", spec[1]))))
    td = _tokenizer_data(tokenizer)
    return _Processor(TokenEnforcer(td, _make_parser(spec)))


def install(server_mod):
    """Patch a loaded ``mlx_lm.server`` module to honor structured-output specs."""
    global _installed
    if _installed:
        return

    _orig_load = server_mod.ModelProvider.load

    def _load(self, *a, **kw):
        model, tok = _orig_load(self, *a, **kw)
        server_mod._optiq_cur_tokenizer = tok
        return model, tok

    server_mod.ModelProvider.load = _load

    # The spec lives on the APIHandler (which has the raw body); the work-queue
    # enqueue happens in ResponseGenerator.generate. Both run in the same
    # per-request handler thread, so a ContextVar carries the spec between them.
    # We then attach it to GenerationArguments, which crosses the queue to the
    # generation thread where _make_logits_processors runs.
    from contextvars import ContextVar

    server_mod._optiq_spec_var = _spec_var = ContextVar("optiq_spec", default=None)

    _orig_hc = server_mod.APIHandler.handle_completion

    def _handle_completion(self, *a, **kw):
        try:
            _spec_var.set(parse_spec(getattr(self, "body", None)))
        except Exception:
            _spec_var.set(None)
        return _orig_hc(self, *a, **kw)

    server_mod.APIHandler.handle_completion = _handle_completion

    _orig_generate = server_mod.ResponseGenerator.generate

    def _generate(self, request, generation_args, progress_callback=None):
        spec = _spec_var.get()
        if spec is not None:
            generation_args.optiq_structured = spec
            # The grammar already forbids any non-JSON/regex text, so a reasoning
            # model can't actually "think" here — but its template would open a
            # <think> block and the constrained output would be mislabeled as
            # `reasoning` instead of `content`. Disable thinking so it lands in
            # `content`. setdefault keeps an explicit client value if given.
            kw = dict(generation_args.chat_template_kwargs or {})
            kw.setdefault("enable_thinking", False)
            generation_args.chat_template_kwargs = kw
        return _orig_generate(self, request, generation_args, progress_callback)

    server_mod.ResponseGenerator.generate = _generate

    _orig_mlp = server_mod._make_logits_processors

    def _mlp(args):
        procs = list(_orig_mlp(args))
        spec = getattr(args, "optiq_structured", None)
        if spec is not None:
            tok = getattr(server_mod, "_optiq_cur_tokenizer", None)
            if tok is not None:
                procs.append(make_processor(tok, spec))
        return procs

    server_mod._make_logits_processors = _mlp
    _installed = True
