"""API-visible model variants for ``optiq serve`` (``model:think`` / ``model:no-think``).

A client selects a variant by suffixing the model id it sends, e.g.
``Qwen3.6-35B-A3B-OptiQ-4bit:no-think``. The suffix is stripped before the
model is resolved and loaded (so it maps to the *real* model at zero extra
memory), and the variant's ``chat_template_kwargs`` are applied to that one
request. Reasoning models thus expose thinking on/off *by name*, so any
OpenAI-compatible client can pick it without having to send the non-standard
``chat_template_kwargs`` body field (which many clients don't expose).

Design: it patches ``APIHandler.validate_model_parameters`` (which runs right
after the body is parsed into ``requested_model`` / ``chat_template_kwargs`` and
before the model is loaded) to rewrite the model id and merge the variant's
template kwargs. ``/v1/models`` is extended to advertise the served model's
variants for discoverability. Unknown suffixes are left untouched, so a model
id that legitimately contains a colon is unaffected.
"""

from __future__ import annotations

import io
import json

# Variant name -> what it applies. ``template`` is merged into the request's
# ``chat_template_kwargs`` (``enable_thinking`` is the Qwen3.x reasoning toggle;
# a no-op on templates that ignore it). ``sampling`` overrides the request's
# sampler attributes (temperature / top_p / top_k / min_p) on the handler, so a
# named profile picks a decoding preset without the client setting those fields.
_VARIANTS: dict[str, dict] = {
    "think": {"template": {"enable_thinking": True}},
    "no-think": {"template": {"enable_thinking": False}},
    "nothink": {"template": {"enable_thinking": False}},
    "precise": {"sampling": {"temperature": 0.0}},
    "creative": {"sampling": {"temperature": 0.8, "top_p": 0.95}},
    "balanced": {"sampling": {"temperature": 0.4, "top_p": 0.9}},
}

# Sampler attributes on APIHandler a ``sampling`` profile may set.
_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p")

# Variants advertised in /v1/models (canonical spellings only).
_ADVERTISED = ("think", "no-think", "precise", "creative")

_installed = False


def split_variant(model_id):
    """Return ``(base_model, variant_config)`` when ``model_id`` ends in a known
    ``:variant`` suffix, else ``(model_id, None)``. Splits on the *last* colon so
    HF repo ids and local paths that contain a colon elsewhere are unaffected."""
    if not isinstance(model_id, str):
        return model_id, None
    base, sep, suffix = model_id.rpartition(":")
    if sep and suffix in _VARIANTS:
        return base, _VARIANTS[suffix]
    return model_id, None


def install(server_mod) -> None:
    """Patch ``mlx_lm.server.APIHandler`` to honor ``model:variant`` selection."""
    global _installed
    if _installed:
        return

    Handler = server_mod.APIHandler

    # 1) Strip the variant suffix + apply its template kwargs, before load.
    _orig_validate = Handler.validate_model_parameters

    def validate_model_parameters(self):
        base, cfg = split_variant(getattr(self, "requested_model", None))
        if cfg is not None:
            self.requested_model = base
            # Template kwargs: variant (the named selection) wins; other
            # request-supplied chat_template_kwargs are preserved.
            tmpl = cfg.get("template")
            if tmpl:
                merged = dict(getattr(self, "chat_template_kwargs", None) or {})
                merged.update(tmpl)
                self.chat_template_kwargs = merged
            # Sampling profile: override the handler's sampler attributes. Set
            # before _orig_validate so the new values get range-validated.
            for k, v in (cfg.get("sampling") or {}).items():
                if k in _SAMPLING_KEYS:
                    setattr(self, k, v)
        return _orig_validate(self)

    Handler.validate_model_parameters = validate_model_parameters

    # 2) Advertise the served model's variants in /v1/models. The upstream
    #    handler writes headers+body straight to the socket and sets no
    #    Content-Length (close-delimited), so we capture its output, splice the
    #    variant entries into the JSON body, and re-emit. Any failure falls back
    #    to the original bytes unchanged — the listing must never break.
    _orig_models = Handler.handle_models_request

    def handle_models_request(self):
        buf = io.BytesIO()
        real_wfile = self.wfile
        self.wfile = buf
        try:
            _orig_models(self)
        finally:
            self.wfile = real_wfile
        raw = buf.getvalue()
        try:
            head, sep, body = raw.partition(b"\r\n\r\n")
            data = json.loads(body)
            served = getattr(self.response_generator.cli_args, "model", None)
            if served and isinstance(data.get("data"), list):
                have = {m.get("id") for m in data["data"]}
                for v in _ADVERTISED:
                    vid = f"{served}:{v}"
                    if vid not in have:
                        data["data"].append(
                            {"id": vid, "object": "model", "created": self.created}
                        )
                raw = head + sep + json.dumps(data).encode()
        except Exception:  # noqa: BLE001 — never break /v1/models
            pass
        real_wfile.write(raw)
        real_wfile.flush()

    Handler.handle_models_request = handle_models_request

    _installed = True
