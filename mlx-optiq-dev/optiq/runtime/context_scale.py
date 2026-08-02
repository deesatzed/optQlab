"""Usage-count scaling for smaller-context models behind context-aware agents.

Agents like Claude Code decide when to auto-compact the conversation by
comparing the cumulative token usage the server *reports* against the context
window they assume the model has (a Claude model is ~200k). Point such a client
at a smaller-context local model and it won't compact until far past the model's
real limit — the model overflows first, and generation fails or truncates.

``--context-scale FACTOR`` multiplies the token counts in the reported ``usage``
by FACTOR, so the client's "compact at N% of the window" logic fires at the
right *real*-token point. For a 32k model behind a client that assumes 200k, use
~6.25 (= 200000 / 32000). Only the **reported** usage is scaled — generation,
KV cache, and the actual prompt are untouched.

It patches the two usage-constructing methods on ``mlx_lm.server.APIHandler``
(``generate_response`` for non-stream, ``completion_usage_response`` for the
streamed ``include_usage`` chunk). The Anthropic ``/v1/messages`` path (Claude
Code) derives its ``input_tokens`` / ``output_tokens`` from the OpenAI response,
so it inherits the scaling from the same place.
"""

from __future__ import annotations

_installed = False


def _scale_usage(resp, factor: float):
    """Scale the ``usage`` token counts in an OpenAI-style response dict in place."""
    if not isinstance(resp, dict):
        return resp
    u = resp.get("usage")
    if isinstance(u, dict):
        for k in ("prompt_tokens", "completion_tokens"):
            v = u.get(k)
            if isinstance(v, (int, float)):
                u[k] = int(round(v * factor))
        p, c = u.get("prompt_tokens"), u.get("completion_tokens")
        if isinstance(p, int) and isinstance(c, int):
            u["total_tokens"] = p + c
    return resp


def install(server_mod, scale) -> None:
    """Patch the server to report ``usage`` token counts scaled by ``scale``.

    No-op when ``scale`` is falsy or 1.0.
    """
    global _installed
    if _installed:
        return
    try:
        factor = float(scale)
    except (TypeError, ValueError):
        return
    if factor <= 0 or factor == 1.0:
        return

    Handler = server_mod.APIHandler

    _orig_generate = Handler.generate_response

    def generate_response(self, *args, **kwargs):
        return _scale_usage(_orig_generate(self, *args, **kwargs), factor)

    Handler.generate_response = generate_response

    _orig_usage = Handler.completion_usage_response

    def completion_usage_response(self, *args, **kwargs):
        return _scale_usage(_orig_usage(self, *args, **kwargs), factor)

    Handler.completion_usage_response = completion_usage_response

    _installed = True
