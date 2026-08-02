"""Reasoning-channel stripping for eval answer extraction.

Some model families wrap their user-facing answer in a "channel" structure and
emit their chain-of-thought in a separate channel that must be discarded before
answer extraction:

* gpt-oss (harmony): ``<|channel|>analysis<|message|>…<|end|>
  <|start|>assistant<|channel|>final<|message|>ANSWER<|return|>``
* Gemma-4: a ``<channel|>thought … <channel|>`` block.

``strip_to_final_channel`` returns only the final user-facing text so the
per-task extractors (GSM8K number, MMLU letter, HumanEval code block, …) don't
trip over the reasoning trace. It's a no-op for models that don't use channels.
"""

from __future__ import annotations

def concise_template_kwargs(tokenizer) -> dict:
    """Return the ``apply_chat_template`` kwargs that make a model answer
    concisely (suppressing or minimising its reasoning trace), probing what the
    tokenizer's chat template actually accepts.

    * Qwen3.5/3.6 honour ``enable_thinking=False``.
    * gpt-oss (harmony) ignores ``enable_thinking`` but honours
      ``reasoning_effort='low'`` — without it, its analysis channel blows past
      the eval token budget and truncates before the final answer.

    Unsupported kwargs are silently dropped, so the result is safe to splat into
    ``apply_chat_template`` for any model.
    """
    out: dict = {}
    if not getattr(tokenizer, "chat_template", None):
        return out
    probe = [{"role": "user", "content": "x"}]
    for key, val in (("enable_thinking", False), ("reasoning_effort", "low")):
        try:
            tokenizer.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=True, **{key: val}
            )
            out[key] = val
        except (TypeError, ValueError):
            pass
    return out


_HARMONY_FINAL = "<|channel|>final<|message|>"
_HARMONY_ANALYSIS = "<|channel|>analysis<|message|>"
_HARMONY_CTRL = ("<|return|>", "<|end|>", "<|start|>", "<|channel|>", "<|message|>")


def strip_to_final_channel(text: str) -> str:
    """Return only the final-channel answer from a channel-structured output.

    No-op when the text contains no recognised channel markers.
    """
    if not text:
        return text
    t = text

    # gpt-oss harmony: keep what follows the LAST 'final' channel marker, cut at
    # the next control token.
    if _HARMONY_FINAL in t:
        t = t.rsplit(_HARMONY_FINAL, 1)[1]
        cut = len(t)
        for stop in _HARMONY_CTRL:
            i = t.find(stop)
            if i != -1:
                cut = min(cut, i)
        return t[:cut].strip()

    # gpt-oss that produced only an analysis channel (e.g. truncated before the
    # final): drop the analysis preamble so the extractor sees the tail.
    if _HARMONY_ANALYSIS in t:
        t = t.split(_HARMONY_ANALYSIS, 1)[1]

    # Gemma-4 thought channel: everything after the closing ``<channel|>``.
    if "<channel|>" in t:
        t = t.rsplit("<channel|>", 1)[1]

    # Remove any stray harmony control tokens left behind.
    for tok in _HARMONY_CTRL:
        t = t.replace(tok, " ")
    return t.strip()
