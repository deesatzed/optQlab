"""Partial provenance capture from chat stream context (Task 7)."""

from __future__ import annotations

from optiq.lab import chat_store, spine
from optiq.lab.provenance_capture import (
    apply_stream_provenance,
    build_partial_provenance,
)


def test_build_partial_provenance_never_fakes_tok_per_sec():
    """tok_per_sec and peak_mem_gb must be null when not measured/passed."""
    env = build_partial_provenance(
        model="qwen-local",
        sampler={"temperature": 0.7, "max_tokens": 1024},
        tools_called=[],
        context_window=8192,
    )
    assert env["tok_per_sec"] is None
    assert env["peak_mem_gb"] is None
    # Canonical keys always present
    for key in (
        "build_id",
        "quant_profile",
        "adapter_stack",
        "kv_bits",
        "sampler",
        "context_used",
        "context_window",
        "tools_enabled",
        "tools_called",
        "retrieved_chunk_ids",
        "healed",
        "retry_hits",
        "thinking_used",
        "thinking_budget",
        "tok_per_sec",
        "peak_mem_gb",
        "server_model_label",
        "captured_at",
    ):
        assert key in env
    assert env["server_model_label"] == "qwen-local"
    assert env["sampler"] == {"temperature": 0.7, "max_tokens": 1024}
    assert env["tools_called"] == []
    assert env["captured_at"] is not None
    assert isinstance(env["captured_at"], str)
    assert len(env["captured_at"]) > 0


def test_build_partial_provenance_completeness_with_required_fields():
    """server_model_label + sampler + context_window + captured_at => complete."""
    env = build_partial_provenance(
        model="local-model",
        sampler={"temperature": 0.2},
        context_window=16384,
        captured_at="2026-08-02T15:30:00Z",
    )
    assert env["server_model_label"] == "local-model"
    assert env["sampler"] == {"temperature": 0.2}
    assert env["context_window"] == 16384
    assert env["captured_at"] == "2026-08-02T15:30:00Z"
    assert env["tok_per_sec"] is None
    assert spine.provenance_complete(env) is True


def test_apply_stream_provenance_none_chat_id_is_noop(lab_home):
    """chat_id=None must not write anything."""
    env = build_partial_provenance(
        model="m",
        sampler={},
        context_window=1,
        captured_at="t",
    )
    result = apply_stream_provenance(
        None,
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        env,
    )
    assert result is None
    # No conversations created
    assert chat_store.list_chat_records() == []


def test_apply_stream_provenance_attaches_to_assistant(lab_home):
    """With chat_id, save messages and attach provenance to last assistant."""
    chat_id = "chat_streamprov1"
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    env = build_partial_provenance(
        model="qwen-local",
        sampler={"temperature": 0.5, "max_tokens": 256},
        tools_called=[{"name": "python", "healed": False}],
        healed=False,
        retry_hits=0,
        context_window=8192,
        captured_at="2026-08-02T18:00:00Z",
    )
    assert spine.provenance_complete(env)

    message_id = apply_stream_provenance(chat_id, messages, env)
    assert message_id is not None
    assert message_id.startswith("msg_")

    # Conversation persisted via chat_store dual-write
    loaded = chat_store.load_chat_record(chat_id)
    assert loaded is not None
    assert loaded["id"] == chat_id
    assert len(loaded["messages"]) == 2
    assert loaded["messages"][-1]["role"] == "assistant"

    got = spine.get_provenance(message_id)
    assert got is not None
    assert got["server_model_label"] == "qwen-local"
    assert got["sampler"] == {"temperature": 0.5, "max_tokens": 256}
    assert got["context_window"] == 8192
    assert got["tools_called"] == [{"name": "python", "healed": False}]
    assert got["tok_per_sec"] is None
    assert got["peak_mem_gb"] is None
    assert got["captured_at"] == "2026-08-02T18:00:00Z"
    assert spine.provenance_complete(got)

    # Message id matches last assistant in conversation
    conv = spine.get_conversation(chat_id)
    assert conv is not None
    asst = [m for m in conv["messages"] if m["role"] == "assistant"][-1]
    assert asst["id"] == message_id
    assert asst.get("provenance") == got
