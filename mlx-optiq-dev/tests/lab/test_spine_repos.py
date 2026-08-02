"""Domain repositories — workspace, build, conversation, provenance (Phase 0)."""

from optiq.lab import events, spine


def test_create_workspace_and_get(lab_home):
    ws_id = spine.create_workspace(
        "Main lab",
        system_prompt="Be concise.",
        sampler_json={"temperature": 0.7, "max_tokens": 1024},
        tools_policy_json={"enabled": ["python"]},
        attached_files_json=["/tmp/a.txt"],
        eval_set_id="eval_1",
    )
    assert ws_id.startswith("ws_")

    row = spine.get_workspace(ws_id)
    assert row is not None
    assert row["id"] == ws_id
    assert row["name"] == "Main lab"
    assert row["system_prompt"] == "Be concise."
    assert row["sampler_json"] == {"temperature": 0.7, "max_tokens": 1024}
    assert row["tools_policy_json"] == {"enabled": ["python"]}
    assert row["attached_files_json"] == ["/tmp/a.txt"]
    assert row["eval_set_id"] == "eval_1"
    assert row["default_build_id"] is None
    assert row["coherence_flags_json"] == {}
    assert row["created_at"]
    assert row["updated_at"]

    assert spine.get_workspace("ws_missing") is None


def test_register_build_and_event(lab_home):
    bld_id = spine.register_build(
        name="Qwen 4bit",
        path="/models/qwen-4bit",
        source_hf_id="Qwen/Qwen2.5-7B",
        quant_profile="4-bit mixed",
        bpw=4.5,
        weights_gb=4.2,
        kv_bits_default=8,
        ctx_default=32768,
        adapter_stack=["clinical-v3"],
        metadata={"note": "local"},
    )
    assert bld_id.startswith("bld_")

    row = spine.get_build(bld_id)
    assert row is not None
    assert row["id"] == bld_id
    assert row["name"] == "Qwen 4bit"
    assert row["path"] == "/models/qwen-4bit"
    assert row["source_hf_id"] == "Qwen/Qwen2.5-7B"
    assert row["quant_profile"] == "4-bit mixed"
    assert row["bpw"] == 4.5
    assert row["weights_gb"] == 4.2
    assert row["kv_bits_default"] == 8
    assert row["ctx_default"] == 32768
    assert row["adapter_stack"] == ["clinical-v3"]
    assert row["metadata"] == {"note": "local"}
    assert row["created_at"]

    # Explicit build_id override
    custom = spine.register_build(
        name="Custom",
        path="/models/custom",
        build_id="bld_custom01",
    )
    assert custom == "bld_custom01"
    assert spine.get_build("bld_custom01")["name"] == "Custom"

    evs = events.iter_after(after_id=0)
    registered = [e for e in evs if e["type"] == "build.registered"]
    assert len(registered) == 2
    assert registered[0]["entity_type"] == "build"
    assert registered[0]["entity_id"] == bld_id
    assert registered[0]["payload"]["path"] == "/models/qwen-4bit"
    assert registered[1]["entity_id"] == "bld_custom01"

    assert spine.get_build("bld_missing") is None


def test_set_workspace_build_coherence_when_not_resident(lab_home):
    ws_id = spine.create_workspace("Coherence")
    bld_id = spine.register_build(name="A", path="/m/a")
    other = spine.register_build(name="B", path="/m/b")

    updated = spine.set_workspace_build(
        ws_id, bld_id, resident_build_id=other
    )
    assert updated["id"] == ws_id
    assert updated["default_build_id"] == bld_id
    assert updated["coherence_flags_json"]["model_not_resident"] is True

    # Persist check
    row = spine.get_workspace(ws_id)
    assert row["default_build_id"] == bld_id
    assert row["coherence_flags_json"]["model_not_resident"] is True

    coh = [e for e in events.iter_after(0) if e["type"] == "workspace.coherence"]
    assert len(coh) == 1
    assert coh[0]["entity_type"] == "workspace"
    assert coh[0]["entity_id"] == ws_id
    assert coh[0]["workspace_id"] == ws_id
    assert coh[0]["payload"]["model_not_resident"] is True
    assert coh[0]["payload"]["default_build_id"] == bld_id
    assert coh[0]["payload"]["resident_build_id"] == other


def test_set_workspace_build_clear_flag_when_resident_matches(lab_home):
    ws_id = spine.create_workspace("Resident ok")
    bld_id = spine.register_build(name="A", path="/m/a")
    other = spine.register_build(name="B", path="/m/b")

    # First: flag set (not resident)
    spine.set_workspace_build(ws_id, bld_id, resident_build_id=other)
    assert spine.get_workspace(ws_id)["coherence_flags_json"]["model_not_resident"] is True

    # Second: resident matches → clear flag
    updated = spine.set_workspace_build(
        ws_id, bld_id, resident_build_id=bld_id
    )
    assert updated["default_build_id"] == bld_id
    assert "model_not_resident" not in updated["coherence_flags_json"]

    row = spine.get_workspace(ws_id)
    assert "model_not_resident" not in row["coherence_flags_json"]
    assert row["default_build_id"] == bld_id


def test_upsert_conversation_with_messages_and_provenance(lab_home):
    ws_id = spine.create_workspace("Chat ws")
    envelope = {
        "build_id": "bld_x",
        "sampler": {"temperature": 0.5},
        "context_window": 8192,
        "captured_at": "2026-08-02T12:00:00Z",
    }
    data = {
        "title": "First thread",
        "model": "qwen-local",
        "workspace_id": ws_id,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there", "provenance": envelope},
        ],
    }
    conv_id = spine.upsert_conversation_from_chat_payload(data)
    assert conv_id.startswith("conv_")

    conv = spine.get_conversation(conv_id)
    assert conv is not None
    assert conv["id"] == conv_id
    assert conv["title"] == "First thread"
    assert conv["model"] == "qwen-local"
    assert len(conv["messages"]) == 2
    assert conv["messages"][0]["role"] == "user"
    assert conv["messages"][0]["content"] == "Hello"
    assert conv["messages"][0]["seq"] == 0
    assert "provenance" not in conv["messages"][0] or conv["messages"][0].get("provenance") is None
    assert conv["messages"][1]["role"] == "assistant"
    assert conv["messages"][1]["content"] == "Hi there"
    assert conv["messages"][1]["seq"] == 1
    assert conv["messages"][1]["provenance"] == envelope
    assert conv["messages"][0]["id"].startswith("msg_")
    assert conv["messages"][1]["id"].startswith("msg_")

    asst_id = conv["messages"][1]["id"]
    assert spine.get_provenance(asst_id) == envelope

    # Upsert same id replaces messages
    data2 = {
        "id": conv_id,
        "title": "Renamed",
        "model": "qwen-local",
        "workspace_id": ws_id,
        "messages": [
            {"role": "user", "content": "Only one now"},
        ],
    }
    again = spine.upsert_conversation_from_chat_payload(data2)
    assert again == conv_id
    conv2 = spine.get_conversation(conv_id)
    assert conv2["title"] == "Renamed"
    assert len(conv2["messages"]) == 1
    assert conv2["messages"][0]["content"] == "Only one now"
    assert conv2["messages"][0]["seq"] == 0
    # Old provenance gone with old message
    assert spine.get_provenance(asst_id) is None

    evs = events.iter_after(0)
    upserted = [e for e in evs if e["type"] == "conversation.upserted"]
    created = [e for e in evs if e["type"] == "message.created"]
    assert len(upserted) == 2
    assert upserted[0]["entity_id"] == conv_id
    assert upserted[0]["workspace_id"] == ws_id
    # first upsert: 2 messages; second: 1 message
    assert len(created) == 3


def test_provenance_complete_true_false_cases():
    # True: build_id path
    assert spine.provenance_complete(
        {
            "build_id": "bld_1",
            "sampler": {"temperature": 0.7},
            "context_window": 32768,
            "captured_at": "2026-08-02T00:00:00Z",
        }
    )
    # True: server_model_label path (no build_id)
    assert spine.provenance_complete(
        {
            "server_model_label": "qwen-local",
            "sampler": {},
            "context_window": 4096,
            "captured_at": "2026-08-02T00:00:00Z",
        }
    )
    # False: missing sampler
    assert not spine.provenance_complete(
        {
            "build_id": "bld_1",
            "context_window": 32768,
            "captured_at": "2026-08-02T00:00:00Z",
        }
    )
    # False: null context_window
    assert not spine.provenance_complete(
        {
            "build_id": "bld_1",
            "sampler": {},
            "context_window": None,
            "captured_at": "2026-08-02T00:00:00Z",
        }
    )
    # False: neither build_id nor server_model_label
    assert not spine.provenance_complete(
        {
            "sampler": {},
            "context_window": 1,
            "captured_at": "x",
        }
    )
    # False: empty build_id and missing label
    assert not spine.provenance_complete(
        {
            "build_id": None,
            "server_model_label": None,
            "sampler": {},
            "context_window": 1,
            "captured_at": "x",
        }
    )


def test_get_provenance_round_trip(lab_home):
    # Need a message row for integrity; insert via conversation upsert
    conv_id = spine.upsert_conversation_from_chat_payload(
        {
            "title": "prov",
            "model": "m",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        }
    )
    asst = spine.get_conversation(conv_id)["messages"][1]
    mid = asst["id"]

    envelope = {
        "server_model_label": "local-model",
        "sampler": {"temperature": 0.2},
        "context_window": 16384,
        "captured_at": "2026-08-02T15:30:00Z",
        "tok_per_sec": None,
    }
    spine.set_message_provenance(mid, envelope)

    got = spine.get_provenance(mid)
    assert got == envelope
    assert spine.provenance_complete(got)

    # Incomplete envelope still stores
    incomplete = {"build_id": "bld_only"}
    spine.set_message_provenance(mid, incomplete)
    assert spine.get_provenance(mid) == incomplete
    assert not spine.provenance_complete(incomplete)

    assert spine.get_provenance("msg_missing") is None


def test_new_id_prefixes():
    assert spine.new_id("ws_").startswith("ws_")
    assert spine.new_id("bld_").startswith("bld_")
    assert spine.new_id("msg_").startswith("msg_")
    assert spine.new_id("conv_").startswith("conv_")
    # without trailing underscore still works
    a = spine.new_id("ws")
    assert a.startswith("ws_")
    assert a != spine.new_id("ws")
