"""Machine state tests — real memory when available; TCP probe logic."""

from __future__ import annotations

from optiq.lab import machine
from optiq.lab.config import ensure_lab_dirs


def test_memory_info_real(lab_home):
    ensure_lab_dirs()
    mem = machine.memory_info()
    assert mem["total_ram_gb"] > 0
    assert mem["free_ram_gb"] >= 0
    assert mem["source"] == "psutil.virtual_memory"


def test_probe_closed_port_false():
    # Port 1 is almost never open for TCP connect on macOS clients
    assert machine.probe_tcp("127.0.0.1", 1) is False


def test_machine_state_shape(lab_home):
    ensure_lab_dirs()
    st = machine.machine_state(
        api_url="http://127.0.0.1:18080",
        lab_port=17860,
        model="/tmp/fake-model",
        api_reachable=False,
    )
    assert st["model"] == "/tmp/fake-model"
    assert "memory" in st
    assert st["ports"]["serve"]["healthy"] is False
    assert st["ports"]["lab"]["port"] == 17860
    assert "running_jobs" in st
