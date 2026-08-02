"""Local network-interface inspection for the OptiQ cluster.

macOS-native, no dependencies. Maps discovered peer addresses back to the
physical link they're reachable on so OptiQ can label a peer "Thunderbolt"
vs "Wi-Fi/Ethernet" and prefer the fast one — the whole point being that the
user never edits an IP or a hostfile.
"""

from __future__ import annotations

import ipaddress
import re
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class LocalIface:
    device: str          # e.g. "bridge0", "en0"
    hardware_port: str   # e.g. "Thunderbolt Bridge", "Wi-Fi"
    inet: Optional[str]  # IPv4 address on this iface, if any
    netmask: Optional[str]
    inet6: list = None   # list[(addr, prefixlen)] IPv6 addresses on this iface

    def __post_init__(self):
        if self.inet6 is None:
            self.inet6 = []

    @property
    def is_thunderbolt(self) -> bool:
        hp = self.hardware_port.lower()
        return "thunderbolt" in hp or self.device.startswith("bridge")

    @property
    def kind(self) -> str:
        hp = self.hardware_port.lower()
        if "thunderbolt" in hp or self.device.startswith("bridge"):
            return "Thunderbolt"
        if "wi-fi" in hp or "airport" in hp:
            return "Wi-Fi"
        if "ethernet" in hp:
            return "Ethernet"
        return self.hardware_port or self.device


def _hardware_ports() -> dict[str, str]:
    """device -> hardware-port name, from `networksetup`."""
    try:
        out = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            capture_output=True, text=True, timeout=6).stdout
    except Exception:
        return {}
    ports, cur = {}, None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Hardware Port:"):
            cur = line.split(":", 1)[1].strip()
        elif line.startswith("Device:"):
            dev = line.split(":", 1)[1].strip()
            if cur and dev:
                ports[dev] = cur
    return ports


def local_interfaces() -> list[LocalIface]:
    """Every interface with its hardware-port label and IPv4 (if assigned)."""
    hw = _hardware_ports()
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True,
                             timeout=6).stdout
    except Exception:
        return []
    ifaces: list[LocalIface] = []
    dev = None
    inet = mask = None
    inet6: list = []
    for line in out.splitlines():
        m = re.match(r"^(\w+):\s", line)
        if m:
            if dev is not None:
                ifaces.append(LocalIface(dev, hw.get(dev, ""), inet, mask, inet6))
            dev, inet, mask, inet6 = m.group(1), None, None, []
            continue
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+).*?netmask (0x[0-9a-fA-F]+)", line)
        if m and dev is not None:
            inet = m.group(1)
            hexmask = int(m.group(2), 16)
            mask = ".".join(str((hexmask >> (8 * i)) & 0xFF) for i in (3, 2, 1, 0))
            continue
        m = re.search(r"inet6 ([0-9a-fA-F:]+)(?:%\w+)?\s+prefixlen (\d+)", line)
        if m and dev is not None:
            inet6.append((m.group(1), int(m.group(2))))
    if dev is not None:
        ifaces.append(LocalIface(dev, hw.get(dev, ""), inet, mask, inet6))
    return ifaces


def _route_interface(ip: str) -> Optional[str]:
    """Ask the routing table which local interface actually reaches ``ip``.
    Authoritative — resolves the ambiguity when several interfaces share the
    169.254/16 link-local range (Thunderbolt bridge0 + other adapters)."""
    try:
        out = subprocess.run(["route", "-n", "get", ip],
                             capture_output=True, text=True, timeout=4).stdout
    except Exception:
        return None
    m = re.search(r"^\s*interface:\s*(\S+)", out, re.MULTILINE)
    return m.group(1) if m else None


def measure_rtt_ms(ip: str, count: int = 4) -> float:
    """Median round-trip latency to ``ip`` in milliseconds via ping. Returns
    ``inf`` if unreachable. This is the honest signal for whether a link is fast
    enough for pipeline inference — a flaky Thunderbolt link that classifies as
    'Thunderbolt' but pings at 40 ms is still too slow."""
    try:
        out = subprocess.run(["ping", "-c", str(count), "-t", "3", ip],
                             capture_output=True, text=True, timeout=count + 6).stdout
    except Exception:
        return float("inf")
    m = re.search(r"= [\d.]+/([\d.]+)/", out)  # min/avg/max
    return float(m.group(1)) if m else float("inf")


# Above this measured RTT, pipeline-parallel inference is slower than a single
# Mac (per-token round-trip dominates), so `optiq cluster serve` refuses by
# default. Thunderbolt is ~0.5-1 ms; Wi-Fi is tens of ms.
FAST_LINK_RTT_MS = 5.0


def classify_peer_address(ip: str, ifaces: Optional[list[LocalIface]] = None) -> str:
    """Return the link kind ("Thunderbolt" / "Wi-Fi" / "Ethernet" / "?") a peer
    at ``ip`` is reachable on, by matching it to the local interface whose
    subnet (IPv4 or IPv6) contains it.

    Only a real subnet match counts — we deliberately do NOT guess "Thunderbolt"
    just because TB hardware exists, because the ports are present even with the
    cable unplugged and every interface carries an fe80:: link-local. So TB is
    reported only when bridge0 actually carries an IP whose subnet holds the
    peer (i.e. the link is genuinely up)."""
    if ifaces is None:
        ifaces = local_interfaces()
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return "?"
    if addr.is_loopback:
        return "?"

    # Authoritative first: the routing table knows exactly which interface
    # reaches this peer, so a shared 169.254/16 range across bridge0 and other
    # adapters can't cause a misclassification.
    dev = _route_interface(ip)
    if dev:
        for iface in ifaces:
            if iface.device == dev:
                return iface.kind

    for iface in ifaces:
        if addr.version == 4 and iface.inet and iface.netmask:
            try:
                net = ipaddress.ip_network(f"{iface.inet}/{iface.netmask}",
                                           strict=False)
                if addr in net:
                    return iface.kind
            except ValueError:
                pass
        if addr.version == 6:
            for a6, plen in iface.inet6:
                try:
                    net6 = ipaddress.ip_network(f"{a6}/{plen}", strict=False)
                    # Skip the fe80::/64 link-local net — it exists identically
                    # on every interface, so it can't disambiguate.
                    if net6.is_link_local:
                        continue
                    if addr in net6:
                        return iface.kind
                except ValueError:
                    pass
    return "?"
