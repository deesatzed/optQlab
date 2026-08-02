"""Zero-config peer discovery for the OptiQ cluster via macOS Bonjour.

The whole seamless story lives here: `pip install mlx-optiq` on two Macs,
`optiq cluster up` on each to advertise an ``_optiq._tcp`` service, and from
either machine `optiq cluster peers` browses Bonjour and resolves peers — no
IPs, no hostfiles, no SSH-key dance. Uses the native ``dns-sd`` (always present
on macOS, goes through the system mDNSResponder) so there are no extra deps,
and ``.local`` names resolve through the OS resolver.
"""

from __future__ import annotations

import platform
import re
import select
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

SERVICE = "_optiq._tcp"
DEFAULT_PORT = 51737

# Link preference when a peer answers on several interfaces: use the fastest.
_LINK_PREF = {"Thunderbolt": 0, "Ethernet": 1, "Wi-Fi": 2, "Network": 3}


@dataclass
class Peer:
    instance: str
    hostname: str                     # e.g. "mac2.local."
    port: int = 0
    txt: dict = field(default_factory=dict)
    addresses: list[str] = field(default_factory=list)
    link: str = "?"                   # "Thunderbolt" / "Wi-Fi" / ...
    preferred_address: str = ""       # address on the fastest available link
    is_self: bool = False

    @property
    def chip(self) -> str:
        return self.txt.get("chip", "?")

    @property
    def ram_gb(self) -> str:
        return self.txt.get("ram", "?")


def _collect(args: list[str], seconds: float) -> list[str]:
    """Run ``dns-sd <args>`` and collect stdout lines for ``seconds``, then stop.
    dns-sd is a long-running browser, so we time-box it ourselves.

    dns-sd block-buffers stdout when it isn't writing to a terminal, so a plain
    pipe hides every record until the buffer fills. We give it a pseudo-tty so
    it line-buffers and each Add/Reply arrives immediately.
    """
    import os
    import pty

    try:
        master, slave = pty.openpty()
    except OSError:
        return []
    try:
        p = subprocess.Popen(["dns-sd", *args], stdout=slave,
                             stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        os.close(master)
        os.close(slave)
        return []
    os.close(slave)

    lines: list[str] = []
    buf = b""
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            r, _, _ = select.select([master], [], [], remaining)
            if not r:
                continue
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                lines.append(raw.decode(errors="replace").rstrip("\r"))
    finally:
        p.terminate()
        try:
            p.wait(timeout=1)
        except Exception:
            p.kill()
        os.close(master)
    return lines


def advertise(port: int = DEFAULT_PORT, txt: Optional[dict] = None,
              instance: Optional[str] = None) -> subprocess.Popen:
    """Register this node as an ``_optiq._tcp`` service. Returns the long-lived
    dns-sd process (the caller/daemon holds it open to keep advertising)."""
    instance = instance or f"optiq@{socket.gethostname().split('.')[0]}"
    txt = txt or {}
    args = ["dns-sd", "-R", instance, SERVICE, "local", str(port)]
    args += [f"{k}={v}" for k, v in txt.items()]
    return subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def _browse_instances(seconds: float) -> list[str]:
    names: list[str] = []
    for line in _collect(["-B", SERVICE, "local"], seconds):
        # ... Add  <flags> <if> local.  _optiq._tcp.  <Instance Name>
        m = re.search(r"\bAdd\b.*?_optiq\._tcp\.\s+(.+?)\s*$", line)
        if m:
            name = m.group(1).strip()
            if name and name not in names:
                names.append(name)
    return names


def _resolve(instance: str, seconds: float) -> tuple[Optional[str], int, dict]:
    """dns-sd -L: instance -> (hostname, port, txt-dict)."""
    hostname, port, txt = None, 0, {}
    for line in _collect(["-L", instance, SERVICE, "local"], seconds):
        m = re.search(r"can be reached at\s+(\S+):(\d+)", line)
        if m:
            hostname = m.group(1).rstrip(".") + "." if not m.group(1).endswith(".") \
                else m.group(1)
            port = int(m.group(2))
            # TXT often trails on the same line after the (interface) note.
            tail = line.split(")", 1)[-1]
            for kv in re.findall(r"(\w+)=([^\s]+)", tail):
                txt[kv[0]] = kv[1]
        else:
            for kv in re.findall(r"(\w+)=([^\s]+)", line):
                txt.setdefault(kv[0], kv[1])
    return hostname, port, txt


def _addresses(hostname: str) -> list[str]:
    """Resolve a .local hostname to IPs via the OS resolver (mDNS on macOS)."""
    ips: list[str] = []
    try:
        for fam, _, _, _, sockaddr in socket.getaddrinfo(hostname.rstrip("."), None):
            ip = sockaddr[0].split("%")[0]  # strip zone id on link-local v6
            if ip not in ips:
                ips.append(ip)
    except socket.gaierror:
        pass
    return ips


def discover(timeout: float = 3.0, include_self: bool = True) -> list[Peer]:
    """Browse + resolve all ``_optiq._tcp`` peers on every interface."""
    from .net import local_interfaces, classify_peer_address

    ifaces = local_interfaces()
    # Detect self by address overlap, not hostname: macOS ComputerName,
    # LocalHostName and gethostname() routinely disagree, so matching names is
    # unreliable. Our own advertised service resolves to our own interface IPs.
    local_addrs = {"127.0.0.1", "::1"}
    local_addrs |= {i.inet for i in ifaces if i.inet}
    try:
        for _f, _s, _p, _c, sa in socket.getaddrinfo(socket.gethostname(), None):
            local_addrs.add(sa[0].split("%")[0])
    except socket.gaierror:
        pass

    peers: list[Peer] = []
    for inst in _browse_instances(timeout):
        hostname, port, txt = _resolve(inst, min(2.0, timeout))
        if not hostname:
            continue
        addrs = _addresses(hostname)
        is_self = bool(set(addrs) & local_addrs)
        # For a remote peer, prefer the FASTEST link it's reachable on: a peer
        # answers on every interface (Wi-Fi + Thunderbolt), and we want to use
        # the Thunderbolt path. Classify all addresses and rank them.
        link = "self" if is_self else "?"
        preferred = ""
        if not is_self:
            scored = []
            for a in addrs:
                k = classify_peer_address(a, ifaces)
                if k != "?":
                    scored.append((_LINK_PREF.get(k, 9), a, k))
            if scored:
                scored.sort(key=lambda t: t[0])
                _, preferred, link = scored[0]
            elif any(not a.startswith(("fe80", "::1", "127.")) for a in addrs):
                link = "Network"
                preferred = next(
                    (a for a in addrs
                     if not a.startswith(("fe80", "::1", "127."))), "")
        peers.append(Peer(instance=inst, hostname=hostname, port=port, txt=txt,
                          addresses=addrs, link=link,
                          preferred_address=preferred, is_self=is_self))
    if not include_self:
        peers = [p for p in peers if not p.is_self]
    return peers


def node_txt() -> dict:
    """TXT record describing this node: chip, RAM, OptiQ version."""
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, timeout=4).stdout.strip()
        chip = chip.replace("Apple ", "").replace(" ", "")
    except Exception:
        chip = platform.processor() or "?"
    try:
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=4).stdout.strip())
        # GiB, not decimal GB: a "36 GB" Mac reports 38.65e9 bytes, and dividing
        # by 1e9 would advertise it as 39 GB — overstating every node and the
        # cluster total.
        ram = str(round(mem / 1024 ** 3))
    except Exception:
        ram = "?"
    try:
        from optiq import __version__ as ver
    except Exception:
        ver = "?"
    import getpass
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    return {"chip": chip, "ram": ram, "ver": ver, "user": user}


def build_ring_hostfile(peers: list, self_ip: str) -> list:
    """Build an mlx.launch ring hostfile from this node + discovered peers.

    Rank 0 is this node (ssh 127.0.0.1). Each peer contributes its
    ``user@hostname`` ssh target and its preferred (fastest-link) address as the
    ring IP. Peers without a resolved address or user are skipped.
    """
    hosts = [{"ssh": "127.0.0.1", "ips": [self_ip]}]
    for p in peers:
        ip = p.preferred_address or next(
            (a for a in p.addresses if not a.startswith(("fe80", "::1", "127."))), "")
        user = p.txt.get("user", "")
        host = p.hostname.rstrip(".")
        if not ip or not host:
            continue
        ssh = f"{user}@{host}" if user else host
        hosts.append({"ssh": ssh, "ips": [ip]})
    return hosts
