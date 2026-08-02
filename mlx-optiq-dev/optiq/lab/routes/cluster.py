"""OptiQ Lab — cluster panel: live Thunderbolt-ring topology + serve control.

Reuses optiq.cluster (discovery, net, serve). The topology viz polls
``/api/cluster/peers``; ``/api/cluster/serve`` launches pipeline-parallel
serving as a subprocess and the Lab chat can be pointed at its endpoint.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request

bp = Blueprint("cluster", __name__)

# One cluster-serve subprocess at a time, tracked at module scope.
_serve = {"proc": None, "endpoint": None, "model": None, "port": None,
          "log": None, "prev_api_url": None, "prev_model": None,
          "activated": False}


def _activate_backend(app, endpoint: str, model: str) -> None:
    """Point the Lab's inference backend at the cluster, so the normal Chat and
    Arena run across the ring. Remembers the previous (single-Mac) backend."""
    _serve["prev_api_url"] = app.config.get("OPTIQ_API_URL")
    _serve["prev_model"] = app.config.get("OPTIQ_LOADED_MODEL")
    app.config["OPTIQ_API_URL"] = endpoint
    app.config["OPTIQ_LOADED_MODEL"] = model


def _endpoint_live(port: int) -> bool:
    """True once the cluster's OpenAI endpoint actually answers (all shards
    loaded). A big model takes minutes, so the backend is only switched once
    this is true — otherwise Chat would hit a not-yet-listening port."""
    try:
        import urllib.request
        urllib.request.urlopen(f"http://127.0.0.1:{port}/cluster/info", timeout=1)
        return True
    except Exception:
        return False


def _activate_once(app) -> None:
    """Switch the Lab backend to the cluster exactly once, when it goes live."""
    if _serve.get("activated") or not _serve.get("endpoint"):
        return
    _activate_backend(app, _serve["endpoint"], _serve["model"])
    _serve["activated"] = True


def _restore_backend(app) -> None:
    """Revert to the single-Mac backend when the cluster stops."""
    app.config["OPTIQ_API_URL"] = _serve.get("prev_api_url") or "http://127.0.0.1:8080"
    app.config["OPTIQ_LOADED_MODEL"] = _serve.get("prev_model")


@bp.route("/cluster")
def cluster_page():
    return render_template("cluster.html", page_title="Cluster", section="cluster")


def _serving() -> bool:
    p = _serve["proc"]
    return p is not None and p.poll() is None


_MEM_CACHE: dict = {}   # key -> (epoch, dict); the topology polls every 4s

# macOS never shrinks its swap file. A Mac that once thrashed keeps a huge swap
# and a depressed free-memory figure until it is rebooted — it will refuse to
# hold its share of a model, and nothing on screen explains why. Past this much
# swap we tell the user to restart that node.
SWAP_DEGRADED_GIB = 4.0


def _parse_swap(text: str) -> float:
    """`total = 36864.00M  used = 35780.88M  free = ...` -> used, in GiB."""
    parts = text.replace("=", " ").split()
    for i, p in enumerate(parts):
        if p == "used" and i + 1 < len(parts):
            v = parts[i + 1]
            mult = {"M": 1 / 1024, "G": 1.0, "K": 1 / 1024 ** 2}.get(v[-1:], 0)
            try:
                return round(float(v[:-1]) * mult, 1)
            except Exception:
                return 0.0
    return 0.0


def _node_mem(ssh_target: str | None) -> dict:
    """Free memory + swap pressure for a node (self when ssh_target is None).

    Availability uses the same ``total - (wired + anonymous + compressed)`` rule
    the sharder uses, so the page can never promise memory the preflight will
    then refuse.
    """
    from optiq.cluster.pipeline import GIB, parse_available_bytes

    key = ssh_target or "@self"
    now = time.time()
    hit = _MEM_CACHE.get(key)
    if hit and now - hit[0] < 15:
        return hit[1]
    cmd = "vm_stat; echo ---; sysctl -n vm.swapusage; echo ---; sysctl -n hw.memsize"
    try:
        if ssh_target:
            out = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=4", "-o", "BatchMode=yes",
                 ssh_target, cmd], capture_output=True, text=True, timeout=8).stdout
        else:
            out = subprocess.run(["sh", "-c", cmd], capture_output=True,
                                 text=True, timeout=6).stdout
        vm, swaptxt, memtxt = (out.split("---") + ["", ""])[:3]
        total = float(memtxt.strip() or 0)
        ab = parse_available_bytes(vm, total) if total else None
        avail = round(ab / GIB, 1) if ab is not None else None
        swap = _parse_swap(swaptxt)
        info = {"avail": avail, "swap": swap,
                "degraded": swap >= SWAP_DEGRADED_GIB}
    except Exception:
        info = {"avail": None, "swap": None, "degraded": False}
    _MEM_CACHE[key] = (now, info)
    return info


@bp.route("/api/cluster/peers")
def cluster_peers():
    """Discovered nodes (self + peers) with link type, measured RTT, and the
    memory each can actually spare, for the topology viz."""
    from optiq.cluster import discover, node_txt
    from optiq.cluster.net import local_interfaces, measure_rtt_ms
    from optiq.cluster.pipeline import (_self_available_bytes, _headroom_bytes, GIB)

    self_txt = node_txt()
    tb = [i for i in local_interfaces() if i.is_thunderbolt and i.inet]
    headroom = _headroom_bytes() / GIB
    me = _node_mem(None)
    nodes = [{
        "name": socket.gethostname().split(".")[0],
        "chip": self_txt.get("chip", "?"), "ram": self_txt.get("ram", "?"),
        "avail": me["avail"], "swap": me["swap"], "degraded": me["degraded"],
        "link": "self", "rtt": 0.0, "is_self": True, "address": "",
    }]
    try:
        peers = discover(timeout=3.0, include_self=True)
    except Exception as e:  # discovery best-effort
        return jsonify({"ok": False, "error": str(e), "nodes": nodes})
    # A crashed `cluster up` can orphan its dns-sd child, leaving a stale mDNS
    # record. Two records for one Mac would show it twice and double-count the
    # cluster's RAM — dedupe on the address that actually carries traffic.
    seen = {n["address"] for n in nodes} | {n["name"] for n in nodes}
    for p in peers:
        if p.is_self:
            continue
        addr = p.preferred_address or ""
        host = p.hostname.rstrip(".")
        if (addr and addr in seen) or host.split(".")[0] in seen:
            continue
        seen.add(addr or host.split(".")[0])
        seen.add(host.split(".")[0])
        rtt = measure_rtt_ms(addr, count=2) if addr else float("inf")
        user = p.txt.get("user", "")
        pm = _node_mem(f"{user}@{host}" if user else host)
        nodes.append({
            "name": host.split(".")[0],
            "chip": p.chip, "ram": p.ram_gb, "link": p.link,
            "avail": pm["avail"], "swap": pm["swap"], "degraded": pm["degraded"],
            "rtt": None if rtt == float("inf") else round(rtt, 2),
            "is_self": False, "address": addr,
        })
    return jsonify({
        "ok": True, "nodes": nodes, "thunderbolt_up": bool(tb),
        "headroom_gb": headroom, "swap_degraded_gb": SWAP_DEGRADED_GIB,
        "serving": _serving(), "endpoint": _serve["endpoint"],
        "model": _serve["model"],
    })


def _cluster_cwd() -> str | None:
    """Directory to run ``optiq cluster serve`` from.

    The CLI resolves the per-host python relative to this. Prefer the
    virtualenv this Lab is running inside, which is by definition an
    environment that has mlx-optiq; ``OPTIQ_CLUSTER_CWD`` overrides. It used to
    hardcode a developer's checkout path, which exists on no user's machine.
    """
    env = os.environ.get("OPTIQ_CLUSTER_CWD")
    if env and Path(env).is_dir():
        return env
    venv_parent = Path(sys.prefix).resolve().parent
    if venv_parent.is_dir():
        return str(venv_parent)
    return None


@bp.route("/api/cluster/serve", methods=["POST"])
def cluster_serve_start():
    data = request.get_json(force=True) or {}
    model = (data.get("model") or "").strip()
    if not model:
        return jsonify({"ok": False, "error": "model required"}), 400
    if _serving():
        return jsonify({"ok": False, "error": "already serving",
                        "endpoint": _serve["endpoint"]}), 409
    port = int(data.get("port", 8100))
    link = data.get("link", "measured")
    cwd = _cluster_cwd()
    if not cwd:
        return jsonify({"ok": False, "error": (
            "Could not locate a cluster working dir with a mlx-optiq venv. "
            "Set OPTIQ_CLUSTER_CWD, or run `optiq cluster serve` in a terminal.")}), 400

    # Append, don't truncate: a serve that loaded and then died is exactly the
    # run whose evidence you want afterwards, and the next attempt used to wipe
    # it. Keep the last ~2 MB.
    log_path = Path(current_app.config["OPTIQ_LAB_PATHS"].root) / "cluster_serve.log"
    try:
        if log_path.exists() and log_path.stat().st_size > 2_000_000:
            tail = log_path.read_bytes()[-1_000_000:]
            log_path.write_bytes(tail)
    except Exception:
        pass
    logf = open(log_path, "a")
    logf.write(f"\n\n===== serve {model} @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    logf.flush()
    cmd = [sys.executable, "-m", "optiq.cli", "cluster", "serve",
           "--model", model, "--port", str(port), "--link", link]
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT)
    _serve.update({"proc": proc, "endpoint": f"http://127.0.0.1:{port}",
                   "model": model, "port": port, "log": str(log_path),
                   "activated": False})

    # Do NOT switch the Lab backend yet — a large model takes minutes to page in
    # across the ring, and pointing Chat at a not-yet-listening port is the
    # "connection refused" bug. The client polls /api/cluster/status; the backend
    # switches only when the endpoint answers (or errors out on an early crash).
    endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
    for _ in range(8):  # catch fast failures (bad model id, gate refusal, scp)
        if proc.poll() is not None:
            tail = Path(log_path).read_text()[-1500:]
            _serve.update({"proc": None, "endpoint": None})
            return jsonify({"ok": False, "error": "serve exited during startup",
                            "log": tail}), 400
        if _endpoint_live(port):
            _activate_once(current_app._get_current_object())
            return jsonify({"ok": True, "status": "ready",
                            "endpoint": endpoint, "model": model})
        time.sleep(1)
    return jsonify({"ok": True, "status": "loading", "endpoint": endpoint,
                    "model": model, "note": "loading shards across the ring…"})


@bp.route("/api/cluster/status")
def cluster_status():
    """Readiness for the topology poll: is the ring alive, and has its endpoint
    come up? Switches the Lab backend to the cluster the moment it goes live."""
    proc = _serve["proc"]
    if proc is None:
        return jsonify({"serving": False, "ready": False})
    if proc.poll() is not None:  # crashed / exited while loading
        tail = ""
        try:
            tail = Path(_serve["log"]).read_text()[-1500:]
        except Exception:
            pass
        _restore_backend(current_app._get_current_object())
        _serve.update({"proc": None, "endpoint": None, "activated": False})
        return jsonify({"serving": False, "ready": False,
                        "error": "serve exited", "log": tail})
    ready = _endpoint_live(_serve["port"])
    if ready:
        _activate_once(current_app._get_current_object())
    return jsonify({"serving": True, "ready": ready,
                    "endpoint": _serve["endpoint"], "model": _serve["model"]})


@bp.route("/api/cluster/stop", methods=["POST"])
def cluster_serve_stop():
    p = _serve["proc"]
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
    # mlx.launch spawns child ranks; sweep stragglers.
    subprocess.run(["pkill", "-f", "optiq_cluster_serve_run"], check=False)
    _restore_backend(current_app._get_current_object())
    _serve.update({"proc": None, "endpoint": None, "model": None,
                   "activated": False})
    return jsonify({"ok": True})
