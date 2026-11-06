"""Sinh hiệu của máy và các dịch vụ cho thanh theo dõi: GPU, commit, Neo4j, Ollama, SQLite, mô-đun các làn."""
import json
import socket
import subprocess
import threading
import time
import urllib.request

from dnx.common import OUT
from dnx.mem import free_commit_gb

from . import data, graph, store
from .lazy import LANES, present

NVIDIA_FIELDS = "name,memory.used,memory.total,utilization.gpu,temperature.gpu"
OLLAMA = "http://127.0.0.1:11434"
GPU_LOCK = OUT / "gpu_heavy.lock"
TTL_S = 1.5
_cache, _lock = {"ts": 0.0, "value": None}, threading.Lock()


def port_open(addr, timeout=0.3):
    try:
        with socket.create_connection(addr, timeout):
            return True
    except OSError:
        return False


def _gpu():
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={NVIDIA_FIELDS}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=3, check=True).stdout
        name, used, total, util, temp = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"ok": True, "name": name, "mem_used_mb": int(used), "mem_total_mb": int(total),
                "util_pct": int(util), "temp_c": int(temp)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as e:
        return {"ok": False, "error": str(e)[:160]}


def _ollama():
    if not port_open(("127.0.0.1", 11434)):
        return {"ok": False, "error": "Ollama chưa chạy"}
    try:
        with urllib.request.urlopen(OLLAMA + "/api/ps", timeout=0.6) as r:
            loaded = [m.get("name") for m in json.loads(r.read()).get("models", [])]
        return {"ok": True, "loaded": loaded}
    except (OSError, ValueError) as e:
        return {"ok": False, "error": str(e)[:160]}


def _gpu_lock():
    try:
        return GPU_LOCK.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _collect():
    return {"ts": time.strftime("%H:%M:%S"), "gpu": _gpu(), "commit_free_gb": round(free_commit_gb(), 2),
            "gpu_lock": _gpu_lock(), "ollama": _ollama(), "neo4j": graph.counts(), "db": data.summary(),
            "modules": {k: {"ready": present(k), "what": v} for k, v in LANES.items()},
            "chunks": store.chunk_count(), "runs": store.run_count(), "metrics": store.metric_count()}


def status():
    with _lock:
        if _cache["value"] is None or time.time() - _cache["ts"] > TTL_S:
            _cache.update(value=_collect(), ts=time.time())
        return _cache["value"]
