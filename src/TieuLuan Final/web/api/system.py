"""Trạng thái máy cho trang Live Logs: GPU qua nvidia-smi, commit, Neo4j, khóa GPU."""
import socket
import subprocess
import time

from tlf.common import OUT
from tlf.mem import free_commit_gb

NVIDIA_FIELDS = "name,memory.used,memory.total,utilization.gpu,temperature.gpu"
NEO4J_ADDR = ("127.0.0.1", 7687)
GPU_LOCK = OUT / "gpu_heavy.lock"


def neo4j_port_open(timeout=0.4):
    """Chỉ thử cổng TCP để trang không treo 30 giây theo thời gian chờ mặc định của driver."""
    try:
        with socket.create_connection(NEO4J_ADDR, timeout):
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


def _commit():
    try:
        return {"ok": True, "free_gb": round(free_commit_gb(), 2)}
    except OSError as e:
        return {"ok": False, "error": str(e)}


def _gpu_lock():
    """Ai đang giữ khóa GPU nặng: cho biết LLM sẽ phải chờ làn nào."""
    try:
        return GPU_LOCK.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def status():
    return {"ts": time.strftime("%H:%M:%S"), "gpu": _gpu(), "commit": _commit(),
            "gpu_lock": _gpu_lock(), "neo4j": {"ok": neo4j_port_open()}}
