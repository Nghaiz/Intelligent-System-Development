"""Hàng đợi DQN: tối đa MAX_RUNNING lượt chạy cùng lúc, tính cả lượt khởi động từ trước; lượt đã có kết quả thì bỏ qua.

Bản thân tiến trình này không nạp torch hay numpy nên gần như không tốn commit; mỗi lượt con tự bọc dnx.mem.slot.
"""
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
MAX_RUNNING = 2
POLL = 20
JOBS = ROOT / "outputs" / "logs" / "rl_dqn_jobs.txt"


def running():
    n = 0
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if "python" in (p.info["name"] or "").lower() and "rl_train_dqn.py" in " ".join(p.info["cmdline"] or []):
                n += 1
        except (psutil.Error, TypeError):
            continue
    return n


def is_done(seed, lam):
    return (ROOT / "outputs" / "metrics" / f"rl_dqn_s{seed}_l{float(lam):g}.json").exists()


def active_tags():
    tags = set()
    for p in psutil.process_iter(["cmdline"]):
        try:
            c = p.info["cmdline"] or []
            if any("rl_train_dqn.py" in x for x in c):
                tags.add((c[c.index("--seed") + 1], c[c.index("--lam") + 1]))
        except (psutil.Error, ValueError, IndexError, TypeError):
            continue
    return tags


def main():
    jobs = [tuple(x.split()) for x in JOBS.read_text(encoding="utf-8").splitlines() if x.strip()]
    for seed, lam in jobs:
        if is_done(seed, lam) or (seed, lam) in active_tags():
            continue
        while running() >= MAX_RUNNING:
            time.sleep(POLL)
        log = open(ROOT / "outputs" / "logs" / f"rl_dqn_s{seed}_l{lam}.log", "a", encoding="utf-8")
        subprocess.Popen([sys.executable, "scripts/rl_train_dqn.py", "--seed", seed, "--lam", lam], cwd=ROOT, stdout=log,
                         stderr=subprocess.STDOUT)
        print("started", seed, lam, flush=True)
        time.sleep(POLL)  # chờ tiến trình con hiện trong bảng tiến trình trước khi đếm lại
    while running():
        time.sleep(POLL)
    missing = [j for j in jobs if not is_done(*j)]
    print("ALLDONE" if not missing else f"THIEU {missing}", flush=True)


if __name__ == "__main__":
    main()
