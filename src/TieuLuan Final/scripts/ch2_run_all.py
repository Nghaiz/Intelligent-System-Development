"""Chạy toàn bộ thực nghiệm Chương 2 tuần tự, mỗi việc một tiến trình để giải phóng bộ nhớ commit.

python scripts/ch2_run_all.py [classic|numpy|torch|keras ...]; việc đã có JSON kết quả thì bỏ qua.
"""
import os
import subprocess
import sys
import time

from tlf.common import LOGS, METRICS

DATASETS = ("housing", "vsfc", "cdc")
ENV = {**os.environ, "PYTHONPATH": ".", "PYTHONUTF8": "1", "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4"}


def jobs(groups):
    for g in groups:
        for ds in DATASETS:
            if g == "classic":
                yield f"ch2_res_{ds}_classic", ["scripts/ch2_classic.py", ds]
            else:
                yield f"ch2_res_{ds}_mlp_{g}", ["scripts/ch2_mlp_run.py", g, ds]


def run(name, argv, tries=8):
    """Windows báo cạn commit bằng cách giết tiến trình ở bất kỳ đâu, nên kết quả được xét qua tệp JSON."""
    log = LOGS / f"{name}.log"
    for i in range(tries):
        with open(log, "a", encoding="utf-8") as f:
            subprocess.run([sys.executable, *argv], env=ENV, stdout=f, stderr=subprocess.STDOUT)
        if (METRICS / f"{name}.json").exists():
            print("xong", name, flush=True)
            return
        print(f"lỗi {name}, lần {i + 1}", flush=True)
        time.sleep(60)
    print("BỎ", name, flush=True)


if __name__ == "__main__":
    for name, argv in jobs(sys.argv[1:] or ("classic", "numpy", "keras", "torch")):
        if not (METRICS / f"{name}.json").exists():
            run(name, argv)
