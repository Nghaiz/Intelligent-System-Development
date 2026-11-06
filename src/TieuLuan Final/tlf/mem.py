"""Cổng bộ nhớ commit: máy chỉ có ~47 GB commit, phần lớn đã bị các ứng dụng khác giữ.

Windows không báo OOM mà báo "paging file too small" (1455) hoặc access violation lúc nạp trọng số, nên mọi việc nặng
phải chờ đủ commit trống TRƯỚC khi khởi động. slot() giữ thêm một tệp đặt chỗ để hai tiến trình cùng chờ không cùng
thấy đủ bộ nhớ rồi khởi động chồng lên nhau.
"""
import ctypes
import json
import os
import shutil
import time
from contextlib import contextmanager

from .common import OUT

RESV = OUT / "mem_slots"
RESV.mkdir(parents=True, exist_ok=True)
DISK_FLOOR_GB = 25  # giữ lại trên ổ D, không để đầy ổ


class _Status(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_commit_gb():
    s = _Status()
    s.dwLength = ctypes.sizeof(_Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
        raise OSError("GlobalMemoryStatusEx thất bại")
    return s.ullAvailPageFile / 2**30


def free_disk_gb(path="D:/"):
    return shutil.disk_usage(path).free / 2**30


def require_disk(gb, what=""):
    """Dừng thẳng thay vì làm đầy ổ D. Gọi trước mỗi lần tải hoặc ghi lớn với dung lượng ước tính."""
    free = free_disk_gb()
    if free - gb < DISK_FLOOR_GB:
        raise OSError(f"[disk] {what}: cần {gb} GB, ổ D còn {free:.1f} GB, sàn an toàn {DISK_FLOOR_GB} GB")


def _reserved_gb():
    """Phần đã đặt chỗ mà các tiến trình còn sống CHƯA dùng tới; phần đã dùng nằm sẵn trong commit nên không trừ lại."""
    import psutil
    total = 0.0
    for f in RESV.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            used = psutil.Process(d["pid"]).memory_info().private / 2**30
        except (OSError, ValueError):
            continue
        except psutil.NoSuchProcess:
            f.unlink(missing_ok=True)
            continue
        total += max(0.0, d["gb"] - used)
    return total


@contextmanager
def slot(gb, what, poll=10.0, margin=0.5):
    """Chờ tới khi commit trống trừ phần đã đặt chỗ ≥ gb + margin, rồi giữ chỗ gb GB cho tới khi thoát khối."""
    require_disk(0, what)
    t0 = time.time()
    last = 0.0
    while True:
        free = free_commit_gb()
        reserved = _reserved_gb()
        if free - reserved >= gb + margin:
            break
        if time.time() - last > 300:
            print(f"[mem] {what}: cần {gb} GB, trống {free:.1f} GB, đã đặt chỗ {reserved:.1f} GB; chờ...", flush=True)
            last = time.time()
        time.sleep(poll)
    f = RESV / f"{os.getpid()}_{int(time.time() * 1000)}.json"
    f.write_text(json.dumps({"pid": os.getpid(), "gb": gb, "what": what}, ensure_ascii=False), encoding="utf-8")
    if time.time() - t0 > 5:
        print(f"[mem] {what}: chờ {time.time() - t0:.0f}s", flush=True)
    try:
        yield
    finally:
        f.unlink(missing_ok=True)
