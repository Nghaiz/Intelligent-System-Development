"""Chuẩn bị ba bộ ảnh Chương 3: chép SVHN, EuroSAT từ Tiểu luận 01, tải PathMNIST; lưu .npy uint8.

    python scripts/ch3_download.py [--only-download] [--ds svhn eurosat pathmnist]
"""
import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlf.ch3_data import DATASETS, PREPARE, RAW, ROOT, raw_sizes  # noqa: E402
from tlf.common import limit_threads  # noqa: E402
from tlf.mem import require_disk, slot  # noqa: E402


def fetch(url, dest):
    """Tải về tệp .part rồi đổi tên, để lần chạy sau biết tệp nào đã trọn vẹn."""
    if dest.exists():
        print("đã có", dest.name, flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "tieuluan-final-ch3"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):     # lô nhỏ: máy hay cạn commit, lô 1 MB từng gây MemoryError
            f.write(chunk)
    tmp.replace(dest)
    print("xong", dest.name, f"{dest.stat().st_size / 2**20:.1f} MB", flush=True)


def drop_raw(ds):
    """Ghi lại dung lượng gốc cho báo cáo rồi xóa dữ liệu thô: ổ D có sàn an toàn, .npy đã đủ dùng."""
    d = RAW / ds
    if d.exists():
        sizes = raw_sizes()
        sizes[ds] = sum(p.stat().st_size for p in d.iterdir()) / 2**20
        (ROOT / "raw_sizes.json").write_text(json.dumps(sizes), encoding="utf-8")
        shutil.rmtree(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", nargs="*", default=list(DATASETS))
    ap.add_argument("--only-download", action="store_true")
    args = ap.parse_args()
    limit_threads(4)
    for ds in args.ds:
        if DATASETS[ds]["reused_from"]:
            if not (ROOT / ds / "reused.ok").exists():
                require_disk(1, f"ch3 chép {ds}")
                with slot(3, f"ch3 chép {ds}"):
                    PREPARE[ds]()
                (ROOT / ds / "reused.ok").write_text(DATASETS[ds]["reused_from"], encoding="utf-8")
        elif not (ROOT / ds / "y_train.npy").exists():
            require_disk(1, f"ch3 tải {ds}")
            for name, url in DATASETS[ds]["files"].items():
                fetch(url, RAW / ds / name)
            if args.only_download:
                continue
            with slot(2.5, f"ch3 chuẩn bị {ds}"):
                PREPARE[ds]()
        print("sẵn sàng", ds, flush=True)
        drop_raw(ds)


if __name__ == "__main__":
    main()
