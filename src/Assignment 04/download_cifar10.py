"""Tải và chuyển đổi CIFAR-10 sang `cifar10/data/cifar10.npz`.

Tệp `.npz` sau khi nén vẫn khoảng 150 MB, vượt giới hạn 100 MB cho một tệp của
GitHub, nên nó không được commit vào kho mã nguồn. Script này dựng lại tệp đó từ
bản gốc của Đại học Toronto.

Chạy:  python download_cifar10.py

Bốn tập dữ liệu còn lại (Comments, Diabetes, House Price, MNIST) đã nằm sẵn trong
kho, không cần tải gì thêm.
"""

from __future__ import annotations

import io
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
EXPECTED_BYTES = 170_498_071
HERE = Path(__file__).resolve().parent
TGZ = HERE / "cifar10" / "data" / "cifar-10-python.tar.gz"
OUT = HERE / "cifar10" / "data" / "cifar10.npz"


def download() -> None:
    """Tải có hỗ trợ tiếp tục giữa chừng, vì tệp 162 MB dễ đứt kết nối."""
    TGZ.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 9):
        have = TGZ.stat().st_size if TGZ.exists() else 0
        if have >= EXPECTED_BYTES:
            return
        req = urllib.request.Request(URL)
        if have:
            req.add_header("Range", f"bytes={have}-")
            print(f"  lần {attempt}: tiếp tục từ {have / 1e6:.1f} MB")
        else:
            print(f"  lần {attempt}: bắt đầu tải")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, \
                 open(TGZ, "ab" if have else "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
                    done = fh.tell()
                    print(f"\r  {done / 1e6:7.1f} / {EXPECTED_BYTES / 1e6:.1f} MB", end="")
            print()
        except Exception as exc:                      # nối lại ở vòng sau
            print(f"\n  đứt kết nối ({type(exc).__name__}), thử lại")
    raise RuntimeError(
        f"Không tải xong sau 8 lần. Tải thủ công từ {URL} rồi đặt vào {TGZ}")


def convert() -> None:
    xs, ys = [], []
    with tarfile.open(TGZ, "r:gz") as tf:
        for i in range(1, 6):
            raw = tf.extractfile(f"cifar-10-batches-py/data_batch_{i}").read()
            d = pickle.load(io.BytesIO(raw), encoding="bytes")
            xs.append(d[b"data"])
            ys.extend(d[b"labels"])
        raw = tf.extractfile("cifar-10-batches-py/test_batch").read()
        d = pickle.load(io.BytesIO(raw), encoding="bytes")
        x_test_flat, y_test = d[b"data"], d[b"labels"]

    # CIFAR-10 lưu ảnh dạng phẳng theo thứ tự (C, H, W); đổi về (H, W, C)
    to_img = lambda a: a.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
    np.savez_compressed(
        OUT,
        x_train=to_img(np.concatenate(xs)), y_train=np.array(ys, dtype=np.uint8),
        x_test=to_img(x_test_flat),         y_test=np.array(y_test, dtype=np.uint8))


def main() -> int:
    if OUT.exists():
        print(f"Đã có sẵn {OUT.name} ({OUT.stat().st_size / 1e6:.0f} MB), không làm gì thêm.")
        return 0
    print("Tải CIFAR-10 từ Đại học Toronto")
    download()
    print("Chuyển sang định dạng .npz")
    convert()
    with np.load(OUT) as z:
        print(f"Xong: train {z['x_train'].shape}, test {z['x_test'].shape}, "
              f"{OUT.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
