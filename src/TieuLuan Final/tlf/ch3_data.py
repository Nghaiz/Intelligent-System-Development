"""Chương 3 — danh mục ba bộ ảnh, chuyển dữ liệu gốc về .npy uint8, nạp bằng memory map.

SVHN và EuroSAT dùng lại tệp của Tiểu luận 01; PathMNIST tải từ Zenodo. Dữ liệu ngoài chỉ đọc bằng numpy
với allow_pickle=False, không chạy mã nào đi kèm dữ liệu.
"""
import json

import numpy as np

from .common import SEED, data_dir

ROOT = data_dir("ch3")
RAW = ROOT / "raw"
TL01 = ROOT.parents[2] / "TieuLuan 01" / "data" / "vision"
SPLITS = ("train", "val", "test")
ZEN_PATH = "https://zenodo.org/api/records/10519652/files/pathmnist.npz/content"

DATASETS = {
    "svhn": {
        "name": "SVHN", "source": "Stanford UFLDL, mục cropped digits; tệp dùng lại từ Tiểu luận 01",
        "url": "https://huggingface.co/datasets/ufldl-stanford/svhn", "license": "Chỉ dùng phi thương mại",
        "task": "Nhận dạng chữ số trên biển số nhà, 10 lớp", "size": 32,
        "classes": [str(i) for i in range(10)], "reused_from": "src/TieuLuan 01/data/vision/svhn.npz",
        "note": "Giữ đúng tệp của Tiểu luận 01 để khớp thứ tự ảnh với dự đoán đã lưu; chưa dùng tập extra.",
    },
    "eurosat": {
        "name": "EuroSAT RGB",
        "source": "Helber và cộng sự, Zenodo record 7711810; tệp dùng lại từ Tiểu luận 01",
        "url": "https://zenodo.org/records/7711810", "license": "MIT",
        "task": "Phân loại lớp phủ mặt đất từ ảnh Sentinel-2, 10 lớp", "size": 64,
        "classes": ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture",
                    "PermanentCrop", "Residential", "River", "SeaLake"],
        "labels": ["Cây hằng năm", "Rừng", "Thảm cỏ", "Đường cao tốc", "Công nghiệp", "Đồng cỏ",
                   "Cây lâu năm", "Dân cư", "Sông", "Biển, hồ"],
        "reused_from": "src/TieuLuan 01/data/vision/eurosat.npz",
        "note": "Bộ gốc không chia sẵn; dùng cách chia phân tầng 80/20 của Tiểu luận 01.",
    },
    "pathmnist": {
        "name": "PathMNIST", "source": "MedMNIST v2, Zenodo record 10519652", "reused_from": None,
        "url": "https://zenodo.org/records/10519652", "license": "CC BY 4.0",
        "task": "Phân loại mô đại trực tràng nhuộm H&E, 9 lớp (chỉ mang tính học thuật)", "size": 28,
        "classes": ["Mô mỡ", "Nền", "Mảnh vụn", "Lympho", "Chất nhầy", "Cơ trơn", "Niêm mạc thường",
                    "Mô đệm ung thư", "Biểu mô ung thư"],
        "files": {"pathmnist.npz": ZEN_PATH},
        "note": "Giữ nguyên ba phần do MedMNIST chia; tập test đến từ trung tâm khác tập train.",
    },
}


def labels(ds):
    """Tên lớp hiển thị; EuroSAT giữ tên thư mục gốc trong "classes" nên có thêm bản tiếng Việt."""
    return DATASETS[ds].get("labels", DATASETS[ds]["classes"])


def raw_sizes():
    """Dung lượng tệp gốc (MB) ghi lại trước khi xóa dữ liệu thô."""
    p = ROOT / "raw_sizes.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save(ds, parts):
    """parts: {split: (x uint8 NHWC, y int64)}; ghi data/ch3/<ds>/{x,y}_<split>.npy."""
    d = ROOT / ds
    d.mkdir(exist_ok=True)
    for split, (x, y) in parts.items():
        np.save(d / f"x_{split}.npy", x, allow_pickle=False)
        np.save(d / f"y_{split}.npy", y.astype(np.int64), allow_pickle=False)


def _stratified(y, frac, seed=SEED):
    """Trả (chỉ số giữ lại, chỉ số tách ra) sao cho mỗi lớp tách đúng tỉ lệ frac."""
    rng = np.random.default_rng(seed)
    keep, out = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        k = int(round(frac * len(idx)))
        out.append(idx[:k])
        keep.append(idx[k:])
    return np.sort(np.concatenate(keep)), np.sort(np.concatenate(out))


def import_tl01(ds):
    """Chuyển npz của Tiểu luận 01 (NCHW nén) về .npy NHWC để nạp bằng memory map; thứ tự ảnh giữ nguyên."""
    with np.load(TL01 / f"{ds}.npz", allow_pickle=False) as z:
        parts = {s: (z[f"x_{s}"].transpose(0, 2, 3, 1), z[f"y_{s}"]) for s in ("train", "test")}
    (ROOT / ds / "x_val.npy").unlink(missing_ok=True)
    (ROOT / ds / "y_val.npy").unlink(missing_ok=True)
    _save(ds, parts)


def prepare_pathmnist():
    with np.load(RAW / "pathmnist" / "pathmnist.npz", allow_pickle=False) as z:
        parts = {s: (z[f"{s}_images"], z[f"{s}_labels"].ravel()) for s in SPLITS}
    _save("pathmnist", parts)


PREPARE = {"svhn": lambda: import_tl01("svhn"), "eurosat": lambda: import_tl01("eurosat"),
           "pathmnist": prepare_pathmnist}


def splits(ds):
    """Các phần có thật của bộ dữ liệu; hai bộ dùng lại từ Tiểu luận 01 không có tập val riêng."""
    return [s for s in SPLITS if (ROOT / ds / f"y_{s}.npy").exists()]


def net_input(x):
    """PathMNIST 28×28 đệm phản chiếu thành 32×32 để cùng kiến trúc chạy được: VGG-11 có năm lần pool."""
    if x.shape[1] == 28:
        return np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), mode="reflect")
    return np.asarray(x)


def load(ds, split, mmap=True):
    """Trả (x uint8 NHWC, y int64); mặc định ánh xạ bộ nhớ, không chiếm commit cho tới khi đọc."""
    d = ROOT / ds
    x = np.load(d / f"x_{split}.npy", mmap_mode="r" if mmap else None, allow_pickle=False)
    return x, np.load(d / f"y_{split}.npy", allow_pickle=False)


def subset(y, n, seed=SEED):
    """n chỉ số phân tầng theo lớp, giữ thứ tự tăng để đọc memmap tuần tự; n=None lấy tất cả."""
    if n is None or n >= len(y):
        return np.arange(len(y))
    return np.sort(_stratified(y, n / len(y), seed)[1])
