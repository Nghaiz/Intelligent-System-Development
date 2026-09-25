"""Mục 3 — nạp và tiền xử lý ba tập dữ liệu.

CIFAR-10/100: đọc thẳng từ tệp tar.gz chính thức, cache thành .npz. Khi huấn luyện,
toàn bộ tập nằm trên GPU dưới dạng uint8; augmentation (crop có padding + lật ngang)
cũng chạy trên GPU, nên không cần DataLoader.
Diabetes: bỏ dòng trùng, chia 70/15/15 phân tầng, one-hot + chuẩn hoá chỉ fit trên train.
"""
import pickle
import tarfile

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .export import DATA

SEED = 42
CIFAR_FILES = {
    "cifar10": ("cifar10/cifar-10-python.tar.gz", "cifar-10-batches-py"),
    "cifar100": ("cifar100/cifar-100-python.tar.gz", "cifar-100-python"),
}


def _read_pickle(tar, member):
    return pickle.load(tar.extractfile(member), encoding="latin1")


def _to_hwc(flat):
    return flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)


def load_cifar(name):
    """Trả về dict: x_train, y_train, x_test, y_test (ảnh uint8 NHWC), class_names;
    riêng CIFAR-100 thêm yc_train, yc_test (20 siêu lớp) và coarse_names."""
    cache = DATA / name / f"{name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: (z[k].tolist() if k.endswith("names") else z[k]) for k in z.files}
    tar_path, root = CIFAR_FILES[name]
    with tarfile.open(DATA / tar_path) as tar:
        if name == "cifar10":
            train = [_read_pickle(tar, f"{root}/data_batch_{i}") for i in range(1, 6)]
            test = _read_pickle(tar, f"{root}/test_batch")
            meta = _read_pickle(tar, f"{root}/batches.meta")
            out = {
                "x_train": _to_hwc(np.concatenate([b["data"] for b in train])),
                "y_train": np.concatenate([b["labels"] for b in train]).astype(np.int64),
                "x_test": _to_hwc(test["data"]),
                "y_test": np.array(test["labels"], dtype=np.int64),
                "class_names": meta["label_names"],
            }
        else:
            train = _read_pickle(tar, f"{root}/train")
            test = _read_pickle(tar, f"{root}/test")
            meta = _read_pickle(tar, f"{root}/meta")
            out = {
                "x_train": _to_hwc(train["data"]),
                "y_train": np.array(train["fine_labels"], dtype=np.int64),
                "yc_train": np.array(train["coarse_labels"], dtype=np.int64),
                "x_test": _to_hwc(test["data"]),
                "y_test": np.array(test["fine_labels"], dtype=np.int64),
                "yc_test": np.array(test["coarse_labels"], dtype=np.int64),
                "class_names": meta["fine_label_names"],
                "coarse_names": meta["coarse_label_names"],
            }
    np.savez(cache, **out)
    return out


def split_train_val(y, n_val=5000, seed=SEED):
    """Tách n_val ảnh làm validation, phân tầng theo nhãn. Trả về (idx_train, idx_val)."""
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=n_val, stratify=y, random_state=seed)


class GPUImageData:
    """Một tập ảnh nằm trọn trên GPU; chuẩn hoá bằng mean/std tính trên nhánh train."""

    def __init__(self, name, device):
        d = load_cifar(name)
        tr, va = split_train_val(d["y_train"])
        self.device = device
        self.class_names = d["class_names"]
        self.n_classes = len(self.class_names)
        self.raw = d
        to_t = lambda x: torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)
        self.x = {"train": to_t(d["x_train"][tr]), "val": to_t(d["x_train"][va]), "test": to_t(d["x_test"])}
        self.y = {"train": torch.from_numpy(d["y_train"][tr]).to(device),
                  "val": torch.from_numpy(d["y_train"][va]).to(device),
                  "test": torch.from_numpy(d["y_test"]).to(device)}
        xf = self.x["train"].float() / 255
        self.mean = xf.mean(dim=(0, 2, 3), keepdim=True)
        self.std = xf.std(dim=(0, 2, 3), keepdim=True)

    def normalize(self, xb):
        # Giữ bố cục NCHW: trên RTX 4060 channels_last đo được chậm hơn 3,6 lần với các mô hình này.
        return (xb.float() / 255 - self.mean) / self.std

    def batches(self, split, batch_size, augment=False, shuffle=False, gen=None):
        x, y = self.x[split], self.y[split]
        n = len(y)
        order = torch.randperm(n, device=self.device, generator=gen) if shuffle else torch.arange(n, device=self.device)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            xb = x[idx]
            if augment:
                xb = gpu_augment(xb, gen)
            yield self.normalize(xb), y[idx]


def gpu_augment(xb, gen=None, pad=4):
    """Crop ngẫu nhiên 32×32 từ ảnh đã đệm 0 thêm `pad` điểm ảnh, rồi lật ngang với xác suất 0,5."""
    B, C, H, W = xb.shape
    dev = xb.device
    xp = torch.nn.functional.pad(xb, (pad, pad, pad, pad))
    oy = torch.randint(0, 2 * pad + 1, (B,), device=dev, generator=gen)
    ox = torch.randint(0, 2 * pad + 1, (B,), device=dev, generator=gen)
    rows = (oy[:, None] + torch.arange(H, device=dev))[:, :, None]
    cols = (ox[:, None] + torch.arange(W, device=dev))[:, None, :]
    out = xp[torch.arange(B, device=dev)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(B, device=dev, generator=gen) < 0.5
    out[flip] = out[flip].flip(3)
    return out


# ---------------------------------------------------------------- Diabetes
DIAB_CSV = "diabetes/diabetes_prediction_dataset.csv"
DIAB_TARGET = "diabetes"
DIAB_NUMERIC = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
DIAB_BINARY = ["hypertension", "heart_disease"]
DIAB_CATEGORICAL = ["gender", "smoking_history"]


def load_diabetes_raw():
    return pd.read_csv(DATA / DIAB_CSV)


def prepare_diabetes(seed=SEED):
    """Trả về dict X/y cho train/val/test (float32) và tên cột sau mã hoá.

    Scaler và danh mục one-hot chỉ học từ nhánh train để không rò rỉ thông tin.
    """
    df = load_diabetes_raw().drop_duplicates().reset_index(drop=True)
    y = df[DIAB_TARGET].to_numpy()
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)

    cats = {c: sorted(df.loc[tr, c].unique()) for c in DIAB_CATEGORICAL}
    scaler = StandardScaler().fit(df.loc[tr, DIAB_NUMERIC])

    def encode(rows):
        part = df.loc[rows]
        cols = [scaler.transform(part[DIAB_NUMERIC]), part[DIAB_BINARY].to_numpy(float)]
        for c in DIAB_CATEGORICAL:
            cols.append(np.stack([(part[c] == v).to_numpy(float) for v in cats[c]], axis=1))
        return np.concatenate(cols, axis=1).astype(np.float32)

    names = DIAB_NUMERIC + DIAB_BINARY + [f"{c}={v}" for c in DIAB_CATEGORICAL for v in cats[c]]
    return {
        "X": {"train": encode(tr), "val": encode(va), "test": encode(te)},
        "y": {"train": y[tr].astype(np.float32), "val": y[va].astype(np.float32), "test": y[te].astype(np.float32)},
        "feature_names": names,
        "n_duplicates": int(len(load_diabetes_raw()) - len(df)),
        "n_rows": int(len(df)),
    }
