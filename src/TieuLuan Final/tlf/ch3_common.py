"""Chương 3 — cấu hình thực nghiệm, chỉ số đánh giá và lưu kết quả từng lượt huấn luyện."""
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .common import model_dir

MODELS = model_dir("ch3")
RESULTS = MODELS / "results"
CKPT = MODELS / "ckpt"
EXAMPLES = MODELS / "examples"
for _d in (RESULTS, CKPT, EXAMPLES):
    _d.mkdir(exist_ok=True)

DS_IDS = ["svhn", "eurosat", "pathmnist"]
ARCHS = ["custom", "vgg11", "resnet18", "mobilenetv2"]
ARCH_NAME = {"custom": "CNN cơ bản", "vgg11": "VGG-11", "resnet18": "ResNet-18", "mobilenetv2": "MobileNetV2"}

# Công thức Tiểu luận 01 cho PathMNIST; tiêu bản mô không có chiều "lên trên" nên lật và chuyển vị.
FULL = dict(recipe="full", epochs=6, batch=256, lr=0.1, momentum=0.9, weight_decay=5e-4, pct_start=0.25)
# So nền tảng trên SVHN đủ dữ liệu: cùng mạng hai tầng, Adam 1e-3, không tăng cường dữ liệu.
PLATFORM = dict(recipe="plain", epochs=3, batch=64)
FRAMEWORKS = ["numpy", "keras", "pytorch"]


def evaluate(y_true, proba):
    """Độ đúng, Precision/Recall/F1 trung bình macro, F1 từng lớp và ma trận nhầm lẫn."""
    y_pred = proba.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision_macro": float(p),
            "recall_macro": float(r), "f1_macro": float(f), "f1_per_class": per_class.tolist(),
            "cm": confusion_matrix(y_true, y_pred, labels=range(proba.shape[1])).tolist()}


def result_path(run):
    return RESULTS / f"{run}.json"


def save_result(run, result, proba=None):
    """Ghi results/<run>.json và xác suất trên tập test; đổi tên nguyên tử để không ai đọc tệp dở."""
    if proba is not None:
        tmp = RESULTS / f"{run}_proba.tmp"
        with open(tmp, "wb") as f:
            np.save(f, proba.astype(np.float32), allow_pickle=False)
        os.replace(tmp, RESULTS / f"{run}_proba.npy")
    js = result_path(run)
    tmp = js.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                              default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    os.replace(tmp, js)


def load_result(run):
    p = result_path(run)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_proba(run):
    return np.load(RESULTS / f"{run}_proba.npy", allow_pickle=False)
