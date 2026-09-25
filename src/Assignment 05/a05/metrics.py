"""Mục 5 — các thước đo so sánh: chất lượng, chi phí, lỗi."""
import time

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from torch import nn


# ---------------------------------------------------------------- chất lượng
def topk_accuracy(logits, y, k=1):
    """Tỉ lệ mẫu có nhãn đúng nằm trong k lớp có điểm cao nhất."""
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())


def classification_report(logits, y):
    pred = logits.argmax(1)
    return {"acc": topk_accuracy(logits, y, 1), "top5": topk_accuracy(logits, y, 5),
            "macro_f1": float(f1_score(y, pred, average="macro"))}


def best_threshold(prob, y):
    """Ngưỡng làm F1 lớn nhất, chọn trên tập validation rồi mới áp cho test."""
    p, r, t = precision_recall_curve(y, prob)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[np.argmax(f1)])


def binary_report(prob, y, thr):
    pred = (prob >= thr).astype(int)
    return {"acc": float((pred == y).mean()),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, prob)),
            "pr_auc": float(average_precision_score(y, prob)),
            "threshold": thr,
            "confusion": confusion_matrix(y, pred).tolist()}


def binary_curves(prob, y, n_points=200):
    """Đường ROC và PR lấy mẫu lại n_points điểm để hình pgfplots không quá nặng."""
    fpr, tpr, _ = roc_curve(y, prob)
    p, r, _ = precision_recall_curve(y, prob)
    grid = np.linspace(0, 1, n_points)
    return {"fpr": grid, "tpr": np.interp(grid, fpr, tpr),
            "recall": grid, "precision": np.interp(grid, r[::-1], p[::-1])}


def grouped_confusion(y, pred, groups, n_groups):
    """Ma trận nhầm lẫn sau khi gộp lớp mịn thành nhóm (100 lớp -> 20 siêu lớp),
    chuẩn hoá theo hàng để mỗi hàng là phân bố dự đoán của một nhóm thật."""
    cm = confusion_matrix(groups[y], groups[pred], labels=range(n_groups)).astype(float)
    return cm / cm.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- chi phí
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(model, x):
    """Đếm phép nhân-cộng (MAC) của các tầng Conv và Linear cho một mẫu.

    Conv: số phần tử đầu ra × (C_in / groups × kích thước bộ lọc); Linear: in × out.
    BatchNorm, ReLU, pooling, cộng đường tắt không tính vì nhỏ hơn vài bậc.
    """
    total = [0]

    def hook(m, inp, out):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            k = int(np.prod(m.kernel_size)) * m.in_channels // m.groups
            total[0] += out[0].numel() * k
        elif isinstance(m, nn.Linear):
            total[0] += m.in_features * m.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(x[:1])
    for h in handles:
        h.remove()
    return total[0]


@torch.no_grad()
def time_inference(model, x, reps=20, warmup=5):
    """Thời gian suy luận trung bình trên một ảnh (ms), đo trên cả lô x."""
    model.eval()
    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
    for _ in range(warmup):
        model(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        model(x)
    sync()
    return (time.perf_counter() - t0) / reps / len(x) * 1000
