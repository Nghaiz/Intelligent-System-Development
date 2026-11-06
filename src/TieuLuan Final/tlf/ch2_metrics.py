"""Chương 2: chỉ số đánh giá, chọn ngưỡng trên validation và khoảng tin cậy bootstrap trên test."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score)

from .common import SEED


def clf_metrics(y, pred, proba=None):
    """Độ đo phân loại; với hai lớp có thêm precision/recall lớp dương và AUC."""
    out = {"accuracy": accuracy_score(y, pred), "f1_macro": f1_score(y, pred, average="macro")}
    if proba is not None and proba.shape[1] == 2:
        out["roc_auc"] = roc_auc_score(y, proba[:, 1])
        out["precision_pos"] = precision_score(y, pred, zero_division=0)
        out["recall_pos"] = recall_score(y, pred)
        out["f1_pos"] = f1_score(y, pred)
    return out


def reg_metrics(y, pred):
    return {"mae": mean_absolute_error(y, pred), "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
            "r2": r2_score(y, pred)}


def best_threshold(y_va, p_va):
    """Quét ngưỡng trên validation, lấy ngưỡng cho F1 macro cao nhất. Test không tham gia."""
    grid = np.arange(0.05, 0.71, 0.01)
    scores = [f1_score(y_va, p_va >= t, average="macro") for t in grid]
    return float(grid[int(np.argmax(scores))])


def macro_f1_fast(y, pred, k):
    """F1 macro từ ma trận nhầm lẫn; nhanh hơn sklearn nhiều lần, dùng trong vòng bootstrap."""
    cm = np.bincount(k * y + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)))


def bootstrap_ci(y, pred, task, n=500, seed=SEED):
    """Khoảng tin cậy 95 % của chỉ số chính (F1 macro hoặc RMSE) khi lấy lại mẫu tập test có hoàn lại."""
    rng = np.random.default_rng(seed)
    k = int(max(y.max(), pred.max())) + 1 if task == "clf" else 0
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if task == "clf":
            vals.append(macro_f1_fast(y[i], pred[i], k))
        else:
            vals.append(float(np.sqrt(np.mean((y[i] - pred[i]) ** 2))))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
