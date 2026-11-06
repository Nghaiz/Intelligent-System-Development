"""Chỉ số phân loại nhị phân, sai số hiệu chỉnh, chọn ngưỡng F1 trên val và khoảng tin cậy bootstrap."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ..common import SEED

N_BINS = 15
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def ece(y, p, bins=N_BINS):
    """Expected Calibration Error với các ngăn đều trên [0, 1]."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    conf = np.bincount(idx, p, minlength=bins)
    acc = np.bincount(idx, y, minlength=bins)
    return float(np.abs(conf - acc).sum() / len(y))


def f1_at(y, p, t):
    pred = p >= t
    tp = float(np.sum(pred & (y == 1)))
    d = pred.sum() + np.sum(y == 1)
    return 2 * tp / d if d else 0.0


def best_f1_threshold(y_val, p_val):
    """Ngưỡng cho F1 lớp dương cao nhất; chỉ gọi trên val."""
    scores = [f1_at(y_val, p_val, t) for t in GRID]
    return float(GRID[int(np.argmax(scores))])


def evaluate(y, p, threshold):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = p >= threshold
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    fn, tn = int(np.sum(~pred & (y == 1))), int(np.sum(~pred & (y == 0)))
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "f1": f1_at(y, p, threshold), "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0, "accuracy": (tp + tn) / len(y),
            "brier": float(brier_score_loss(y, p)), "ece": ece(y, p), "threshold": threshold,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}, "n": len(y), "positive_rate": float(y.mean())}


def calibrate(y_val, p_val):
    """Isotonic fit trên val."""
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)


def full_eval(yv, pv, yt, pt):
    """Chuỗi chuẩn cho một mô hình: hiệu chỉnh và chọn ngưỡng chỉ trên val, rồi chấm một lần trên test."""
    iso = calibrate(yv, pv)
    pvc, ptc = iso.predict(pv), iso.predict(pt)
    th = best_f1_threshold(yv, pvc)
    return {"val": evaluate(yv, pvc, th), "val_raw_pr_auc": float(average_precision_score(yv, pv)),
            "test_raw_brier": float(brier_score_loss(yt, pt)), "test_raw_ece": ece(yt, pt),
            "test": evaluate(yt, ptc, th)}, iso


def bootstrap_auc(y, p, n=500, seed=SEED):
    """Khoảng tin cậy 95 % của ROC-AUC và PR-AUC trên test."""
    rng = np.random.default_rng(seed)
    roc, pr = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        roc.append(roc_auc_score(y[i], p[i]))
        pr.append(average_precision_score(y[i], p[i]))
    return {"roc_auc": [float(v) for v in np.percentile(roc, [2.5, 97.5])],
            "pr_auc": [float(v) for v in np.percentile(pr, [2.5, 97.5])], "n_boot": n}
