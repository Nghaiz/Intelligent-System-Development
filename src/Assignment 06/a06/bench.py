"""Hạng mục 4 (Ch.4): chỉ số đánh giá, đo độ trễ suy luận và lưu kết quả của mỗi lượt huấn luyện.

Mọi chỉ số của cả hai framework tính bằng cùng hàm trong tệp này, từ dự báo đã lưu ở outputs/preds/.
"""
import json
import time

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from .data_stock import to_price
from .export import MODELS, ROOT, _to_builtin, write_dat

PREDS = ROOT / "outputs" / "preds"
PREDS.mkdir(parents=True, exist_ok=True)
META = MODELS / "metadata.json"


# ---------------------------------------------------------------- hồi quy (AMZN)
def regression_metrics(p_true, p_pred, r_true=None, r_pred=None):
    """RMSE, MAE (USD) và R² trên giá; R² trên return nếu có (khó hơn nhiều so với R² trên giá)."""
    err = p_pred - p_true
    out = {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
           "r2": float(1 - np.sum(err ** 2) / np.sum((p_true - p_true.mean()) ** 2))}
    if r_true is not None:
        out["r2_ret"] = float(1 - np.sum((r_pred - r_true) ** 2) / np.sum((r_true - r_true.mean()) ** 2))
        out["dir_acc"] = float(np.mean(np.sign(r_pred) == np.sign(r_true)))
    return out


def eval_stock(data, pred_test):
    s = data["scaler"]
    p = to_price(s, pred_test, data["p_now"]["test"])
    return {**regression_metrics(data["p_next"]["test"], p, data["ret"]["test"], s.inverse_y(pred_test)),
            "price_pred": p}


def naive_stock(data):
    """Mốc ngây thơ: giá ngày mai bằng giá hôm nay (return dự báo = 0, nên không có tỉ lệ đúng chiều)."""
    p_now = data["p_now"]["test"]
    out = regression_metrics(data["p_next"]["test"], p_now, data["ret"]["test"], np.zeros_like(p_now))
    del out["dir_acc"]
    return out


# ---------------------------------------------------------------- phân loại (KKBox)
def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=np.float64)))


def best_threshold(y, prob):
    """Ngưỡng làm F1 lớn nhất, chọn trên VALIDATION rồi mới áp cho test."""
    prec, rec, thr = precision_recall_curve(y, prob)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def classification_metrics(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"acc": float((tp + tn) / len(y)), "precision": float(precision), "recall": float(recall),
            "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
            "roc_auc": float(roc_auc_score(y, prob)), "pr_auc": float(average_precision_score(y, prob)),
            "threshold": threshold, "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def eval_churn(data, pred_val, pred_test):
    thr = best_threshold(data["y"]["val"], sigmoid(pred_val))
    return classification_metrics(data["y"]["test"].astype(int), sigmoid(pred_test), thr)


def curves(y, prob, n=120):
    """ROC và PR tỉa còn khoảng n điểm để pgfplots vẽ nhanh."""
    fpr, tpr, _ = roc_curve(y, prob)
    prec, rec, _ = precision_recall_curve(y, prob)
    ri = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    pi = np.unique(np.linspace(0, len(rec) - 1, n).astype(int))
    return {"fpr": fpr[ri], "tpr": tpr[ri]}, {"recall": rec[pi][::-1], "precision": prec[pi][::-1]}


# ---------------------------------------------------------------- lưu kết quả một lượt huấn luyện
def save_run(dataset, framework, out, extra_meta=None):
    """Ghi history (.dat), dự báo (.npz) và models/meta_<tập>_<framework>.json cho 4 mô hình.

    Mỗi lượt một tệp riêng vì notebook PyTorch và Keras chạy song song; merge_metadata() gộp sau.
    """
    meta = {}
    for cell, r in out.items():
        key = f"{dataset}_{framework}_{cell}"
        h = r["history"]
        write_dat(f"hist_{key}", {k: h[k] for k in ("epoch", "train_loss", "val_loss")})
        np.savez(PREDS / f"{key}.npz", val=r["pred_val"], test=r["pred_test"])
        meta[key] = {"file": r["file"], "params": r["params"], "device": r["device"],
                     "best_epoch": h["best_epoch"], "epochs_run": len(h["epoch"]),
                     "epoch_s": float(np.mean(h["epoch_s"])), "train_s": float(np.sum(h["epoch_s"])),
                     "best_val_loss": float(min(h["val_loss"])), **(extra_meta or {})}
    path = MODELS / f"meta_{dataset}_{framework}.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def merge_metadata(extra):
    """Gộp 4 tệp meta_*.json thành models/metadata.json, kèm scaler, ngưỡng và cấu hình (extra)."""
    models = {}
    for p in sorted(MODELS.glob("meta_*.json")):
        models.update(json.loads(p.read_text(encoding="utf-8")))
    META.write_text(json.dumps(_to_builtin({"models": models, **extra}), ensure_ascii=False, indent=2), encoding="utf-8")
    return models


def load_preds(key):
    z = np.load(PREDS / f"{key}.npz")
    return z["val"], z["test"]


# ---------------------------------------------------------------- độ trễ suy luận trên CPU
def _median_time(fn, x, warmup=20, repeat=200):
    for _ in range(warmup):
        fn(x)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(x)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def latency(fn, sample, batch_big=256):
    """fn nhận mảng (B, T, D). Trả về ms/mẫu với lô 1 và mẫu/giây với lô 256 (trung vị)."""
    x1 = sample[:1]
    xb = np.repeat(sample[:1], batch_big, axis=0) if len(sample) < batch_big else sample[:batch_big]
    t1 = _median_time(fn, x1)
    tb = _median_time(fn, xb, warmup=5, repeat=30)
    return {"latency_ms": 1000 * t1, "throughput": batch_big / tb}
