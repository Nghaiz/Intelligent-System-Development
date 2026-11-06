"""Chương 4: chỉ số dự báo trên đơn vị gốc và kiểm định Diebold-Mariano so với baseline."""
import numpy as np
from scipy import stats


def forecast_metrics(y, yhat, last=None):
    """MAE, RMSE, MAPE, sMAPE (%) và tỉ lệ đoán đúng hướng thay đổi so với điểm cuối cửa sổ."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = y - yhat
    out = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
           "mape": float(100 * np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-9))),
           "smape": float(100 * np.mean(2 * np.abs(err) / np.maximum(np.abs(y) + np.abs(yhat), 1e-9)))}
    if last is not None:
        out["dir_acc"] = direction_accuracy(y, yhat, np.asarray(last, float))
    return out


def direction_accuracy(y, yhat, last):
    """Chỉ tính các điểm chuỗi thật có đổi và mô hình có dự báo đổi; Persistence không đoán hướng nên trả None."""
    moved = (y != last) & (yhat != last)
    if not moved.any():
        return None
    return float(np.mean(np.sign(yhat - last)[moved] == np.sign(y - last)[moved]))


def majority_direction(y, last):
    """Mốc so sánh cho hướng đúng: tỉ lệ của hướng phổ biến hơn trong tập test."""
    moved = y != last
    up = float(np.mean(y[moved] > last[moved]))
    return max(up, 1 - up)


def horizon_rmse(truth, pred):
    """RMSE theo từng bước dự báo, cỡ H."""
    return np.sqrt(np.mean((truth - pred) ** 2, axis=0))


def diebold_mariano(e_model, e_base, h=1):
    """Thống kê DM với hàm mất mát bình phương; DM < 0 nghĩa là mô hình có sai số nhỏ hơn baseline.

    Phương sai dài hạn cộng tự hiệp phương sai tới trễ h − 1 vì sai số dự báo h bước tương quan với nhau.
    """
    d = np.asarray(e_model) ** 2 - np.asarray(e_base) ** 2
    n, dc = len(d), d - d.mean()
    lrv = dc @ dc / n + 2 * sum(dc[k:] @ dc[:-k] / n for k in range(1, h))
    dm = float(d.mean() / np.sqrt(max(lrv, 1e-18) / n))
    return dm, float(2 * stats.norm.sf(abs(dm)))


def coverage(truth, lo, hi):
    return float(np.mean((truth >= lo) & (truth <= hi)))
