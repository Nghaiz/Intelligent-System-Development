"""Chạy Dummy, tuyến tính, SVM, cây, rừng, XGBoost trên một dataset: python scripts/ch2_classic.py cdc"""
import sys
import time

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from tlf.ch2_data import TITLES, get_split, housing_random_split
from tlf.ch2_metrics import best_threshold, bootstrap_ci, clf_metrics, reg_metrics
from tlf.ch2_models import JOBS, NAMES, grid, make
from tlf.common import limit_threads, model_dir, save_json, seed_all
from tlf.mem import require_disk, slot
from tlf.runlog import log_run

FAMILIES = ("dummy", "linear", "svm", "tree", "rf", "xgb")
MODELS = model_dir("ch2")
# Giới hạn thời gian và CPU: CDC và VSFC chỉ chạy một cấu hình mỗi họ; Housing đã quét đủ lưới trước đó.
ONE_CONFIG = {"linear": {"C": 1}, "svm": {"C": 0.1}, "tree": {"max_depth": 10, "min_samples_leaf": 20},
              "rf": {"max_depth": 20, "min_samples_leaf": 2}, "xgb": {"max_depth": 5, "reg_lambda": 1}}
FAST = len(sys.argv) > 1 and sys.argv[1] in ("cdc", "vsfc")
if len(sys.argv) > 1 and sys.argv[1] == "vsfc":
    # Văn bản thưa cần cây sâu: độ sâu 20 làm F1 macro tụt rõ so với cây không giới hạn.
    ONE_CONFIG.update(rf={"max_depth": None, "min_samples_leaf": 2}, tree={"max_depth": 14, "min_samples_leaf": 5})


def fit_one(family, p, s, dense, balanced):
    m = make(family, p, s.task, balanced)
    d = dense if family in ("rf", "xgb", "tree") else s
    kw = {"eval_set": [(d.X_va, d.y_va)], "verbose": False} if family == "xgb" else {}
    if family == "xgb" and balanced:
        c = np.bincount(d.y_tr)
        kw["sample_weight"] = (len(d.y_tr) / (len(c) * c))[d.y_tr]
    t0 = time.perf_counter()
    m.fit(d.X_tr, d.y_tr, **kw)
    return m, d, time.perf_counter() - t0


def score_val(m, d, tune_thr):
    """Trả (điểm chọn mô hình, ngưỡng): F1 macro hoặc -RMSE trên validation."""
    if d.task == "reg":
        return -reg_metrics(d.y_va, m.predict(d.X_va))["rmse"], None
    if tune_thr:
        p = m.predict_proba(d.X_va)[:, 1]
        thr = best_threshold(d.y_va, p)
        return clf_metrics(d.y_va, (p >= thr).astype(int))["f1_macro"], thr
    return clf_metrics(d.y_va, m.predict(d.X_va))["f1_macro"], None


def search(family, s, dense, balanced, tune_thr):
    """Thử mọi tổ hợp của lưới; giữ tổ hợp có điểm validation cao nhất."""
    params = [{}] if family == "dummy" else grid(family, s.task)
    if family in ONE_CONFIG and FAST:
        params = [ONE_CONFIG[family]]
    best, tried = None, []
    for p in params:
        m, d, sec = fit_one(family, p, s, dense, balanced)
        val, thr = score_val(m, d, tune_thr and family != "dummy")
        tried.append({**p, "val": val, "seconds": sec})
        print(f"  {family} {p} val={val:.4f} {sec:.1f}s", flush=True)
        if best is None or val > best[0]:
            best = (val, m, d, p, thr, sec)
    return best, tried


def evaluate(m, d, thr):
    """Đánh giá trên test đúng một lần; trả (hàng số liệu, dự đoán, xác suất)."""
    if d.task == "reg":
        pred, proba = m.predict(d.X_te), None
        row = reg_metrics(d.y_te, pred)
    else:
        proba = m.predict_proba(d.X_te)
        pred = (proba[:, 1] >= thr).astype(int) if thr is not None else proba.argmax(1)
        row = clf_metrics(d.y_te, pred, proba)
    row["ci_lo"], row["ci_hi"] = bootstrap_ci(np.asarray(d.y_te), np.asarray(pred), d.task)
    return row, pred, proba


def importance(m, d):
    """Hoán vị trên test lấy mẫu 6.000 dòng: mức giảm AUC hoặc R² khi xáo trộn một đặc trưng."""
    rng = np.random.default_rng(0)
    i = rng.choice(len(d.y_te), min(6000, len(d.y_te)), replace=False)
    r = permutation_importance(m, d.X_te[i], d.y_te[i], n_repeats=3, random_state=0, n_jobs=1,
                               scoring="r2" if d.task == "reg" else "roc_auc")
    order = np.argsort(-r.importances_mean)[:12]
    return [{"feature": d.names[j], "drop": float(r.importances_mean[j])} for j in order]


def extras(ds, fitted, s):
    """Phần riêng từng bộ: tầm quan trọng đặc trưng, từ đặc trưng lớp, tác động của cách chia."""
    out = {}
    if ds in ("cdc", "housing"):
        out["importance"] = {"rows": importance(*fitted["xgb"][:2])}
    if ds == "housing":
        xgb = fitted["xgb"][0]
        r = housing_random_split()
        keep = {k: v for k, v in xgb.get_params().items() if k in ("max_depth", "reg_lambda")}
        m = make("xgb", keep, "reg")
        m.fit(r.X_tr, r.y_tr, eval_set=[(r.X_va, r.y_va)], verbose=False)
        out["random_split"] = reg_metrics(r.y_te, m.predict(r.X_te))
    if ds == "vsfc":
        lr = fitted["linear"][0]
        names = np.array(s.names)
        out["terms"] = {str(c): {"rows": [str(names[j]) for j in np.argsort(-lr.coef_[c])[:10]]} for c in range(3)}
    return out


def save_best(ds, fam, fitted, thr_by):
    """Lưu Pipeline tiền xử lý + mô hình tốt nhất theo validation; không lưu dựa trên test."""
    require_disk(0.5, f"ch2 lưu mô hình {ds}")
    m, d, _ = fitted[fam]
    joblib.dump(Pipeline([("pre", d.pre), ("model", m)]), MODELS / f"{ds}_best.joblib", compress=3)
    return {"family": fam, "name": NAMES[fam], "threshold": thr_by.get(fam)}


def main(ds):
    seed_all()
    limit_threads(JOBS)
    s, dense = get_split(ds), get_split(ds, dense=True)
    tune_thr, balanced = ds == "cdc", ds == "vsfc"
    rows, preds, fitted, thr_by, vals = [], {"y": np.asarray(s.y_te)}, {}, {}, {}
    for fam in FAMILIES:
        (val, m, d, p, thr, sec), tried = search(fam, s, dense, balanced, tune_thr)
        row, pred, proba = evaluate(m, d, thr)
        row.update(dataset=TITLES[ds], model=NAMES[fam], family=fam, framework="scikit-learn", seconds=sec,
                   val_score=val, params=p, threshold=thr, grid_size=len(tried), search_seconds=sum(
                       t["seconds"] for t in tried))
        if fam == "xgb":
            row["framework"], row["n_trees"] = "xgboost", int(m.best_iteration) + 1
        rows.append(row)
        preds[fam] = pred
        if proba is not None and proba.shape[1] == 2:
            preds[f"{fam}_p"] = proba[:, 1]
        fitted[fam], thr_by[fam], vals[fam] = (m, d, sec), thr, val
        log_run("ch2", TITLES[ds], NAMES[fam], row["framework"], {k: row[k] for k in row if k in (
            "accuracy", "f1_macro", "roc_auc", "mae", "rmse", "r2")}, sec, "cpu", p)
        print(TITLES[ds], fam, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}, flush=True)
    best_fam = max((f for f in FAMILIES if f != "dummy"), key=vals.get)
    best = save_best(ds, best_fam, fitted, thr_by)
    np.savez_compressed(MODELS / f"pred_{ds}.npz", **preds)
    save_json(f"ch2_res_{ds}_classic", {"rows": rows, "best": best, **extras(ds, fitted, s)})


if __name__ == "__main__":
    ds = sys.argv[1]
    with slot({"cdc": 2.5}.get(ds, 1.5), f"ch2 classic {ds}"):
        main(ds)
