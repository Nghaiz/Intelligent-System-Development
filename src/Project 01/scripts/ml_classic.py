"""Huấn luyện LR, RF, XGBoost cho một bài toán trên CPU; lưu xác suất val/test và mô hình XGB. Dùng: ml_classic.py <task>"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import limit_threads, save_json, seed_all, timer  # noqa: E402

limit_threads(8)
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from dnx.mem import slot  # noqa: E402
from dnx.ml.metrics import full_eval  # noqa: E402
from dnx.ml.tasks import MODEL_DIR, PRED_DIR, splits  # noqa: E402
from dnx.runlog import log_run  # noqa: E402

XGB_GRID = [dict(max_depth=d, min_child_weight=m) for d in (4, 6) for m in (1, 10)]


def fit_xgb(Xtr, ytr, Xva, yva):
    """Lưới nhỏ trên val; dừng sớm theo PR-AUC val."""
    best, best_ap, trials = None, -1.0, []
    for g in XGB_GRID:
        m = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=1.0, tree_method="hist", n_jobs=12, eval_metric="aucpr",
                              early_stopping_rounds=50, random_state=42, **g)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        ap = float(average_precision_score(yva, m.predict_proba(Xva)[:, 1]))
        trials.append({**g, "best_iteration": int(m.best_iteration), "val_pr_auc": ap})
        print("xgb", trials[-1], flush=True)
        if ap > best_ap:
            best, best_ap = m, ap
    return best, trials


def main(task):
    seed_all()
    with slot(4, f"ml_classic {task}"):
        pre, _, _, data = splits(task)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]
        print(task, "train", Xtr.shape, "val", Xva.shape, "test", Xte.shape, "dương", ytr.mean(), flush=True)
        joblib.dump(pre, MODEL_DIR / f"{task}_pre.joblib")
        out, times = {}, {}
        models = {
            "logreg": lambda: LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr),
            "random_forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=5,
                                                            max_samples=0.6, n_jobs=12, random_state=42).fit(Xtr, ytr),
            "xgboost": lambda: fit_xgb(Xtr, ytr, Xva, yva),
        }
        for name, fit in models.items():
            with timer(times, name):
                r = fit()
            m, extra = r if isinstance(r, tuple) else (r, None)
            pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
            res, _ = full_eval(yva, pv, yte, pt)
            res.update(seconds=times[name], framework="sklearn" if name != "xgboost" else "xgboost")
            if extra:
                res["grid"] = extra
                m.get_booster().save_model(MODEL_DIR / f"{task}_xgb.json")
            out[name] = res
            np.savez_compressed(PRED_DIR / f"{task}_{name}.npz", yv=yva, pv=pv, yt=yte, pt=pt)
            log_run("ml", task, name, res["framework"],
                    {k: res["test"][k] for k in ("roc_auc", "pr_auc", "f1", "brier", "ece")}, times[name], "cpu",
                    {"val_pr_auc": res["val"]["pr_auc"]})
            print(name, f"val PR {res['val']['pr_auc']:.4f} test AUC {res['test']['roc_auc']:.4f} "
                        f"({times[name]:.0f}s)", flush=True)
            del m
        save_json(f"ml_classic_{task}", {"task": task, "ts": time.strftime("%FT%T"), "n_features": Xtr.shape[1],
                                         "models": out})


if __name__ == "__main__":
    main(sys.argv[1])
