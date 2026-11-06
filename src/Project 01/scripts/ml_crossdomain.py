"""Đối chuẩn chéo miền trên đặc trưng chung (tuổi, BMI, đường huyết): 100k, Pima, Frankfurt, thêm ROC ngoài miền."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import limit_threads, save_json, seed_all  # noqa: E402

limit_threads(8)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from dnx.db import connect  # noqa: E402
from dnx.mem import slot  # noqa: E402
from dnx.ml.metrics import best_f1_threshold, evaluate  # noqa: E402
from dnx.runlog import log_run  # noqa: E402

COMMON = ["age", "bmi", "glucose"]
DOMAINS = {"patients_lab": "100k", "pima": "Pima", "frankfurt": "Frankfurt"}
MODELS = {
    "logreg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)),
    "xgboost": lambda: xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.8, n_jobs=12, random_state=42),
}


def main():
    seed_all()
    with slot(2, "ml_crossdomain"), connect() as con:
        D = {t: pd.read_sql(f"select age, bmi, glucose, diabetes, split from {t}", con) for t in DOMAINS}
    out = {"features": COMMON, "note": "Ngưỡng chọn trên val của miền nguồn; miền đích dùng toàn bộ dòng, miền nguồn chấm trên test.",
           "results": []}
    for mname, make in MODELS.items():
        for src, ds in D.items():
            tr, va = ds[ds.split == "train"], ds[ds.split == "val"]
            m = make().fit(tr[COMMON].values, tr.diabetes.values)
            th = best_f1_threshold(va.diabetes.values, m.predict_proba(va[COMMON].values)[:, 1])
            for dst, dd in D.items():
                ev = dd[dd.split == "test"] if dst == src else dd
                r = evaluate(ev.diabetes.values, m.predict_proba(ev[COMMON].values)[:, 1], th)
                out["results"].append({"model": mname, "source": src, "target": dst, "n": len(ev),
                                       "target_positive_rate": r["positive_rate"],
                                       **{k: r[k] for k in ("roc_auc", "pr_auc", "f1", "precision", "recall", "brier", "ece")}})
                log_run("ml", f"{src}->{dst}", mname, "sklearn", {k: r[k] for k in ("roc_auc", "pr_auc", "f1")}, 0.0, "cpu")
                print(mname, src, "->", dst, f"AUC {r['roc_auc']:.3f} F1 {r['f1']:.3f}", flush=True)
    save_json("ml_crossdomain", out)


if __name__ == "__main__":
    main()
