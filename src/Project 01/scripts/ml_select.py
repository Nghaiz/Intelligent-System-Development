"""Tổng hợp mọi mô hình, chọn theo val, hiệu chỉnh mô hình phục vụ, tính SHAP tổng và sinh thẻ mô hình."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import limit_threads, load_json, save_json  # noqa: E402

limit_threads(8)
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402

from dnx.mem import slot  # noqa: E402
from dnx.ml.metrics import bootstrap_auc, full_eval  # noqa: E402
from dnx.ml.serve import LABEL_VI, PROB_RANGE  # noqa: E402
from dnx.ml.tasks import MODEL_DIR, PRED_DIR, TASKS, feature_groups, splits  # noqa: E402

SERVED = "xgboost"  # hợp đồng yêu cầu SHAP qua pred_contribs nên mô hình phục vụ là XGBoost
KEYS = ("roc_auc", "pr_auc", "f1", "precision", "recall", "brier", "ece")
TITLE = {"full": "Dự đoán đái tháo đường có xét nghiệm", "screening": "Sàng lọc không cần xét nghiệm",
         "readmission": "Tái nhập viện dưới 30 ngày"}


def compare(task):
    rows = {**load_json(f"ml_classic_{task}")["models"], **load_json(f"dl_{task}")["models"]}
    table = {m: {"val_pr_auc_raw": r["val_raw_pr_auc"], "threshold": r["val"]["threshold"],
                 **{f"test_{k}": r["test"][k] for k in KEYS}, "test_ece_raw": r["test_raw_ece"],
                 "test_brier_raw": r["test_raw_brier"], "seconds": r["seconds"], "framework": r["framework"]}
             for m, r in rows.items()}
    best = max(table, key=lambda m: table[m]["val_pr_auc_raw"])
    return table, best, rows


def shap_global(task, n=4000):
    """SHAP trên mẫu test: trung bình trị tuyệt đối và mẫu cột để vẽ, gộp cột one-hot về đặc trưng gốc."""
    pre, _, _, data = splits(task)
    X = data["test"][0]
    idx = np.random.default_rng(42).choice(len(X), min(n, len(X)), replace=False)
    b = xgb.Booster()
    b.load_model(MODEL_DIR / f"{task}_xgb.json")
    con = b.predict(xgb.DMatrix(X[idx]), pred_contribs=True)[:, :-1]
    groups = feature_groups(pre, task)
    names = list(dict.fromkeys(groups))
    arr = np.stack([con[:, [i for i, g in enumerate(groups) if g == f]].sum(1) for f in names], 1)
    return {"n": len(idx), "scale": "log-odds", "mean_abs": sorted(
        [{"feature": f, "label_vi": LABEL_VI.get(f, f), "mean_abs_shap": float(np.abs(arr[:, i]).mean())}
         for i, f in enumerate(names)], key=lambda r: -r["mean_abs_shap"])}


def main():
    summary, card = {}, {"name": "DiaNexus risk models", "generated": time.strftime("%FT%T"),
                         "serving_prob_range": list(PROB_RANGE),
                         "serving_note": "prob phục vụ được kẹp vào serving_prob_range; chỉ số thẻ tính trên xác suất chưa kẹp.",
                         "disclaimer": "Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ.",
                         "tasks": {}}
    with slot(3, "ml_select"):
        for task in TASKS:
            table, best, _ = compare(task)
            p = np.load(PRED_DIR / f"{task}_{SERVED}.npz")
            res, iso = full_eval(p["yv"], p["pv"], p["yt"], p["pt"])
            ptc = iso.predict(p["pt"])
            ci = bootstrap_auc(p["yt"], ptc)
            th = res["val"]["threshold"]
            joblib.dump(iso, MODEL_DIR / f"{task}_calib.joblib")
            meta = {"task": task, "model": SERVED, "threshold": th, "selected_on": "val", "features": TASKS[task]}
            (MODEL_DIR / f"{task}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            shap = shap_global(task)
            save_json(f"ml_shap_{task}", shap)
            summary[task] = {"title": TITLE[task], "best_on_val": best, "served": SERVED,
                             "val_gap_best_minus_served": table[best]["val_pr_auc_raw"] - table[SERVED]["val_pr_auc_raw"],
                             "served_test": res["test"], "served_val": res["val"], "bootstrap_ci_test": ci,
                             "models": table}
            card["tasks"][task] = {
                "title": TITLE[task], "dataset": TASKS[task]["table"], "target": TASKS[task]["target"], "model": SERVED,
                "calibration": "isotonic trên val", "threshold": th, "threshold_rule": "F1 lớn nhất trên val",
                "n_test": res["test"]["n"], "metrics_test": {k: res["test"][k] for k in KEYS}, "ci95_test": ci,
                "top_features": [r["label_vi"] for r in shap["mean_abs"][:5]]}
            print(task, "best val", best, "served test AUC", res["test"]["roc_auc"], flush=True)
    save_json("ml_summary", summary)
    (MODEL_DIR / "model_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
