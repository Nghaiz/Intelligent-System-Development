"""Gộp kết quả thành ch2_compare.json, ch2_summary.json và sinh mọi bảng generated/ch2_*.tex."""
import numpy as np

from sklearn.metrics import f1_score

from tlf.ch2_data import TITLES, VSFC_LABELS, load_vsfc
from tlf.ch2_mlp import SEEDS
from tlf.ch2_mlp_numpy import grad_check
from tlf.ch2_models import NAMES
from tlf.common import load_json, model_dir, save_json
from tlf.texfmt import num, pct, table, tex_escape

FWS = ("numpy", "keras", "torch")
METRIC_KEYS = ("accuracy", "f1_macro", "roc_auc", "precision_pos", "recall_pos", "f1_pos", "mae", "rmse", "r2")
COL = {"cdc": ("f1_macro", "roc_auc", "recall_pos", "precision_pos"), "housing": ("mae", "rmse", "r2"),
       "vsfc": ("accuracy", "f1_macro")}
HEAD = {"f1_macro": "F1 macro", "roc_auc": "AUC", "recall_pos": "Recall dương", "precision_pos": "Precision dương",
        "mae": "MAE", "rmse": "RMSE", "r2": "$R^2$", "accuracy": "Accuracy"}
LOWER_BETTER = ("mae", "rmse")


def mlp_row(ds, fw):
    """Trung bình ba seed; ci_lo/ci_hi là khoảng bootstrap của seed cuối, seed_std là độ lệch giữa các seed."""
    d = load_json(f"ch2_res_{ds}_mlp_{fw}")
    rows = d["rows"]
    keys = [k for k in METRIC_KEYS if k in rows[0]]
    main = "rmse" if ds == "housing" else "f1_macro"
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out.update(dataset=TITLES[ds], model=NAMES[f"mlp_{fw}"], family=f"mlp_{fw}", framework=fw,
               seconds=float(np.mean([r["seconds"] for r in rows])), ci_lo=rows[-1]["ci_lo"], ci_hi=rows[-1]["ci_hi"],
               seed_std=float(np.std([r[main] for r in rows])), n_seeds=len(rows),
               epochs_run=float(np.mean([r["epochs_run"] for r in rows])), n_params=d["n_params"])
    return out


def all_rows(ds):
    classic = load_json(f"ch2_res_{ds}_classic")
    return classic, classic["rows"] + [mlp_row(ds, fw) for fw in FWS]


def fmt(key, r):
    return num(r[key], 4) if r.get(key) is not None else "--"


def compare_table(ds, rows):
    keys = COL[ds]
    cols = [[fmt(k, r) for r in rows] for k in keys]
    for j, k in enumerate(keys):
        vals = [r.get(k, np.nan) for r in rows]
        best = np.nanmin(vals) if k in LOWER_BETTER else np.nanmax(vals)
        cols[j] = [f"\\textbf{{{c}}}" if v == best else c for c, v in zip(cols[j], vals)]
    body = []
    for i, r in enumerate(rows):
        ci = f"[{num(r['ci_lo'], 3)}; {num(r['ci_hi'], 3)}]" if "ci_lo" in r else "--"
        body.append([r["model"], *[c[i] for c in cols], ci, num(r["seconds"], 1)])
    table(f"ch2_{ds}_compare", "l" + "r" * len(keys) + "cr",
          ["Mô hình", *[HEAD[k] for k in keys], "KTC 95 \\%", "Giây"], body)


def sample_tables(meta):
    cdc, house, vsfc = meta["datasets"]
    keys = list(cdc["sample_rows"][0])
    table("ch2_cdc_sample", "r" * len(keys), [k.replace("Diabetes_binary", "Nhãn") for k in keys],
          [[num(r[k], 0) for k in keys] for r in cdc["sample_rows"]])
    table("ch2_housing_sample", "p{6.6cm}rrrlr", ["Địa chỉ", "m²", "Tầng", "PN", "Pháp lý", "Tỷ"],
          [[tex_escape(r["Address"]), num(r["Area"], 0), _opt(r["Floors"]), _opt(r["Bedrooms"]),
            _legal(r["Legal status"]), num(r["Price"], 2)] for r in house["sample_rows"]])
    table("ch2_vsfc_sample", "p{11.6cm}l", ["Nội dung", "Nhãn"],
          [[tex_escape(r["sentence"]), r["sentiment"]] for r in vsfc["sample_rows"]])


def _opt(v):
    return "--" if v != v else num(v, 0)


def _legal(v):
    return {"Have certificate": "Sổ đỏ", "Sale contract": "HĐMB"}.get(v, "--")


def size_table(meta):
    body = []
    for d in meta["datasets"]:
        body.append([d["name"], d["task"], num(d["n_samples"], 0), num(d["n_features"], 0), num(d["size_mb"], 2)])
    table("ch2_overview", "llrrr", ["Dataset", "Bài toán", "Số mẫu", "Đặc trưng", "MB"], body)


def mlp_table():
    body = []
    for ds in ("cdc", "housing", "vsfc"):
        main = "rmse" if ds == "housing" else "f1_macro"
        for fw in FWS:
            r = mlp_row(ds, fw)
            body.append([TITLES[ds], r["model"], num(r[main], 4),
                         num(r["seed_std"], 4) if r["n_seeds"] > 1 else "--", num(r["epochs_run"], 1),
                         num(r["seconds"], 1)])
    table("ch2_mlp", "llrrrr", ["Dataset", "Nền tảng", "Chỉ số chính", "Độ lệch seed", "Epoch TB", "Giây"], body)


def summary(all_data):
    """Khóa cho \\SL: mô hình tốt nhất theo test (chỉ để bình luận), hiệu với Dummy, tốc độ."""
    out = {"seeds": len(SEEDS)}
    for ds, (classic, rows) in all_data.items():
        reg = ds == "housing"
        key = "rmse" if reg else "f1_macro"
        real = [r for r in rows if r["family"] != "dummy"]
        top = min(real, key=lambda r: r[key]) if reg else max(real, key=lambda r: r[key])
        s = {"top_model": top["model"], "top": top[key], "dummy": rows[0][key], "best_val": classic["best"]["name"],
             "threshold": classic["best"]["threshold"]}
        for r in rows:
            s[r["family"]] = {k: r[k] for k in (*METRIC_KEYS, "ci_lo", "ci_hi", "seconds", "n_trees", "seed_std")
                              if r.get(k) is not None}
        out[ds] = s
    return out


def cdc_threshold():
    """So F1 macro của XGBoost ở ngưỡng 0,5 và ngưỡng chọn trên validation."""
    d = np.load(model_dir("ch2") / "pred_cdc.npz")
    thr = load_json("ch2_res_cdc_classic")["best"]["threshold"]
    y, p = d["y"], d["xgb_p"]
    f = {t: {"f1_macro": float(f1_score(y, p >= t, average="macro")), "recall": float(np.mean(p[y == 1] >= t))}
         for t in (0.5, thr)}
    return {"at_05": f[0.5], "at_best": f[thr]}


def vsfc_detail():
    """F1 từng lớp, bảng từ đặc trưng và bốn câu XGBoost đoán sai, lấy theo thứ tự trong tập test."""
    d = np.load(model_dir("ch2") / "pred_vsfc.npz")
    y, sentences = d["y"], load_vsfc("test")["sentence"].to_numpy()
    f1 = {fam: f1_score(y, d[fam], average=None).tolist() for fam in ("linear", "svm", "xgb")}
    terms = load_json("ch2_res_vsfc_classic")["terms"]
    table("ch2_vsfc_terms", "lp{11.6cm}", ["Lớp", "Từ nổi bật"],
          [[VSFC_LABELS[c], tex_escape(", ".join(terms[str(c)]["rows"]))] for c in range(3)])
    wrong = []
    for true, pred in ((0, 2), (2, 0), (1, 0), (1, 2)):
        i = np.flatnonzero((y == true) & (d["xgb"] == pred) & (np.char.str_len(sentences.astype(str)) < 90))[0]
        wrong.append([tex_escape(sentences[i]), VSFC_LABELS[true], VSFC_LABELS[pred]])
    table("ch2_vsfc_errors", "p{9.2cm}ll", ["Nội dung", "Nhãn thật", "Dự đoán"], wrong)
    return {"f1_class": f1}


if __name__ == "__main__":
    data = {ds: all_rows(ds) for ds in COL}
    save_json("ch2_vsfc_detail", vsfc_detail())
    save_json("ch2_cdc_threshold", cdc_threshold())
    meta = load_json("ch2_datasets")
    rows = [{k: v for k, v in r.items() if k not in ("params",)} for _, rs in data.values() for r in rs]
    save_json("ch2_compare", {"rows": rows, "primary": "f1_macro",
                              "primary_by_dataset": {TITLES[d]: "rmse" if d == "housing" else "f1_macro" for d in COL}})
    save_json("ch2_summary", summary(data))
    err, n_params = grad_check()
    save_json("ch2_gradcheck", {"rel_error": err, "n_params": n_params})
    for ds, (_, rs) in data.items():
        compare_table(ds, rs)
    sample_tables(meta)
    size_table(meta)
    mlp_table()
    print("xong bảng", pct(1.0))
