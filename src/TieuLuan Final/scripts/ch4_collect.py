"""Chương 4: gom các lượt chạy thành ch4_compare.json, ch4_horizon.json, ch4_summary.json, ch4_errors.json.

"Mô hình chọn" của mỗi chuỗi có MSE validation thấp nhất, không nhìn test; hạt giống đại diện cũng chọn theo validation.
"""
import json

import numpy as np
import pandas as pd

from tlf.ch4_data import OUT_DIR, SPECS, TICKERS, Prepared
from tlf.ch4_metrics import (coverage, diebold_mariano, forecast_metrics, horizon_rmse, majority_direction)
from tlf.ch4_torch import ARCH_NAMES, ARCHS
from tlf.common import OUT, ROOT, model_dir, save_json
from tlf.runlog import log_run

RUNS = OUT_DIR / "runs"
TL02 = ROOT.parent / "TieuLuan 02" / "outputs" / "metrics" / "forecast.json"
METRIC_KEYS = ("mae", "rmse", "mape", "smape", "dir_acc")
LABEL = {**ARCH_NAMES}
TERCILES = (1 / 3, 2 / 3)


def _load(tag):
    meta = json.loads((RUNS / f"{tag}.json").read_text(encoding="utf-8"))
    return np.load(RUNS / f"{tag}.npz")["test"], meta


def _seeds(series, arch):
    return sorted(RUNS.glob(f"{series}__{arch}__s*.json"))


def _row(series, model, framework, metrics, **extra):
    return {"dataset": series, "model": model, "framework": framework, **metrics, **extra}


def baseline_rows(prep, tt):
    truth, last = prep.truth(tt), prep.level[tt]
    rows = [_row(prep.spec.id, "Persistence", "baseline", forecast_metrics(truth[:, 0], last, last), seconds=0.0,
                 rmse_last=float(horizon_rmse(truth, prep.persistence(tt))[-1]))]
    sea = prep.seasonal(tt)
    if sea is not None:
        rows.append(_row(prep.spec.id, "Mùa vụ 24 giờ", "baseline", forecast_metrics(truth[:, 0], sea[:, 0], last),
                         seconds=0.0, rmse_last=float(horizon_rmse(truth, sea)[-1])))
    return rows


def torch_row(prep, arch, tt, pers_rmse):
    """Trung bình chỉ số trên các hạt giống; DM và độ phủ tính trên hạt giống đã lưu thành mô hình phục vụ."""
    truth, last, runs = prep.truth(tt), prep.level[tt], []
    for f in _seeds(prep.spec.id, arch):
        test, meta = _load(f.stem)
        runs.append((meta["val_mse"], prep.to_level(test, tt), meta))
    ms = [forecast_metrics(truth[:, 0], lv[:, 0], last) for _, lv, _ in runs]
    avg = {k: float(np.mean([m[k] for m in ms])) for k in METRIC_KEYS}
    _, lv, meta = min(runs, key=lambda r: r[0])
    dm, p = diebold_mariano(truth[:, 0] - lv[:, 0], truth[:, 0] - last)
    return _row(prep.spec.id, LABEL[arch], "pytorch", avg, rmse_sd=float(np.std([m["rmse"] for m in ms])),
                skill=1 - avg["rmse"] / pers_rmse, dm=dm, dm_p=p, seeds=len(runs),
                val_mse=float(np.mean([r[0] for r in runs])), seconds=float(np.mean([r[2]["seconds"] for r in runs])),
                n_params=meta["n_params"], epochs=meta["epochs"],
                rmse_last=float(horizon_rmse(truth, lv)[-1])), lv


def side_rows(prep, tt, pers_rmse):
    """Các lượt NumPy và Keras của chuỗi này (mỗi lượt một hạt giống)."""
    rows = []
    for f in sorted(RUNS.glob(f"{prep.spec.id}__*__numpy.json")) + sorted(RUNS.glob(f"{prep.spec.id}__*__keras*.json")):
        test, meta = _load(f.stem)
        _, arch, fw = f.stem.split("__")
        lv = prep.to_level(test, tt)
        m = forecast_metrics(prep.truth(tt)[:, 0], lv[:, 0], prep.level[tt])
        name = LABEL[arch] + (" + recurrent dropout" if fw.endswith("_rd") else "")
        rows.append(_row(prep.spec.id, name, fw.removesuffix("_rd"), m, skill=1 - m["rmse"] / pers_rmse,
                         seconds=meta["seconds"], n_params=meta["n_params"], epochs=meta["epochs"],
                         rmse_last=float(horizon_rmse(prep.truth(tt), lv)[-1])))
    return rows


def _interval_cover(prep, arch, lv, tt):
    import torch
    q = torch.load(model_dir("ch4") / f"{prep.spec.id}__{arch}.pt", map_location="cpu", weights_only=True)["q"].numpy()
    z = np.log(lv) if prep.spec.log_target else lv
    lo, hi = z + q[0], z + q[1]
    lo, hi = (np.exp(lo), np.exp(hi)) if prep.spec.log_target else (lo, hi)
    return coverage(prep.truth(tt), lo, hi), float(np.mean(hi - lo))


def error_rows(prep, tt, lv):
    """Phân tích lỗi một bước: MAE theo giờ (power), tháng (weather) hoặc ba mức biến động (cổ phiếu)."""
    idx, truth = prep.index[tt + 1], prep.truth(tt)[:, 0]
    if prep.spec.log_target:
        vol = pd.Series(prep.x[tt, prep.columns.index("volat")])
        group = pd.qcut(vol, 3, labels=["thấp", "vừa", "cao"]).astype(str).to_numpy()
    else:
        group = idx.hour if prep.spec.id == "power" else idx.month
    frame = pd.DataFrame({"g": group, "model": np.abs(truth - lv[:, 0]), "base": np.abs(truth - prep.level[tt])})
    g = frame.groupby("g", sort=False)[["model", "base"]].mean().reset_index()
    return [{"group": r.g if isinstance(r.g, str) else int(r.g), "model": r.model, "base": r.base}
            for r in g.itertuples()]


def summarize(prep, tt, rows, preds):
    """Chọn kiến trúc theo MSE validation rồi mới đọc kết quả test của nó."""
    s = prep.spec.id
    nn = [r for r in rows if r["framework"] == "pytorch"]
    best = min(nn, key=lambda r: r["val_mse"])
    arch = next(a for a in ARCHS if LABEL[a] == best["model"])
    pers = next(r for r in rows if r["model"] == "Persistence")
    cov, width = _interval_cover(prep, arch, preds[arch], tt)
    y = prep.truth(tt)[:, 0]
    return {"series": s, "best": best["model"], "rmse": best["rmse"], "rmse_pers": pers["rmse"],
            "skill_pct": 100 * best["skill"], "beat": bool(best["rmse"] < pers["rmse"]), "dm_p": best["dm_p"],
            "significant": bool(best["dm_p"] < 0.05 and best["rmse"] < pers["rmse"]), "dir_acc": best["dir_acc"],
            "majority": majority_direction(y, prep.level[tt]), "coverage": cov, "width": width,
            "n_rows": len(prep.level), "n_train": len(prep.last_index("train")), "n_val": len(prep.last_index("val")),
            "n_test": len(tt), "start": str(prep.index[0].date()), "end": str(prep.index[-1].date()),
            "val_start": str(prep.index[prep.a].date()), "test_start": str(prep.index[prep.b].date())}


def collect(series):
    prep = Prepared(series)
    tt = prep.last_index("test")
    rows = baseline_rows(prep, tt)
    pers = rows[0]["rmse"]
    truth, preds = prep.truth(tt), {}
    for arch in ARCHS:
        row, preds[arch] = torch_row(prep, arch, tt, pers)
        rows.append(row)
    rows += side_rows(prep, tt, pers)
    curve = {LABEL[a]: horizon_rmse(truth, p).tolist() for a, p in preds.items()}
    curve["Persistence"] = horizon_rmse(truth, prep.persistence(tt)).tolist()
    if prep.seasonal(tt) is not None:
        curve["Mùa vụ 24 giờ"] = horizon_rmse(truth, prep.seasonal(tt)).tolist()
    summary = summarize(prep, tt, rows, preds)
    arch = next(a for a in ARCHS if LABEL[a] == summary["best"])
    return rows, curve, summary, error_rows(prep, tt, preds[arch])


def log_baselines(rows):
    runs = OUT / "runs.jsonl"
    if '"chapter": "ch4", "dataset": "FPT", "model": "Persistence"' in runs.read_text(encoding="utf-8"):
        return
    for r in rows:
        if r["framework"] == "baseline":
            m = {k: r[k] for k in METRIC_KEYS if r.get(k) is not None}
            log_run("ch4", r["dataset"], r["model"], "baseline", m, 0.0, "cpu")


def tl02_reference():
    """Số thật của Tiểu luận 02 trên cùng bốn mã (cửa sổ 30, hai lớp, chia theo mốc ngày), dùng làm đối chiếu độc lập."""
    d = json.loads(TL02.read_text(encoding="utf-8"))
    rows = [{"ticker": t, "arch": a, "rmse_ratio": d[t][a]["rmse_ratio"], "dir_acc": d[t][a]["dir_acc"]}
            for t in TICKERS for a in ARCHS if a in d.get(t, {})]
    ratio = [r["rmse_ratio"] for r in rows]
    return {"reused_from": TL02.relative_to(ROOT.parent).as_posix(), "config": d["config"], "rows": rows,
            "ratio_min": min(ratio), "ratio_max": max(ratio), "n": len(rows)}


def main():
    rows, curves, summ, errors = [], {}, {}, {}
    for s in SPECS:
        r, c, sm, e = collect(s)
        rows += r
        curves[s], summ[s], errors[s] = c, sm, e
    log_baselines(rows)
    wins = [s for s in TICKERS if summ[s]["significant"]]
    save_json("ch4_tl02", tl02_reference())
    save_json("ch4_compare", {"rows": rows, "primary": "rmse", "lower_is_better": True})
    save_json("ch4_horizon", {"series": curves})
    save_json("ch4_errors", {"rows": errors})
    save_json("ch4_summary", {**summ, "stocks_beat": sum(summ[s]["beat"] for s in TICKERS),
                              "stocks_significant": len(wins), "n_tickers": len(TICKERS),
                              "numpy_rmse": next(r["rmse"] for r in rows if r["framework"] == "numpy")})


if __name__ == "__main__":
    main()
