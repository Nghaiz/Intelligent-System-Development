"""Chương 4: sinh bảng generated/ch4_*.tex từ ch4_datasets.json, ch4_compare.json và ch4_summary.json."""
import json

from tlf.ch4_data import OUT_DIR, TICKERS
from tlf.ch4_torch import ARCH_NAMES
from tlf.common import load_json, save_json
from tlf.texfmt import best, num, table, tex_escape

LABEL = {"FPT": "FPT", "VCB": "VCB", "HPG": "HPG", "VIC": "VIC", "weather": "Nhiệt độ", "power": "Công suất"}
LAG = 48
DEC = {"FPT": 0, "VCB": 0, "HPG": 0, "VIC": 0, "weather": 3, "power": 3}


def _p(v):
    return "<0{,}001" if v < 0.001 else num(v, 3)


def _opt(v, f):
    return "—" if v is None else f(v)


def sample_tables(ds):
    specs = {"vn30": (("Ngày", "Open", "High", "Low", "Close", "Volume"), ("Open", "High", "Low", "Close", "Volume"), 0),
             "weather": (("Giờ", "Nhiệt độ", "Ẩm \\%", "Áp suất", "Gió", "Mưa"),
                         ("temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
                          "precipitation"), 1),
             "power": (("Giờ", "P (kW)", "Q (kW)", "U (V)", "Điều hòa (Wh)"),
                       ("Global_active_power", "Global_reactive_power", "Voltage", "Sub_metering_3"), 2)}
    for d in ds:
        head, cols, dec = specs[d["id"]]
        fmt_t = (lambda s: s[:10]) if d["id"] == "vn30" else (lambda s: s[:16])
        rows = [[fmt_t(r["time"])] + [num(r[c], dec) for c in cols] for r in d["sample_rows"]]
        table(f"ch4_sample_{d['id']}", "l" + "r" * len(cols), head, rows)


def detail_table(rows):
    """Đủ năm chỉ số cho FPT và hai chuỗi theo giờ; in đậm RMSE thấp nhất của từng chuỗi."""
    out = []
    for s in ("FPT", "weather", "power"):
        part = [r for r in rows if r["dataset"] == s and r["framework"] in ("baseline", "pytorch")]
        rmse = [r["rmse"] for r in part]
        f_rmse = best(rmse, [num(v, DEC[s]) for v in rmse], higher=False)
        out.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{LABEL[s]}}}}} \\\\")
        for r, fr in zip(part, f_rmse):
            out.append([r["model"], num(r["mae"], DEC[s]), fr, num(r["mape"], 2), num(r["smape"], 2),
                        _opt(r.get("dir_acc"), lambda v: num(100 * v, 1)),
                        _opt(r.get("skill"), lambda v: num(100 * v, 2, sign=True)), _opt(r.get("dm_p"), _p)])
    table("ch4_detail", "lrrrrrrr", ["Mô hình", "MAE", "RMSE", "MAPE \\%", "sMAPE \\%", "Hướng \\%", "Giảm RMSE \\%", "DM $p$"], out)


def ticker_table(rows, summ):
    out = []
    for s in TICKERS:
        part = {r["model"]: r for r in rows if r["dataset"] == s and r["framework"] in ("baseline", "pytorch")}
        m = summ[s]
        out.append([s, num(part["Persistence"]["rmse"], 0)] + [num(part[a]["rmse"], 0) for a in ARCH_NAMES.values()]
                   + [m["best"], num(m["skill_pct"], 2, sign=True), _p(m["dm_p"]), num(100 * m["dir_acc"], 1),
                      num(100 * m["majority"], 1)])
    table("ch4_tickers", "lrrrrrlrrrr",
          ["Mã", "Pers.", *ARCH_NAMES.values(), "Chọn", "Giảm \\%", "DM $p$", "Hướng \\%", "Đa số \\%"], out)


def framework_table(rows):
    out = []
    for r in rows:
        if r["framework"] not in ("numpy", "keras"):
            continue
        d = DEC[r["dataset"]]
        base = next(x for x in rows if x["dataset"] == r["dataset"] and x["model"] == r["model"].split(" +")[0]
                    and x["framework"] == "pytorch")
        out.append([LABEL[r["dataset"]], tex_escape(r["model"]), r["framework"], num(r["rmse"], d), num(base["rmse"], d),
                    num(100 * r["skill"], 2, sign=True), r["epochs"], num(r["seconds"], 0)])
    table("ch4_frameworks", "lllrrrrr", ["Chuỗi", "Mô hình", "Khung", "RMSE", "RMSE PyTorch", "Giảm RMSE \\%", "Epoch", "Giây"], out)


def horizon_table(rows, summ):
    out = []
    for s in LABEL:
        m = summ[s]
        part = {r["model"]: r for r in rows if r["dataset"] == s and r["framework"] in ("baseline", "pytorch")}
        out.append([LABEL[s], m["best"], num(part["Persistence"]["rmse_last"], DEC[s]),
                    num(part[m["best"]]["rmse_last"], DEC[s]), num(100 * m["coverage"], 1), num(m["width"], DEC[s])])
    table("ch4_horizon", "llrrrr", ["Chuỗi", "Mô hình", "Pers. bước $H$", "Mô hình bước $H$", "Phủ 90 \\%", "Rộng"], out)


def theory_facts():
    """Số liệu của phần lý thuyết: độ nhạy theo trễ và kiểm tra gradient, đo trên mô hình đã huấn luyện."""
    sens = load_json("ch4_sensitivity")["series"]
    run = json.loads((OUT_DIR / "runs" / "FPT__rnn__numpy.json").read_text(encoding="utf-8"))
    norms = run["delta_norms"]
    save_json("ch4_theory", {"lag": LAG, "weather": {a: v[LAG] for a, v in sens["weather"].items()},
                             "power": {a: v[LAG] for a, v in sens["power"].items()},
                             "grad_check_max": max(run["grad_check"].values()), "numpy_params": run["n_params"],
                             "numpy_epochs": run["epochs"], "numpy_seconds": run["seconds"],
                             "delta_ratio": norms[-1] / norms[0]})


def main():
    theory_facts()
    rows, summ = load_json("ch4_compare")["rows"], load_json("ch4_summary")
    sample_tables(load_json("ch4_datasets")["datasets"])
    detail_table(rows)
    ticker_table(rows, summ)
    framework_table(rows)
    horizon_table(rows, summ)


if __name__ == "__main__":
    main()
