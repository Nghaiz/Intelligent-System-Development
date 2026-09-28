"""Kiểm chứng Web App bằng test client của Flask (không cần mở cổng).

Gọi 3 endpoint chính cho cả hai framework, dừng ngay nếu có mã khác 200 hoặc thiếu khoá; lưu phản hồi thật vào
outputs/metrics/app.json và dữ liệu cho hai khung giao diện vẽ lại bằng TikZ trong báo cáo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a06.export import save_metrics, write_dat  # noqa: E402
from app import app  # noqa: E402

KEYS = {"history": {"dates", "close", "ma20"},
        "stock": {"pred_close", "last_close", "change_pct", "latency_ms", "model", "rsi14", "rsi_zone", "note"},
        "churn": {"prob", "risk", "threshold", "latency_ms", "sequence", "advice", "trend"}}


def call(client, method, path, body=None):
    r = client.get(path) if method == "GET" else client.post(path, json=body)
    if r.status_code != 200:
        raise SystemExit(f"{method} {path} {body} -> {r.status_code}: {r.get_json()}")
    return r.get_json()


def check(name, data):
    missing = KEYS[name] - set(data)
    if missing:
        raise SystemExit(f"{name}: thiếu khoá {missing}")
    return data


def main():
    c = app.test_client()
    hist = check("history", call(c, "GET", "/api/stock/history"))
    stock = {fw: check("stock", call(c, "POST", "/api/stock/predict", {"framework": fw})) for fw in ("torch", "keras")}
    churn = {f"{p}_{fw}": check("churn", call(c, "POST", "/api/churn/predict", {"preset": p, "framework": fw}))
             for p in ("churn", "loyal") for fw in ("torch", "keras")}
    bad = c.post("/api/churn/predict", json={"sequence": [[0] * 8] * 5}).status_code   # shape sai phải bị từ chối

    n = len(hist["close"])
    write_dat("app_stock", {"i": list(range(n)), "close": hist["close"], "ma20": hist["ma20"]})
    seq_c, seq_l = churn["churn_torch"]["sequence"], churn["loyal_torch"]["sequence"]
    write_dat("app_churn", {"day": list(range(1, 32)), "churn_secs": [d[6] for d in seq_c],
                            "loyal_secs": [d[6] for d in seq_l], "churn_unq": [d[5] for d in seq_c],
                            "loyal_unq": [d[5] for d in seq_l]})
    save_metrics("app", {"history": {"n": n, "first": hist["dates"][0], "last": hist["dates"][-1],
                                     "close_last": hist["close"][-1]},
                         "stock": stock, "churn": {k: {kk: v for kk, v in r.items() if kk != "sequence"}
                                                   for k, r in churn.items()},
                         "bad_shape_status": bad, "n_calls": 1 + len(stock) + len(churn)})
    print("3 endpoint trả 200:", 1 + len(stock) + len(churn), "lượt gọi · sai shape ->", bad)
    for k, r in stock.items():
        print(f"  stock {k}: {r['model_label']} {r['last_close']} -> {r['pred_close']} USD ({r['latency_ms']} ms)")
    for k, r in churn.items():
        print(f"  churn {k}: {r['model_label']} p = {r['prob']} · rủi ro {r['risk']} ({r['latency_ms']} ms)")
    if bad != 400:
        raise SystemExit("chuỗi sai shape phải trả 400")


if __name__ == "__main__":
    main()
