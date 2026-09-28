"""Kiểm chứng Web App bằng test client của Flask (không cần mở cổng).

Gọi 4 endpoint chính cho cả hai framework, dừng ngay nếu có mã khác 200 hoặc thiếu khoá; lưu phản hồi thật vào
outputs/metrics/app.json để báo cáo trích số liệu.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a06.export import save_metrics  # noqa: E402
from app import app  # noqa: E402

KEYS = {"history": {"dates", "close", "ma20", "rsi14", "last"},
        "meta": {"models", "best", "cards", "presets", "stats", "naive_rmse"},
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
    meta = check("meta", call(c, "GET", "/api/meta"))
    if len(meta["cards"]) != 16:
        raise SystemExit(f"meta: cần 16 thẻ mô hình, có {len(meta['cards'])}")
    hist = check("history", call(c, "GET", "/api/stock/history"))
    for n in (60, 250):
        if len(check("history", call(c, "GET", f"/api/stock/history?n={n}"))["close"]) != n:
            raise SystemExit(f"history?n={n} trả sai số phiên")
    stock = {fw: check("stock", call(c, "POST", "/api/stock/predict", {"framework": fw})) for fw in ("torch", "keras")}
    churn = {f"{p}_{fw}": check("churn", call(c, "POST", "/api/churn/predict", {"preset": p, "framework": fw}))
             for p in ("churn", "loyal") for fw in ("torch", "keras")}
    bad = c.post("/api/churn/predict", json={"sequence": [[0] * 8] * 5}).status_code   # shape sai phải bị từ chối
    bad_n = c.get("/api/stock/history?n=7").status_code                                # dải lịch sử ngoài 60/100/250

    n = len(hist["close"])
    save_metrics("app", {"history": {"n": n, "first": hist["dates"][0], "last": hist["dates"][-1],
                                     "close_last": hist["close"][-1]},
                         "stock": stock, "churn": {k: {kk: v for kk, v in r.items() if kk != "sequence"}
                                                   for k, r in churn.items()},
                         "bad_shape_status": bad, "bad_n_status": bad_n,
                         "n_calls": 4 + len(stock) + len(churn)})
    print("4 endpoint trả 200:", 4 + len(stock) + len(churn), "lượt gọi · sai shape ->", bad, "· n sai ->", bad_n)
    for k, r in stock.items():
        print(f"  stock {k}: {r['model_label']} {r['last_close']} -> {r['pred_close']} USD ({r['latency_ms']} ms)")
    for k, r in churn.items():
        print(f"  churn {k}: {r['model_label']} p = {r['prob']} · rủi ro {r['risk']} ({r['latency_ms']} ms)")
    if bad != 400 or bad_n != 400:
        raise SystemExit("chuỗi sai shape và n sai phải trả 400")


if __name__ == "__main__":
    main()
