"""Chương 8 — hệ đa tác tử mua sắm: huấn luyện bộ phân loại ý định trên GPU, chạy demo, kiểm thử,
mô phỏng lỗi kho và so sánh pipeline đơn khối với MAS. Ghi outputs/metrics/ch8_shop.json.

    python scripts/ch8_run.py && python scripts/ch8_figures.py && python scripts/ch8_tables.py
"""
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ag2.common import save_json  # noqa: E402
from ag2.shop_agents import CatalogAgent, IntentAgent, InventoryAgent, RecommendationAgent  # noqa: E402
from ag2.shop_data import INTENTS, build_catalog, build_intent_dataset, build_stock  # noqa: E402
from ag2.shop_intent import predict_proba, train_intent  # noqa: E402
from ag2.shop_mas import Coordinator, pipeline_handle  # noqa: E402

torch.set_num_threads(4)
MODELS = ROOT / "models" / "ch8"
FAIL_RATES = (0.0, 0.1, 0.3)
DEMO = ["Tìm giúp mình laptop dưới 20 triệu", "so sanh tai nghe Sony voi JBL", "giày Adidas còn hàng không ạ",
        "Xin chào shop"]


def choose_threshold(proba, y, target=0.95):
    """Ngưỡng tin cậy thấp nhất (0,30…0,90) mà độ chính xác trên phần câu được chấp nhận của tập val ≥ target (0,95)."""
    conf, pred = proba.max(1).values.numpy(), proba.argmax(1).numpy()
    for tau in np.round(np.arange(0.30, 0.901, 0.05), 2):
        keep = conf >= tau
        if keep.any() and (pred[keep] == y[keep]).mean() >= target:
            return float(tau)
    return 0.90


def classification_report(proba, y, tau):
    """Accuracy, macro-F1, P/R/F1 từng ý định, ma trận nhầm lẫn, độ phủ và độ chính xác sau ngưỡng."""
    pred, conf = proba.argmax(1).numpy(), proba.max(1).values.numpy()
    k = len(INTENTS)
    cm = np.zeros((k, k), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    per = {}
    for i, name in enumerate(INTENTS):
        prec = cm[i, i] / max(cm[:, i].sum(), 1)
        rec = cm[i, i] / max(cm[i].sum(), 1)
        per[name] = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
                     "support": int(cm[i].sum())}
    keep = conf >= tau
    return {"accuracy": float((pred == y).mean()), "macro_f1": float(np.mean([v["f1"] for v in per.values()])),
            "per_class": per, "coverage": float(keep.mean()), "accepted_acc": float((pred[keep] == y[keep]).mean()),
            "conf": {"rows": cm.tolist()}}


class ScriptedRng(random.Random):
    """Dãy lỗi định trước cho demo: mỗi lời gọi kho lấy phần tử kế tiếp (None = không lỗi)."""

    def __init__(self, script):
        super().__init__(0)
        self.script = list(script)

    def random(self):
        self.fault = self.script.pop(0) if self.script else None
        return 0.0 if self.fault else 1.0

    def choice(self, _):
        return self.fault


def evaluate(system, queries, oracle, now):
    """So từng phản hồi với đáp án chuẩn; 'khẳng định sai' = báo còn hàng/tồn kho khác kho thật.
    system(câu, ý định đúng) — hệ tự quyết có dùng ý định đúng hay tự phân loại."""
    ok = unsafe = 0
    lat, msgs, status = [], [], {}
    for (q, lab), ref in zip(queries, oracle):
        r = system(q, INTENTS[lab])
        ok += (r["status"], r["items"]) == (ref["status"], ref["items"])
        unsafe += any(now[i] == 0 for i in r["items"]) if r["intent"] == "tim_san_pham" else \
            any(now[i] != v for i, v in r["stock"].items())
        lat.append(r["latency_ms"])
        msgs.append(r["messages"])
        status[r["status"]] = status.get(r["status"], 0) + 1
    n = len(queries)
    return {"completion": ok / n, "unsafe": unsafe, "unsafe_rate": unsafe / n, "msgs_mean": float(np.mean(msgs)),
            "lat_mean": float(np.mean(lat)), "lat_p95": float(np.percentile(lat, 95)), "status": status,
            "unverified": status.get("chua_xac_minh", 0), "errors": status.get("loi", 0)}


def main():
    data = build_intent_dataset()
    model, hist, info = train_intent(data)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS / "intent_net.pt")
    yv = np.array([lab for _, lab in data["val"]])
    tau = choose_threshold(predict_proba(model, [t for t, _ in data["val"]]), yv)
    yt = np.array([lab for _, lab in data["test"]])
    test = classification_report(predict_proba(model, [t for t, _ in data["test"]]), yt, tau)
    print(f"{info['device']} · {info['train_seconds']:.1f} s · test acc {test['accuracy']:.4f} · tau {tau}")

    catalog = build_catalog()
    now, old = build_stock(catalog)
    intent = IntentAgent(model, predict_proba, tau)
    cat_a, rec_a = CatalogAgent(catalog), RecommendationAgent(catalog)

    inv = InventoryAgent(now, old, fail_rate=1.0)
    inv.rng = ScriptedRng(["timeout", None, None])
    demo_coord = Coordinator([intent, cat_a, rec_a, inv], seed=1, verbose=True)
    demo = []
    for q in DEMO:
        r = demo_coord.handle(q)
        demo.append({"query": q, "status": r["status"], "items": r["items"], "messages": r["messages"],
                     "latency_ms": r["latency_ms"], "log": demo_coord.log})

    clean = Coordinator([cat_a, rec_a, InventoryAgent(now, old)], seed=2)
    queries = data["test"]
    oracle = [clean.handle(q, intent=INTENTS[lab]) for q, lab in queries]
    e2e = Coordinator([intent, cat_a, rec_a, InventoryAgent(now, old)], seed=4)
    end_to_end = evaluate(lambda q, _: e2e.handle(q), queries, oracle, now)
    faults = {}
    for rate in FAIL_RATES:
        inv_m, inv_p = InventoryAgent(now, old, rate, seed=11), InventoryAgent(now, old, rate, seed=11)
        mas = Coordinator([intent, cat_a, rec_a, inv_m], seed=3)
        rng = random.Random(3)
        key = f"r{int(rate * 100)}"
        # cô lập phần điều phối: cả hai hệ nhận ý định đúng, chỉ khác cách xử lý lỗi kho
        faults[key] = {"rate": rate,
                       "mas": evaluate(lambda q, it: mas.handle(q, it), queries, oracle, now)
                       | {"faults": dict(inv_m.faults)},
                       "pipe": evaluate(lambda q, it: pipeline_handle(q, intent, cat_a, rec_a, inv_p, rng, it),
                                        queries, oracle, now) | {"faults": dict(inv_p.faults)}}
        print(key, {s: round(faults[key][s]["completion"], 4) for s in ("mas", "pipe")})

    suite = unittest.defaultTestLoader.loadTestsFromName("ch8_tests")
    tr = unittest.TextTestRunner(verbosity=0).run(suite)
    save_json("ch8_shop", {
        "data": {s: len(v) for s, v in data.items()} | {"total": sum(len(v) for v in data.values()),
                                                       "n_intents": len(INTENTS), "n_products": len(catalog),
                                                       "in_stock": sum(q > 0 for q in now.values())},
        "train": info | {"tau": tau, "best_val_acc": max(h["val_acc"] for h in hist), "history": hist},
        "test": test, "e2e": end_to_end, "demo": {"rows": demo}, "faults": faults,
        "tests": {"run": tr.testsRun, "failed": len(tr.failures) + len(tr.errors)}})


if __name__ == "__main__":
    main()
