"""Kiểm thử đơn vị hệ mua sắm (Chương 8): python scripts/ch8_tests.py"""
import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ag2.shop_agents import CatalogAgent, InventoryAgent, Message, RecommendationAgent, parse_query  # noqa: E402
from ag2.shop_data import build_catalog, build_stock  # noqa: E402
from ag2.shop_mas import ContractError, Coordinator, pipeline_handle  # noqa: E402


class StubIntent:
    """Bộ phân loại giả luôn trả 'tim_san_pham' với độ tin cậy 0,9 để kiểm thử không cần GPU."""
    name, model, threshold = "intent", None, 0.5

    @staticmethod
    def predict_proba(_, texts):
        return torch.tensor([[0.9, 0.02, 0.02, 0.02, 0.02, 0.02]] * len(texts))


def make(fault=None, fail_rate=0.0):
    cat = build_catalog()
    now, old = build_stock(cat)
    inv = InventoryAgent(now, old, fail_rate=1.0 if fault else fail_rate)
    if fault:
        inv.FAULTS = (fault,)
    return cat, now, inv, Coordinator([CatalogAgent(cat), RecommendationAgent(cat), inv])


class ShopTests(unittest.TestCase):
    def test_parse_budget_and_category(self):
        self.assertEqual(parse_query("tai nghe Sony 500k"), ("tai_nghe", ["Sony"], 0.5))
        self.assertEqual(parse_query("dien thoai duoi 1 trieu ruoi")[2], 1.5)

    def test_unknown_query_returns_nothing(self):
        self.assertEqual(CatalogAgent(build_catalog()).search("thời tiết hôm nay")["ids"], [])

    def test_budget_respected(self):
        cat = build_catalog()
        by_id = {p["id"]: p for p in cat}
        ids = [p["id"] for p in cat if p["category"] == "laptop"]
        for i in RecommendationAgent(cat).rank(ids, budget=20.0):
            self.assertLessEqual(by_id[i]["price"] * (1 - by_id[i]["promo"] / 100), 20.0)

    def test_out_of_stock_excluded(self):
        _, now, _, coord = make()
        res = coord.handle("tìm laptop", intent="tim_san_pham")
        self.assertTrue(all(now[i] > 0 for i in res["items"]))

    def test_faults_never_claim_stock(self):
        for fault in ("timeout", "malformed", "stale"):
            _, _, _, coord = make(fault)
            res = coord.handle("tìm laptop", intent="tim_san_pham")
            self.assertEqual((res["status"], res["items"], res["retries"]), ("chua_xac_minh", [], 2), fault)

    def test_contract_rejects_wrong_sender(self):
        with self.assertRaises(ContractError):
            Coordinator.check(Message("catalog", "coordinator", "STOCK", {}), "inventory", "STOCK")

    def test_greeting_needs_no_agents(self):
        res = make()[3].handle("xin chào", intent="chao_hoi")
        self.assertEqual((res["status"], res["messages"]), ("chao", 0))

    def test_pipeline_trusts_stale_stock(self):
        cat, now, inv, _ = make("stale")
        res = pipeline_handle("tìm balo", StubIntent, CatalogAgent(cat), RecommendationAgent(cat), inv, random.Random(0))
        self.assertEqual(res["status"], "ok")  # điểm yếu được ghi nhận: pipeline không kiểm tuổi dữ liệu


if __name__ == "__main__":
    unittest.main(verbosity=2)
