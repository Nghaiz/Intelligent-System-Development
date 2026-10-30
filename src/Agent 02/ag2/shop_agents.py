"""Thông điệp và bốn tác tử chuyên trách của hệ mua sắm (Chương 8). Mỗi tác tử chỉ nhận đúng loại thông điệp của
mình qua handle(msg) và trả một Message cho coordinator; dữ liệu nội bộ chỉ đọc, không tác tử nào sửa kho."""
import itertools
import random
import re
from dataclasses import dataclass, field

from .shop_data import CATEGORIES, INTENTS, strip_accents

_IDS = itertools.count(1)


@dataclass
class Message:
    sender: str
    receiver: str
    kind: str
    payload: dict
    msg_id: int = field(default_factory=lambda: next(_IDS))
    deadline_ms: float = 80.0


def reply(msg, kind, payload):
    """Phản hồi đúng hợp đồng: đảo người gửi/nhận, giữ nguyên hạn chót của yêu cầu."""
    return Message(msg.receiver, msg.sender, kind, payload, deadline_ms=msg.deadline_ms)


class IntentAgent:
    """CLASSIFY -> INTENT {intent, conf}. Dưới ngưỡng tin cậy thì đánh dấu cần làm rõ thay vì đoán."""
    name = "intent"

    def __init__(self, model, predict_proba, threshold):
        self.model, self.predict_proba, self.threshold = model, predict_proba, threshold

    def handle(self, msg):
        p = self.predict_proba(self.model, [msg.payload["query"]])[0]
        conf, k = float(p.max()), int(p.argmax())
        return reply(msg, "INTENT", {"intent": INTENTS[k], "conf": conf, "confident": conf >= self.threshold})


BUDGET_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(trieu|tr|million|k|nghin)(\s*ruoi)?")


def parse_query(query):
    """Hiểu câu truy vấn: loại hàng, thương hiệu và ngân sách (triệu đồng) từ dạng không dấu."""
    q = strip_accents(query.lower())
    cat = next((c for c, (keys, *_) in CATEGORIES.items() if any(strip_accents(k) in q for k in keys)), None)
    brands = [b for b in (CATEGORIES[cat][2] if cat else ()) if strip_accents(b.lower()) in q]
    budget = None
    m = BUDGET_RE.search(q)
    if m:
        v = float(m.group(1).replace(",", "."))
        budget = v / 1000 if m.group(2) in ("k", "nghin") else v
        budget += 0.5 if m.group(3) else 0.0
    return cat, brands, budget


class CatalogAgent:
    """SEARCH {query} -> CANDIDATES {category, brands, budget, ids}. Không nhận ra loại hàng thì trả rỗng,
    không bao giờ trả cả catalog."""
    name = "catalog"

    def __init__(self, catalog):
        self.catalog = catalog

    def search(self, query):
        cat, brands, budget = parse_query(query)
        pool = [p for p in self.catalog if p["category"] == cat]
        hit = [p for p in pool if p["brand"] in brands] or pool
        return {"category": cat, "brands": brands, "budget": budget, "ids": [p["id"] for p in hit]}

    def handle(self, msg):
        return reply(msg, "CANDIDATES", self.search(msg.payload["query"]))


class RecommendationAgent:
    """RANK {ids, budget, brands, mode} -> RANKED {ids}. Chỉ chọn trong ứng viên Catalog đưa; lọc theo giá sau
    khuyến mãi, xếp theo điểm đánh giá giảm dần rồi giá tăng dần; mode='compare' lấy mẫu tốt nhất mỗi thương hiệu."""
    name = "recommend"

    def __init__(self, catalog, k=3):
        self.by_id, self.k = {p["id"]: p for p in catalog}, k

    def rank(self, ids, budget=None, brands=(), mode="top"):
        items = [self.by_id[i] for i in ids]
        if budget is not None:
            items = [p for p in items if p["price"] * (1 - p["promo"] / 100) <= budget]
        items.sort(key=lambda p: (-p["rating"], p["price"]))
        if mode == "compare" and len(brands) >= 2:
            items = [next(p for p in items if p["brand"] == b) for b in brands[:2] if any(p["brand"] == b for p in items)]
        return [p["id"] for p in items[: (2 if mode == "compare" else self.k)]]

    def handle(self, msg):
        pl = msg.payload
        return reply(msg, "RANKED", {"ids": self.rank(pl["ids"], pl.get("budget"), pl.get("brands", ()), pl["mode"])})


class InventoryAgent:
    """CHECK_STOCK {ids} -> STOCK {stock, age_s}. Mô phỏng dịch vụ kho thật: với xác suất fail_rate, lời gọi gặp
    một trong ba lỗi đều nhau — quá hạn (TimeoutError), thông điệp hỏng (thiếu 'stock'), dữ liệu cũ (age 3600 s)."""
    name = "inventory"
    FAULTS = ("timeout", "malformed", "stale")

    def __init__(self, stock, stale_stock, fail_rate=0.0, seed=0):
        self.stock, self.stale_stock, self.fail_rate = stock, stale_stock, fail_rate
        self.rng = random.Random(seed)
        self.faults = {f: 0 for f in self.FAULTS}

    def lookup(self, ids):
        fault = self.rng.choice(self.FAULTS) if self.rng.random() < self.fail_rate else None
        if fault:
            self.faults[fault] += 1
        if fault == "timeout":
            raise TimeoutError("inventory không trả lời trước hạn chót")
        if fault == "malformed":
            return {"qty": [self.stock[i] for i in ids]}
        src, age = (self.stale_stock, 3600) if fault == "stale" else (self.stock, 0)
        return {"stock": {i: src[i] for i in ids}, "age_s": age}

    def handle(self, msg):
        return reply(msg, "STOCK", self.lookup(msg.payload["ids"]))
