"""Coordinator điều phối tập trung và pipeline đơn khối để so sánh (Chương 8).

Độ trễ dùng đồng hồ ảo: mỗi chặng thông điệp tốn một khoảng trễ mạng lognormal (trung vị 2 ms), cộng thời gian
tính toán thật của tác tử; lời gọi quá hạn tốn trọn hạn chót 80 ms. Nhờ vậy số đo không phụ thuộc máy đang bận.
"""
import math
import random
import time

from .shop_agents import Message
from .shop_data import INTENTS

MAX_AGE_S = 60


class ContractError(Exception):
    """Phản hồi sai hợp đồng: sai lớp, sai người gửi/nhận, sai loại, thiếu trường hoặc dữ liệu kho quá cũ."""


class Coordinator:
    """Nhận câu của người dùng, gọi lần lượt Intent -> Catalog -> Recommend -> Inventory qua thông điệp,
    kiểm hợp đồng từng phản hồi, thử lại tối đa max_retries lần khi lỗi; không bao giờ khẳng định còn hàng
    khi chưa xác minh được tồn kho."""

    def __init__(self, agents, max_retries=2, seed=0, verbose=False):
        self.agents = {a.name: a for a in agents}
        self.max_retries, self.verbose = max_retries, verbose
        self.rng = random.Random(seed)

    def _hop(self, msg):
        self.messages += 1
        self.clock += self.rng.lognormvariate(math.log(2.0), 0.35)
        if self.verbose:
            self.log.append(f"[{self.clock:6.1f} ms] {msg.sender:>11} -> {msg.receiver:<11} {msg.kind:<12} "
                            f"{str(msg.payload)[:58]}")

    @staticmethod
    def check(resp, receiver, expect):
        if not isinstance(resp, Message) or resp.sender != receiver or resp.receiver != "coordinator":
            raise ContractError("sai người gửi/nhận")
        if resp.kind != expect or not isinstance(resp.payload, dict):
            raise ContractError(f"chờ {expect}, nhận {resp.kind}")
        if expect == "STOCK":
            if not isinstance(resp.payload.get("stock"), dict):
                raise ContractError("STOCK thiếu trường stock")
            if resp.payload["age_s"] > MAX_AGE_S:
                raise ContractError(f"dữ liệu kho cũ {resp.payload['age_s']} s")

    def call(self, receiver, kind, payload, expect):
        """Gửi một yêu cầu, chờ phản hồi đúng hợp đồng; lỗi thì ghi sự kiện và thử lại. Hết lượt -> None."""
        for attempt in range(self.max_retries + 1):
            msg = Message("coordinator", receiver, kind, payload)
            self._hop(msg)
            t0 = time.perf_counter()
            try:
                resp = self.agents[receiver].handle(msg)
                self.clock += 1000 * (time.perf_counter() - t0)
                self.check(resp, receiver, expect)
                self._hop(resp)
                return resp.payload
            except (TimeoutError, ContractError) as e:
                self._fault(msg, e, attempt)
        return None

    def _fault(self, msg, err, attempt):
        """Ghi sự kiện lỗi; quá hạn thì đồng hồ chạy hết hạn chót. Còn lượt thì tính là một lần thử lại."""
        self.clock += msg.deadline_ms if isinstance(err, TimeoutError) else 0.0
        self.events.append(f"{msg.receiver}: {err}")
        if self.verbose:
            self.log.append(f"[{self.clock:6.1f} ms]  !! {msg.receiver}: {err} (lần {attempt + 1})")
        self.retries += attempt < self.max_retries

    def handle(self, query, intent=None):
        """Một yêu cầu trọn vẹn -> {status, intent, items, stock, messages, retries, latency_ms}."""
        self.messages, self.retries, self.clock, self.events, self.log = 0, 0, 0.0, [], []
        if intent is None:
            out = self.call("intent", "CLASSIFY", {"query": query}, "INTENT")
            intent = out["intent"] if out["confident"] else "lam_ro"
        res = {"intent": intent, "items": [], "stock": {}}
        if intent in ("chao_hoi", "khac", "lam_ro"):
            res["status"] = {"chao_hoi": "chao", "khac": "chuyen_nhan_vien", "lam_ro": "can_lam_ro"}[intent]
            return self._done(res)
        cand = self.call("catalog", "SEARCH", {"query": query}, "CANDIDATES")
        if not cand["ids"]:
            res["status"] = "can_lam_ro"
            return self._done(res)
        mode = "compare" if intent == "so_sanh" else "top"
        budget = cand["budget"] if intent == "tim_san_pham" else None
        ranked = self.call("recommend", "RANK", {"ids": cand["ids"], "budget": budget, "brands": cand["brands"],
                                                 "mode": mode}, "RANKED")["ids"]
        res["items"], res["status"] = ranked, "ok" if ranked else "khong_co"
        if intent in ("tim_san_pham", "hoi_ton_kho") and ranked:
            inv = self.call("inventory", "CHECK_STOCK", {"ids": ranked}, "STOCK")
            if inv is None:
                res.update(status="chua_xac_minh", items=[])
            else:
                res["stock"] = inv["stock"]
                res["items"] = [i for i in ranked if inv["stock"][i] > 0] if intent == "tim_san_pham" else ranked
        return self._done(res)

    def _done(self, res):
        res.update(messages=self.messages, retries=self.retries, latency_ms=self.clock, events=list(self.events))
        return res


def pipeline_handle(query, intent_agent, catalog, rec, inventory, rng, intent=None):
    """Pipeline đơn khối: cùng logic nhưng gọi hàm trực tiếp, không kiểm hợp đồng, không thử lại. Chỉ kho là dịch vụ
    ngoài (một chặng đi, một chặng về). Lỗi kho -> 'loi'; dữ liệu cũ thì vẫn dùng (khẳng định sai nếu kho đã đổi)."""
    t0, msgs = time.perf_counter(), 0
    if intent is None:
        intent = intent_agent_label(intent_agent.predict_proba(intent_agent.model, [query])[0], intent_agent.threshold)
    res = {"intent": intent, "items": [], "stock": {}, "status": None}
    extra = 0.0
    if intent in ("chao_hoi", "khac", "lam_ro"):
        res["status"] = {"chao_hoi": "chao", "khac": "chuyen_nhan_vien", "lam_ro": "can_lam_ro"}[intent]
    else:
        cand = catalog.search(query)
        ranked = rec.rank(cand["ids"], cand["budget"] if intent == "tim_san_pham" else None, cand["brands"],
                          "compare" if intent == "so_sanh" else "top") if cand["ids"] else []
        res["items"], res["status"] = ranked, ("ok" if ranked else "khong_co") if cand["ids"] else "can_lam_ro"
        if intent in ("tim_san_pham", "hoi_ton_kho") and ranked:
            msgs = 2
            extra = sum(rng.lognormvariate(math.log(2.0), 0.35) for _ in range(2))
            try:
                stock = inventory.lookup(ranked)["stock"]
                res["stock"] = stock
                res["items"] = [i for i in ranked if stock[i] > 0] if intent == "tim_san_pham" else ranked
            except (TimeoutError, KeyError):
                extra += 80.0
                res.update(status="loi", items=[])
    res.update(messages=msgs, retries=0, latency_ms=1000 * (time.perf_counter() - t0) + extra, events=[])
    return res


def intent_agent_label(proba, threshold):
    """Nhãn ý định từ vector xác suất; dưới ngưỡng thì 'lam_ro' (cần hỏi lại người dùng)."""
    return INTENTS[int(proba.argmax())] if float(proba.max()) >= threshold else "lam_ro"
