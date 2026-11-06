"""Đồ thị trạng thái tự viết: blackboard + cạnh điều kiện khai báo bằng dữ liệu + giới hạn vòng lặp.

GRAPH là nguồn duy nhất của cấu trúc: mỗi nút có danh sách cạnh (điều kiện, nút đích, tên cạnh) thử theo thứ tự.
"""
import time
import uuid

from . import nodes as A
from . import textcheck as C
from . import writer as W
from .state import CONFIGS, DISCLAIMER, Board, Bus

END = "end"
MAX_STEPS = 18
MAX_VISITS = {"supervisor": 1, "intake": 3, "risk": 2, "retina": 1, "knowledge": 4, "rule_guard": 2, "advisor": 4,
              "critic": 3}
AGENT_NAME = {"supervisor": "Supervisor", "intake": "Intake", "risk": "Risk", "retina": "Retina", "knowledge": "Knowledge",
              "rule_guard": "RuleGuard", "advisor": "Advisor", "critic": "Critic"}
NODES = {"supervisor": A.supervisor, "intake": A.intake, "risk": A.risk, "retina": A.retina, "knowledge": A.knowledge,
         "rule_guard": A.rule_guard, "advisor": W.advisor, "critic": W.critic}


def _rules_on(b):
    return b.config != "no_rules"


def _critic_on(b):
    return b.config != "no_critic"


def _first_guard(b):
    return _rules_on(b) and b.guard_runs == 0


def _final_guard(b):
    return _rules_on(b) and b.guard_runs < 2


def _need_risk(b):
    return b.want_risk and b.risk is None and not b.urgent and "risk" not in b.path


def _need_retina(b):
    return b.image is not None and b.retina is None


def _asked(b):
    return bool(b.question) and not b.question.get("stop")


GRAPH = {
    "supervisor": [(lambda b: b.intent == "out_of_scope", "advisor", "ngoài phạm vi"),
                   (lambda b: b.intent != "knowledge" or A.PERSONAL.search(b.query.lower()) is not None, "intake", "có người dùng"),
                   (lambda b: True, "knowledge", "hỏi kiến thức")],
    "intake": [(lambda b: _asked(b), END, "hỏi lại người dùng"),
               (lambda b: b.question is not None and b.risk is None, "risk", "RL dừng hỏi"),
               (_first_guard, "rule_guard", "kiểm khẩn cấp"),
               (_need_risk, "risk", "cần ước lượng"),
               (_need_retina, "retina", "có ảnh"),
               (lambda b: True, "knowledge", "tra cứu")],
    "rule_guard": [(lambda b: b.guard_runs >= 2 or bool(b.evidence), "advisor", "viết câu trả lời"),
                   (_need_risk, "risk", "cần ước lượng"),
                   (_need_retina, "retina", "có ảnh"),
                   (lambda b: True, "knowledge", "tra cứu")],
    "risk": [(lambda b: b.need_question, "intake", "thiếu dữ liệu, hỏi lại"),   # quay lui 1
             (_need_retina, "retina", "có ảnh"),
             (lambda b: True, "knowledge", "tra cứu")],
    "retina": [(lambda b: True, "knowledge", "tra cứu")],
    "knowledge": [(lambda b: b.weak_evidence, "knowledge", "mở rộng truy vấn"),  # quay lui 2
                  (_final_guard, "rule_guard", "soát luật"),
                  (lambda b: True, "advisor", "viết câu trả lời")],
    "advisor": [(_critic_on, "critic", "kiểm tra"),
                (lambda b: True, END, "trả lời")],
    "critic": [(lambda b: b.critic_ok, END, "đạt"),
               (lambda b: b.need_evidence, "knowledge", "thiếu bằng chứng"),     # quay lui 3
               (lambda b: True, "advisor", "viết lại")],                         # quay lui 4
}


def edge_list():
    """Danh sách cạnh cho báo cáo và Web App."""
    return [{"source": n, "target": t, "label": lb} for n, edges in GRAPH.items() for _, t, lb in edges]


def _next(b):
    for cond, target, label in GRAPH[b.path[-1]]:
        if cond(b):
            return target, label
    raise RuntimeError(f"Không có cạnh đi ra từ {b.path[-1]}")


def _walk(b, bus):
    node, visits = "supervisor", {}
    for _ in range(MAX_STEPS):
        visits[node] = visits.get(node, 0) + 1
        if visits[node] > MAX_VISITS[node]:
            bus.emit("Graph", "error", f"vượt giới hạn lượt ở {node}", {"visits": visits})
            break
        b.path.append(node)
        agent = AGENT_NAME[node]
        bus.emit(agent, "start", node, {})
        NODES[node](b, bus)
        bus.emit(agent, "end", node, {})
        node, label = _next(b)
        bus.emit("Graph", "decision", f"{b.path[-1]} -> {node}: {label}", {"from": b.path[-1], "to": node})
        if node == END:
            return
    if not b.answer:
        b.answer = f"Hệ thống chưa hoàn tất được yêu cầu này, bạn vui lòng thử lại. {DISCLAIMER}"


def _citations(b):
    used = {int(x) for x in C.CITE.findall(b.answer)}
    return [{"n": e["n"], "title": e["title"], "url": e["url"], "quote": e["text"][:240]} for e in b.evidence if e["n"] in used]


def _question_answer(b):
    opts = ", ".join(o["label"] for o in b.question["options"])
    return f'Để ước lượng nguy cơ, DiaNexus cần thêm thông tin. {b.question["question_vi"]} ({opts}) {DISCLAIMER}'


def run(query, profile=None, image=None, config="full", on_event=None, session_id=None):
    """Điểm vào theo hợp đồng; kết quả có thêm khóa question khi cần người dùng trả lời thêm."""
    if config not in CONFIGS:
        raise ValueError(f"config không hợp lệ: {config}; chọn trong {CONFIGS}")
    t0 = time.perf_counter()
    b = Board(query=query, profile=dict(profile or {}), image=image, config=config)
    bus = Bus(on_event, session_id)
    if config == "single_agent":
        b.path.append("single_agent")
        bus.emit("SingleAgent", "start", "single_agent", {})
        W.single_agent(b, bus)
        bus.emit("SingleAgent", "end", "single_agent", {})
    else:
        _walk(b, bus)
    question = b.question if b.question and not b.question.get("stop") else None
    if question:
        b.answer = _question_answer(b)
    if session_id is not None and b.risk:
        from ..db import save_prediction
        save_prediction(session_id, b.risk)
    return {"answer": b.answer, "citations": _citations(b), "risk": b.risk, "rules": b.rules, "retina": b.retina,
            "trace": bus.trace, "path": b.path, "seconds": round(time.perf_counter() - t0, 3), "question": question,
            "profile": b.profile, "sources": W.allowed_sources(b), "n_evidence": len(b.evidence), "run_id": uuid.uuid4().hex[:8]}
