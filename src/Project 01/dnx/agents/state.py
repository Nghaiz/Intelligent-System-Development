"""Bảng đen dùng chung của các tác tử và kênh phát Event theo hợp đồng."""
import time
from dataclasses import dataclass, field

DISCLAIMER = "Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ."
CONFIGS = ("full", "no_critic", "vector_only", "single_agent", "no_rules")


@dataclass
class Board:
    query: str
    profile: dict = field(default_factory=dict)
    image: bytes | None = None
    config: str = "full"
    intent: str = "knowledge"         # knowledge | risk | retina | prescription | out_of_scope
    want_risk: bool = False
    risk: dict | None = None
    retina: dict | None = None
    rules: list = field(default_factory=list)
    rule_flags: dict = field(default_factory=dict)
    guard_runs: int = 0
    question: dict | None = None      # câu hỏi lại do RL chọn
    need_question: bool = False
    evidence: list = field(default_factory=list)   # [{"n","title","url","text","kind"}]
    expansions: int = 0
    weak_evidence: bool = False
    answer: str = ""
    draft: str = ""
    feedback: list = field(default_factory=list)
    critic_rounds: int = 0
    critic_ok: bool = False
    need_evidence: bool = False
    sanitized: bool = False
    path: list = field(default_factory=list)

    @property
    def urgent(self):
        return bool(self.rule_flags.get("urgent_any"))


class Bus:
    """Mọi Event đi qua đây: thêm vào vết, gọi on_event, ghi agent_events khi có session_id."""

    def __init__(self, on_event=None, session_id=None):
        self.t0 = time.perf_counter()
        self.trace, self.on_event, self.session_id = [], on_event, session_id

    def emit(self, agent, kind, detail, data=None):
        event = {"t": round(time.perf_counter() - self.t0, 3), "agent": agent, "kind": kind, "detail": detail,
                 "data": data or {}}
        self.trace.append(event)
        if self.session_id is not None:
            from ..db import log_event
            log_event(self.session_id, event)
        if self.on_event is not None:
            self.on_event(event)
        return event
