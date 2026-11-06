"""Stub theo hợp đồng cho KB, mô hình và LLM: kiểm thử chạy nhanh, xác định, không cần GPU hay Neo4j."""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DOC = {"chunk_id": "c1", "text": "HbA1c từ 6,5% là ngưỡng chẩn đoán đái tháo đường. Vận động đều giúp giảm nguy cơ.",
       "title": "Hướng dẫn chẩn đoán", "source": "stub", "url": "http://stub/a", "lang": "vi", "score": 0.9,
       "retrievers": ["bm25", "dense"]}


def fake_search(query, k=8, mode="hybrid"):
    return [dict(DOC, chunk_id=f"c{i}", url=f"http://stub/{i}") for i in range(min(k, 3))]


@pytest.fixture
def stubs(monkeypatch):
    """Thay kb.search, ml.serve, ml.serve_retina, rl.policy và LLM. Trả dict ghi các prompt LLM đã nhận."""
    kb = types.ModuleType("dnx.kb.search")
    kb.search = fake_search
    kb.entities = lambda text: [{"id": "e1", "name": "HbA1c", "label": "Test"}]
    kb.subgraph = lambda ids, hops=1, limit=120: {"nodes": [{"id": "e2", "name": "Glucose", "label": "Test"}], "links": []}
    kb.thresholds = lambda test="": [{"test": "HbA1c", "op": ">=", "value": 6.5, "unit": "%", "meaning": "đái tháo đường",
                                      "evidence": "ADA", "url": "http://stub/t", "aliases": ["hba1c"]}]
    serve = types.ModuleType("dnx.ml.serve")
    serve.predict = lambda profile, mode="auto": {
        "mode": "full", "prob": 0.62, "label": 1, "threshold": 0.5, "model": "stub", "calibrated": True,
        "shap": [{"feature": "hba1c", "label_vi": "HbA1c", "value": 7, "contribution": 1.0}]}
    ret = types.ModuleType("dnx.ml.serve_retina")
    ret.grade = lambda b: {"grade": 2, "label_vi": "Trung bình", "probs": [0, 0, 1, 0, 0], "gradcam_png_b64": "x"}
    pol = types.ModuleType("dnx.rl.policy")
    pol.next_question = lambda a: {"stop": False, "feature": "age", "question_vi": "Bạn bao nhiêu tuổi?",
                                   "options": [{"value": 1, "label": "18–24"}], "prob": 0.15, "q_values": {}}
    for name, mod in (("dnx.kb.search", kb), ("dnx.ml.serve", serve), ("dnx.ml.serve_retina", ret), ("dnx.rl.policy", pol)):
        monkeypatch.setitem(sys.modules, name, mod)
    log = {"prompts": [], "replies": []}

    def fake_ask(user, system=None, max_new_tokens=256):
        log["prompts"].append(user)
        text = log["replies"].pop(0) if log["replies"] else "Nên đi xét nghiệm để bác sĩ đánh giá nguy cơ [1]."
        return text, {"prompt_tokens": 1, "completion_tokens": 1, "seconds": 0.0}

    from dnx.agents import llmio, tools
    monkeypatch.setattr(llmio, "ask", fake_ask)
    monkeypatch.setitem(tools.REGISTRY, "llm", fake_ask)
    return log
