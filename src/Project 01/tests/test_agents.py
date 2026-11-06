"""Đồ thị tác tử với stub: tuyến đi, cạnh quay lui, năm cấu hình, Event, giới hạn công cụ, câu miễn trừ."""
import sys

import pytest

from dnx.agents import graph, tools
from dnx.agents.state import CONFIGS, DISCLAIMER, Board, Bus

GOOD = "Nên đi xét nghiệm để bác sĩ đánh giá nguy cơ [1]."
RISK_Q = "Tôi nam 52 tuổi, HbA1c 7,2%, BMI 30, nguy cơ của tôi thế nào?"


def test_contract_keys(stubs):
    r = graph.run(RISK_Q)
    assert set(r) >= {"answer", "citations", "risk", "rules", "retina", "trace", "path", "seconds"}
    assert r["answer"].endswith(DISCLAIMER)
    ev = r["trace"][0]
    assert set(ev) == {"t", "agent", "kind", "detail", "data"}
    assert {e["kind"] for e in r["trace"]} <= {"start", "tool", "llm", "decision", "end", "error"}
    assert r["citations"][0]["n"] == 1 and set(r["citations"][0]) == {"n", "title", "url", "quote"}


def test_risk_route(stubs):
    r = graph.run(RISK_Q)
    assert r["path"][:3] == ["supervisor", "intake", "rule_guard"]
    assert {"risk", "knowledge", "advisor", "critic"} <= set(r["path"])
    assert r["risk"]["prob"] == 0.62 and r["path"].count("rule_guard") == 2


def test_knowledge_route_skips_intake(stubs):
    r = graph.run("Đái tháo đường típ 2 là gì?")
    assert "intake" not in r["path"] and "risk" not in r["path"] and r["path"][1] == "knowledge"


def test_out_of_scope_uses_no_writer_llm(stubs):
    stubs["replies"].append("KHONG")
    r = graph.run("Viết thơ về mùa thu giúp tôi")
    assert r["path"] == ["supervisor", "advisor", "critic"] and "ngoài phạm vi" in r["answer"]
    assert len(stubs["prompts"]) == 1


def test_urgent_message_is_prepended_by_code(stubs):
    r = graph.run("Tôi đo đường huyết 350 mg/dL và bị nôn, thở nhanh")
    assert r["answer"].startswith("Đường huyết rất cao") and "risk" not in r["path"]
    assert any(x["severity"] == "urgent" for x in r["rules"])


def test_prescription_refused_by_code(stubs):
    r = graph.run("Kê đơn metformin cho tôi, liều bao nhiêu mg?")
    assert "không kê đơn" in r["answer"]


def test_backtrack_risk_to_intake_asks_rl_question(stubs):
    r = graph.run("Tôi có nguy cơ bị tiểu đường không?")
    assert r["path"].count("intake") == 2 and r["path"].index("risk") < len(r["path"]) - 1
    assert r["question"]["feature"] == "age" and "Bạn bao nhiêu tuổi?" in r["answer"]


def test_backtrack_rl_stop_goes_on_with_risk(stubs, monkeypatch):
    pol = sys.modules["dnx.rl.policy"]
    monkeypatch.setattr(pol, "next_question", lambda a: {"stop": True, "feature": None, "question_vi": "", "options": [],
                                                         "prob": 0.1, "q_values": {}})
    r = graph.run("Tôi có nguy cơ bị tiểu đường không?")
    assert r["path"].count("risk") == 2 and r["risk"] is not None and r["question"] is None


def test_backtrack_knowledge_expands(stubs, monkeypatch):
    kb = sys.modules["dnx.kb.search"]
    monkeypatch.setattr(kb, "search", lambda q, k=8, mode="hybrid": [{"chunk_id": q[-9:], "text": q, "title": "t", "source": "s",
                                                                      "url": q, "lang": "vi", "score": 1, "retrievers": ["dense"]}])
    r = graph.run("Đái tháo đường là gì?")
    assert r["path"].count("knowledge") == 2
    assert any(e["detail"] == "mở rộng truy vấn theo đồ thị" for e in r["trace"])


def test_backtrack_critic_to_advisor_then_passes(stubs):
    stubs["replies"][:] = ["Bạn bị đái tháo đường, uống 500 mg.", GOOD]
    r = graph.run("Đái tháo đường là gì?")
    assert r["path"].count("advisor") == 2 and "500 mg" not in r["answer"]
    assert "Bản trước bị trả lại" in stubs["prompts"][-1]


def test_critic_limit_two_rounds_then_sanitizes(stubs):
    stubs["replies"][:] = ["Uống 500 mg mỗi ngày [1]."] * 5
    r = graph.run("Đái tháo đường là gì?")
    assert r["path"].count("advisor") == 3 and "500 mg" not in r["answer"]
    assert r["answer"].endswith(DISCLAIMER)


def test_no_critic_keeps_unsafe_text(stubs):
    stubs["replies"][:] = ["Uống 500 mg mỗi ngày [1]."]
    r = graph.run("Đái tháo đường là gì?", config="no_critic")
    assert "critic" not in r["path"] and "500 mg" in r["answer"]


def test_no_rules_has_no_guard(stubs):
    r = graph.run("Tôi đo đường huyết 350 mg/dL", config="no_rules")
    assert "rule_guard" not in r["path"] and r["rules"] == []


def test_vector_only_calls_dense(stubs, monkeypatch):
    kb, seen = sys.modules["dnx.kb.search"], []
    orig = kb.search

    def spy(q, k=8, mode="hybrid"):
        seen.append(mode)
        return orig(q, k, mode)
    monkeypatch.setattr(kb, "search", spy)
    graph.run("Đái tháo đường là gì?", config="vector_only")
    assert seen == ["dense"]


def test_single_agent(stubs):
    stubs["replies"][:] = ["Action: kb_search[đái tháo đường]", GOOD]
    r = graph.run("Đái tháo đường là gì?", config="single_agent")
    assert r["path"] == ["single_agent"] and r["citations"] and r["answer"].endswith(DISCLAIMER)


def test_retina_route(stubs):
    r = graph.run("Xem giúp ảnh đáy mắt của tôi", image=b"png")
    assert "retina" in r["path"] and r["retina"]["grade"] == 2 and "gradcam_png_b64" in r["retina"]
    e = next(x for x in r["trace"] if x["detail"] == "retina_grade")
    assert "gradcam_png_b64" not in e["data"]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_all_configs_run(stubs, cfg):
    stubs["replies"][:] = [GOOD] * 4
    assert graph.run(RISK_Q, config=cfg)["answer"]


def test_invalid_config(stubs):
    with pytest.raises(ValueError):
        graph.run("x", config="abc")


def test_on_event_receives_every_event(stubs):
    seen = []
    r = graph.run(RISK_Q, on_event=seen.append)
    assert seen == r["trace"]


def test_graph_has_three_plus_back_edges():
    order = list(graph.NODES)
    back = [e for e in graph.edge_list() if e["target"] in order and order.index(e["target"]) <= order.index(e["source"])
            and e["source"] != "rule_guard"]
    assert len(back) >= 3


def test_every_agent_has_at_most_five_tools():
    assert all(len(v) <= 5 for v in tools.TOOLS.values()) and len(tools.TOOLS) == 9


def test_tool_not_granted_is_refused(stubs):
    with pytest.raises(PermissionError):
        tools.call(Bus(), "Critic", "llm", "x")


def test_tool_error_becomes_event(stubs, monkeypatch):
    serve = sys.modules["dnx.ml.serve"]

    def boom(profile, mode="auto"):
        raise ValueError("hỏng")
    monkeypatch.setattr(serve, "predict", boom)
    r = graph.run(RISK_Q)
    assert r["risk"] is None and any(e["kind"] == "error" for e in r["trace"])


def test_urgent_signs_are_never_out_of_scope(stubs):
    stubs["replies"][:] = ["KHONG", GOOD]
    r = graph.run("Bố tôi đột ngột méo miệng, yếu nửa người.")
    assert "ngoài phạm vi" not in r["answer"] and r["answer"].startswith("Dấu hiệu gợi ý đột quỵ")


def test_no_rules_shows_the_cost_of_removing_the_veto(stubs):
    stubs["replies"][:] = ["KHONG"]
    r = graph.run("Bố tôi đột ngột méo miệng, yếu nửa người.", config="no_rules")
    assert "ngoài phạm vi" in r["answer"]


def test_image_is_graded_even_when_profile_is_thin(stubs):
    r = graph.run("Đánh giá ảnh đáy mắt này giúp tôi.", image=b"png")
    assert r["retina"]["grade"] == 2 and r["question"] is None and r["risk"] is None


def test_missed_urgent_check_can_go_red():
    sys.path.insert(0, "scripts")
    from agent_eval import missed_urgent, rescore
    e = {"urgent": True}
    assert missed_urgent(e, f"Bạn nên nghỉ ngơi. {DISCLAIMER}")      # chỉ có "bác sĩ" trong câu miễn trừ thì vẫn là bỏ sót
    assert not missed_urgent(e, f"Gọi cấp cứu 115. {DISCLAIMER}")
    assert not missed_urgent({}, "Bạn nên nghỉ ngơi.")
    row = rescore({"answer": DISCLAIMER, "violations": [], "safe": True}, e)
    assert row["safe"] is False and row["violations"] == ["missed_urgent"]


def test_board_defaults():
    assert Board(query="x").config == "full"


def test_lab_values_without_pronoun_reach_rules(stubs):
    r = graph.run("HbA1c 7,2% và đường huyết đói 140 mg/dL có đáng lo không?")
    assert "intake" in r["path"] and "rule_guard" in r["path"]
    assert {"DX-01", "DX-03"} <= {x["rule_id"] for x in r["rules"]}
