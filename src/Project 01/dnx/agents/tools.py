"""Khai báo công cụ: mỗi tác tử chỉ gọi được công cụ trong danh sách của mình, tối đa 5 công cụ."""
import base64

from .. import rules as rule_engine
from . import intake_parse, llmio


def _predict(profile):
    from ..ml.serve import predict
    return predict(profile, mode="auto")


def _grade(image):
    from ..ml.serve_retina import grade
    return grade(image)


def _search(query, k=6, mode="hybrid"):
    from ..kb.search import search
    return search(query, k=k, mode=mode)


def _entities(text):
    from ..kb.search import entities
    return entities(text)


def _subgraph(ids, hops=1, limit=30):
    from ..kb.search import subgraph
    return subgraph(ids, hops=hops, limit=limit)


def _thresholds(test):
    from ..kb.search import thresholds
    return thresholds(test)


def _next_question(answers):
    from ..rl.policy import next_question
    return next_question(answers)


REGISTRY = {"parse_profile": intake_parse.parse, "rl_next_question": _next_question, "llm": llmio.ask,
            "predict": _predict, "retina_grade": _grade, "kb_search": _search, "kg_entities": _entities,
            "kg_subgraph": _subgraph, "kb_thresholds": _thresholds, "rules_evaluate": rule_engine.evaluate}
TOOLS = {"Supervisor": ["llm", "rules_evaluate"],
         "Intake": ["parse_profile", "rl_next_question", "llm"],
         "Risk": ["predict"],
         "Retina": ["retina_grade"],
         "Knowledge": ["kb_search", "kg_entities", "kg_subgraph", "kb_thresholds"],
         "RuleGuard": ["rules_evaluate"],
         "Advisor": ["llm"],
         "Critic": [],
         "SingleAgent": ["kb_search", "predict", "retina_grade", "llm"]}


def _brief(tool, value):
    """Tóm tắt kết quả cho Event, không đưa nguyên ảnh hay văn bản dài vào vết."""
    if tool == "retina_grade":
        return {k: v for k, v in value.items() if k != "gradcam_png_b64"}
    if tool == "kb_search":
        return {"hits": [{"title": h["title"], "score": h["score"], "retrievers": h["retrievers"]} for h in value]}
    if tool == "llm":
        return {"chars": len(value[0]), **value[1]}
    if isinstance(value, list):
        return {"count": len(value), "head": value[:5]}
    return value if isinstance(value, dict) else {"value": value}


def call(bus, agent, tool, *args, **kw):
    """Gọi công cụ thay mặt tác tử, ghi Event; lỗi công cụ được phát thành Event error rồi ném tiếp."""
    if tool not in TOOLS[agent]:
        raise PermissionError(f"{agent} không được dùng công cụ {tool}")
    kind = "llm" if tool == "llm" else "tool"
    try:
        value = REGISTRY[tool](*args, **kw)
    except Exception as e:
        bus.emit(agent, "error", f"{tool}: {type(e).__name__}: {str(e)[:200]}", {"tool": tool})
        raise
    bus.emit(agent, kind, tool, _brief(tool, value))
    return value


def image_b64(image):
    return base64.b64encode(image).decode("ascii") if image else None
