"""Sáu tác tử không viết văn: Supervisor, Intake, Risk, Retina, Knowledge, RuleGuard."""
import re

from .. import rules as R
from .intake_parse import parse
from .state import Board, Bus
from .tools import call

DOMAIN = ("đái tháo đường", "tiểu đường", "đường huyết", "đường máu", "glucose", "hba1c", "a1c", "insulin", "metformin",
          "tiền đái", "võng mạc", "biến chứng", "bmi", "huyết áp", "mỡ máu", "cholesterol", "thuốc", "tiểu nhiều", "khát",
          "sàng lọc", "nguy cơ", "ăn kiêng", "chỉ số đường", "bệnh thận", "biến chứng", "egfr", "loét", "kali", "ogtt",
          "cân nặng", "béo phì", "sức khỏe", "bệnh", "triệu chứng", "đau ngực", "khó thở", "hôn mê", "lơ mơ", "ngất")
RISK_ASK = ("nguy cơ", "khả năng bị", "có bị", "sàng lọc", "đánh giá", "có mắc", "dự đoán", "tôi có", "kiểm tra giúp")
PERSONAL = re.compile(r"\b(tôi|mình|em|mẹ|bố|ba|chồng|vợ|con|bác|ông|bà|anh|chị)\b")
DOSE_ASK = ("kê đơn", "kê thuốc", "liều", "bao nhiêu mg", "mấy viên", "tiêm bao nhiêu", "đơn thuốc", "ngừng thuốc",
            "bỏ thuốc", "dừng thuốc", "ngừng insulin", "bỏ insulin")
MIN_FEATURES = 3          # dưới mức này mô hình gần như đoán theo tỉ lệ nền, phải hỏi thêm
LAB_KEYS = ("hba1c", "glucose")
K_DOCS = 5
QUOTE = 700
MIN_STRONG = 2            # số đoạn tối thiểu đạt ngưỡng tin cậy trước khi dừng mở rộng truy vấn
MAX_EXPAND = 1
THRESH_TESTS = (("hba1c", "hba1c"), ("a1c", "hba1c"), ("đường huyết lúc đói", "fpg"), ("lúc đói", "fpg"),
                ("glucose", "fpg"), ("đường huyết", "fpg"), ("đường máu", "fpg"), ("ogtt", "ogtt"),
                ("nghiệm pháp", "ogtt"), ("bmi", "bmi"), ("chẩn đoán", "hba1c"), ("tiền đái tháo đường", "hba1c"))
OUT_PROMPT = ("Câu hỏi sau có liên quan tới sức khỏe, y tế hoặc bệnh đái tháo đường không? "
              "Chỉ trả lời một từ: CO hoặc KHONG.\n\nCâu hỏi: {q}")


def supervisor(b: Board, bus: Bus):
    """Định tuyến: mã kiểm từ khóa trước, chỉ gọi LLM khi không có từ nào thuộc miền y tế."""
    t = R.norm_text(b.query)
    has_profile = any(v is not None for v in b.profile.values())
    in_domain = any(w in t for w in DOMAIN) or has_profile or b.image is not None
    if not in_domain and b.config != "no_rules":
        # dấu hiệu khẩn không bao giờ được xem là ngoài phạm vi, kể cả khi câu không nhắc từ nào thuộc đái tháo đường
        in_domain = R.has_urgent(call(bus, "Supervisor", "rules_evaluate", {"text": b.query}))
    if not in_domain:
        text, _ = call(bus, "Supervisor", "llm", OUT_PROMPT.format(q=b.query[:300]), None, 6)
        in_domain = not R.strip_accents(text).upper().startswith("KHONG")
    if not in_domain:
        b.intent = "out_of_scope"
    elif any(w in t for w in DOSE_ASK):
        b.intent = "prescription"
    elif has_profile or _states_lab(b.query) or (PERSONAL.search(t) and (any(w in t for w in RISK_ASK) or _has_data(t))):
        b.intent = "risk"
    elif b.image is not None:
        b.intent = "retina"
    else:
        b.intent = "knowledge"
    b.want_risk = b.intent == "risk"
    bus.emit("Supervisor", "decision", b.intent, {"want_risk": b.want_risk, "image": b.image is not None})


def _has_data(t):
    return bool(re.search(r"\d", t))


def _states_lab(text):
    """Câu nêu giá trị xét nghiệm là câu về một người, kể cả khi không có từ xưng hô; bỏ qua là bỏ luật ngưỡng."""
    return any(k in parse(text) for k in LAB_KEYS)


def intake(b: Board, bus: Bus):
    """Lượt đầu trích hồ sơ; lượt sau, khi Risk báo thiếu dữ liệu, RL chọn một câu hỏi lại."""
    if b.need_question:
        q = call(bus, "Intake", "rl_next_question", {k: v for k, v in b.profile.items()})
        b.need_question = False
        if q["stop"]:
            bus.emit("Intake", "decision", "RL dừng hỏi, dùng dữ liệu hiện có", {"prob": q["prob"]})
            b.question = {"stop": True}
        else:
            b.question = q
            bus.emit("Intake", "decision", f"hỏi thêm: {q['feature']}", {"prob": q["prob"]})
        return
    parsed = call(bus, "Intake", "parse_profile", b.query)
    # hồ sơ do người dùng nhập có quyền ưu tiên hơn phần suy ra từ câu văn
    b.profile = {**parsed, **{k: v for k, v in b.profile.items() if v is not None}}
    if b.want_risk is False and _known(b.profile) >= 2:
        b.want_risk = b.intent != "prescription"
    bus.emit("Intake", "decision", f"hồ sơ có {len(b.profile)} trường", {"profile": b.profile})


def _known(profile):
    from ..rl.features import answer_value
    return sum(answer_value(k, v) is not None for k, v in profile.items())


def risk(b: Board, bus: Bus):
    if b.question is None and not any(b.profile.get(k) is not None for k in LAB_KEYS) and _known(b.profile) < MIN_FEATURES:
        if b.image is not None:     # ảnh là yêu cầu chính: chấm ảnh trước, không chặn lại để hỏi thêm
            bus.emit("Risk", "decision", "thiếu dữ liệu nhưng có ảnh, bỏ qua ước lượng", {"known": _known(b.profile)})
            return
        b.need_question = True
        bus.emit("Risk", "decision", "thiếu dữ liệu, trả về Intake", {"known": _known(b.profile)})
        return
    try:
        b.risk = call(bus, "Risk", "predict", b.profile)
    except (FileNotFoundError, ValueError, KeyError):
        b.risk = None     # lỗi đã phát thành Event; Advisor sẽ không nêu con số nguy cơ


def retina(b: Board, bus: Bus):
    try:
        b.retina = call(bus, "Retina", "retina_grade", b.image)
    except Exception:
        b.retina = None


def _add_hits(b, hits):
    seen = {e["url"] + e["text"][:60] for e in b.evidence}
    for h in hits:
        key = h["url"] + h["text"][:60]
        if key not in seen and len(b.evidence) < K_DOCS + 3:
            seen.add(key)
            b.evidence.append({"n": len(b.evidence) + 1, "title": h["title"], "url": h["url"], "text": h["text"][:QUOTE],
                               "kind": "doc", "retrievers": h.get("retrievers", [])})


def _query_text(b):
    extra = {"prescription": " thuốc điều trị an toàn", "risk": " yếu tố nguy cơ sàng lọc"}.get(b.intent, "")
    if b.rule_flags.get("urgent"):
        extra = " dấu hiệu cấp cứu xử trí"
    return (b.query + extra).strip()


def knowledge(b: Board, bus: Bus):
    """Truy xuất lai; bằng chứng yếu thì tự mở rộng truy vấn qua lân cận đồ thị rồi quay lại chính nó."""
    vector_only = b.config == "vector_only"
    mode = "dense" if vector_only else "hybrid"
    query = _query_text(b)
    if b.expansions and not vector_only:
        ents = call(bus, "Knowledge", "kg_entities", query)
        if ents:
            sub = call(bus, "Knowledge", "kg_subgraph", [e["id"] for e in ents[:4]], 1, 24)
            names = [n["name"] for n in sub["nodes"] if n["id"] not in {e["id"] for e in ents}][:6]
            query = f"{query} {' '.join(names)}"
            bus.emit("Knowledge", "decision", "mở rộng truy vấn theo đồ thị", {"added": names})
    if not b.evidence:
        _thresholds(b, bus, vector_only)
    hits = call(bus, "Knowledge", "kb_search", query, K_DOCS, mode)
    _add_hits(b, hits)
    strong = sum(len(h.get("retrievers", [])) >= 2 for h in hits)
    b.weak_evidence = not vector_only and strong < MIN_STRONG and b.expansions < MAX_EXPAND
    if b.need_evidence:     # Critic trả về vì thiếu bằng chứng: đã tìm lại xong
        b.need_evidence = False
    bus.emit("Knowledge", "decision", f"{len(b.evidence)} bằng chứng", {"strong": strong, "weak": b.weak_evidence})
    if b.weak_evidence:
        b.expansions += 1


def _thresholds(b, bus, vector_only):
    """Ngưỡng chẩn đoán có nguồn vào thẳng danh sách bằng chứng để câu trả lời trích dẫn được."""
    if vector_only:
        return
    t = R.norm_text(b.query)
    tests = {code for word, code in THRESH_TESTS if word in t}
    tests |= {code for key, code in (("hba1c", "hba1c"), ("glucose", "fpg")) if b.profile.get(key) is not None}
    for test in sorted(tests):
        for row in call(bus, "Knowledge", "kb_thresholds", test)[:3]:
            value = "–".join(map(str, row["value"])) if isinstance(row["value"], list) else row["value"]
            text = f'{row["test"]} {row["op"]} {value} {row["unit"]}: {row["meaning"]}. {row.get("evidence", "")}'
            b.evidence.append({"n": len(b.evidence) + 1, "title": f'Ngưỡng {row["test"]}', "url": row.get("url", ""),
                               "text": text[:QUOTE], "kind": "threshold", "retrievers": ["thresholds"]})


def rule_guard(b: Board, bus: Bus):
    """Chạy luật trên toàn bộ dữ kiện hiện có; lượt đầu sau Intake để bắt khẩn cấp, lượt sau trước khi viết."""
    facts = {**b.profile, "text": b.query}
    if b.risk:
        facts["risk_prob"] = b.risk["prob"]
    if b.retina:
        facts["retina_grade"] = b.retina["grade"]
    b.rules = call(bus, "RuleGuard", "rules_evaluate", facts)
    b.rule_flags = {**R.flags(b.rules), "urgent_any": R.has_urgent(b.rules)}
    b.guard_runs += 1
    bus.emit("RuleGuard", "decision", f"{len(b.rules)} luật cháy", {"ids": [r["rule_id"] for r in b.rules],
                                                                    "urgent": b.rule_flags["urgent_any"]})
