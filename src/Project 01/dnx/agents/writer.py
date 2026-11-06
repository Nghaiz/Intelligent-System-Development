"""Advisor viết câu trả lời có trích dẫn, Critic kiểm bằng mã, SingleAgent là đối chứng một tác tử."""
import json
import re

from ..common import vn
from . import textcheck as C
from .state import DISCLAIMER, Board, Bus
from .tools import call

MAX_NEW = 320
MAX_CRITIC = 2
RULES_LIMIT = 5
ACTION = re.compile(r"Action:\s*(\w+)\[(.*?)\]", re.S)
SYSTEM = ("Bạn là DiaNexus, trợ lý thông tin về đái tháo đường, viết tiếng Việt, giọng y tế cẩn trọng.\n"
          "Quy tắc bắt buộc:\n"
          "1. Chỉ dùng thông tin trong BẰNG CHỨNG và DỮ LIỆU, không tự thêm số.\n"
          "2. Mỗi câu kết thúc bằng trích dẫn dạng [1] hoặc [2][3] trỏ tới đúng bằng chứng.\n"
          "3. Không nêu liều thuốc cụ thể (mg, viên, đơn vị).\n"
          "4. Không khẳng định người dùng đã mắc bệnh; dùng cách nói 'có thể', 'nguy cơ', 'nên đi khám'.\n"
          "5. Viết 3 đến 5 câu ngắn, không tiêu đề, không lặp câu hỏi, không viết câu miễn trừ.")
OUT_OF_SCOPE = "Câu hỏi này nằm ngoài phạm vi của DiaNexus. Hệ thống chỉ hỗ trợ thông tin về đái tháo đường và sức khỏe liên quan."
REFUSE = "DiaNexus không kê đơn và không nêu liều thuốc; việc chọn thuốc, liều và thay đổi điều trị do bác sĩ điều trị quyết định."
SINGLE_SYSTEM = (SYSTEM + "\nCông cụ: kb_search[truy vấn], predict[], retina[]. Mỗi lượt viết một dòng 'Action: tên[đầu vào]' "
                 "để gọi công cụ, hoặc viết thẳng câu trả lời khi đã đủ thông tin.")


def _rule_ids(b):
    return [r["rule_id"] for r in b.rules]


def _evidence_block(b):
    return "\n".join(f'[{e["n"]}] {e["title"]}: {e["text"]}' for e in b.evidence) or "(không có)"


def _data_block(b):
    lines = []
    if b.profile:
        keep = {k: v for k, v in b.profile.items() if k != "meds" or v}
        lines.append("Hồ sơ: " + json.dumps(keep, ensure_ascii=False))
    if b.risk:
        lines.append(f'Mô hình ước lượng nguy cơ {vn(b.risk["prob"] * 100, 1)}% (chế độ {b.risk["mode"]}).')
        top = [s["label_vi"] for s in b.risk.get("shap", [])[:3]]
        if top:
            lines.append("Yếu tố ảnh hưởng nhiều nhất: " + ", ".join(top) + ".")
    if b.retina:
        lines.append(f'Ảnh võng mạc: mức {b.retina["grade"]} ({b.retina["label_vi"]}), chỉ là gợi ý từ mô hình ảnh.')
    if b.rules:
        lines.append("Cảnh báo cần lồng vào câu trả lời: " + " ".join(r["message"] for r in b.rules[:RULES_LIMIT]))
        ids = _rule_ids(b)
        if set(ids) & C.DIABETES_RANGE and not set(ids) & C.PRE_RULES:
            lines.append("Phân loại bắt buộc: xét nghiệm ở ngưỡng chẩn đoán đái tháo đường, không phải tiền đái tháo đường.")
    return "\n".join(lines)


def allowed_sources(b):
    """Mọi nguồn số hợp lệ: bằng chứng, hồ sơ, xác suất, luật, câu hỏi của người dùng."""
    src = [e["text"] for e in b.evidence] + [str(v) for v in b.profile.values()] + [b.query]
    src += [r["message"] for r in b.rules]
    if b.risk:
        pct = b.risk["prob"] * 100
        src += [f"{pct:.0f}", f"{pct:.1f}", f"{pct:.2f}"]
    if b.retina:
        src.append(str(b.retina["grade"]))
    return src


def _prefix(b):
    """Câu luật khẩn và câu từ chối do mã thêm vào đầu, để thông điệp an toàn không phụ thuộc LLM."""
    parts = [r["message"] for r in b.rules if r["severity"] == "urgent"][:3]
    if b.rule_flags.get("no_dose"):
        parts.append(REFUSE)
    return " ".join(parts)


def compose(b, body):
    return " ".join(p for p in (_prefix(b), body.strip(), DISCLAIMER) if p)


def advisor(b: Board, bus: Bus):
    if b.intent == "out_of_scope":
        b.draft, b.answer = OUT_OF_SCOPE, f"{OUT_OF_SCOPE} {DISCLAIMER}"
        bus.emit("Advisor", "decision", "từ chối vì ngoài phạm vi", {})
        return
    user = f"BẰNG CHỨNG:\n{_evidence_block(b)}\n\nDỮ LIỆU:\n{_data_block(b)}\n\nCÂU HỎI: {b.query}"
    if b.intent == "prescription":
        user += "\n\nNgười dùng hỏi về thuốc: chỉ giải thích nguyên tắc chung, nhắc gặp bác sĩ, không nêu liều."
    if b.feedback:
        user += f"\n\nBản trước:\n{b.draft}\n\nBản trước bị trả lại vì: " + "; ".join(p["detail"] for p in b.feedback) + \
                "\nViết lại, sửa đúng các lỗi này."
    b.draft = call(bus, "Advisor", "llm", user, SYSTEM, MAX_NEW)[0]
    b.answer = compose(b, b.draft)


def critic(b: Board, bus: Bus):
    """Kiểm bằng mã; còn lỗi thì trả Advisor tối đa 2 vòng, thiếu bằng chứng thì trả Knowledge, cuối cùng cắt câu lỗi."""
    if b.intent == "out_of_scope":
        b.critic_ok = True
        return
    problems = C.check(b.draft, len(b.evidence), allowed_sources(b), rule_ids=_rule_ids(b))
    b.critic_ok = not problems
    bus.emit("Critic", "decision", "đạt" if b.critic_ok else "; ".join(p["detail"] for p in problems),
             {"round": b.critic_rounds, "kinds": [p["kind"] for p in problems]})
    if b.critic_ok:
        return
    b.feedback = problems
    if b.critic_rounds < MAX_CRITIC:
        b.critic_rounds += 1
        b.need_evidence = not b.evidence and b.expansions <= 1
        if b.need_evidence:
            b.expansions += 1
        return
    clean = C.sanitize(b.draft, len(b.evidence), _rule_ids(b))
    b.sanitized = True
    b.critic_ok = True
    b.answer = compose(b, clean or "Hiện chưa đủ bằng chứng trong kho tri thức để trả lời câu hỏi này; bạn nên hỏi bác sĩ.")
    bus.emit("Critic", "decision", "hết lượt sửa, đã bỏ các câu lỗi", {"kept_chars": len(clean)})


def single_agent(b: Board, bus: Bus):
    """Đối chứng: một LLM tự gọi công cụ, không luật, không kiểm tra; tối đa 3 lượt."""
    log = f"CÂU HỎI: {b.query}"
    for _ in range(3):
        text = call(bus, "SingleAgent", "llm", log, SINGLE_SYSTEM, MAX_NEW)[0]
        act = ACTION.search(text)
        if not act:
            b.draft = text
            break
        name, arg = act[1].lower(), act[2].strip()
        obs = _single_tool(b, bus, name, arg)
        log += f"\n{act[0]}\nObservation: {obs}"
    else:
        b.draft = call(bus, "SingleAgent", "llm", f"{log}\nDừng gọi công cụ, viết câu trả lời ngay.", SYSTEM, MAX_NEW)[0]
    b.answer = f"{b.draft.strip()} {DISCLAIMER}"


def _single_tool(b, bus, name, arg):
    from .intake_parse import parse
    try:
        if name == "kb_search":
            base = len(b.evidence)
            hits = call(bus, "SingleAgent", "kb_search", arg or b.query, 4, "hybrid")
            for h in hits:
                b.evidence.append({"n": len(b.evidence) + 1, "title": h["title"], "url": h["url"], "text": h["text"][:500],
                                   "kind": "doc", "retrievers": h.get("retrievers", [])})
            return "\n".join(f'[{e["n"]}] {e["title"]}: {e["text"]}' for e in b.evidence[base:]) or "Không thấy tài liệu."
        if name == "predict":
            b.profile = {**parse(b.query), **b.profile}
            b.risk = call(bus, "SingleAgent", "predict", b.profile)
            return f'Xác suất nguy cơ {b.risk["prob"]:.3f}'
        if name == "retina" and b.image:
            b.retina = call(bus, "SingleAgent", "retina_grade", b.image)
            return f'Mức {b.retina["grade"]}: {b.retina["label_vi"]}'
    except Exception as e:
        return f"Lỗi công cụ: {type(e).__name__}"
    return "Không có công cụ này."
