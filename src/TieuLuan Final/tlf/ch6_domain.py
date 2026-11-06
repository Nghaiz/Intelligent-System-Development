"""Ba miền bài toán của Chương 6: định dạng đề, công cụ, đội chuyên gia, cách đọc và chấm đáp án."""
import json
import random
import re
import unicodedata
from dataclasses import dataclass, field

from .ch6_tools import calc, find, guideline
from .common import SEED, data_dir

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def to_number(text):
    """Lấy số cuối trong văn bản ('$1,250.50.' -> 1250.5); None nếu không có."""
    found = NUM_RE.findall(str(text).replace("$", ""))
    if not found:
        return None
    try:
        return float(found[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def parse_number(text):
    v = to_number(text)
    return None if v is None else f"{v:g}"


def conclusion_word(text, pattern):
    """Luật đọc nhãn duy nhất, dùng cả lúc chạy lẫn lúc chấm: từ đầu tiên sau 'Final Answer:', hoặc từ đứng ngay
    đầu câu trả lời hay ngay sau dấu hai chấm của nhãn tên. Từ nằm giữa câu ('Chỗ sai ở đây...') không phải nhãn:
    trả None để tác tử bị hỏi lại."""
    text = unicodedata.normalize("NFC", str(text)).lower().replace("*", "").strip()
    marked = "final answer:" in text
    part = text.rsplit("final answer:", 1)[-1].strip()
    found = re.search(pattern, part)
    if found is None:
        return None
    head = part[:found.start()].strip()
    return found.group(1) if marked or not head or (head.endswith(":") and len(head) < 24) else None


def parse_decision(text):
    return conclusion_word(text, r"\b(yes|no|maybe)\b")


def parse_verdict(text):
    word = conclusion_word(text, r"(?<!\w)(đúng|dung|sai)(?!\w)")
    return None if word is None else ("sai" if word == "sai" else "dung")


@dataclass(frozen=True)
class Domain:
    id: str
    lang: str
    task: str                    # mô tả việc cho mọi tác tử
    answer_hint: str             # dạng đáp án sau "Final Answer:"
    tools: dict = field(default_factory=dict)
    tool_doc: str = ""
    roster: dict = field(default_factory=dict)   # chuyên gia -> vai trò
    parse: object = None
    max_new: int = 200

    def prompt(self, item):
        ctx = f"Abstract:\n{item['context']}\n\n" if item.get("context") else ""
        return f"{ctx}{item['question']}"


GSM8K = Domain(
    "gsm8k", "en", "You solve grade-school math word problems.", "a single number", {"calc": calc},
    "calc[expression] evaluates arithmetic such as calc[(3+4)*2].",
    {"Solver": "work out the quantities and do each computation with the calculator"},
    parse_number)
PUBMED = Domain(
    "pubmed", "en", "You answer a biomedical research question from the given abstract only.",
    "one word: yes, no or maybe", {"find": find}, "find[keyword] returns abstract sentences with that keyword.",
    {"Methodologist": "judge the study design, sample size and how strong the evidence is",
     "Clinician": "read the results and say what they imply for the question"},
    parse_decision, 160)
CLINICAL = Domain(
    "clinical", "vi", "Bạn kiểm tra đơn thuốc trong ca bệnh giả lập theo bộ luật có sẵn, không dùng kiến thức "
    "khác. Đơn ĐÚNG nếu không vi phạm luật nào của thuốc đó, SAI nếu vi phạm.", "ĐÚNG hoặc SAI",
    {"guideline": guideline}, "guideline[tên thuốc] trả luật của thuốc, ví dụ guideline[metformin]. Luật không có trong đề, "
    "phải tra trước khi kết luận.",
    {"Thận-Nội tiết": "so eGFR, liều và kali với luật của thuốc",
     "Tim mạch": "so huyết áp, nhịp tim và bệnh nền với luật của thuốc",
     "Dược lâm sàng": "tra luật của thuốc và kiểm tương tác như rượu, warfarin, thai kỳ"},
    parse_verdict, 180)
DOMAINS = {d.id: d for d in (GSM8K, PUBMED, CLINICAL)}


def load_items(domain_id, n, seed=SEED):
    """Tập con cố định n câu: mọi kiến trúc gặp đúng cùng câu hỏi, nên so sánh theo cặp được."""
    path = data_dir("ch6") / f"{domain_id}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split(chr(10)) if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def is_correct(domain_id, pred, gold):
    if pred is None:
        return False
    if domain_id == "gsm8k":
        return abs(float(pred) - float(gold)) < 1e-4
    return pred == gold
