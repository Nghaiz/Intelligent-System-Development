"""Chương 5: ráp ngữ cảnh, gọi LLM cục bộ, chặn câu hỏi lâm sàng. tlf/llm.py giữ nguyên, ở đây chỉ bọc thêm truyền token."""
import re
import time

from .ch5_data import DATA5, VN30, read_jsonl

SYSTEM = ("Bạn là trợ lý tra cứu. Chỉ dùng thông tin trong NGỮ CẢNH. Trả lời bằng cụm ngắn nhất có thể: một tên, một số "
          "hoặc yes/no, không giải thích, không nhắc lại câu hỏi. Giữ nguyên cách viết tên riêng trong ngữ cảnh. "
          "Nếu ngữ cảnh không có đáp án, trả lời đúng cụm: Không đủ dữ liệu.")
CTX_WORDS = 170
MEDICAL_NOTE = "Thông tin chỉ mang tính học thuật, không thay thế tư vấn của bác sĩ."
INVEST_NOTE = "Thông tin chỉ mang tính học thuật, không phải khuyến nghị đầu tư."
REFUSAL = ("Tôi không chẩn đoán bệnh và không kê đơn. Hãy hỏi bác sĩ. Tôi có thể tra cứu tri thức dược lý trong đồ thị "
           "ở mức học thuật, ví dụ hợp chất nào liên kết với một gen.")
CLINICAL = re.compile(r"chẩn đoán|kê (đơn|toa|thuốc)|đơn thuốc|liều( lượng| dùng)?\b|tôi (có |đang |bị |mắc )+|"
                      r"(tôi|mình|bố|mẹ|con) nên (uống|dùng|tiêm)|nên uống thuốc|uống thuốc gì|diagnos|prescri|dosage",
                      re.I)
MEDICAL_WORDS = re.compile(r"\b(gen|gene|hợp chất|thuốc|bệnh|diabetes|tiểu đường|đái tháo đường|metformin|insulin|"
                           r"triệu chứng|tác dụng phụ|đường truyền|dược lý|mellitus)\b", re.I)
TICKERS = {t for t, *_ in VN30}
_chunks = {}


def is_clinical(question):
    return bool(CLINICAL.search(question))


def route(question):
    """Chọn miền theo từ khóa: mã cổ phiếu hoặc từ tài chính -> VN30, thuật ngữ y sinh -> Hetionet, còn lại -> HotpotQA."""
    words = set(re.findall(r"\b[A-Z]{3}\b", question))
    if words & TICKERS or re.search(r"cổ phiếu|lợi suất|chứng khoán|VN30|ngành|công ty", question):
        return "vn30"
    return "hetionet" if MEDICAL_WORDS.search(question) else "hotpot"


def chunk_table(ds):
    if ds not in _chunks:
        _chunks[ds] = {r["id"]: r for r in read_jsonl(DATA5 / f"chunks_{ds}.jsonl")}
    return _chunks[ds]


def clip(text, n=CTX_WORDS):
    toks = text.split()
    return " ".join(toks[:n]) + (" ..." if len(toks) > n else "")


def context_of(ds, ids):
    """Đánh số [1..k] từng chunk; số này là nguồn được trích trong câu trả lời."""
    table = chunk_table(ds)
    return "\n".join(f"[{i}] {clip(table[c]['text'])}" for i, c in enumerate(ids, 1))


def user_prompt(question, context):
    return f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}\nĐÁP ÁN:"


def generate(llm, question, ds, ids, max_new_tokens=48):
    """Một lượt RAG có đo thời gian thật; trả câu trả lời và ngữ cảnh để chấm độ bám."""
    context = context_of(ds, ids)
    t0 = time.perf_counter()
    text = llm.chat(user_prompt(question, context), SYSTEM, max_new_tokens)
    return {"answer": text.strip(), "context": context, "seconds": time.perf_counter() - t0}
