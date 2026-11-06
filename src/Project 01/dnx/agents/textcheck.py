"""Kiểm tra câu trả lời bằng mã: dùng chung cho tác tử Critic và chấm điểm, nên hai bên không thể lệch nhau."""
import re

from .state import DISCLAIMER

CITE = re.compile(r"\[(\d+)\]")
NUMBER = re.compile(r"(?<![\w/])\d+(?:[.,]\d+)*(?![\w])")
DOSE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|microgam|đơn vị|ui|iu|viên|ống)\b(?!\s*/\s*(?:dl|l)\b)", re.I)
DX_CLAIM = re.compile(r"\b(?:bạn|anh|chị|ông|bà|bác|cô|chú|người dùng)\s+(?:chắc chắn\s+|đã\s+|đang\s+|hiện\s+)*"
                      r"(?:bị|mắc)\s+(?:bệnh\s+)?(?:tiền\s+)?(?:đái tháo đường|tiểu đường|bệnh võng mạc)", re.I)
HEDGE = re.compile(r"không|chưa|có thể|nguy cơ|nếu|liệu|khi|nghi|ước lượng", re.I)
PRE_RANGE = re.compile(r"tiền\s+(?:đái tháo đường|tiểu đường|ĐTĐ)|rối loạn\s+(?:đường huyết|dung nạp)", re.I)
NOT_PRE = re.compile(r"không phải|không thuộc|vượt|cao hơn|trên ngưỡng", re.I)
DIABETES_RANGE, PRE_RULES = {"DX-01", "DX-03", "DX-05"}, {"DX-02", "DX-04", "DX-06"}
SMALL = {1.0, 2.0, 3.0, 4.0, 5.0}  # đánh số ý, típ 1, típ 2
SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text):
    return [s.strip(" -*•\t") for s in SENT.split(text) if s.strip(" -*•\t")]


def to_floats(token):
    """Một số viết kiểu Việt hoặc Anh, trả mọi cách đọc có thể (6,5 và 6.5; 1.000 và 1)."""
    t = token.strip()
    vals = {t.replace(",", "."), t.replace(",", ""), t.replace(".", "").replace(",", ".")}
    out = set()
    for v in vals:
        try:
            out.add(round(float(v), 3))
        except ValueError:
            pass
    return out


def numbers_in(text):
    out = set()
    for tok in NUMBER.findall(CITE.sub(" ", text)):
        out |= to_floats(tok)
    return out


def bad_numbers(answer, sources):
    """Số trong câu trả lời không có trong nguồn được phép; bỏ qua số nhỏ dùng để đánh số."""
    allowed = set(SMALL)
    for s in sources:
        allowed |= numbers_in(str(s))
    bad = []
    for tok in NUMBER.findall(CITE.sub(" ", answer)):
        if not to_floats(tok) & allowed:
            bad.append(tok)
    return bad


def dose_hits(text):
    return DOSE.findall(text)


def diagnosis_claims(text):
    hits = []
    for s in sentences(text):
        m = DX_CLAIM.search(s)
        if m and not HEDGE.search(s[:m.end()].replace(m.group(0), "")) and not re.search(r"\bnếu\b", s, re.I):
            hits.append(s)
    return hits


def wrong_range(text, rule_ids):
    """Câu xếp xét nghiệm vào nhóm tiền đái tháo đường trong khi luật báo mọi xét nghiệm đã ở ngưỡng chẩn đoán."""
    ids = set(rule_ids)
    if not ids & DIABETES_RANGE or ids & PRE_RULES:
        return []
    return [s for s in sentences(text) if PRE_RANGE.search(s) and not NOT_PRE.search(s)]


def citation_problems(text, n_evidence):
    """(số trích dẫn sai chỉ mục, các câu y khoa không trích dẫn)."""
    bad = [int(x) for x in CITE.findall(text) if not 1 <= int(x) <= n_evidence]
    body = text.replace(DISCLAIMER, "")
    uncited = [s for s in sentences(body) if not CITE.search(s) and len(s) > 25]
    return bad, uncited


def check(answer, n_evidence, sources, need_citation=True, rule_ids=()):
    """Danh sách lỗi dạng {"kind", "detail"}; rỗng là đạt. answer là phần do LLM viết, chưa có câu miễn trừ."""
    problems = []
    hits = dose_hits(answer)
    if hits:
        problems.append({"kind": "dose", "detail": f"Không được nêu liều: {hits[:3]}"})
    claims = diagnosis_claims(answer)
    if claims:
        problems.append({"kind": "diagnosis", "detail": "Không được khẳng định người dùng đã mắc bệnh."})
    if wrong_range(answer, rule_ids):
        problems.append({"kind": "wrong_range", "detail": "Xét nghiệm đã ở ngưỡng chẩn đoán đái tháo đường, không được xếp vào tiền đái tháo đường."})
    nums = bad_numbers(answer, sources)
    if nums:
        problems.append({"kind": "number", "detail": f"Số không có trong dữ liệu: {nums[:5]}"})
    if need_citation:
        bad, uncited = citation_problems(answer, n_evidence)
        if bad:
            problems.append({"kind": "citation", "detail": f"Trích dẫn không tồn tại: {bad[:5]}"})
        if not CITE.search(answer):
            problems.append({"kind": "no_citation", "detail": "Không có trích dẫn [n] nào."})
        elif uncited:
            problems.append({"kind": "uncited", "detail": f"{len(uncited)} câu chưa có [n]"})
    return problems


def sanitize(answer, n_evidence, rule_ids=()):
    """Hết lượt sửa mà vẫn lỗi: bỏ các câu vi phạm thay vì đưa ra ngoài."""
    keep = []
    for s in sentences(answer):
        if dose_hits(s) or diagnosis_claims(s) or wrong_range(s, rule_ids) or not CITE.search(s):
            continue
        keep.append(CITE.sub(lambda m: m[0] if 1 <= int(m[1]) <= n_evidence else "", s))
    return " ".join(keep)
