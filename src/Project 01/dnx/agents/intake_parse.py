"""Trích hồ sơ từ câu tiếng Việt bằng biểu thức chính quy: số liệu phải đúng tuyệt đối nên không giao cho LLM."""
import re

from ..rules import MMOL_TO_MGDL, norm_text, strip_accents

NUM = r"(\d+(?:[.,]\d+)?)"
MEDS = {"metformin": ("metformin", "glucophage"), "insulin": ("insulin",),
        "sglt2": ("empagliflozin", "dapagliflozin", "canagliflozin", "sglt2"),
        "sulfonylurea": ("gliclazid", "glimepirid", "glibenclamid", "sulfonylurea"),
        "acei_arb": ("lisinopril", "enalapril", "perindopril", "losartan", "valsartan", "telmisartan"),
        "nsaid": ("ibuprofen", "diclofenac", "meloxicam", "naproxen")}


def _f(s):
    return float(s.replace(",", "."))


def _after(text, label, tail=""):
    """Số đứng sau nhãn, cho phép vài từ nối ngắn giữa nhãn và số."""
    m = re.search(rf"(?:{label})[^\d]{{0,28}}?{NUM}{tail}", text)
    return m


def _glucose(t, out):
    m = re.search(rf"(?:đường huyết|đường máu|glucose|đường trong máu)[^\d]{{0,36}}?{NUM}\s*(mmol|mg)?", t)
    if not m:
        return
    v = _f(m[1])
    # không ghi đơn vị thì suy từ độ lớn: dưới 35 là mmol/L, trên đó là mg/dL
    if m[2] == "mmol" or (m[2] is None and v < 35):
        v = round(v * MMOL_TO_MGDL, 1)
    out["glucose"] = v
    if re.search(r"lúc đói|khi đói|buổi sáng đói|nhịn đói|fpg|(?:huyết|máu|glucose)\s+đói", t):
        out["glucose_kind"] = "fasting"
    elif re.search(r"nghiệm pháp|ogtt|sau uống", t):
        out["glucose_kind"] = "ogtt"
    else:
        out["glucose_kind"] = "random"


def _body(t, out):
    m = re.search(rf"bmi[^\d]{{0,12}}?{NUM}", t)
    if m:
        out["bmi"] = _f(m[1])
        return
    w = re.search(rf"{NUM}\s*(?:kg|ký|cân)\b", t)
    h = re.search(r"(?<![\d.,])(1)\s*m\s*(\d{1,2})|(?<![\d.,])1[.,](\d{1,2})\s*m\b|(?<![\d.,])(1\d{2})\s*cm", t)
    if w and h:
        cm = int(h[4]) if h[4] else 100 + int((h[2] or h[3]).ljust(2, "0"))
        out["bmi"] = round(_f(w[1]) / (cm / 100) ** 2, 1)


def _pressure(t, out):
    m = re.search(r"(?:huyết áp|ha)[^\d]{0,16}?(\d{2,3})\s*/\s*(\d{2,3})", t)
    if m:
        out["sbp"], out["dbp"] = int(m[1]), int(m[2])
        out["high_bp"] = int(out["sbp"] >= 140 or out["dbp"] >= 90)
        out["hypertension"] = out["high_bp"]
    elif re.search(r"(?:tăng|cao) huyết áp|huyết áp cao", t) and not re.search(r"không (?:bị )?(?:tăng|cao)", t):
        out["hypertension"], out["high_bp"] = 1, 1


def _flags(t, out):
    if re.search(r"\b(?:nữ|phụ nữ|chị|bà|cô)\b", t):
        out["sex"] = "female"
    if re.search(r"\b(?:nam|đàn ông|anh|ông|chú|bác trai)\b", t) and "việt nam" not in t:
        out["sex"] = "male"
    if re.search(r"không hút thuốc|chưa hút thuốc", t):
        out["smoking"] = "never"
    elif re.search(r"(?:bỏ thuốc lá|cai thuốc lá|đã cai thuốc|từng hút thuốc)", t):
        out["smoking"] = "former"
    elif re.search(r"hút thuốc|hút \d+ điếu", t):
        out["smoking"] = "current"
    if re.search(r"mỡ máu cao|cholesterol cao|rối loạn mỡ máu", t):
        out["high_chol"] = 1
    if re.search(r"bệnh tim|nhồi máu cơ tim|bệnh mạch vành", t) and "không" not in t.split("bệnh")[0][-12:]:
        out["heart_disease"] = 1
    if re.search(r"mang thai|có thai|có bầu|đang bầu", t):
        out["pregnant"] = True
    if re.search(r"ít vận động|không tập|không vận động", t):
        out["phys_activity"] = 0
    elif re.search(r"tập thể dục|chạy bộ|đi bộ mỗi|bơi", t):
        out["phys_activity"] = 1


def parse(text):
    """Trả dict hồ sơ đúng khóa hợp đồng; chỉ có khóa nào câu văn thật sự nêu."""
    t, out = norm_text(text), {}
    bare = strip_accents(t)
    m = re.search(r"(\d{1,3})\s*tuổi", t)
    if m:
        out["age"] = int(m[1])
    m = _after(t, r"hba1c|a1c|hemoglobin a1c")
    if m:
        out["hba1c"] = _f(m[1])
    _glucose(t, out)
    _body(t, out)
    _pressure(t, out)
    _flags(t, out)
    for key, label in (("egfr", "egfr"), ("creatinine", "creatinin\\w*"), ("potassium", "kali(?: máu)?")):
        m = _after(t, label)
        if m:
            out[key] = _f(m[1])
    meds = [k for k, names in MEDS.items() if any(n in bare for n in names)]
    if meds:
        out["meds"] = meds
    return out
