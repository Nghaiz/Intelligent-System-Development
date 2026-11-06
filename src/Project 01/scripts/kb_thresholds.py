"""Ngưỡng chẩn đoán có nguồn: giá trị chỉ được nhận khi câu bằng chứng có thật trong chunk. Thiếu bằng chứng thì dừng báo lỗi."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import save_json  # noqa: E402
from dnx.kb import store  # noqa: E402

AL = {"hba1c": ["hba1c", "a1c", "hemoglobin a1c", "glycated hemoglobin"],
      "fpg": ["fpg", "fasting glucose", "fasting plasma glucose", "glucose lúc đói", "đường huyết lúc đói", "glucose"],
      "ogtt": ["ogtt", "oral glucose tolerance test", "nghiệm pháp dung nạp glucose", "2h glucose"],
      "rpg": ["rpg", "random glucose", "random plasma glucose", "glucose bất kỳ"],
      "bp": ["bp", "blood pressure", "huyết áp", "high_bp", "hypertension"],
      "bmi": ["bmi", "body mass index", "chỉ số khối cơ thể"],
      "chol": ["total cholesterol", "cholesterol", "high_chol"], "ldl": ["ldl", "ldl cholesterol"], "tg": ["triglyceride", "triglycerides"]}
NAME = {"hba1c": "HbA1c", "fpg": "Fasting plasma glucose", "ogtt": "OGTT 2 giờ", "rpg": "Random plasma glucose", "bp": "Blood pressure",
        "bmi": "BMI", "chol": "Total cholesterol", "ldl": "LDL cholesterol", "tg": "Triglyceride"}
TEST_ID = {"hba1c": "T:hba1c", "fpg": "T:fpg", "ogtt": "T:ogtt", "rpg": "T:rpg", "bp": "T:bp", "bmi": "T:bmi", "chol": "T:lipid", "ldl": "T:lipid", "tg": "T:lipid"}
Row = r"Diabetes \| 6\.5% or above \| 126 mg/dL or above \| 200 mg/dL or above \| 200 mg/dL or above"
Pre = r"Prediabetes \| 5\.7% to 6\.4% \| 100 to 125 mg/dL \| 140 to 199 mg/dL \| N/A"
Nor = r"Normal \| below 5\.7% \| 99 mg/dL or below \| 139 mg/dL or below \| N/A"
TABLE = "tests-diagnosis"
# (khóa, op, giá trị, đơn vị, ý nghĩa, mẫu URL, regex bằng chứng)
SPEC = [
    ("hba1c", ">=", 6.5, "%", "đái tháo đường", TABLE, Row),
    ("hba1c", "between", [5.7, 6.4], "%", "tiền đái tháo đường", TABLE, Pre),
    ("hba1c", "<", 5.7, "%", "bình thường", TABLE, Nor),
    ("hba1c", ">=", 48, "mmol/mol", "đái tháo đường", "2cb3ab68", r"HbA1c ≥ 6\.5% \(48 mmol/mol\)"),
    ("fpg", ">=", 126, "mg/dL", "đái tháo đường", TABLE, Row),
    ("fpg", "between", [100, 125], "mg/dL", "tiền đái tháo đường", TABLE, Pre),
    ("fpg", "<=", 99, "mg/dL", "bình thường", TABLE, Nor),
    ("fpg", ">=", 7.0, "mmol/L", "đái tháo đường", "2cb3ab68", r"fasting plasma glucose ≥ 7\.0 mmol/L"),
    ("fpg", ">=", 7.0, "mmol/L", "đái tháo đường", "Q-BYT", r"glucose huyết tương lúc đói ≥ 7,0 mmol/l \(≥ 126 mg/dl\)"),
    ("ogtt", ">=", 200, "mg/dL", "đái tháo đường", TABLE, Row),
    ("ogtt", "between", [140, 199], "mg/dL", "tiền đái tháo đường", TABLE, Pre),
    ("ogtt", "<=", 139, "mg/dL", "bình thường", TABLE, Nor),
    ("ogtt", ">=", 11.1, "mmol/L", "đái tháo đường", "2cb3ab68", r"2-hour post-load plasma glucose ≥ 11\.1 mmol/L"),
    ("rpg", ">=", 200, "mg/dL", "đái tháo đường khi có triệu chứng tăng đường huyết", TABLE, Row),
    ("rpg", ">=", 11.1, "mmol/L", "đái tháo đường khi có triệu chứng tăng đường huyết", "2cb3ab68", r"random blood glucose ≥ 11\.1 mmol/L"),
    ("bp", ">=", "140/90", "mmHg", "tăng huyết áp", "fact-sheets/detail/hypertension", r"140/90 mmHg or higher"),
    ("bmi", ">=", 25, "kg/m2", "thừa cân", "obesity-and-overweight", r"overweight is a BMI (?:greater than or equal to|≥) ?25"),
    ("bmi", ">=", 30, "kg/m2", "béo phì", "obesity-and-overweight", r"obesity is a BMI (?:greater than or equal to|≥) ?30"),
    ("bmi", ">=", 23, "kg/m2", "ngưỡng BMI người châu Á dùng để sàng lọc, theo quy trình lâm sàng tại Việt Nam", "duytan", r"BMI ≥ 23 kg/m2"),
    ("chol", "<", 200, "mg/dL", "mức mong muốn ở người lớn", "cholesterollevels", r"Total Cholesterol \| Less than 200 mg/dL"),
    ("ldl", "<", 100, "mg/dL", "mức mong muốn", "ldlthebadcholesterol", r"Less than 100 mg/dL \| Optimal"),
    ("tg", ">=", 150, "mg/dL", "tăng, có thể cần điều trị", "cholesterollevels", r"borderline high \(150-199 mg/dL\)"),
]


def sentence_around(text, start, end):
    a = max(text.rfind(". ", 0, start), start - 200)
    b = text.find(". ", end)
    b = min(b + 1 if b >= 0 else len(text), end + 160)
    return text[a + 2 if text[max(a, 0):a + 2] == ". " else max(a, 0):b].strip()


if __name__ == "__main__":
    rows = [json.loads(x) for x in open(store.CHUNKS, encoding="utf-8")]
    out, missing = [], []
    for n, (k, op, val, unit, meaning, url_pat, rx) in enumerate(SPEC):
        hit = None
        for c in rows:
            if url_pat in c["url"]:
                m = re.search(rx, c["text"].replace("\n", " "), re.I)
                if m:
                    hit = (c, m)
                    break
        if not hit:
            missing.append({"test": k, "url": url_pat, "regex": rx})
            continue
        c, m = hit
        out.append({"id": f"TH:{n:02d}", "test": NAME[k], "test_id": TEST_ID[k], "aliases": AL[k], "op": op, "value": val, "unit": unit,
                    "meaning": meaning, "evidence": sentence_around(c["text"], m.start(), m.end()), "url": c["url"],
                    "source": c["source"], "chunk_id": c["chunk_id"]})
    store.THRESH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    save_json("kb_thresholds", {"count": len(out), "missing": missing, "rows": out})
    print("ngưỡng", len(out), "thiếu bằng chứng", len(missing), missing, flush=True)
    if missing:
        sys.exit(1)
