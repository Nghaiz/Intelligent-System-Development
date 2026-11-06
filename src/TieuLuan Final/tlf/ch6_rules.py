"""Bộ luật và bộ sinh ca lâm sàng TỔNG HỢP của Chương 6. Dữ liệu giả lập để đo tác tử, không phải tư vấn y tế.

Ngưỡng lấy cảm hứng từ nhãn thuốc và khuyến cáo phổ biến nhưng được rút gọn thành luật cứng; nhãn đúng/sai
là kết quả
của các hàm dưới đây, không phải đánh giá của bác sĩ.
"""
import random

from .common import SEED

GUIDELINES = {
    "metformin": "Metformin: chống chỉ định khi eGFR < 30. Khi 30 <= eGFR < 45, tổng liều tối đa 1000 mg/ngày. "
                 "Không dùng khi uống rượu từ 3 ly/ngày trở lên (nguy cơ nhiễm toan lactic).",
    "lisinopril": "Lisinopril: chống chỉ định khi có thai. Không dùng khi kali máu >= 5,5 mmol/L.",
    "spironolactone": "Spironolactone: không dùng khi kali máu > 5,0 mmol/L hoặc eGFR < 30.",
    "ibuprofen": "Ibuprofen: không dùng khi eGFR < 30 hoặc đang dùng warfarin (tăng nguy cơ chảy máu).",
    "propranolol": "Propranolol: chống chỉ định khi có hen phế quản hoặc nhịp tim < 50 lần/phút.",
    "hydrochlorothiazide": "Hydrochlorothiazide: không dùng khi eGFR < 30 vì giảm hiệu lực.",
    "amlodipine": "Amlodipine: không dùng khi huyết áp tâm thu < 100 mmHg.",
}
DOSES = {"metformin": [(500, 2), (850, 2), (1000, 2), (500, 1)], "lisinopril": [(10, 1), (20, 1)],
         "spironolactone": [(25, 1), (50, 1)], "ibuprofen": [(400, 3), (600, 2)], "propranolol": [(40, 2)],
         "hydrochlorothiazide": [(25, 1)], "amlodipine": [(5, 1), (10, 1)]}
INDICATION = {"metformin": "đái tháo đường típ 2", "lisinopril": "tăng huyết áp", "spironolactone": "suy tim",
              "ibuprofen": "viêm khớp", "propranolol": "tăng huyết áp", "hydrochlorothiazide": "tăng huyết áp",
              "amlodipine": "tăng huyết áp"}


def verdict(case):
    """Luật gán nhãn: trả (đúng thuốc?, mã luật)."""
    f = case
    checks = {
        "metformin": [(f["egfr"] < 30, "metformin_egfr30"),
                      (30 <= f["egfr"] < 45 and f["daily_mg"] > 1000, "metformin_dose45"),
                      (f["alcohol"] >= 3, "metformin_alcohol")],
        "lisinopril": [(f["pregnant"], "acei_pregnancy"), (f["k"] >= 5.5, "acei_potassium")],
        "spironolactone": [(f["k"] > 5.0, "mra_potassium"), (f["egfr"] < 30, "mra_egfr30")],
        "ibuprofen": [(f["egfr"] < 30, "nsaid_egfr30"), (f["warfarin"], "nsaid_warfarin")],
        "propranolol": [(f["asthma"], "bb_asthma"), (f["hr"] < 50, "bb_bradycardia")],
        "hydrochlorothiazide": [(f["egfr"] < 30, "thiazide_egfr30")],
        "amlodipine": [(f["sbp"] < 100, "ccb_hypotension")],
    }[f["drug"]]
    for hit, rule in checks:
        if hit:
            return False, rule
    return True, "none"


def _sample_case(rng):
    """Chỉ số lấy mẫu dày quanh ngưỡng để ca khó hơn một phép so sánh hiển nhiên."""
    drug = rng.choice(list(GUIDELINES))
    mg, times = rng.choice(DOSES[drug])
    sex = rng.choice(["nam", "nữ"])
    age = rng.randint(32, 84)
    return {"drug": drug, "mg": mg, "times": times, "daily_mg": mg * times, "sex": sex, "age": age,
            "egfr": rng.choice([24, 27, 29, 30, 33, 38, 42, 44, 45, 48, 55, 62, 75, 90]),
            "k": rng.choice([3.9, 4.2, 4.6, 4.9, 5.0, 5.1, 5.3, 5.4, 5.5, 5.7]),
            "alcohol": rng.choice([0, 0, 1, 2, 3, 4]), "hr": rng.choice([46, 49, 50, 54, 62, 74, 88]),
            "sbp": rng.choice([92, 98, 100, 108, 126, 142, 156]), "dbp": rng.choice([60, 70, 80, 90]),
            "asthma": rng.random() < 0.2, "warfarin": rng.random() < 0.2,
            "pregnant": sex == "nữ" and age < 45 and rng.random() < 0.5}


def render_case(c):
    """Tóm tắt bệnh án; mọi thuốc đều nhận cùng bộ chỉ số nên tác tử phải tự chọn chỉ số liên quan."""
    extra = [t for flag, t in [(c["asthma"], "hen phế quản"), (c["warfarin"], "đang dùng warfarin"),
                               (c["pregnant"], "đang có thai")] if flag]
    return (f"Bệnh nhân {c['sex']} {c['age']} tuổi, {INDICATION[c['drug']]}"
            f"{', ' + ', '.join(extra) if extra else ''}. eGFR {c['egfr']} mL/phút/1,73m2, "
            f"kali {str(c['k']).replace('.', ',')} mmol/L, huyết áp {c['sbp']}/{c['dbp']} mmHg, nhịp tim {c['hr']} "
            f"lần/phút, uống {c['alcohol']} ly rượu/ngày. Đơn đề xuất: {c['drug'].capitalize()} {c['mg']} mg, "
            f"{c['times']} lần/ngày.")


def generate_cases(n=100, seed=SEED):
    """Sinh n ca cân bằng đúng/sai; câu trùng bị loại, id cố định theo hạt giống."""
    rng, need = random.Random(seed), {True: n // 2, False: n - n // 2}
    rows, seen = [], set()
    while need[True] or need[False]:
        c = _sample_case(rng)
        ok, rule = verdict(c)
        text = render_case(c)
        if need[ok] == 0 or text in seen:
            continue
        seen.add(text)
        need[ok] -= 1
        rows.append({**c, "id": len(rows), "question": text, "context": "",
                     "gold": "dung" if ok else "sai", "rule": rule})
    rng.shuffle(rows)
    return rows
