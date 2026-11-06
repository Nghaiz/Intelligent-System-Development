"""Bộ kiểm tra văn bản của Critic: mỗi loại lỗi phải bắt được, câu hợp lệ không bị bắt nhầm."""
from dnx.agents import textcheck as C
from dnx.agents.intake_parse import parse


def test_dose_detected_but_lab_units_are_not():
    assert C.dose_hits("Uống metformin 500 mg mỗi ngày")
    assert C.dose_hits("tiêm 10 đơn vị insulin")
    assert not C.dose_hits("Đường huyết 126 mg/dL là ngưỡng chẩn đoán")


def test_diagnosis_claim():
    assert C.diagnosis_claims("Bạn bị đái tháo đường típ 2.")
    assert C.diagnosis_claims("Bạn đã mắc tiểu đường rồi.")
    assert not C.diagnosis_claims("Bạn có thể có nguy cơ bị đái tháo đường.")
    assert not C.diagnosis_claims("Nếu bạn bị đái tháo đường, hãy theo dõi đường huyết.")
    assert not C.diagnosis_claims("Bạn không bị đái tháo đường theo số liệu này.")


def test_numbers_vietnamese_and_english_forms():
    assert C.bad_numbers("HbA1c 6,5% [1]", ["HbA1c 6.5%"]) == []
    assert C.bad_numbers("HbA1c 7,9% [1]", ["HbA1c 6.5%"]) == ["7,9"]
    assert C.bad_numbers("xem [12]", []) == []


def test_citation_problems():
    bad, uncited = C.citation_problems("Câu một có đủ nội dung để tính [1]. Câu hai dài nhưng lại không có trích dẫn nào cả.", 1)
    assert bad == [] and len(uncited) == 1
    assert C.citation_problems("Câu [9]", 2)[0] == [9]


def test_check_ok_and_each_problem_kind():
    src = ["6,5"]
    assert C.check("HbA1c từ 6,5% là ngưỡng tham chiếu [1].", 1, src) == []
    kinds = {p["kind"] for p in C.check("Bạn bị đái tháo đường, uống 500 mg, HbA1c 9,9%.", 1, src)}
    assert kinds >= {"dose", "diagnosis", "number", "no_citation"}


def test_sanitize_removes_only_bad_sentences():
    text = "Nên đi khám sớm [1]. Uống 500 mg mỗi ngày [1]. Câu sai chỉ mục [7]."
    out = C.sanitize(text, 1)
    assert "500 mg" not in out and "Nên đi khám" in out and "[7]" not in out


def test_parse_profile_vietnamese():
    p = parse("Tôi nam 52 tuổi, cao 1m68 nặng 80kg, HbA1c 7,2%, đường huyết lúc đói 8,1 mmol/L, huyết áp 150/95, đang hút thuốc")
    assert p["age"] == 52 and p["sex"] == "male" and p["hba1c"] == 7.2 and p["bmi"] == 28.3
    assert p["glucose"] == 145.8 and p["glucose_kind"] == "fasting" and p["sbp"] == 150 and p["hypertension"] == 1
    assert p["smoking"] == "current"


def test_parse_does_not_confuse_stopping_medicine_with_smoking():
    assert "smoking" not in parse("Em có nên bỏ thuốc không")
    assert parse("Tôi đã bỏ thuốc lá")["smoking"] == "former"


def test_parse_meds_and_pregnancy():
    p = parse("Tôi đang có thai, dùng losartan và metformin, eGFR 25")
    assert p["pregnant"] is True and set(p["meds"]) == {"metformin", "acei_arb"} and p["egfr"] == 25


def test_wrong_range_flags_prediabetes_label_on_diabetic_values():
    text = "HbA1c 6,6% đã ở ngưỡng chẩn đoán tiền đái tháo đường [1]."
    assert C.wrong_range(text, ["DX-01"])
    kinds = [p["kind"] for p in C.check(text, 1, ["6,6"], rule_ids=["DX-01"])]
    assert "wrong_range" in kinds


def test_wrong_range_stays_quiet_when_a_prediabetes_rule_fired_or_label_is_negated():
    assert not C.wrong_range("Đường huyết đói ở mức rối loạn đường huyết đói [1].", ["DX-01", "DX-04"])
    assert not C.wrong_range("HbA1c đã vượt khoảng tiền đái tháo đường [1].", ["DX-01"])
    assert not C.wrong_range("HbA1c ở khoảng tiền đái tháo đường [1].", ["DX-02"])


def test_sanitize_drops_wrong_range_sentence():
    text = "HbA1c ở ngưỡng tiền đái tháo đường [1]. Nên đi khám sớm [1]."
    assert C.sanitize(text, 1, ["DX-01"]) == "Nên đi khám sớm [1]."


def test_parse_reads_fasting_without_luc():
    assert parse("đường huyết đói 140 mg/dL")["glucose_kind"] == "fasting"
    assert parse("đường máu đói 7 mmol/L")["glucose_kind"] == "fasting"
    assert parse("đường huyết 140 mg/dL")["glucose_kind"] == "random"
