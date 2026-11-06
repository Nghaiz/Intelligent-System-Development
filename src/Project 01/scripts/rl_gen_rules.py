"""Sinh dnx/rules.json từ bảng gọn: rule là dữ liệu, tệp JSON là SSOT lúc chạy; chạy lại khi sửa bảng dưới đây."""
import json
from pathlib import Path

ADA_DX = "ADA Standards of Care in Diabetes 2025, Section 2 (Diagnosis and Classification)"
ADA_HYPO = "ADA Standards of Care in Diabetes 2025, Section 6 (Glycemic Goals and Hypoglycemia)"
ADA_HOSP = "ADA Standards of Care in Diabetes 2025, Section 16; đồng thuận ADA/EASD 2024 về cấp cứu tăng đường huyết"
ADA_CVD = "ADA Standards of Care in Diabetes 2025, Section 10 (Cardiovascular Disease and Risk Management)"
ADA_MICRO = "ADA Standards of Care in Diabetes 2025, Section 12 (Retinopathy, Neuropathy, and Foot Care)"
ADA_CKD = "ADA Standards of Care in Diabetes 2025, Section 11 (Chronic Kidney Disease)"
ADA_LIFE = "ADA Standards of Care in Diabetes 2025, Section 5 (Lifestyle Behavior Change)"
FDA_MET = "FDA Drug Safety Communication 2016: metformin và chức năng thận"
FDA_SGLT = "FDA Drug Safety Communication 2015: chất ức chế SGLT2 và nhiễm toan ceton"
LABEL_ACE = "Tờ hướng dẫn thuốc ức chế men chuyển (FDA boxed warning: độc tính thai nhi)"
KDIGO = "KDIGO 2022 Clinical Practice Guideline for Diabetes Management in CKD"
AHA = "AHA/ACC 2017 Hypertension Guideline: cơn tăng huyết áp"
AHA_STROKE = "AHA/ASA: nhận biết đột quỵ theo BE FAST"
POLICY = "Chính sách an toàn DiaNexus"


def r(id, group, sev, pri, when, msg, src, then=None):
    row = {"id": id, "group": group, "severity": sev, "priority": pri, "when": when, "message": msg, "source": src}
    if then:
        row["then"] = then
    return row


def f(fact, op, value=None):
    return {"fact": fact, "op": op, "value": value}


def th(fact, test, op, value, unit, kind):
    """Điều kiện ngưỡng: value là giá trị dự phòng; lúc chạy ưu tiên giá trị trong KB qua thresholds(test)."""
    return {"fact": fact, "op": op, "value": value, "threshold": {"test": test, "unit": unit, "kind": kind}}


KETO = ["nôn", "đau bụng", "thở nhanh", "thở sâu", "hơi thở mùi trái cây", "ketone", "ceton"]
ALTERED = ["lơ mơ", "lú lẫn", "hôn mê", "bất tỉnh", "ngất", "co giật", "không tỉnh", "không đánh thức"]
CHEST = ["đau ngực", "tức ngực", "khó thở", "đau thắt ngực", "nhồi máu"]
STROKE = ["méo miệng", "yếu nửa người", "liệt nửa người", "nói ngọng đột ngột", "không nói được", "tê nửa người"]
VISION = ["mất thị lực", "mù đột ngột", "đột ngột không nhìn thấy", "mờ mắt đột ngột", "đột ngột mờ", "màn đen che mắt"]
FOOT = ["loét chân", "vết loét", "hoại tử", "chân đen", "vết thương ở bàn chân", "sưng đỏ bàn chân", "chảy mủ"]
SYMPT = ["khát nhiều", "tiểu nhiều", "sụt cân", "uống nhiều", "ăn nhiều", "mờ mắt"]
DOSE_ASK = ["kê đơn", "kê thuốc", "liều", "bao nhiêu mg", "mấy viên", "uống mấy", "tiêm bao nhiêu", "đơn thuốc",
            "bao nhiêu đơn vị", "tăng thuốc", "giảm thuốc"]
STOP_ASK = ["ngừng thuốc", "bỏ thuốc", "dừng thuốc", "ngưng thuốc", "ngừng insulin", "bỏ insulin", "thay thuốc bằng"]
DX_ASK = ["tôi có bị", "tôi bị tiểu đường không", "tôi bị đái tháo đường không", "có phải tôi", "chẩn đoán giúp"]
OUT = ["thời tiết", "giá vàng", "chứng khoán", "bóng đá", "viết thơ", "viết code", "lập trình", "nấu ăn", "du lịch",
       "xổ số", "tỉ giá", "bộ phim"]
ALCOHOL = ["uống rượu nhiều", "nhậu nhiều", "nghiện rượu"]
CONTRAST = ["chụp cản quang", "thuốc cản quang"]

RULES = [
    # --- Nhóm khẩn: ưu tiên cao nhất, đi thẳng sang lời khuyên cấp cứu ---
    r("URG-01", "urgent", "urgent", 100, {"all": [f("glucose", ">=", 600)]},
      "Đường huyết từ 600 mg/dL có thể là tăng áp lực thẩm thấu do tăng đường huyết: cần tới cấp cứu ngay.", ADA_HOSP,
      {"urgent": True}),
    r("URG-02", "urgent", "urgent", 99, {"all": [f("glucose", ">=", 250), f("text", "contains_any", KETO)]},
      "Đường huyết rất cao kèm triệu chứng gợi ý nhiễm toan ceton: cần tới cấp cứu ngay.", ADA_HOSP, {"urgent": True}),
    r("URG-03", "urgent", "urgent", 98, {"all": [f("glucose", ">=", 300)]},
      "Đường huyết từ 300 mg/dL là mức rất cao: cần liên hệ cơ sở y tế trong ngày, tới cấp cứu nếu có triệu chứng.",
      ADA_HOSP, {"urgent": True}),
    r("URG-04", "urgent", "urgent", 97, {"all": [f("glucose", "<", 54)]},
      "Đường huyết dưới 54 mg/dL là hạ đường huyết mức 2: cần xử trí ngay và gọi hỗ trợ y tế.", ADA_HYPO,
      {"urgent": True, "hypoglycemia": True}),
    r("URG-05", "urgent", "urgent", 96, {"all": [f("hypoglycemia", "is", True), f("text", "contains_any", ALTERED)]},
      "Hạ đường huyết kèm rối loạn ý thức là mức 3: gọi cấp cứu 115, không cho người bệnh ăn uống khi không tỉnh.",
      ADA_HYPO),
    r("URG-06", "urgent", "urgent", 95, {"all": [f("text", "contains_any", ALTERED)]},
      "Rối loạn ý thức ở người nghi đái tháo đường là tình huống cấp cứu: gọi 115 ngay.", ADA_HOSP, {"urgent": True}),
    r("URG-07", "urgent", "urgent", 94, {"all": [f("text", "contains_any", CHEST)]},
      "Đau ngực hoặc khó thở có thể là biến cố tim mạch: gọi cấp cứu 115.", ADA_CVD, {"urgent": True}),
    r("URG-08", "urgent", "urgent", 93, {"all": [f("text", "contains_any", STROKE)]},
      "Dấu hiệu gợi ý đột quỵ: gọi cấp cứu 115 và ghi lại thời điểm khởi phát.", AHA_STROKE, {"urgent": True}),
    r("URG-09", "urgent", "urgent", 92, {"all": [f("text", "contains_any", VISION)]},
      "Giảm thị lực đột ngột cần được bác sĩ mắt khám cấp cứu trong ngày.", ADA_MICRO, {"urgent": True}),
    r("URG-10", "urgent", "urgent", 91, {"any": [f("sbp", ">=", 180), f("dbp", ">=", 120)]},
      "Huyết áp từ 180/120 mmHg là cơn tăng huyết áp: đo lại và liên hệ cơ sở y tế ngay.", AHA, {"urgent": True}),
    r("URG-11", "urgent", "urgent", 90, {"all": [f("text", "contains_any", FOOT)]},
      "Loét hoặc nhiễm trùng bàn chân ở người đái tháo đường cần được khám trong 24 giờ.", ADA_MICRO, {"urgent": True}),
    r("URG-12", "urgent", "warn", 80, {"all": [f("glucose", "<", 70), f("glucose", ">=", 54)]},
      "Đường huyết dưới 70 mg/dL là hạ đường huyết mức 1: cần xử trí và đo lại theo hướng dẫn của nhân viên y tế.",
      ADA_HYPO, {"hypoglycemia": True}),

    # --- Nhóm ngưỡng chẩn đoán: giá trị đọc từ KB lúc chạy, không bao giờ kết luận thay bác sĩ ---
    r("DX-01", "diagnostic", "warn", 70, {"all": [th("hba1c", "hba1c", ">=", 6.5, "%", "diabetes")]},
      "HbA1c ở mức của ngưỡng chẩn đoán đái tháo đường: cần xét nghiệm lặp lại để bác sĩ xác nhận.", ADA_DX,
      {"dx_range": "diabetes"}),
    r("DX-02", "diagnostic", "info", 60, {"all": [th("hba1c", "hba1c", ">=", 5.7, "%", "prediabetes"),
                                                  th("hba1c", "hba1c", "<", 6.5, "%", "diabetes")]},
      "HbA1c nằm trong khoảng tiền đái tháo đường theo tiêu chuẩn tham chiếu.", ADA_DX, {"dx_range": "prediabetes"}),
    r("DX-03", "diagnostic", "warn", 70, {"all": [f("glucose_kind", "is", "fasting"),
                                                  th("glucose", "fpg", ">=", 126, "mg/dL", "diabetes")]},
      "Đường huyết lúc đói ở mức của ngưỡng chẩn đoán: cần xét nghiệm lặp lại để bác sĩ xác nhận.", ADA_DX,
      {"dx_range": "diabetes"}),
    r("DX-04", "diagnostic", "info", 60, {"all": [f("glucose_kind", "is", "fasting"),
                                                  th("glucose", "fpg", ">=", 100, "mg/dL", "prediabetes"),
                                                  th("glucose", "fpg", "<", 126, "mg/dL", "diabetes")]},
      "Đường huyết lúc đói nằm trong khoảng rối loạn đường huyết đói.", ADA_DX, {"dx_range": "prediabetes"}),
    r("DX-05", "diagnostic", "warn", 70, {"all": [f("glucose_kind", "is", "ogtt"),
                                                  th("glucose", "ogtt", ">=", 200, "mg/dL", "diabetes")]},
      "Đường huyết 2 giờ sau nghiệm pháp ở mức của ngưỡng chẩn đoán: cần bác sĩ xác nhận.", ADA_DX,
      {"dx_range": "diabetes"}),
    r("DX-06", "diagnostic", "info", 60, {"all": [f("glucose_kind", "is", "ogtt"),
                                                  th("glucose", "ogtt", ">=", 140, "mg/dL", "prediabetes"),
                                                  th("glucose", "ogtt", "<", 200, "mg/dL", "diabetes")]},
      "Đường huyết 2 giờ sau nghiệm pháp nằm trong khoảng rối loạn dung nạp glucose.", ADA_DX,
      {"dx_range": "prediabetes"}),
    r("DX-07", "diagnostic", "warn", 70, {"all": [f("glucose_kind", "is", "random"),
                                                  th("glucose", "rpg", ">=", 200, "mg/dL", "diabetes"),
                                                  f("text", "contains_any", SYMPT)]},
      "Đường huyết bất kỳ cao kèm triệu chứng kinh điển: cần gặp bác sĩ sớm để xác nhận.", ADA_DX,
      {"dx_range": "diabetes"}),
    r("DX-08", "diagnostic", "info", 40, {"all": [f("bmi", ">=", 23)]},
      "BMI từ 23 ở người châu Á là một yếu tố nên cân nhắc sàng lọc đái tháo đường.", ADA_DX),
    r("DX-09", "diagnostic", "info", 40, {"all": [f("age", ">=", 35)]},
      "Từ 35 tuổi nên sàng lọc đái tháo đường định kỳ.", ADA_DX),

    # --- Nhóm an toàn thuốc: nêu cảnh báo, không nêu liều ---
    r("MED-01", "medication", "urgent", 85, {"all": [f("meds", "has", "metformin"), f("egfr", "<", 30)]},
      "Metformin chống chỉ định khi eGFR dưới 30: cần liên hệ bác sĩ điều trị để xem lại thuốc, không tự ý đổi.",
      FDA_MET),
    r("MED-02", "medication", "warn", 75, {"all": [f("meds", "has", "metformin"), f("egfr", ">=", 30), f("egfr", "<", 45)]},
      "Metformin khi eGFR từ 30 đến dưới 45 cần bác sĩ đánh giá lại lợi ích và nguy cơ.", FDA_MET),
    r("MED-03", "medication", "warn", 74, {"all": [f("meds", "has", "metformin"), f("text", "contains_any", ALCOHOL)]},
      "Uống nhiều rượu khi dùng metformin làm tăng nguy cơ nhiễm toan lactic.", FDA_MET),
    r("MED-04", "medication", "warn", 74, {"all": [f("meds", "has", "metformin"), f("text", "contains_any", CONTRAST)]},
      "Trước khi chụp có thuốc cản quang cần báo bác sĩ đang dùng metformin.", FDA_MET),
    r("MED-05", "medication", "warn", 74, {"all": [f("meds", "has", "sglt2"), f("text", "contains_any", KETO)]},
      "Đang dùng thuốc ức chế SGLT2 mà có nôn, đau bụng, thở nhanh cần được kiểm tra ceton dù đường huyết không cao.",
      FDA_SGLT),
    r("MED-06", "medication", "warn", 73, {"all": [f("meds", "has", "sulfonylurea"), f("age", ">=", 65)]},
      "Người cao tuổi dùng sulfonylurea có nguy cơ hạ đường huyết cao hơn, cần trao đổi với bác sĩ.", ADA_HYPO),
    r("MED-07", "medication", "warn", 76, {"all": [f("meds", "has", "insulin"), f("hypoglycemia", "is", True)]},
      "Đang dùng insulin mà hạ đường huyết: cần báo bác sĩ để xem lại phác đồ, không tự chỉnh.", ADA_HYPO),
    r("MED-08", "medication", "urgent", 86, {"all": [f("meds", "has", "acei_arb"), f("pregnant", "is", True)]},
      "Thuốc ức chế men chuyển hoặc chẹn thụ thể không dùng khi mang thai: cần liên hệ bác sĩ ngay.", LABEL_ACE),
    r("MED-09", "medication", "warn", 75, {"all": [f("meds", "has", "acei_arb"), f("potassium", ">=", 5.5)]},
      "Kali máu từ 5,5 mmol/L khi dùng ức chế men chuyển cần bác sĩ xem lại.", LABEL_ACE),
    r("MED-10", "medication", "warn", 75, {"all": [f("meds", "has", "nsaid"), f("egfr", "<", 30)]},
      "Thuốc kháng viêm không steroid cần tránh khi eGFR dưới 30.", KDIGO),
    r("MED-11", "medication", "warn", 65, {"all": [f("egfr", "<", 60)]},
      "eGFR dưới 60 gợi ý giảm chức năng thận: cần theo dõi và bác sĩ xem lại các thuốc đang dùng.", ADA_CKD),

    # --- Nhóm chính sách: ràng buộc cách trả lời ---
    r("POL-01", "policy", "warn", 88, {"all": [f("text", "contains_any", DOSE_ASK)]},
      "Yêu cầu về kê đơn hoặc liều thuốc: hệ thống không nêu liều cụ thể, chỉ bác sĩ điều trị mới được quyết định.",
      POLICY, {"no_dose": True}),
    r("POL-02", "policy", "warn", 87, {"all": [f("text", "contains_any", STOP_ASK)]},
      "Không tự ý ngừng hoặc thay thuốc đang dùng: cần trao đổi với bác sĩ điều trị.", POLICY, {"no_dose": True}),
    r("POL-03", "policy", "info", 50, {"all": [f("text", "contains_any", DX_ASK)]},
      "Hệ thống chỉ ước lượng nguy cơ, không khẳng định người dùng đã mắc bệnh.", POLICY, {"no_diagnosis": True}),
    r("POL-04", "policy", "info", 45, {"all": [f("text", "contains_any", OUT)]},
      "Nội dung nằm ngoài phạm vi đái tháo đường của hệ thống.", POLICY, {"out_of_scope": True}),
    r("POL-05", "policy", "warn", 84, {"all": [f("pregnant", "is", True)]},
      "Đang mang thai cần bác sĩ sản khoa theo dõi đường huyết, ngưỡng và thuốc khác với người thường.", POLICY),

    # --- Nhóm yếu tố nguy cơ: gợi ý trao đổi, luật sau dùng kết quả luật trước và mô hình ---
    r("RSK-01", "risk", "info", 35, {"all": [f("smoking", "is", "current")]},
      "Hút thuốc làm tăng nguy cơ biến chứng tim mạch: nên trao đổi về hỗ trợ cai thuốc.", ADA_LIFE),
    r("RSK-02", "risk", "info", 35, {"all": [f("bmi", ">=", 30)]},
      "BMI từ 30 thuộc nhóm béo phì: giảm cân có giám sát giúp cải thiện chuyển hóa.", ADA_LIFE),
    r("RSK-03", "risk", "info", 36, {"any": [f("hypertension", "is", 1), f("high_bp", "is", 1)]},
      "Có tăng huyết áp: cần theo dõi huyết áp vì đây là yếu tố nguy cơ tim mạch quan trọng.", ADA_CVD,
      {"has_hypertension": True}),
    r("RSK-04", "risk", "warn", 55, {"all": [f("risk_prob", ">=", 0.5)]},
      "Mô hình ước lượng nguy cơ ở mức cao: nên đi xét nghiệm HbA1c hoặc đường huyết lúc đói.", ADA_DX,
      {"high_risk": True}),
    r("RSK-05", "risk", "warn", 56, {"all": [f("high_risk", "is", True), f("has_hypertension", "is", True)]},
      "Nguy cơ cao kèm tăng huyết áp: nên khám sớm thay vì chỉ sàng lọc định kỳ.", ADA_CVD),
    r("RSK-06", "risk", "warn", 57, {"all": [f("dx_range", "is", "diabetes"), f("retina_grade", ">=", 2)]},
      "Chỉ số đường huyết cao kèm ảnh võng mạc gợi ý tổn thương mức trung bình trở lên: nên khám mắt sớm.", ADA_MICRO),
    r("RSK-07", "risk", "info", 37, {"all": [f("retina_grade", ">=", 1)]},
      "Ảnh võng mạc gợi ý có thay đổi: nên khám mắt theo chỉ định.", ADA_MICRO),
]

if __name__ == "__main__":
    ids = [x["id"] for x in RULES]
    assert len(ids) == len(set(ids))
    out = Path(__file__).resolve().parents[1] / "dnx" / "rules.json"
    out.write_text(json.dumps({"version": 1, "rules": RULES}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(RULES), "luật ->", out)
