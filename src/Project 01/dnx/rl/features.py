"""21 đặc trưng CDC BRFSS: tên cột, khóa hồ sơ, câu hỏi tiếng Việt, lựa chọn trả lời, chi phí hỏi.

Chi phí phản ánh công sức của người trả lời: biết ngay 0,2; tự đánh giá 0,3–0,4; thông tin nhạy cảm 0,5; cần đo hoặc
nhớ kết quả khám 0,5–1,0. Đây là giả định thiết kế, đổi bảng này thì phải huấn luyện lại chính sách.
"""
from dataclasses import dataclass

YES_NO = ((1, "Có"), (0, "Không"))


@dataclass(frozen=True)
class Feature:
    col: str
    key: str
    cost: float
    question_vi: str
    options: tuple


AGE_BINS = ("18–24", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54", "55–59", "60–64", "65–69", "70–74", "75–79",
            "từ 80")
FEATURES = (
    Feature("HighBP", "high_bp", 1.0, "Bạn đã từng được nhân viên y tế nói là bị huyết áp cao chưa?", YES_NO),
    Feature("HighChol", "high_chol", 1.0, "Bạn đã từng được nói là cholesterol máu cao chưa?", YES_NO),
    Feature("CholCheck", "chol_check", 0.3, "Trong 5 năm qua bạn có kiểm tra cholesterol không?", YES_NO),
    Feature("BMI", "bmi", 0.5, "Chỉ số BMI (cân nặng chia bình phương chiều cao) của bạn gần mức nào?",
            ((17, "Dưới 18,5"), (22, "18,5–24,9"), (27, "25–29,9"), (32, "30–34,9"), (37, "35–39,9"), (44, "Từ 40"))),
    Feature("Smoker", "smoker", 0.3, "Bạn đã hút ít nhất 100 điếu thuốc trong đời chưa?", YES_NO),
    Feature("Stroke", "stroke", 0.3, "Bạn đã từng bị đột quỵ chưa?", YES_NO),
    Feature("HeartDiseaseorAttack", "heart_disease_or_attack", 0.5,
            "Bạn đã từng bị bệnh mạch vành hoặc nhồi máu cơ tim chưa?", YES_NO),
    Feature("PhysActivity", "phys_activity", 0.3, "Trong 30 ngày qua bạn có vận động thể lực ngoài công việc không?",
            YES_NO),
    Feature("Fruits", "fruits", 0.2, "Bạn có ăn trái cây ít nhất một lần mỗi ngày không?", YES_NO),
    Feature("Veggies", "veggies", 0.2, "Bạn có ăn rau ít nhất một lần mỗi ngày không?", YES_NO),
    Feature("HvyAlcoholConsump", "hvy_alcohol_consump", 0.4,
            "Bạn có uống nhiều rượu bia không (nam trên 14, nữ trên 7 ly mỗi tuần)?", YES_NO),
    Feature("AnyHealthcare", "any_healthcare", 0.2, "Bạn có bảo hiểm y tế hoặc nguồn chi trả khám chữa bệnh không?",
            YES_NO),
    Feature("NoDocbcCost", "no_docbc_cost", 0.3, "Năm qua có lần nào bạn cần khám mà không đi vì lo chi phí không?",
            YES_NO),
    Feature("GenHlth", "gen_hlth", 0.3, "Bạn tự đánh giá sức khỏe chung của mình thế nào?",
            ((1, "Rất tốt"), (2, "Tốt"), (3, "Khá"), (4, "Trung bình"), (5, "Kém"))),
    Feature("MentHlth", "ment_hlth", 0.4, "Trong 30 ngày qua có bao nhiêu ngày tinh thần bạn không tốt?",
            ((0, "0 ngày"), (3, "1–5 ngày"), (10, "6–14 ngày"), (20, "15–25 ngày"), (30, "Gần như cả tháng"))),
    Feature("PhysHlth", "phys_hlth", 0.4, "Trong 30 ngày qua có bao nhiêu ngày thể chất bạn không tốt?",
            ((0, "0 ngày"), (3, "1–5 ngày"), (10, "6–14 ngày"), (20, "15–25 ngày"), (30, "Gần như cả tháng"))),
    Feature("DiffWalk", "diff_walk", 0.3, "Bạn có gặp khó khăn nghiêm trọng khi đi bộ hoặc leo cầu thang không?",
            YES_NO),
    Feature("Sex", "sex", 0.2, "Giới tính sinh học của bạn là gì?", ((1, "Nam"), (0, "Nữ"))),
    Feature("Age", "age", 0.2, "Bạn thuộc nhóm tuổi nào?", tuple((i + 1, b) for i, b in enumerate(AGE_BINS))),
    Feature("Education", "education", 0.3, "Trình độ học vấn cao nhất của bạn?",
            ((1, "Chưa đi học"), (2, "Tiểu học"), (3, "Trung học cơ sở"), (4, "Tốt nghiệp trung học phổ thông"),
             (5, "Cao đẳng hoặc đại học dở dang"), (6, "Tốt nghiệp đại học"))),
    Feature("Income", "income", 0.5, "Thu nhập hộ gia đình của bạn thuộc nhóm nào?",
            ((1, "Thấp nhất"), (2, "Thấp"), (3, "Dưới trung bình"), (4, "Trung bình thấp"), (5, "Trung bình"),
             (6, "Trung bình cao"), (7, "Khá"), (8, "Cao nhất"))),
)
N = len(FEATURES)
STOP = N  # chỉ số hành động dừng
COLS = [f.col for f in FEATURES]
KEYS = [f.key for f in FEATURES]
COSTS = [f.cost for f in FEATURES]
BY_KEY = {f.key: i for i, f in enumerate(FEATURES)}
ALIASES = {"hypertension": "high_bp", "heart_disease": "heart_disease_or_attack"}


def age_bin(years):
    """Tuổi thật sang nhóm 1–13 của CDC."""
    return 1 if years < 25 else min(13, 2 + int((years - 25) // 5))


def answer_value(key, value):
    """Quy giá trị hồ sơ về thang của CDC; trả (chỉ số đặc trưng, giá trị) hoặc None nếu khóa không thuộc 21 câu."""
    if value is None:
        return None
    if key == "smoking":  # khóa hồ sơ của hợp đồng
        if value == "unknown":
            return None
        return BY_KEY["smoker"], float(value in ("former", "current"))
    key = ALIASES.get(key, key)
    if key not in BY_KEY:
        return None
    if key == "sex" and isinstance(value, str):
        value = {"male": 1, "female": 0, "nam": 1, "nữ": 0}[value.lower()]
    if key == "age" and float(value) > 13:
        value = age_bin(float(value))
    return BY_KEY[key], float(value)
