"""Ontology đóng của đồ thị tri thức DiaNexus: nhãn, quan hệ, ràng buộc miền-trị cho cạnh do LLM trích."""

# Nhãn thực thể y khoa (mọi nút này còn mang siêu nhãn Entity để dùng chung ràng buộc id và chỉ mục toàn văn)
ENTITY_LABELS = ("Disease", "Gene", "Compound", "Symptom", "SideEffect", "Anatomy", "Pathway", "PharmacologicClass",
                 "BiologicalProcess", "MolecularFunction", "CellularComponent", "MeshTerm", "Test", "RiskFactor",
                 "Lifestyle", "Complication")
# Nhãn cấu trúc
STRUCT_LABELS = ("Article", "Document", "Chunk", "Threshold", "Feature", "Rule", "Source")
ALL_LABELS = ENTITY_LABELS + STRUCT_LABELS

# Ký hiệu cạnh Hetionet -> kiểu quan hệ Neo4j
HETIONET_EDGES = {
    "CtD": "TREATS", "CpD": "PALLIATES", "DaG": "ASSOCIATES", "DuG": "UPREGULATES", "DdG": "DOWNREGULATES",
    "DlA": "LOCALIZES", "DpS": "PRESENTS", "DrD": "RESEMBLES", "CrC": "RESEMBLES", "CbG": "BINDS", "CcSE": "CAUSES",
    "PCiC": "INCLUDES", "GpPW": "PARTICIPATES", "GpBP": "PARTICIPATES", "GpMF": "PARTICIPATES", "GpCC": "PARTICIPATES",
    "GiG": "INTERACTS", "AeG": "EXPRESSES", "AuG": "UPREGULATES", "AdG": "DOWNREGULATES", "Gr>G": "REGULATES",
    "GcG": "COVARIES", "CuG": "UPREGULATES", "CdG": "DOWNREGULATES",
}
HETIONET_KIND = {"Biological Process": "BiologicalProcess", "Cellular Component": "CellularComponent",
                 "Molecular Function": "MolecularFunction", "Pharmacologic Class": "PharmacologicClass",
                 "Side Effect": "SideEffect"}
# Bệnh chuyển hóa và tim mạch-thận làm hạt lân cận; tên DOID đối chiếu trong hetionet-nodes.tsv
SEED_DISEASES = ("type 2 diabetes mellitus", "type 1 diabetes mellitus", "gestational diabetes", "obesity", "metabolic syndrome X",
                 "hypertension", "coronary artery disease", "atherosclerosis", "chronic kidney failure", "gout", "hypothyroidism")

# Quan hệ cấu trúc tạo bằng mã
STRUCT_RELS = {"HAS_MESH": "Article -> MeshTerm", "PART_OF": "Chunk -> Article|Document", "FROM_SOURCE": "Article|Document -> Source", "MENTIONS": "Chunk -> Entity",
               "SAME_AS": "MeshTerm -> Entity", "HAS_THRESHOLD": "Test -> Threshold",
               "EVIDENCED_BY": "Threshold -> Chunk", "MAPS_TO": "Feature -> Test|RiskFactor",
               "USES_FEATURE": "Rule -> Feature|Test", "CO_OCCURS": "MeshTerm -> MeshTerm"}

# Ontology đóng cho LLM: loại thực thể
LLM_TYPES = ("Disease", "Complication", "Compound", "Symptom", "Test", "RiskFactor", "Lifestyle", "Anatomy", "Gene")
# quan hệ -> (tập chủ thể, tập đối tượng)
_CLIN = ("Disease", "Complication")
LLM_RELATIONS = {
    "TREATS": (("Compound", "Lifestyle"), _CLIN),
    "PREVENTS": (("Compound", "Lifestyle"), _CLIN),
    "CAUSES": (("Disease", "Complication", "Compound", "RiskFactor"), ("Disease", "Complication", "Symptom")),
    "RISK_FACTOR_FOR": (("RiskFactor", "Lifestyle", "Disease", "Complication"), _CLIN),
    "COMPLICATION_OF": (("Complication", "Disease"), _CLIN),
    "PRESENTS": (_CLIN, ("Symptom",)),
    "DIAGNOSED_BY": (_CLIN, ("Test",)),
    "MONITORED_BY": (_CLIN + ("Compound",), ("Test",)),
    "AFFECTS": (_CLIN + ("Compound",), ("Anatomy",)),
    "ASSOCIATED_WITH": (_CLIN + ("Gene", "RiskFactor"), _CLIN + ("Gene", "RiskFactor")),
    "CONTRAINDICATED_IN": (("Compound",), _CLIN),
}
LLM_REL_VI = {"TREATS": "điều trị", "PREVENTS": "phòng ngừa", "CAUSES": "gây ra", "RISK_FACTOR_FOR": "là yếu tố nguy cơ của",
              "COMPLICATION_OF": "là biến chứng của", "PRESENTS": "có biểu hiện", "DIAGNOSED_BY": "được chẩn đoán bằng",
              "MONITORED_BY": "được theo dõi bằng", "AFFECTS": "ảnh hưởng tới", "ASSOCIATED_WITH": "liên quan tới",
              "CONTRAINDICATED_IN": "chống chỉ định trong"}

# Thực thể lâm sàng do dự án định nghĩa: (id, nhãn, tên, tên Việt, bí danh)
CLINICAL = [
    ("T:hba1c", "Test", "HbA1c", "xét nghiệm HbA1c", ("a1c", "hemoglobin a1c", "glycated hemoglobin", "glycated haemoglobin", "glycohemoglobin")),
    ("T:fpg", "Test", "Fasting plasma glucose", "glucose huyết tương lúc đói", ("fpg", "fasting glucose", "fasting blood glucose", "fasting blood sugar", "đường huyết lúc đói")),
    ("T:ogtt", "Test", "Oral glucose tolerance test", "nghiệm pháp dung nạp glucose đường uống", ("ogtt", "glucose tolerance test")),
    ("T:rpg", "Test", "Random plasma glucose", "glucose huyết tương bất kỳ", ("random glucose", "random blood sugar")),
    ("T:bp", "Test", "Blood pressure measurement", "đo huyết áp", ("blood pressure",)),
    ("T:lipid", "Test", "Lipid panel", "xét nghiệm mỡ máu", ("lipid profile", "cholesterol test")),
    ("T:egfr", "Test", "Estimated glomerular filtration rate", "mức lọc cầu thận ước tính", ("egfr",)),
    ("T:uacr", "Test", "Urine albumin-to-creatinine ratio", "tỉ số albumin/creatinin niệu", ("uacr", "albuminuria")),
    ("T:bmi", "Test", "Body mass index", "chỉ số khối cơ thể", ("bmi",)),
    ("R:obesity", "RiskFactor", "Overweight or obesity", "thừa cân, béo phì", ("overweight",)),
    ("R:smoking", "RiskFactor", "Tobacco smoking", "hút thuốc lá", ("smoking", "cigarette smoking")),
    ("R:hypertension", "RiskFactor", "High blood pressure", "tăng huyết áp", ("hypertension",)),
    ("R:dyslipidemia", "RiskFactor", "High cholesterol", "rối loạn lipid máu", ("dyslipidemia", "hypercholesterolemia")),
    ("R:inactivity", "RiskFactor", "Physical inactivity", "ít vận động", ("sedentary lifestyle",)),
    ("R:age", "RiskFactor", "Older age", "tuổi cao", ("aging",)),
    ("R:family", "RiskFactor", "Family history of diabetes", "tiền sử gia đình mắc đái tháo đường", ()),
    ("R:heart", "RiskFactor", "History of heart disease", "tiền sử bệnh tim", ("heart disease",)),
    ("R:stroke", "RiskFactor", "History of stroke", "tiền sử đột quỵ", ()),
    ("R:alcohol", "RiskFactor", "Heavy alcohol use", "uống nhiều rượu bia", ()),
    ("R:gdm", "RiskFactor", "Gestational diabetes history", "tiền sử đái tháo đường thai kỳ", ("gestational diabetes",)),
    ("R:prediabetes", "RiskFactor", "Prediabetes", "tiền đái tháo đường", ("impaired glucose tolerance",)),
    ("R:insulin_res", "RiskFactor", "Insulin resistance", "kháng insulin", ()),
    ("X:retinopathy", "Complication", "Diabetic retinopathy", "bệnh võng mạc đái tháo đường", ()),
    ("X:nephropathy", "Complication", "Diabetic nephropathy", "bệnh thận đái tháo đường", ("diabetic kidney disease",)),
    ("X:neuropathy", "Complication", "Diabetic neuropathy", "bệnh thần kinh đái tháo đường", ()),
    ("X:foot", "Complication", "Diabetic foot", "bàn chân đái tháo đường", ("foot ulcer",)),
    ("X:dka", "Complication", "Diabetic ketoacidosis", "nhiễm toan ceton", ("ketoacidosis", "dka")),
    ("X:hypo", "Complication", "Hypoglycemia", "hạ đường huyết", ("hypoglycaemia", "low blood sugar")),
    ("L:diet", "Lifestyle", "Healthy diet", "chế độ ăn lành mạnh", ("dietary change",)),
    ("L:activity", "Lifestyle", "Physical activity", "hoạt động thể chất", ("exercise",)),
    ("L:weightloss", "Lifestyle", "Weight loss", "giảm cân", ()),
]

# Cầu cột dữ liệu -> thực thể lâm sàng (e). id Feature là tên khóa Profile trong hợp đồng.
FEATURE_MAP = {
    "hba1c": ("HbA1c", ["T:hba1c"]), "glucose": ("Đường huyết", ["T:fpg", "T:rpg"]), "bmi": ("Chỉ số khối cơ thể", ["T:bmi", "R:obesity"]),
    "age": ("Tuổi", ["R:age"]), "hypertension": ("Tăng huyết áp", ["R:hypertension", "T:bp"]), "high_bp": ("Huyết áp cao", ["R:hypertension", "T:bp"]),
    "heart_disease": ("Bệnh tim", ["R:heart"]), "heart_diseaseor_attack": ("Bệnh tim hoặc nhồi máu", ["R:heart"]),
    "smoking": ("Hút thuốc", ["R:smoking"]), "smoker": ("Hút thuốc", ["R:smoking"]), "high_chol": ("Cholesterol cao", ["R:dyslipidemia", "T:lipid"]),
    "chol_check": ("Đã kiểm tra cholesterol", ["T:lipid"]), "phys_activity": ("Hoạt động thể chất", ["R:inactivity", "L:activity"]),
    "stroke": ("Đột quỵ", ["R:stroke"]), "hvy_alcohol_consump": ("Uống nhiều rượu bia", ["R:alcohol"]),
    "fruits": ("Ăn trái cây", ["L:diet"]), "veggies": ("Ăn rau", ["L:diet"]), "gen_hlth": ("Sức khỏe tổng quát", ["R:age"]),
    "blood_pressure": ("Huyết áp tâm trương", ["T:bp"]), "insulin": ("Insulin huyết thanh", ["R:insulin_res"]),
    "diabetes_pedigree": ("Chỉ số phả hệ", ["R:family"]), "pregnancies": ("Số lần mang thai", ["R:gdm"]),
}

# Bí danh thường gặp không có sẵn trong tên Hetionet hoặc MeSH: bí danh -> id nút
GAZ_ALIASES = {
    "type 1 diabetes": "H:Disease::DOID:9744", "t1dm": "H:Disease::DOID:9744", "t1d": "H:Disease::DOID:9744",
    "type 2 diabetes": "H:Disease::DOID:9352", "t2dm": "H:Disease::DOID:9352", "t2d": "H:Disease::DOID:9352",
    "diabetes": "M:Diabetes Mellitus", "sglt2 inhibitors": "M:Sodium-Glucose Transporter 2 Inhibitors",
    "sglt2 inhibitor": "M:Sodium-Glucose Transporter 2 Inhibitors", "sglt-2 inhibitors": "M:Sodium-Glucose Transporter 2 Inhibitors",
    "sglt2 inhibition": "M:Sodium-Glucose Transporter 2 Inhibitors", "glp-1 receptor agonists": "M:Glucagon-Like Peptide-1 Receptor Agonists",
    "glp-1 receptor agonist": "M:Glucagon-Like Peptide-1 Receptor Agonists", "dpp-4 inhibitors": "M:Dipeptidyl-Peptidase IV Inhibitors",
    "sulfonylurea": "M:Sulfonylurea Compounds", "sulfonylureas": "M:Sulfonylurea Compounds",
}
# Từ MeSH một tiếng quá chung, không dùng để liên kết thực thể trong văn bản
GAZ_GENERIC = {"risk", "affect", "eye", "fasting", "health", "time", "ulcer", "pain", "diet", "exercise", "fatigue", "emotions", "memory", "language",
               "motivation", "attention", "cognition", "patients", "white", "prevalence", "incidence", "goals", "technology", "algorithms", "mortality", "prognosis", "phenotype", "mutation", "hospitals", "nurses", "communication", "forecasting", "climate", "seasons", "workflow", "fear", "registries", "behavior", "sex", "income", "poverty", "pressure", "temperature", "light", "water", "food", "weight"}
