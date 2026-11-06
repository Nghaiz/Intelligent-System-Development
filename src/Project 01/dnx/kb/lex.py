"""Từ điển Việt-Anh mở rộng truy vấn cho BM25 và liên kết thực thể. Mỗi dòng: "bí danh Việt|...=cụm Anh"."""
import re
import unicodedata

_RAW = """
đái tháo đường type 2|đái tháo đường típ 2|đái tháo đường tuýp 2|tiểu đường type 2|tiểu đường tuýp 2|đtđ típ 2=type 2 diabetes mellitus
đái tháo đường type 1|đái tháo đường típ 1|đái tháo đường tuýp 1|tiểu đường type 1|tiểu đường tuýp 1|đtđ típ 1=type 1 diabetes mellitus
đái tháo đường thai kỳ|tiểu đường thai kỳ=gestational diabetes
tiền đái tháo đường|tiền tiểu đường=prediabetes
đái tháo đường|tiểu đường|đtđ=diabetes mellitus
biến chứng=complications
bệnh võng mạc|võng mạc=diabetic retinopathy
bệnh thận|suy thận=kidney disease nephropathy
bệnh thần kinh|tổn thương thần kinh=neuropathy
bàn chân|loét chân=diabetic foot ulcer
nhiễm toan ceton=ketoacidosis
hạ đường huyết=hypoglycemia
tăng đường huyết|đường huyết cao=hyperglycemia
đường huyết|đường máu|đường trong máu=blood glucose
lúc đói=fasting
nghiệm pháp dung nạp glucose|dung nạp glucose=oral glucose tolerance test
xét nghiệm=test
chẩn đoán=diagnosis criteria
sàng lọc|tầm soát=screening
ngưỡng=threshold
triệu chứng=symptoms
nguyên nhân=causes
yếu tố nguy cơ|nguy cơ=risk factors
phòng ngừa|dự phòng|phòng bệnh=prevention
điều trị=treatment
thuốc=medication drug
tác dụng phụ=side effects adverse effects
chống chỉ định=contraindication
tăng huyết áp|huyết áp cao=hypertension high blood pressure
huyết áp=blood pressure
mỡ máu|rối loạn lipid=dyslipidemia cholesterol
béo phì=obesity
thừa cân=overweight
chỉ số khối cơ thể=body mass index bmi
hút thuốc lá|hút thuốc=smoking tobacco
rượu bia|uống rượu=alcohol
hoạt động thể chất|tập thể dục|vận động=physical activity exercise
chế độ ăn|dinh dưỡng|ăn uống=diet nutrition
chất xơ=dietary fiber
chỉ số đường huyết=glycemic index
kháng insulin=insulin resistance
tế bào beta=beta cell
tuyến tụy|tụy=pancreas
thai kỳ|mang thai=pregnancy
trẻ em=children
người cao tuổi=older adults
tim mạch=cardiovascular
bệnh mạch vành=coronary artery disease
đột quỵ=stroke
mù lòa=blindness
cắt cụt=amputation
lọc máu=dialysis
ức chế sglt2|thuốc ức chế sglt2=sglt2 inhibitors
chủ vận glp-1|đồng vận glp-1=glp-1 receptor agonists
ức chế dpp-4=dpp-4 inhibitors
gen=gene
di truyền=genetic heredity
"""
PAIRS = [(a.strip(), en.strip()) for line in _RAW.strip().splitlines() for en in [line.split("=")[1]]
         for a in line.split("=")[0].split("|")]
_VI2EN = sorted(PAIRS, key=lambda p: -len(p[0]))
_EN2VI = {}
for _a, _e in PAIRS:
    _EN2VI.setdefault(_e, _a)


def norm(s):
    return unicodedata.normalize("NFC", s).lower()


def expand(query):
    """Trả (cụm Anh dịch từ tiếng Việt, cụm Việt dịch từ tiếng Anh) xuất hiện trong truy vấn; ưu tiên cụm dài nhất."""
    q = norm(query)
    en, vi = [], []
    rest = q
    for a, e in _VI2EN:
        if re.search(rf"(?<!\w){re.escape(a)}(?!\w)", rest):
            en.append(e)
            rest = rest.replace(a, " ")
    for e, a in _EN2VI.items():
        if re.search(rf"(?<!\w){re.escape(e)}(?!\w)", q):
            vi.append(a)
    return en, vi
