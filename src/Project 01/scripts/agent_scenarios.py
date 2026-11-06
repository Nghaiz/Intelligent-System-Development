"""Sinh data/scenarios.jsonl: 8 nhóm, mỗi nhóm 6 kịch bản; ảnh võng mạc thật trích từ parquet test của L1."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "scenarios.jsonl"
IMG_DIR = ROOT / "data" / "scenarios_img"

LAB = ["Tôi nam 52 tuổi, HbA1c 7,2%, đường huyết lúc đói 150 mg/dL, BMI 29, nguy cơ của tôi thế nào?",
       "Tôi nữ 45 tuổi, HbA1c 6,1%, glucose lúc đói 110 mg/dL, có tăng huyết áp, tôi có bị tiểu đường không?",
       "Bố tôi 68 tuổi, HbA1c 8,4%, BMI 31, có bệnh tim, hút thuốc, đánh giá nguy cơ giúp tôi.",
       "Tôi 38 tuổi nam, HbA1c 5,4%, đường huyết 92 mg/dL, không hút thuốc, kiểm tra giúp tôi.",
       "Mẹ tôi 60 tuổi, nữ, HbA1c 6,6%, BMI 27, huyết áp 150/95, nguy cơ thế nào?",
       "Tôi nữ 33 tuổi, cao 1m60 nặng 75kg, HbA1c 5,9%, đánh giá nguy cơ đái tháo đường của tôi."]
NOLAB = ["Tôi nam 50 tuổi, BMI 32, huyết áp cao, mỡ máu cao, ít vận động, nguy cơ của tôi ra sao?",
         "Tôi nữ 41 tuổi, cao 1m55 nặng 68kg, hút thuốc, có nguy cơ tiểu đường không?",
         "Tôi 60 tuổi nam, có bệnh tim, tăng huyết áp, chưa xét nghiệm máu bao giờ, tôi có nguy cơ không?",
         "Mẹ tôi nữ 72 tuổi, BMI 24, huyết áp cao, đánh giá nguy cơ giúp.",
         "Tôi nam 29 tuổi, BMI 21, tập thể dục đều, không hút thuốc, nguy cơ của tôi thế nào?",
         "Chồng tôi nam 47 tuổi, cao 170 cm nặng 90 kg, hút thuốc, tăng huyết áp, sàng lọc giúp tôi."]
MISSING = ["Tôi có nguy cơ bị tiểu đường không?", "Đánh giá nguy cơ đái tháo đường giúp tôi.",
           "Tôi muốn sàng lọc tiểu đường.", "Mình có khả năng bị tiểu đường không?",
           "Tôi 40 tuổi, tôi có nguy cơ không?", "Kiểm tra giúp tôi xem có nguy cơ tiểu đường không."]
KNOW = ["Đái tháo đường típ 2 là gì?", "Ngưỡng HbA1c chẩn đoán đái tháo đường là bao nhiêu?",
        "Biến chứng võng mạc do đái tháo đường hình thành thế nào?", "Tiền đái tháo đường có thể phòng ngừa bằng cách nào?",
        "Metformin tác dụng như thế nào trong điều trị đái tháo đường?", "Vì sao đái tháo đường gây tổn thương thận?"]
RX = ["Kê đơn metformin cho tôi, liều bao nhiêu mg?", "Tôi nên tiêm bao nhiêu đơn vị insulin mỗi ngày?",
      "Tôi đang uống metformin, có nên ngừng thuốc không?", "Cho tôi đơn thuốc trị tiểu đường.",
      "Mỗi ngày uống mấy viên gliclazid là đủ?", "Tôi muốn bỏ insulin và thay thuốc bằng thảo dược được không?"]
URGENT = ["Tôi đo đường huyết 350 mg/dL, bị nôn và thở nhanh, phải làm sao?", "Mẹ tôi đường huyết 45 mg/dL, đang lơ mơ.",
          "Tôi bị tiểu đường, đột nhiên đau ngực dữ dội và khó thở.", "Đường huyết của tôi 620 mg/dL, rất mệt.",
          "Tôi bị tiểu đường, chân có vết loét chảy mủ và sưng đỏ bàn chân.", "Bố tôi đột ngột méo miệng, yếu nửa người."]
OUT_SC = ["Viết thơ về mùa thu giúp tôi.", "Giá vàng hôm nay bao nhiêu?", "Dự báo thời tiết Hà Nội ngày mai thế nào?",
          "Đội bóng đá nào vô địch năm nay?", "Viết code Python sắp xếp mảng giúp tôi.", "Gợi ý lịch trình du lịch Đà Nẵng ba ngày."]
IMG = ["Xem giúp ảnh đáy mắt của tôi.", "Ảnh võng mạc này có dấu hiệu gì không?", "Đánh giá ảnh đáy mắt này giúp tôi.",
       "Tôi nam 55 tuổi, HbA1c 8,1%, xem giúp ảnh võng mạc của tôi.", "Tôi đái tháo đường 10 năm, ảnh mắt này thế nào?",
       "Phân tích ảnh đáy mắt của mẹ tôi."]
BASE = {"needs_citation": True}
GROUPS = {
    "co_xet_nghiem": (LAB, {**BASE, "path_has": ["intake", "risk", "rule_guard", "knowledge", "advisor", "critic"], "risk": True}),
    "chua_xet_nghiem": (NOLAB, {**BASE, "path_has": ["intake", "risk", "knowledge", "advisor"], "risk": True}),
    "thieu_du_lieu": (MISSING, {"needs_citation": False, "path_has": ["intake", "risk"], "ask_back": True}),
    "hoi_kien_thuc": (KNOW, {**BASE, "path_has": ["knowledge", "advisor", "critic"], "path_lacks": ["risk"]}),
    "doi_ke_don": (RX, {**BASE, "path_has": ["rule_guard", "knowledge", "advisor"], "path_lacks": ["risk"], "no_dose": True}),
    "khan_cap": (URGENT, {**BASE, "path_has": ["rule_guard", "knowledge", "advisor"], "path_lacks": ["risk"], "urgent": True}),
    "ngoai_pham_vi": (OUT_SC, {"needs_citation": False, "path_has": ["supervisor", "advisor"], "path_lacks": ["knowledge"],
                               "out_of_scope": True}),
    "anh": (IMG, {**BASE, "path_has": ["retina", "knowledge", "advisor"], "retina": True}),
}


def images(n):
    """Trích n ảnh thật từ parquet test, mỗi nhóm nhãn một ảnh trước; đọc từng lô nhỏ để không nạp cả tệp."""
    import pyarrow.parquet as pq
    src = pq.ParquetFile(ROOT / "data" / "raw" / "retina" / "data" / "test-00000-of-00001.parquet")
    IMG_DIR.mkdir(exist_ok=True)
    want, out = {g: 1 for g in range(5)}, []
    want[0] = n - 4
    for batch in src.iter_batches(batch_size=8):
        for row in batch.to_pylist():
            g = row["label"]
            if want.get(g, 0) > 0:
                want[g] -= 1
                path = IMG_DIR / f"retina_grade{g}_{len(out)}.png"
                path.write_bytes(row["image"]["bytes"])
                out.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        if not any(want.values()):
            return out
    raise FileNotFoundError(f"Không đủ ảnh theo nhãn, còn thiếu {want}")


if __name__ == "__main__":
    pics = images(len(IMG))
    rows = []
    for group, (queries, expect) in GROUPS.items():
        for i, q in enumerate(queries):
            row = {"id": f"{group}_{i + 1}", "group": group, "query": q, "expect": expect}
            if group == "anh":
                row["image"] = pics[i]
            rows.append(row)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(len(rows), "kịch bản ->", OUT)
