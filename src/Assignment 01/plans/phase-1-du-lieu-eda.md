# Phase 1 — Dữ liệu và Phân tích khám phá

**Mục tiêu:** làm sạch dữ liệu, xây dựng biểu diễn đặc trưng, sinh toàn bộ biểu
đồ khám phá cho báo cáo. Đáp ứng yêu cầu R3 và R4.

**Tệp sở hữu:** `src/config.py`, `src/preprocess.py`, `src/viz.py`

## Bài toán 1 — Tiểu đường

- Năm cột sinh học (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`,
  `BMI`) dùng số 0 để mã hoá "không đo được" → chuyển thành khuyết.
- **KHÔNG điền khuyết ở bước này.** Việc điền do pipeline đảm nhiệm — xem mục
  "Bài học" bên dưới.
- Bốn đặc trưng thiết kế thủ công: `Glucose_BMI_Risk`, `Glucose_Level`,
  `BMI_Class`, `Age_Group`.

## Bài toán 2 — Bất động sản

- Tách `Address` thành `Province` và `District` theo quy ước hành chính Việt Nam
  (đọc từ phải sang: phần tử cuối là cấp tỉnh, áp chót là cấp huyện).
- Loại cột thiếu trên 80% (`Balcony direction`, thiếu 82.6%).
- Ba đặc trưng thiết kế: `Total_Area`, `Room_Density`, `Frontage_Ratio`.

## Biểu đồ phải sinh

Sơ đồ hệ thống · KDE · Histogram · Boxplot đơn và lưới · Pie chart · Heatmap
tương quan · KDE theo lớp · Scatter · Bar theo tỉnh/pháp lý · Bar tỷ lệ khuyết.

## Tiêu chí hoàn thành

- [x] Tối thiểu 12 tệp PNG trong `figures/`
- [x] Mỗi biểu đồ có tiêu đề và nhãn trục bằng tiếng Việt

## Bài học rút ra trong phase này

**Điền khuyết bằng trung vị nhóm theo nhãn `Outcome` là rò rỉ dữ liệu.**
Cách làm đó khiến Accuracy đạt 0.8705 — vượt xa trần 0.77–0.79 mà các nghiên
cứu công bố trên tập Pima. Nguyên nhân: `Insulin` khuyết 374/768 và
`SkinThickness` khuyết 227/768, nên gần một nửa dữ liệu bị điền bằng giá trị
mã hoá thẳng nhãn cần dự đoán. Đã sửa bằng cách đưa `SimpleImputer` vào trong
pipeline, chỉ học trung vị từ tập huấn luyện. Kết quả sau khi sửa: 0.7914.
