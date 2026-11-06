# Phase 1 — DB, ML/DL, dự đoán (làn L1)

Đọc trước: `contract.md`. Dùng lại: `src/TieuLuan Final/tlf/ch2_*.py`, `src/TieuLuan Final/data/ch2/cdc_diabetes.csv`,
`src/Assignment 03/diabetes_large/data/diabetes_prediction_dataset.csv`, `src/Assignment 01/data/raw/diabetes.csv` (Pima),
`src/Assignment 01/data/external/diabetes_hospital_frankfurt_2000.csv`.

## Việc
1. **Dữ liệu vào SQLite** `data/dianexus.db` (`dnx/db.py`, `scripts/data_load.py`): 5 bảng dữ liệu + bảng vận hành theo hợp đồng. Tải mới UCI 130-US Hospitals
   (`https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip`). Làm sạch, khử trùng, ghi `data_summary.json`
   (số dòng thô/sạch, cột, tỉ lệ dương, thiếu, trùng).
2. **EDA** `scripts/data_eda.py`: 8–10 hình đẹp, to chữ (phân bố nhãn cả 5 bộ, tương quan, phân bố theo nhãn, tỉ lệ bệnh theo tuổi/BMI, nhóm thuốc viện).
3. **Bài toán & mô hình**, split train/val/test cố định, chọn ngưỡng và mô hình chỉ trên val, test khóa:
   - T1 `full` (100k, có HbA1c, đường huyết): dự đoán đái tháo đường.
   - T2 `screening` (CDC, không xét nghiệm): sàng lọc.
   - T3 `readmission` (130 viện): tái nhập viện < 30 ngày; split theo `patient_nbr`, không rò rỉ.
   - Pima + Frankfurt: đối chuẩn chéo miền bằng các đặc trưng chung.
   - Cổ điển (CPU): Logistic Regression, Random Forest, XGBoost. Sâu (GPU, PyTorch): MLP, ResNet bảng, FT-Transformer.
   - Hiệu chỉnh isotonic, chọn ngưỡng F1 trên val; chỉ số: ROC-AUC, PR-AUC, F1, Brier, ECE. SHAP qua `pred_contribs` của XGBoost.
4. **Võng mạc** (`dnx/ml/retina.py`): tìm bộ HF phù hợp đã kiểm (ví dụ `youssefedweqd/Diabetic_Retinopathy_Detection`, `Rami/Diabetic_Retinopathy_Preprocessed_Dataset_256x256`,
   xem kích thước trước, ≤ 6 GB, 5 cấp APTOS/EyePACS). So 2 mô hình ImageNet (ResNet-34, EfficientNet-B0 hoặc ConvNeXt-Tiny), augmentation, cân bằng lớp, ~15 epoch.
   Chỉ số: accuracy, macro-F1, quadratic weighted kappa, ma trận nhầm lẫn; Grad-CAM. Lưu ảnh mẫu `web/static/img/retina/` (6–10 ảnh) và ảnh mẫu các lớp.
5. **Phục vụ**: `dnx/ml/serve.py`, `dnx/ml/serve_retina.py` đúng hợp đồng, nhanh (nạp lười, CPU suy luận cũng được).
6. Hình: ROC/PR, reliability, loss, so sánh mô hình, SHAP tổng, ma trận nhầm lẫn, Grad-CAM, lưới ảnh võng mạc.

## Song song CPU/GPU
Chạy cổ điển CPU cùng lúc với MLP GPU. Võng mạc CNN giữ `gpu.heavy`.

## Xong khi
`ml_*.json dl_*.json retina_*.json data_*.json` đủ, `serve.predict` và `grade` chạy thử đúng. Báo cáo cuối: bảng số chính, đường dẫn hình, quyết định dữ liệu.
