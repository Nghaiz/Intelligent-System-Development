# Phase 2 — Huấn luyện mô hình

**Mục tiêu:** huấn luyện mô hình cơ sở và 5 mô hình Học máy cho mỗi bài toán.
Đáp ứng yêu cầu R6, R7, R9.

**Tệp sở hữu:** `src/train.py`, `src/evaluate.py`

## Chia dữ liệu ba phần

`OOD (10%)` cắt ra trước tiên, rồi phần còn lại chia `train 80% / test 20%`.
Tập ngoại kiểm không tham gia bất kỳ bước phát triển nào, kể cả tinh chỉnh siêu
tham số, nên đo được khả năng tổng quát hoá thật sự.

## Danh mục mô hình

| Phân lớp | Hồi quy |
|----------|---------|
| Logistic Regression | Linear Regression |
| K-Nearest Neighbors | Decision Tree Regressor |
| Decision Tree | Random Forest Regressor |
| Random Forest | Support Vector Regression |
| XGBoost | XGBoost Regressor |

Mô hình cơ sở: `DummyClassifier(most_frequent)` và `DummyRegressor(median)`.

## Nguyên tắc bắt buộc

Mọi mô hình gói trong `Pipeline` gồm bước tiền xử lý và bước học. Bước tiền xử
lý chỉ `fit` trên tập huấn luyện, còn tập kiểm tra chỉ đi qua `transform`. Nhờ
đó cùng một đối tượng dùng lại được cho dữ liệu người dùng nhập trên web, bảo
đảm biểu diễn lúc dự đoán khớp biểu diễn lúc huấn luyện.

## Tiêu chí hoàn thành

- [x] 13 tệp `.joblib` trong `models/`
- [x] Mọi mô hình học được đều vượt mô hình cơ sở
- [x] Bảng so sánh lưu tại `outputs/*_model_comparison.csv`
