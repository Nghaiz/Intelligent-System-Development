# Phase 4 — Jupyter Notebook

**Mục tiêu:** hai notebook chạy được từ đầu đến cuối, đủ 22 mục mà đề bài liệt
kê ở Mục 21. Đáp ứng yêu cầu R14.

**Tệp sở hữu:** `notebooks/01_diabetes_classification.ipynb`,
`notebooks/02_housing_regression.ipynb`

## 22 mục bắt buộc theo đề bài

1. Định nghĩa hệ thống và bài toán
2. Sơ đồ hệ thống thông minh
3. Nguồn dữ liệu
4. Mô tả dữ liệu
5. Biểu diễn dữ liệu
6. Phân tích đặc trưng và mục tiêu
7. Phân tích khám phá
8. Chia train/test
9. Mô hình cơ sở
10. Mô hình 1
11. Mô hình 2
12. Mô hình 3
13. Mô hình 4
14. Đánh giá
15. Thực nghiệm 1 — So sánh mô hình
16. Thực nghiệm 2 — Siêu tham số
17. Thực nghiệm 3 — Biểu diễn dữ liệu
18. Mô hình cuối cùng
19. Ứng dụng
20. Minh hoạ hệ thống
21. Phản chiếu
22. Kết luận

## Nguyên tắc

Notebook **gọi lại** các hàm trong `src/`, không chép lại logic. Như vậy notebook
và ứng dụng web luôn dùng chung một biểu diễn, và sửa một chỗ là sửa mọi nơi.

## Tiêu chí hoàn thành

- [x] `jupyter nbconvert --execute --to notebook` trả về mã thoát 0 — cả hai tệp
- [x] Đủ 22 mục, mỗi mục có tiêu đề Markdown rõ ràng
- [x] Mọi ô có đầu ra đã lưu, người chấm không cần chạy lại mới thấy kết quả

## Đã hoàn thành — ghi chú bàn giao

Notebook 01 có 89 ô (44 ô mã), notebook 02 có 86 ô (40 ô mã). Thời gian chạy lại
từ đầu: 15 giây và 95 giây.

Biểu đồ được sinh bằng cách gọi hàm trong `src/viz.py` rồi hiển thị lại tệp PNG
qua `IPython.display.Image`, vì `viz` dùng backend `Agg` nên không vẽ trực tiếp
vào notebook. Cách này giữ nguyên SSOT: hình trong notebook, trong `figures/` và
trong báo cáo LaTeX là **cùng một tệp**.

Notebook **không** ghi đè `models/` hay `outputs/`. Mục 18 của cả hai tệp nạp lại
pipeline mà `run_pipeline.py` đã lưu rồi đối chiếu dự đoán — vừa chứng minh tính
tái lập (R14) vừa tránh 17 MB thay đổi vô nghĩa trong git mỗi lần chạy notebook.

**Nhãn phương án trong Thực nghiệm 3 phải trùng khít với `run_pipeline.py`.**
Notebook và script cùng ghi ra `figures/dia_representation.png` và
`figures/hou_representation.png`; đặt nhãn khác nhau sẽ khiến hình phụ thuộc vào
việc chạy cái nào sau cùng.

### Ba phát hiện khoa học mới, phát sinh khi viết notebook

1. **Biểu diễn KHÔNG luôn quan trọng hơn thuật toán.** Ở bài toán tiểu đường,
   biên độ F1 do đổi biểu diễn là 0.0300 còn do đổi thuật toán là 0.1809 — gấp 6
   lần. Ở bài toán bất động sản thì ngược lại. Điều phân biệt: biểu diễn có làm
   đổi *lượng thông tin* hay chỉ đổi *dạng*. Bốn phương án biểu diễn của bài toán
   1 chỉ phân loại khác nhau 1–9 ca trên 139, tức nằm trong nhiễu lấy mẫu.
2. **`Price` của tập bất động sản đã bị cắt biên từ nguồn**: nằm gọn trong
   1 – 11.5 tỷ VNĐ, skewness −0.03, kurtosis −0.85. Giả định "giá nhà lệch phải
   mạnh" — đúng với thị trường thật — **sai** với tập dữ liệu này.
3. **Mô hình bất động sản có độ chệch đơn điệu theo phân khúc giá**: định giá cao
   hơn thực tế +1.05 tỷ ở nhóm rẻ nhất, thấp hơn −1.48 tỷ ở nhóm đắt nhất. Hồi
   quy về trung bình, cộng với việc cây quyết định không ngoại suy được.

Cả ba đã viết vào Mục 21 của notebook tương ứng và cần đưa vào phần Phân tích
khoa học của báo cáo LaTeX (Phase 8).
