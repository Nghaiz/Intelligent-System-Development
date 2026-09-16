# Phase 3 — Thực nghiệm có kiểm soát

**Mục tiêu:** trả lời bốn câu hỏi thực nghiệm được phát biểu **trước** khi chạy.
Đáp ứng yêu cầu R8 và R10.

**Tệp sở hữu:** `run_pipeline.py`

## Thực nghiệm 1 — So sánh mô hình

*Câu hỏi:* Trong cùng một giao thức đánh giá, mô hình nào cho kết quả tốt nhất
và chênh lệch có đáng kể so với mô hình cơ sở không?

Điều kiện giữ nguyên: cùng tập chia, cùng hạt giống, cùng bộ độ đo.

## Thực nghiệm 2 — Siêu tham số

*Câu hỏi 2a:* Số láng giềng k ảnh hưởng thế nào tới đánh đổi thiên lệch–phương sai?

*Câu hỏi 2b:* Hạ ngưỡng quyết định theta có tăng được Recall tới mức chấp nhận
được trong sàng lọc y tế không, và trả giá bằng bao nhiêu Precision?

*Câu hỏi 2c:* Độ sâu tối đa của Random Forest bão hoà ở đâu?

Chỉ một siêu tham số thay đổi giữa các lần chạy.

## Thực nghiệm 3 — Biểu diễn dữ liệu

*Câu hỏi:* Giữ nguyên thuật toán và siêu tham số, thay đổi cách biểu diễn X thì
kết quả đổi bao nhiêu?

Bốn phương án cho tiểu đường: thô / làm sạch / làm sạch + chuẩn hoá / cộng thêm
đặc trưng thiết kế. Ba phương án cho bất động sản: chỉ định lượng / thêm định
danh / cộng thêm đặc trưng thiết kế.

## Thực nghiệm 4 — Kiểm chuẩn ngoại kiểm

*Câu hỏi:* Hiệu năng trên tập chưa từng thấy có tụt so với tập kiểm tra không?
Chênh lệch lớn là dấu hiệu mô hình khớp quá mức vào đặc thù tập kiểm tra.

## Tiêu chí hoàn thành

- [x] 9 bảng CSV trong `outputs/`
- [x] Quét ngưỡng cho ra ít nhất 5 mức TP khác nhau (đạt 12 trên 13)
- [x] Cực trị siêu tham số nằm trong dải khảo sát, không chạm biên

## Bài học rút ra trong phase này

**Quét ngưỡng phải chạy trên mô hình cho xác suất liên tục.** Lần chạy đầu dùng
mô hình tốt nhất theo F1 là Decision Tree sâu 6 tầng. Cây nông chỉ trả về vài
giá trị xác suất rời rạc theo độ thuần của lá, nên bảy ngưỡng từ 0.50 xuống
0.20 cho ra đúng một kết quả — đường cong phẳng lì và thực nghiệm vô nghĩa.
Chuyển sang XGBoost thì mười ngưỡng cho mười kết quả khác nhau, Recall tăng từ
0.458 lên 0.854.

**Cực trị chạm biên dải khảo sát thì chưa phải cực trị.** Lần chạy đầu dải độ
sâu dừng ở 24 và kết quả tốt nhất rơi đúng vào 24 — không kết luận được. Mở
rộng dải tới 40 mới thấy điểm bão hoà thật ở 24.

Lỗi tương tự lặp lại ở thực nghiệm ngưỡng: dải quét ban đầu dừng ở 0.15 và cực
đại F1 rơi đúng vào 0.15. Nới dải xuống 0.05 mới xác nhận được đó là cực trị
thật — F1 đạt đỉnh 0.6508 tại 0.15 rồi tụt xuống 0.6324 tại 0.10.
