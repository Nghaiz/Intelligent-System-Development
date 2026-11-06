# Phase 4 — Website DiaNexus (làn L4)

Đọc trước: `contract.md`. Dùng lại khung: `src/TieuLuan Final/web/` (FastAPI, SSE logs, 3d-force-graph, ECharts, command palette, nền canvas, `web/vendor.mjs`; `node_modules` sao chép hoặc cài lại trong `src/Project 01/web`).
Phải là sản phẩm mới, nhìn khác hẳn Đồng Tâm AI Lab: thương hiệu y tế, nền tối xanh ngọc với điểm nhấn san hô, chế độ sáng, chữ tiếng Việt rõ (font đóng gói cục bộ qua @fontsource). Tránh gradient tím và thẻ bo tròn lặp lại. Không CDN: mọi thứ phục vụ cục bộ.

## Trang
1. **Tổng quan**: số sống từ `/api/status`, sơ đồ kiến trúc động (hai nhánh DB và KB hội tụ vào Agent), trạng thái Neo4j, Ollama, GPU.
2. **Hội chẩn** (chat): dòng thời gian tác tử trực tiếp qua SSE, câu trả lời gõ dần, trích dẫn `[n]` bấm được, đồ thị con mini, thẻ nguy cơ và SHAP, phản hồi người dùng.
3. **Sàng lọc thích ứng**: từng câu hỏi do RL chọn, thước xác suất nhảy theo đáp án, hiển thị Q-value, so với hỏi đủ 21 câu.
4. **Phòng thí nghiệm nguy cơ**: thanh trượt what-if cập nhật tức thì, SHAP thác nước, "song sinh số" so hồ sơ hiện tại và mục tiêu.
5. **Võng mạc**: kéo-thả ảnh hoặc chọn mẫu, phân cấp, Grad-CAM với thanh trượt so sánh trước-sau.
6. **Đồ thị tri thức 3D**: Neo4j thật, tìm thực thể, lọc nhãn, bấm mở rộng, hiện Cypher đã chạy.
7. **Tra cứu**: so 5 chế độ truy xuất cạnh nhau, tô sáng đoạn khớp.
8. **Dữ liệu**: duyệt bảng SQLite, phân trang, thống kê, ECharts EDA.
9. **Thí nghiệm**: bảng so sánh, biểu đồ từ `outputs/metrics` và `runs.jsonl`.
10. **Nhật ký**: log trực tiếp SSE từ `outputs/logs`, bộ lọc mức, vết tác tử phát lại được.

Hiệu ứng: chuyển trang, số đếm chạy, dòng hạt dữ liệu chạy qua sơ đồ, hover có ý nghĩa; `prefers-reduced-motion`. Bàn phím: Ctrl+K. Responsive tới 390 px.

## Xong khi
`python -m uvicorn web.server:app --host 127.0.0.1 --port 8097` chạy, mọi trang lấy dữ liệu thật qua HTTP đúng hợp đồng (phần chưa sẵn báo "đang chuẩn bị" rõ ràng, không dữ liệu giả). Kiểm bằng Python playwright (đã có trình duyệt trong `D:/DevCache/playwright`) hoặc công cụ có sẵn: mở từng trang, không lỗi console. Script `web/screenshots.py` chụp mọi trang vào `Report/Project 01/figures/web_*.png` ở 1440×900, cả ca thao tác thật (chat xong, sàng lọc xong, what-if, Grad-CAM, KG). Báo cáo cuối: danh sách trang, API, tệp ảnh.
