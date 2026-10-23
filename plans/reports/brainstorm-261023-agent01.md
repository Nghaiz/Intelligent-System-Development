# Brainstorm — Bài tập Agent 01: Discovering AI Agents + Q-Learning

**Trạng thái:** đang thực hiện (một phiên, người dùng giao toàn quyền chọn phương án).
**Ràng buộc người dùng:** chỉ làm đúng những gì bài mẫu `tmp/agent01_CT_02_tupv.pdf` có; cấm thêm phần thừa.
**Đầu ra:** `Report/Agent 01/agent01_CT_07_nghiand.pdf`; mã `src/Agent 01/`.

## Đề bài (suy từ bài mẫu, 11 trang)
Đọc Google Cloud, IBM Think, McKinsey (AI agent) + GeeksforGeeks (RL); dựng Q-Learning Agent trên Jupyter
Notebook; chụp màn hình mã (copy screen code) và kết quả (copy screen output).

## Bố cục (giữ đúng 10 mục của mẫu)
Mở đầu · 1 Bản chất, ba trụ cột · 2 Tiến hóa Chatbot → Copilot → Agent → Agentic Mesh · 3 IBM: chu trình,
ReAct vs ReWOO, mô hình suy luận khác · 4 Năm loại tác tử R&N, Learning Agent, MDP · 5 RL theo GfG
(ε-greedy, Bellman, bảng trường phái) · 6 Taxonomy doanh nghiệp, rủi ro, quản trị · 7 Thực nghiệm notebook
(mô tả bài toán, ảnh chụp mã) · 8 Kết quả (ảnh chụp đầu ra, phân tích hội tụ, trích Q-Table) · 9 LLM Agent × RL
(RLHF, MCTS, neuro-symbolic) · 10 Tổng kết · Tài liệu tham khảo.

## Khác mẫu (chỉ ở hình thức, theo yêu cầu "của riêng tôi", "to rõ hơn")
- Mê cung, tham số, số liệu của riêng mình; mọi số trong báo cáo sinh từ JSON do notebook ghi.
- Khuôn LaTeX Tiểu luận 03: extreport 14pt, bìa A04, mã ≥ 12pt, hình rộng đúng vùng chữ.
- Sơ đồ TikZ nhiều màu, khối 3D cho chính các hình của mẫu (kiến trúc nhận thức, lộ trình 4 giai đoạn,
  ReAct/ReWOO, Learning Agent) và vài sơ đồ minh họa cho các mục lý thuyết sẵn có (MDP, ε-greedy, mesh...).
- Mã nguồn trích nguyên văn vào báo cáo ngoài ảnh chụp màn hình.

## CPU / GPU
Q-Learning dạng bảng chỉ cần CPU (vài giây). Không đặt việc giả lên GPU.

## Loại bỏ (theo người dùng)
DQN, SARSA/so sánh thuật toán, ablation, thí nghiệm LLM ReAct/ReWOO, Web App.
