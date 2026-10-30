# Brainstorm — Bài tập Agent 02: Tác tử học tăng cường và tác tử theo luật (hệ đa tác tử)

**Trạng thái:** đang thực hiện (một phiên; người dùng giao toàn quyền chọn phương án, được spawn sub-agent).
**Đầu ra:** `Report/Agent 02/agent02_CT_07_nghiand.pdf` (50–55 trang, cấm vượt 55); mã `src/Agent 02/`.
Kế hoạch thực thi: `plans/261030-agent02/plan.md`.

## Đề bài (suy từ 5 bài mẫu `tmp/agent02_*.pdf`; đề gốc `assignment_Agent02_due_29.10.pdf` không có trong máy)
- Đọc 3 tài liệu: Tutorial MAS (Python + PyTorch, hệ mua sắm Catalog/Recommendation/Inventory/Coordinator,
  bộ phân loại ý định), Naeem et al. 2020 (IEEE Access, RL nhập môn: MDP, Bellman, Q-learning, SARSA, ứng dụng,
  thách thức), Tran et al. 2025 (arXiv 2501.06322, cơ chế cộng tác đa tác tử LLM: actors, kiểu, chiến lược,
  cấu trúc, điều phối).
- Laboratory 02: hai cửa hàng TMĐT, mỗi bước chọn giảm giá/không. s=(d, a_đối thủ(t−1)), p=clip(0,30+0,25d
  +0,15a_i−0,10a_−i; 0,05; 0,95), N=10/20, S~Bin(N,p), r=S[P(1−δa)−C], P=10, C=5, δ=0,2, 20 bước/episode,
  5.000 episode, α=0,1, γ=0,9, ε=max(0,02; 0,2·0,995^e). Tác tử luật: d=0→0; d=1∧c=1→1; d=1∧c=0→0.
- Thí nghiệm A baseline · B quá trình học · C α∈{0,01; 0,1; 0,5} · D lịch khám phá (thấp/cơ sở/cao) ·
  E so sánh công bằng nhiều seed. Mười câu hỏi (mô hình hóa; học vs luật; tính 1 bước cập nhật Q; khám phá;
  reward vs mục tiêu kinh doanh; đánh giá hiệu năng; α và ổn định; đa tác tử & phi dừng; giới hạn + một thay
  đổi môi trường; tác tử tốt hơn).

## Phương án chọn (tự quyết, không hỏi)
| Hạng mục | Chọn | Lý do |
|---|---|---|
| Phạm vi | Lab 02 đầy đủ + 3 chương lý thuyết + hệ mua sắm MAS có bộ phân loại PyTorch | Gộp mọi yêu cầu xuất hiện trong 5 mẫu (02 có MAS + lý thuyết; 03/04 có Q9–10 cài thật) |
| Đánh giá | Tách huấn luyện/đánh giá, ε=0, Q đóng băng, seed riêng, số ngẫu nhiên chung (paired) + đánh giá chính xác (lan truyền phân phối) + best-response liệt kê 16 chính sách | Tốt hơn mẫu: tách nhiễu lấy mẫu khỏi chất lượng chính sách |
| Seed | Của riêng mình: baseline 2026; 10 seed 7,17,…,97 (nhóm 07) | "Bản của riêng tôi" |
| Mở rộng | F δ∈{0,05…0,30}; G nhu cầu bền (Markov p_giữ=0,8); H hai Q-learning cùng học; Q10: Q-learning trạng thái mở rộng + tác tử dựa trên mô hình | Mỗi mục truy về câu 8/9/10 |
| CPU/GPU | Mọi lượt RL trên CPU, multiprocessing (≤ 24 tiến trình); GPU chỉ cho bộ phân loại ý định PyTorch, chạy song song với quét CPU | Q-table 8 ô không có lợi gì trên GPU |
| Khuôn | Sao khuôn Agent 01 (extreport 14pt, bìa A04, \SL, \maNguon, TikZ nhiều màu/3D), thêm Mục lục (tocdepth 1) | Đã được người dùng chấp nhận |
| Song song | 2 làn fork: (L1) Ch.1–3 lý thuyết + TikZ; (L2) Ch.8 hệ mua sắm (mã + GPU + chương). Lead: mã RL, thực nghiệm, Ch.0, 4–7, 9, dựng PDF, commit | Lý thuyết và MAS độc lập với số liệu RL |

## Ngân sách trang (14pt)
Bìa+mục lục 3 · Ch0 Mở đầu 1,5 · Ch1 MAS 4,5 · Ch2 RL (Naeem) 5,5 · Ch3 MAS-LLM (Tran) 4 · Ch4 Mô hình 5 ·
Ch5 Cài đặt 7 · Ch6 Kết quả A–H 10 · Ch7 Mười câu hỏi 7 · Ch8 Hệ mua sắm 4 · Ch9 Thảo luận+KL 2 · TLTK 1 → ≈ 54.
Kiểm `pdfinfo` sau mỗi lượt dựng; > 55 thì cắt mã dài trước, hình sau.

## Rủi ro
- Không có đề gốc/mã tham chiếu: dựng lại từ công thức chung của 5 mẫu (thống nhất tuyệt đối).
- Tutorial MAS không có trong máy: mô tả theo mẫu 02, tự cài hệ mua sắm của mình.
- Vượt 55 trang: ngân sách cứng ở trên, rà sau lượt dựng đầu.
