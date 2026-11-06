# Phase 3 — Rule, RL, hệ đa tác tử (làn L3)

Đọc trước: `contract.md`. Dùng lại: `src/TieuLuan Final/tlf/ch6_{core,rules,tools,arch}.py`, `ch5_agent.py`, `dnx/llm.py`, `dnx/db.py` (L1 viết).
L1, L2 làm song song: khi `dnx/ml/serve.py` hoặc `dnx/kb/search.py` chưa có, kiểm thử bằng stub trong `tests/`, không sửa tệp của họ.

## Việc
1. **Rule** `dnx/rules.py`: suy diễn tiến, có id, nguồn, ưu tiên: dấu hiệu khẩn (đường huyết rất cao/thấp, triệu chứng cấp), ngưỡng chẩn đoán (đọc từ `thresholds()` khi có), an toàn thuốc (ví dụ metformin với eGFR thấp), chính sách không nêu liều. Bảng luật dữ liệu ở `dnx/rules.json`. ≥ 25 luật. Test pytest.
2. **RL sàng lọc thích ứng** `dnx/rl/`: bài toán chính, RL trên dữ liệu thật CDC (bảng `brfss`, tách theo split của L1 nếu có, nếu không dùng seed 42 theo tỉ lệ 70/15/15):
   - Mô hình dự đoán chịu được thiếu đặc trưng (MLP nhận giá trị + mặt nạ, huấn luyện che ngẫu nhiên, GPU).
   - MDP: trạng thái = đáp án đã có + mặt nạ + xác suất hiện tại; hành động = hỏi đặc trưng k, hoặc dừng và kết luận; thưởng = −chi phí câu hỏi (chi phí khác nhau) + thưởng cuối (log-score hoặc F-beta).
   - Double DQN (replay, target net, ε giảm dần) so với Q-learning bảng (rời rạc hóa), thứ tự cố định theo độ quan trọng, ngẫu nhiên, hỏi đủ 21 câu, tham lam theo thông tin. So AUC/F1 theo số câu; đường cong đánh đổi chi phí-chính xác; ≥ 5 seed cho DQN.
   - `dnx/rl/policy.py` đúng hợp đồng, câu hỏi tiếng Việt và lựa chọn trả lời cho từng đặc trưng. `rl_compare.json`, `rl_history.json`.
3. **Hệ đa tác tử** `dnx/agents/`: đồ thị trạng thái tự viết (blackboard + cạnh điều kiện + giới hạn vòng lặp) hoặc LangGraph nếu gọn hơn. 8 vai trò:
   Supervisor (định tuyến), Intake (trích hồ sơ tiếng Việt, hỏi lại, RL chọn câu), Risk, Retina, Knowledge (HybridRAG + KG, tự mở rộng truy vấn), RuleGuard, Advisor (LLM Qwen2.5-7B, trích dẫn `[n]`), Critic (kiểm bằng code: số, trích dẫn, liều thuốc, khẳng định chẩn đoán; rồi trả về Advisor tối đa 2 vòng).
   Ba cạnh quay lui trở lên. Phát `Event` qua `on_event`, ghi `agent_events`. Công cụ khai báo rõ cho từng agent, mỗi agent ≤ 5 công cụ.
4. **Đánh giá** `scripts/agent_eval.py`, chạy sau khi L1, L2 xong: ≥ 40 kịch bản `data/scenarios.jsonl` chia nhóm (có xét nghiệm, chưa xét nghiệm, thiếu dữ liệu, hỏi kiến thức, đòi kê đơn, khẩn cấp, ngoài phạm vi, ảnh). 5 cấu hình: full, no_critic, vector_only, no_rules, single_agent. Chấm bằng code: đúng tuyến, số trung thực, trích dẫn hợp lệ, vi phạm an toàn, thời gian. Có cache để chạy tiếp. `agent_ablation.json`, `agent_runs.jsonl`.
5. Hình: đường học DQN, chi phí-chính xác, tần suất câu hỏi, heatmap Q, ablation, thời gian, sơ đồ vết thực.

## Xong khi
pytest xanh, `run()` chạy thật có LLM, metrics đủ. Báo cáo cuối: số chính, danh sách luật theo nhóm, sơ đồ cạnh, tệp hình.
