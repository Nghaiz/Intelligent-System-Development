# Brainstorm — Tiểu luận Final

**Trạng thái:** người dùng giao toàn quyền chọn phương án và làm hết trong một phiên, được phép dùng sub-agent, không cần hỏi lại.
**Kế hoạch:** `plans/261105-tieuluan-final/plan.md`.

## Suy ra từ bốn bài tham khảo
- Thứ tự chương chung: lịch sử AI, ML, CNN, RNN, KG/RAG, Agent. tupv thêm MAS và GNN, nên lấy bố cục tupv làm chính và thêm Agent 02.
- Mỗi chương 2-3 dataset, 3 mô hình, NumPy + Keras + PyTorch cho CNN và RNN; nguồn, mô tả, phân bố, mã, kết quả, deploy.
- Điểm yếu cần vượt: chỉ một bài có Neo4j thật và không có Cypher, số node/cạnh hay số đo truy xuất; KG của tupv dùng NetworkX với luật gán tay; nhiều bảng đối chuẩn MAS không có mã đo đi kèm.

## Quyết định
| Hạng mục | Chọn | Lý do |
|---|---|---|
| Bố cục | 8 chương + Web App + Kết luận | tupv + Agent 02 |
| Dataset | Mỗi chương 3 bộ từ nguồn khác nhau, gồm UCI, HF, Kaggle, Zenodo, Yahoo, Open-Meteo, GitHub, PyG | Yêu cầu đa dạng nguồn |
| KG | Neo4j thật, Docker, volume có tên | Yêu cầu của đề bài |
| LLM | Ollama, Qwen2.5-7B, Gemma3-4B, Qwen2.5-1.5B | Người dùng cho phép Ollama; Qwen3-4B bị loại vì in phần suy nghĩ vào câu trả lời; 3 LLM cho đánh giá chéo |
| Web | FastAPI + giao diện tĩnh giàu hiệu ứng, 3D KG, nhật ký trực tiếp, Experiment Tracker | Yêu cầu đẹp, mới lạ, có logs/data |
| Cỡ chữ | extreport 14pt, mã 12pt, hình chữ 13pt | Yêu cầu mọi thứ to hơn; 160 trang là trần |
| Thực thi | 8 làn sub-agent theo tiền tố chương, chỉ lead commit | Nhanh, tránh ghi đè |

## Rủi ro
- Commit chỉ trống 4-7 GB do các ứng dụng khác: `tlf.mem.slot` xếp hàng, làn CPU tối đa 8 luồng.
- LLM 4-bit có thể sinh chậm: giảm số câu đánh giá, ghi rõ cỡ mẫu.
- Trang Stanford SVHN không truy cập được: lấy qua Hugging Face.
