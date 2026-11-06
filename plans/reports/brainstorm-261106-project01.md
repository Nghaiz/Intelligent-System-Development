# Brainstorm — Project 01 (DiaNexus)

**Trạng thái:** người dùng giao toàn quyền chọn phương án, làm hết trong một phiên, cho phép sub-agent, không hỏi lại.
**Kế hoạch:** `plans/261106-project01/plan.md` · **Hợp đồng tích hợp:** `plans/261106-project01/contract.md`.

## Suy ra từ 4 bài tham khảo (`tmp/*.pdf`, 19–40 trang)
Đề là sơ đồ trên bảng của giảng viên: `DB → ML/DL → Dự đoán/Phân loại` và `PDF/web → KB → RAG → Agent` (kèm Rule, Tool, RL),
hai nhánh hợp lại thành câu trả lời. 3/4 bài chọn đái tháo đường, 1 bài CARLA.

| Bài | Mạnh | Yếu cần vượt |
|---|---|---|
| ducba | 7 vai trò, MessageBus, Q-learning 8 trạng thái | KB 12 ghi chú, đồ thị 12 khái niệm, không Neo4j |
| khoipv | Neo4j, 816 thực thể, 6 agent LangGraph, ablation 96 lượt, SHAP, isotonic | 20 trang web, không RL, Gradio |
| dungvt | HybridRAG BM25 + vector + Neo4j + RRF, SQLite | 19 trang, chỉ Logistic Regression, không RL |
| anhnt | Đa tác tử CARLA, chống rò rỉ | Khác miền |

## Quyết định
| Hạng mục | Chọn | Lý do |
|---|---|---|
| Đề tài | **DiaNexus**: hệ đa tác tử sàng lọc, tư vấn, tra cứu đái tháo đường | Đúng sơ đồ, nhiều tài sản nhất trong repo (A01–A05, TL03, TLF) |
| Điểm riêng | Sàng lọc thích ứng bằng RL thật trên dữ liệu thật; ảnh võng mạc; KG lớn | Không bài mẫu nào có; RL không phải simulator giả lập |
| DB | SQLite (hồ sơ, hội thoại, vết, dự đoán) ~450k dòng | Đúng nhánh DB, KISS |
| Dữ liệu bảng | 100k + CDC 253k + UCI 130 viện 101k + Pima + Frankfurt | Lớn, dùng lại 4/5 bộ |
| Ảnh | Bệnh võng mạc đái tháo đường (HF), 5 cấp | Đa phương thức, nhiều hình |
| KB | Wikipedia vi, MedlinePlus, NIDDK, WHO, CDC, PDF, PubMed/Europe PMC | Quy mô vượt hẳn 20 trang |
| KG | Neo4j: Hetionet + MeSH + trích LLM + cầu Feature | Đã có Hetionet trên đĩa |
| RAG | BM25 + e5 + đồ thị + RRF + rerank | Dùng lại `tlf/ch5_retrieval.py` |
| LLM | Qwen2.5-7B qua Ollama | Đã có, tiếng Việt tốt |
| Agent | Đồ thị trạng thái tự viết, cạnh điều kiện và quay lui | Mã ngắn đưa vào báo cáo được |
| Web | FastAPI + giao diện tĩnh, dùng lại khung TLF, thương hiệu mới | Đã có 3D KG, SSE, ECharts |
| Cỡ chữ | extreport 14pt, mã ≥11pt, hình chữ ≥12pt | Yêu cầu chữ to; trần 55 trang |
| Thực thi | Tối đa 4 làn, Opus, effort medium, không commit | Ràng buộc commit memory, luật chủ máy |

## Phân CPU/GPU
- **CPU (32 luồng):** XGBoost, RF, LR, EDA, BM25, nạp Neo4j, thu thập web, RL env.
- **GPU (RTX 4060 8 GB), tuần tự qua `gpu.heavy`:** MLP/FT-Transformer, CNN võng mạc, embedding, LLM trích xuất, DQN, chạy ablation.

## Rủi ro
- Trần 55 trang ở 14pt: `build.py` đếm trang; mỗi chương có ngân sách cứng.
- PubMed không phản hồi lúc thử: dùng Europe PMC REST.
- Commit trống ~15 GB: `mem.slot`, tối đa 4 làn; LLM độc chiếm khi trích xuất.
- Trang y khoa chặn tải: báo số trang lấy được thực tế, không đoán.
- Không viết con số nào ngoài `outputs/metrics/*.json`.
