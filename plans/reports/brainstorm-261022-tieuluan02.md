# Brainstorm — Tiểu luận 02: Knowledge Graph, RAG và AI Agent

**Trạng thái:** đang thực hiện (người dùng giao toàn quyền chọn phương án, làm từ đầu đến cuối trong một phiên).
**Đầu ra:** `Report/TieuLuan 02/tieuluan02_CT_07_nghiand.pdf`; mã nguồn `src/TieuLuan 02`.

## Đề bài (suy từ hai bài mẫu)
- Mẫu chính tupv (A07, 43 tr): KG & RAG → kiến trúc KG/KB, RDF, ontology, Graph DB vs Vector DB → các thế hệ RAG (Naive, chunking 5 chiến lược, Advanced, GraphRAG) → hệ thống y tế (đái tháo đường VN) → hệ thống tài chính (VN30, vàng SJC) → AI Agent ReAct (DL + KG + RAG) → Web App → kết luận.
- Tham khảo dungvt (25 tr): lịch sử KG/RAG, KB từ dữ liệu bảng (bản ghi → tài liệu + bộ ba), TF-IDF / embedding / truy xuất đồ thị / hybrid, đánh giá RAG, rủi ro, deploy chatbot cho 3 bộ dữ liệu.

## Điểm yếu của mẫu cần làm tốt hơn
- "Mô hình học sâu" của tupv là luật if-else, "BiLSTM" nhân 1,018; RAG là đếm từ trùng trên 2 câu. → Bản này huấn luyện thật, truy xuất thật, có đánh giá định lượng.
- Một số dữ kiện y khoa bị phóng đại (GI gạo lứt 50, "tuyệt đối chống chỉ định") → dùng số liệu bảng GI quốc tế 2008, diễn đạt đúng mức khuyến cáo.

## Phương án chọn (khuyến nghị, không hỏi lại)
| Hạng mục | Chọn | Lý do |
|---|---|---|
| Kho tri thức | Tự biên soạn ~16 tài liệu Markdown tiếng Việt (8 y tế, 8 tài chính), có tiêu đề phân cấp | Cần văn bản thật để chunking theo cấu trúc có nghĩa |
| KG | NetworkX MultiDiGraph + rdflib (Turtle, SPARQL, suy luận RDFS tự viết) | Đủ cả LPG lẫn RDF, không cần Neo4j server |
| KG từ bảng | Theo cách dungvt trên 100k hồ sơ tiểu đường và phiên giao dịch | Trực tiếp yêu cầu của mẫu tham khảo |
| Truy xuất | TF-IDF, BM25, dense (3 mô hình e5/MiniLM), hybrid RRF, rerank bge-reranker-v2-m3, mở rộng đồ thị | Đối chuẩn đầy đủ các thế hệ RAG |
| Đánh giá | Bộ câu hỏi có đáp án span, Recall@k, MRR, nDCG; chunking 5 chiến lược | "Đánh giá RAG" của dungvt, có số thật |
| GraphRAG | Louvain trên KG hợp nhất, tóm tắt cộng đồng, local/global search | Đúng mục 3.4 tupv |
| Học sâu | MLP PyTorch (GPU) trên 100k hồ sơ; BiLSTM (GPU) dự báo FPT và vàng quy đổi VND | Thay mô hình giả của tupv |
| LLM | Qwen2.5-1.5B-Instruct chạy cục bộ trên GPU, chỉ dùng để tổng hợp câu trả lời có căn cứ | Nội bộ, không gọi API ngoài |
| Agent | ReAct: bộ lập kế hoạch tất định chọn công cụ, LLM viết câu trả lời, kiểm chứng trích dẫn | LLM 1,5B gọi công cụ không ổn định; vết suy luận vẫn đủ Thought/Action/Observation |
| Web | Flask 4 phân hệ (Y tế, Tài chính, Khám phá KG, Tra cứu RAG), chụp bằng Playwright | Giống TL01 |
| Báo cáo | Khuôn LaTeX TL01: extreport 14pt, minted ≥ 12pt, TikZ nhiều màu/3D, matplotlib rộng 6,3 in | Theo phản hồi "mọi thứ phải to" |

## Phân chia CPU / GPU
- GPU: huấn luyện MLP, BiLSTM; mã hóa embedding; reranker; LLM sinh câu trả lời.
- CPU (song song): dựng KG, SPARQL, BM25/TF-IDF, Louvain, thực nghiệm chunking, vẽ hình, biên dịch LaTeX.

## Rủi ro
- LLM 1,5B tiếng Việt có thể yếu → câu trả lời được ràng buộc bởi bằng chứng, có bộ kiểm chứng; ghi rõ giới hạn.
- Dữ liệu SJC không có nguồn miễn phí → dùng giá vàng thế giới quy đổi VND/lượng; phần bù SJC chỉ nêu như tri thức trong KG.
- Hệ thống mang tính học thuật, không thay lời khuyên y tế/đầu tư (ghi trong báo cáo và giao diện).
