# Phase 2 — KB, Neo4j KG, RAG (làn L2)

Đọc trước: `contract.md`. Dùng lại: `src/TieuLuan Final/tlf/ch5_{chunk,kg,neo,retrieval,rag,schema,qa}.py`,
`src/TieuLuan Final/data/ch5/raw/hetionet-*`, HF `e5-base`, `bge-reranker`, `dnx/llm.py`.
Neo4j `dnx-neo4j` đã chạy. Bật Ollama bằng `scripts/ollama_serve.ps1` (chạy nền, kiểm `127.0.0.1:11434` trước).

## Việc
1. **Thu thập** `scripts/kb_fetch.py` vào `data/kb/raw/` (tôn trọng robots, tốc độ thấp, User-Agent chung, không dùng email): Wikipedia vi (≥ 40 bài về đái tháo đường, biến chứng, thuốc, xét nghiệm, dinh dưỡng),
   MedlinePlus, NIDDK, CDC, WHO, ít nhất 1 PDF hướng dẫn (Bộ Y tế nếu lấy được, không thì WHO/NIDDK PDF), PubMed E-utilities **hoặc** Europe PMC REST
   (`https://www.ebi.ac.uk/europepmc/webservices/rest/search`) lấy ≥ 20.000 tóm tắt về đái tháo đường kèm MeSH. Ghi nguồn, URL, ngày lấy, số tài liệu thật vào `kb_sources.json`.
2. **Chunk + chỉ mục**: làm sạch, chunk theo câu; BM25 (`rank_bm25` hoặc tự viết nhanh hơn), e5 embedding GPU (float16 npy). `kb_chunks.json`.
3. **KG Neo4j** `scripts/kb_graph.py`: ontology đóng ≥ 12 nhãn, ≥ 15 loại quan hệ. Nguồn cạnh:
   (a) Hetionet: lân cận của các bệnh chuyển hóa, gene, compound, side effect, symptom, anatomy, pathway, giới hạn theo ngân sách;
   (b) MeSH + Article + Chunk từ PubMed; (c) trích bằng LLM có bằng chứng từ tài liệu y khoa (Qwen2.5-7B, ontology đóng, kiểm câu trích có trong văn bản, ≥ 300 chunk, chạy GPU độc chiếm, có cache để chạy tiếp);
   (d) ngưỡng chẩn đoán có op/value/unit/evidence/url; (e) cầu `Feature`–`MAPS_TO`–`Test/RiskFactor` nối cột dữ liệu với KG; (f) nút `Rule` đọc từ `dnx/rules.py` nếu có.
   Chỉ mục fulltext + ràng buộc. Mục tiêu ≥ 30k nút, ≥ 200k cạnh. `kg_stats.json` (đếm theo nhãn và quan hệ từ chính Neo4j, Cypher mẫu). Chấm mẫu 50 cạnh trích LLM theo seed, ghi precision và rõ ai chấm.
4. **Truy xuất** `dnx/kb/search.py`: BM25, dense, graph (thực thể → lân cận → chunk), RRF, rerank; đúng hợp đồng. Dịch hoặc mở rộng truy vấn Việt-Anh bằng từ điển/entities khi cần.
5. **Đánh giá** `scripts/kb_eval.py`: ≥ 150 câu vi/en sinh từ chunk bằng LLM + ≥ 40 câu tự viết, có chunk vàng. Recall@k, MRR, nDCG cho 5 chế độ; độ trễ; sinh RAG vs không RAG trên mẫu nhỏ (faithfulness bằng code: số trong câu trả lời có trong ngữ cảnh). `rag_eval.json`, `rag_per_q.jsonl`.
6. Hình: phân bố nguồn, độ dài chunk, phân bố nhãn/quan hệ KG, ảnh đồ thị con từ Neo4j (networkx vẽ đẹp), biểu đồ Recall/MRR/nDCG, độ trễ.

## Xong khi
`search` chạy < 1 s sau nạp, Neo4j đạt mục tiêu hoặc ghi rõ số thực, metrics đủ. Báo cáo cuối: số liệu chính, ontology, Cypher mẫu, tệp hình.
