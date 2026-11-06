// DiaNexus - truy vấn mẫu để demo đồ thị tri thức. Mỗi khối chạy độc lập trong Neo4j Browser.
// Chạy cả tệp: docker exec -i dnx-neo4j cypher-shell -u neo4j -p dianexus2026 < neo4j/queries.cypher

// 1. Quy mô đồ thị
MATCH (n) RETURN count(n) AS so_nut;
MATCH ()-[r]->() RETURN count(r) AS so_canh;

// 2. Số nút theo nhãn
MATCH (n) UNWIND labels(n) AS nhan RETURN nhan, count(*) AS so_nut ORDER BY so_nut DESC;

// 3. Số cạnh theo loại quan hệ
MATCH ()-[r]->() RETURN type(r) AS quan_he, count(*) AS so_canh ORDER BY so_canh DESC;

// 4. Thuốc điều trị đái tháo đường típ 2 (Hetionet)
MATCH (c:Compound)-[:TREATS]->(d:Disease {name:'type 2 diabetes mellitus'})
RETURN c.name AS thuoc ORDER BY thuoc LIMIT 20;

// 5. Nối hai nhánh: đặc trưng ML -> xét nghiệm -> ngưỡng -> đoạn văn bằng chứng
MATCH (f:Feature {name:'hba1c'})-[:MAPS_TO]->(t:Test)-[:HAS_THRESHOLD]->(h)-[:EVIDENCED_BY]->(c:Chunk)
RETURN f.name_vi AS dac_trung, t.name AS xet_nghiem, h.op AS op, h.value AS gia_tri, h.unit AS don_vi,
       h.meaning AS y_nghia, c.source AS nguon LIMIT 10;

// 6. Luật dùng đặc trưng nào
MATCH (r:Rule)-[:USES_FEATURE]->(f:Feature) RETURN r.id AS luat, collect(f.name) AS dac_trung ORDER BY luat;

// 7. Đường đi gen: Metformin -> gen <- đái tháo đường típ 2
MATCH p=(:Compound {name:'Metformin'})-[:BINDS]->(:Gene)<-[:ASSOCIATES]-(:Disease {name:'type 2 diabetes mellitus'})
RETURN [n IN nodes(p) | n.name] AS duong_di LIMIT 10;

// 8. Chủ đề MeSH hay đi cùng đái tháo đường típ 2
MATCH (a:MeshTerm {name:'Diabetes Mellitus, Type 2'})-[r:CO_OCCURS]-(b)
RETURN b.name AS mesh, r.articles AS so_bai ORDER BY so_bai DESC LIMIT 10;

// 9. Nguồn nào nhắc Metformin nhiều nhất
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name:'Metformin'})
RETURN c.source AS nguon, count(*) AS so_doan ORDER BY so_doan DESC;

// 10. Cạnh do LLM trích, kèm câu bằng chứng (chỉ để gợi ý, precision mẫu 0,54)
MATCH (a)-[r {origin:'llm'}]->(b)
RETURN a.name AS chu_the, type(r) AS quan_he, b.name AS doi_tuong, left(r.evidence, 120) AS bang_chung LIMIT 10;

// 11. Tìm thực thể theo tên (full-text, có tiếng Việt)
CALL db.index.fulltext.queryNodes('entity_names', 'võng mạc OR retinopathy') YIELD node, score
RETURN labels(node) AS nhan, node.name AS ten, score LIMIT 10;

// 12. Đồ thị con để xem hình (bấm tab Graph trong Browser)
MATCH p=(d:Disease {name:'type 2 diabetes mellitus'})-[:TREATS|CAUSES|RISK_FACTOR_FOR|COMPLICATION_OF|PRESENTS]-(x)
RETURN p LIMIT 40;
