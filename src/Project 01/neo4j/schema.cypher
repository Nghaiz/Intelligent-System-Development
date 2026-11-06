// DiaNexus - lược đồ đồ thị tri thức trên Neo4j 5.26
// Trùng với scripts/kb_graph.py (step_schema). Chạy được nhiều lần nhờ IF NOT EXISTS.
// Chạy: dán vào Neo4j Browser http://127.0.0.1:7474 (neo4j / dianexus2026), hoặc
//   docker exec -i dnx-neo4j cypher-shell -u neo4j -p dianexus2026 < neo4j/schema.cypher

CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT article_id IF NOT EXISTS FOR (n:Article) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT threshold_id IF NOT EXISTS FOR (n:Threshold) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT feature_id IF NOT EXISTS FOR (n:Feature) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT rule_id IF NOT EXISTS FOR (n:Rule) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT source_id IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE;

CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.name_vi];
CREATE FULLTEXT INDEX chunk_titles IF NOT EXISTS FOR (n:Chunk) ON EACH [n.title, n.preview];
CREATE INDEX article_year IF NOT EXISTS FOR (n:Article) ON (n.year);

// Nhãn tri thức (đều mang thêm siêu nhãn Entity):
//   Disease, Gene, Compound, Symptom, SideEffect, Anatomy, Pathway, PharmacologicClass,
//   BiologicalProcess, MolecularFunction, CellularComponent, MeshTerm, Test, RiskFactor, Lifestyle, Complication
// Nhãn cấu trúc: Article, Document, Chunk, Threshold, Feature, Rule, Source
// Quan hệ chính: MENTIONS (Chunk->Entity), PART_OF (Chunk->Document), HAS_MESH, CO_OCCURS,
//   TREATS, CAUSES, RISK_FACTOR_FOR, COMPLICATION_OF, HAS_THRESHOLD, EVIDENCED_BY, MAPS_TO, USES_FEATURE
