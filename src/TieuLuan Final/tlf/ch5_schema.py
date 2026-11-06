"""Chương 5: lược đồ Neo4j và các truy vấn Cypher có tên. Báo cáo trích nguyên văn các chuỗi này."""

DATASETS = {"hetionet": "ChunkHetionet", "vn30": "ChunkVn30", "hotpot": "ChunkHotpot"}
EMBED_DIM = 768

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
    "CREATE CONSTRAINT year_stat_id IF NOT EXISTS FOR (s:YearStat) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT month_stat_id IF NOT EXISTS FOR (s:MonthStat) REQUIRE s.id IS UNIQUE",
]
INDEXES = [
    "CREATE INDEX entity_dataset IF NOT EXISTS FOR (e:Entity) ON (e.dataset)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX year_stat_year IF NOT EXISTS FOR (s:YearStat) ON (s.year)",
    "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.ticker]",
]
VECTOR_INDEX = """CREATE VECTOR INDEX {name} IF NOT EXISTS
FOR (c:{label}) ON (c.embedding)
OPTIONS {{indexConfig: {{`vector.dimensions`: {dim},
                        `vector.similarity_function`: 'cosine'}}}}"""

LOAD_NODES = """UNWIND $rows AS r
MERGE (n:Entity {id: r.id})
SET n += r.props
WITH n, r CALL apoc.create.addLabels(n, [r.label]) YIELD node
RETURN count(node) AS n"""

LOAD_EDGES = """UNWIND $rows AS r
MATCH (a:Entity {id: r.source}), (b:Entity {id: r.target})
CALL apoc.merge.relationship(a, r.type, {}, r.props, b, {}) YIELD rel
RETURN count(rel) AS n"""

LOAD_STATS = """UNWIND $rows AS r
MATCH (c:Company {ticker: r.ticker}), (p:Entity {id: r.period})
CALL apoc.merge.node([r.label], {id: r.id}, r.props) YIELD node
CALL apoc.merge.relationship(c, r.has, {}, {}, node, {}) YIELD rel AS a
CALL apoc.merge.relationship(node, r.of, {}, {}, p, {}) YIELD rel AS b
RETURN count(*) AS n"""

LOAD_CHUNKS = """UNWIND $rows AS r
MERGE (c:Chunk {id: r.id})
SET c += r.props
WITH c, r CALL apoc.create.addLabels(c, [r.label]) YIELD node
UNWIND r.about AS eid
MATCH (e:Entity {id: eid})
MERGE (node)-[:ABOUT]->(e)
RETURN count(*) AS n"""

LOAD_MENTIONS = """UNWIND $rows AS r
MATCH (c:Chunk {id: r.chunk}), (e:Entity {id: r.entity})
MERGE (c)-[:MENTIONS]->(e)
RETURN count(*) AS n"""

SET_EMBEDDINGS = """UNWIND $rows AS r
MATCH (c:Chunk {id: r.id})
CALL db.create.setNodeVectorProperty(c, 'embedding', r.vec)
RETURN count(*) AS n"""

LINK_ENTITIES = """CALL db.index.fulltext.queryNodes('entity_names', $lucene)
YIELD node AS e, score
WHERE e.dataset = $ds
RETURN e.id AS id, e.name AS name, score
ORDER BY score DESC LIMIT $n"""

VECTOR_SEARCH = """CALL db.index.vector.queryNodes($index, $k, $vec)
YIELD node AS c, score
RETURN c.id AS id, score"""

SEED_ENTITIES = """UNWIND $chunks AS cid
MATCH (:Chunk {id: cid})-[:ABOUT]->(e:Entity)
RETURN DISTINCT e.id AS id"""

GRAPH_RETRIEVE = """UNWIND $seeds AS sid
MATCH (s:Entity {id: sid})
CALL (s) {
  MATCH (s)-[r]-(m:Entity)
  RETURN m ORDER BY type(r), m.id LIMIT $fanout
}
WITH collect(DISTINCT s) + collect(DISTINCT m) AS nodes
UNWIND nodes AS n
MATCH (c:Chunk)-[:ABOUT|MENTIONS]->(n)
WITH c, count(DISTINCT CASE WHEN n.id IN $linked THEN n END) AS hits
WITH c, hits, vector.similarity.cosine(c.embedding, $vec) AS sim
RETURN c.id AS id, sim + $boost * hits AS score, hits, sim
ORDER BY score DESC LIMIT $k"""

SUBGRAPH = """MATCH (s:Entity)
WHERE s.id = $seed OR toLower(s.name) = toLower($seed)
WITH s LIMIT 1
CALL apoc.path.subgraphAll(s, {maxLevel: $depth, limit: $limit})
YIELD nodes, relationships
RETURN nodes, relationships"""

COUNT_LABELS = """MATCH (n)
UNWIND labels(n) AS label
RETURN label, count(*) AS n ORDER BY n DESC"""

COUNT_TYPES = """MATCH ()-[r]->()
RETURN type(r) AS type, count(*) AS n ORDER BY n DESC"""

DEGREES = """MATCH (n:Entity {dataset: $ds})
RETURN n.name AS name, labels(n) AS labels, COUNT { (n)--() } AS degree
ORDER BY degree DESC"""

# Mẫu câu hỏi trên đồ thị: đáp án của bộ đánh giá được tính từ chính các truy vấn này.
Q_DRUG_TARGET_ASSOC = """MATCH (c:Compound)-[:TREATS]->(d:Disease {id: $disease})
MATCH (c)-[:BINDS]->(g:Gene)<-[:ASSOCIATES]-(d)
RETURN c.name AS compound, g.name AS gene
ORDER BY compound, gene"""

Q_TOP_YEAR_IN_SECTOR = """MATCH (c:Company)-[:IN_SECTOR]->(s:Sector)
MATCH (c)-[:HAS_YEAR]->(y:YearStat {year: $year})
WITH s, c, y ORDER BY y.ret_pct DESC
WITH s, collect({ticker: c.ticker, ret: y.ret_pct})[0] AS top
RETURN s.name AS sector, top.ticker AS ticker, top.ret AS ret_pct
ORDER BY ret_pct DESC"""

Q_TOP_CORRELATED = """MATCH (a:Company {ticker: $ticker})-[r:CORRELATED]-(b:Company)
RETURN b.ticker AS peer, r.rho AS rho
ORDER BY rho DESC LIMIT $k"""

Q_BRIDGE = """MATCH (a:Chunk)-[:MENTIONS]->(e:Entity)<-[:ABOUT]-(b:Chunk)
WHERE a.dataset = 'hotpot' AND a <> b
RETURN a.title AS from_doc, e.name AS bridge, b.title AS to_doc
LIMIT $k"""
