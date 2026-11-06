"""Nạp đồ thị tri thức vào Neo4j theo lô UNWIND. Chạy: kb_graph.py <bước...> | all; mọi bước MERGE nên chạy lại an toàn.

Bước: wipe schema hetionet pubmed mentions clinical llm rules stats.
"""
import argparse
import gzip
import json
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import ROOT, limit_threads, save_json  # noqa: E402
from dnx.kb import lex, ontology as O, store  # noqa: E402
from dnx.kb.text import nfc  # noqa: E402
from dnx.mem import slot  # noqa: E402
from dnx.runlog import log_run  # noqa: E402

HET = ROOT.parent / "TieuLuan Final" / "data" / "ch5" / "raw"
RAW = store.KB / "raw"
LLM_EDGES = store.KB / "llm_edges.jsonl"
HET_EDGE_BUDGET = 260_000
OUTWARD_ALL = {"CtD", "CpD", "DaG", "DuG", "DdG", "DlA", "DpS", "DrD"}       # mọi cạnh chạm hạt bệnh
OUTWARD_NEAR = {"CbG", "CcSE", "PCiC", "CrC", "GpPW", "GpMF", "GpCC", "GpBP", "GiG"}  # lan từ lân cận bậc 1, ngân sách theo thứ tự này
CHECK_TAGS = {"humans", "male", "female", "adult", "aged", "middle aged", "young adult", "adolescent", "child", "animals", "mice",
              "rats", "aged, 80 and over", "infant", "child, preschool", "pregnancy", "cohort studies", "retrospective studies",
              "prospective studies", "cross-sectional studies", "treatment outcome", "risk factors", "time factors", "follow-up studies",
              "surveys and questionnaires", "case-control studies", "disease models, animal", "mice, inbred c57bl", "rats, sprague-dawley"}
TOK = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*", re.U)
TOKEN_MAX = 6


def run(q, rows, size=4000, **extra):
    t0 = time.time()
    for i in range(0, len(rows), size):
        store.cypher(q, rows=rows[i:i + size], **extra)
    return len(rows), time.time() - t0


def load_nodes(by_label):
    for label, rows in by_label.items():
        sup = ":Entity" if label in O.ENTITY_LABELS else ""
        key = label if not sup else "Entity"
        run(f"UNWIND $rows AS r MERGE (n:{key} {{id: r.id}}) SET n += r.p, n:{label}", rows)
        print(f"  nút {label}: {len(rows)}", flush=True)


def load_rels(triples, a="Entity", b="Entity"):
    """triples: (nguồn, đích, kiểu, thuộc tính)."""
    groups = defaultdict(list)
    for s, t, ty, p in triples:
        groups[ty].append({"s": s, "t": t, "p": p or {}})
    for ty, rows in groups.items():
        n, sec = run(f"UNWIND $rows AS r MATCH (a:{a} {{id: r.s}}) MATCH (b:{b} {{id: r.t}}) MERGE (a)-[e:{ty}]->(b) SET e += r.p", rows)
        print(f"  cạnh {ty}: {n} ({sec:.0f}s)", flush=True)


def step_wipe():
    while True:
        n = store.cypher("MATCH (n) WITH n LIMIT 20000 DETACH DELETE n RETURN count(*) AS c")[0]["c"]
        if not n:
            break


def step_schema():
    for label in ("Entity", "Article", "Chunk", "Document", "Threshold", "Feature", "Rule", "Source"):
        store.cypher(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")
    store.cypher("CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.name_vi]")
    store.cypher("CREATE FULLTEXT INDEX chunk_titles IF NOT EXISTS FOR (n:Chunk) ON EACH [n.title, n.preview]")
    store.cypher("CREATE INDEX article_year IF NOT EXISTS FOR (n:Article) ON (n.year)")


def het_kind(i):
    k = i.split("::")[0]
    return O.HETIONET_KIND.get(k, k)


def step_hetionet():
    names, kinds = {}, {}
    for line in open(HET / "hetionet-nodes.tsv", encoding="utf-8"):
        i, n, k = line.rstrip("\n").split("\t")[:3]
        names[i], kinds[i] = n, k
    seeds = {i for i, n in names.items() if kinds[i] == "Disease" and n in O.SEED_DISEASES}
    missing = set(O.SEED_DISEASES) - {names[i] for i in seeds}
    assert not missing, missing
    with gzip.open(HET / "hetionet-edges.sif.gz", "rt", encoding="utf-8") as f:
        next(f)
        edges = [ln.rstrip("\n").split("\t") for ln in f]
    hop1 = set(seeds)
    for s, m, t in edges:
        if s in seeds or t in seeds:
            hop1 |= {s, t}
    kept, later = [], defaultdict(list)
    for s, m, t in edges:
        if m not in O.HETIONET_EDGES:
            continue
        if (s in seeds or t in seeds) or (s in hop1 and t in hop1):
            kept.append((s, m, t))
        elif (s in hop1 or t in hop1) and m in OUTWARD_NEAR:
            later[m].append((s, m, t))
    core = len(kept)
    for m in sorted(OUTWARD_NEAR, key=lambda x: ("CbG", "CcSE", "PCiC", "CrC", "GpPW", "GpMF", "GpCC", "GiG", "GpBP").index(x)):
        room = HET_EDGE_BUDGET - len(kept)
        kept += later[m][:max(room, 0)]
    deg = Counter(x for s, _, t in kept for x in (s, t))
    by = defaultdict(list)
    for i in deg:
        by[het_kind(i)].append({"id": f"H:{i}", "p": {"name": names[i], "hid": i, "origin": "hetionet", "degree": deg[i]}})
    load_nodes(by)
    load_rels([(f"H:{s}", f"H:{t}", O.HETIONET_EDGES[m], {"metaedge": m, "origin": "hetionet"}) for s, m, t in kept])
    print(f"hetionet: toàn bộ {len(edges)} cạnh, hạt {len(seeds)}, bậc 1 {len(hop1)}, lõi {core}, giữ {len(kept)}", flush=True)
    return {"seeds": sorted(names[i] for i in seeds), "edges_full": len(edges), "hop1_nodes": len(hop1), "core_edges": core,
            "kept_edges": len(kept), "kept_nodes": len(deg), "budget": HET_EDGE_BUDGET}


def docs():
    for f in sorted(RAW.glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            yield json.loads(line)


def mesh_key(name):
    """'Diabetes Mellitus, Type 2' -> 'type 2 diabetes mellitus' để so với tên Hetionet."""
    n = name.lower()
    if ", " in n:
        a, b = n.split(", ", 1)
        return f"{b} {a}"
    return n


def step_pubmed():
    arts, documents, mesh_rows, has, co = [], [], {}, [], Counter()
    sources = {}
    src_rel = []
    for d in docs():
        sources[d["source"]] = 1
        if d["source"] == "europepmc":
            m = d["meta"]
            aid = f"A:{m['pmid']}"
            arts.append({"id": aid, "p": {"title": d["title"][:300], "pmid": m["pmid"], "year": int(m["year"] or 0), "journal": m.get("journal") or "", "url": d["url"]}})
            major = []
            for h in m["mesh"]:
                n = nfc(h["name"])
                mesh_rows.setdefault(f"M:{n}", {"id": f"M:{n}", "p": {"name": n, "origin": "mesh"}})
                has.append((aid, f"M:{n}", "HAS_MESH", {"major": h["major"]}))
                if h["major"] and n.lower() not in CHECK_TAGS:
                    major.append(f"M:{n}")
            co.update(combinations(sorted(set(major)), 2))
            src_rel.append((aid, "S:europepmc", "FROM_SOURCE", {}))
        else:
            documents.append({"id": d["doc_id"], "p": {"title": d["title"][:300], "url": d["url"], "lang": d["lang"]}})
            src_rel.append((d["doc_id"], f"S:{d['source']}", "FROM_SOURCE", {}))
    load_nodes({"Source": [{"id": f"S:{s}", "p": {"name": s}} for s in sources], "Article": arts, "Document": documents, "MeshTerm": list(mesh_rows.values())})
    load_rels(has, a="Article")
    load_rels([x for x in src_rel if x[0].startswith("A:")], a="Article", b="Source")
    load_rels([x for x in src_rel if not x[0].startswith("A:")], a="Document", b="Source")
    load_rels([(a, b, "CO_OCCURS", {"articles": c}) for (a, b), c in co.items() if c >= 5])
    rows = [json.loads(x) for x in open(store.CHUNKS, encoding="utf-8")]
    nodes = [{"id": r["chunk_id"], "p": {"title": r["title"][:200], "source": r["source"], "lang": r["lang"], "words": r["words"], "preview": r["text"][:220]}} for r in rows]
    load_nodes({"Chunk": nodes})
    load_rels([(r["chunk_id"], "A:" + r["doc_id"].split(":", 1)[1], "PART_OF", {}) for r in rows if r["source"] == "europepmc"], a="Chunk", b="Article")
    load_rels([(r["chunk_id"], r["doc_id"], "PART_OF", {}) for r in rows if r["source"] != "europepmc"], a="Chunk", b="Document")


def build_gazetteer():
    """Tên thường -> [id, tên, nhãn]. Ưu tiên thực thể lâm sàng, rồi Hetionet, rồi MeSH; ký hiệu gen phải chủ yếu viết hoa trong kho."""
    gaz = {}

    def add(name, hit):
        key = " ".join(TOK.findall(nfc(name).lower()))
        if len(key) >= 3:
            gaz.setdefault(key, hit)

    for i, label, name, vi, aliases in O.CLINICAL:
        hit = [i, name, label]
        for x in (name, vi) + tuple(aliases):
            add(x, hit)
    het = store.cypher("MATCH (n:Entity) WHERE n.origin='hetionet' RETURN n.id AS id, n.name AS name, [x IN labels(n) WHERE x<>'Entity'][0] AS label")
    rows = [json.loads(x) for x in open(store.CHUNKS, encoding="utf-8")]
    upper, anyc = Counter(), Counter()
    genes = {r["name"].lower() for r in het if r["label"] == "Gene"}
    for r in rows:
        for t in TOK.findall(r["text"]):
            low = t.lower()
            if low in genes:
                anyc[low] += 1
                upper[low] += t.isupper() or any(ch.isdigit() for ch in t)
    for r in sorted(het, key=lambda x: x["label"] == "Gene"):
        n = r["name"].lower()
        if r["label"] == "Gene":
            if len(n) >= 3 and anyc[n] >= 1 and upper[n] / anyc[n] >= 0.8:
                add(r["name"], [r["id"], r["name"], "Gene"])
        elif r["label"] not in ("BiologicalProcess", "CellularComponent", "MolecularFunction"):
            add(r["name"], [r["id"], r["name"], r["label"]])
    mesh = store.cypher("MATCH (n:MeshTerm) RETURN n.id AS id, n.name AS name")
    same = []
    for r in mesh:
        low = r["name"].lower()
        if low in CHECK_TAGS or low in O.GAZ_GENERIC:
            continue
        key = " ".join(TOK.findall(mesh_key(r["name"])))
        if key in gaz and gaz[key][0] != r["id"]:
            same.append((r["id"], gaz[key][0], "SAME_AS", {}))
        add(mesh_key(r["name"]), [r["id"], r["name"], "MeshTerm"])
        add(r["name"], [r["id"], r["name"], "MeshTerm"])
    names = {r["id"]: r for r in mesh} | {r["id"]: r for r in het}
    for alias, target in O.GAZ_ALIASES.items():
        hit = names.get(target)
        assert hit, f"bí danh trỏ tới nút không tồn tại: {alias} -> {target}"
        add(alias, [target, hit["name"], hit.get("label", "MeshTerm")])
    for vi, en in lex.PAIRS:   # bí danh Việt trỏ về thực thể tiếng Anh
        key = " ".join(TOK.findall(en.lower()))
        if key in gaz:
            add(vi, gaz[key])
    store.GAZ.write_text(json.dumps(gaz, ensure_ascii=False), encoding="utf-8")
    print("gazetteer", len(gaz), "SAME_AS", len(same), flush=True)
    return gaz, same, rows


def scan(tokens, gaz, cap=20):
    found, i = {}, 0
    while i < len(tokens) and len(found) < cap:
        for n in range(min(TOKEN_MAX, len(tokens) - i), 0, -1):
            hit = gaz.get(" ".join(tokens[i:i + n]))
            if hit:
                found[hit[0]] = 1
                i += n - 1
                break
        i += 1
    return found


def step_mentions():
    step_clinical_nodes()
    gaz, same, rows = build_gazetteer()
    load_rels(same)
    while store.cypher("MATCH ()-[r:MENTIONS]->() WITH r LIMIT 20000 DELETE r RETURN count(*) AS c")[0]["c"]:
        pass
    mesh_of = {}
    for d in docs():
        if d["source"] == "europepmc":
            mesh_of[d["doc_id"]] = [f"M:{nfc(h['name'])}" for h in d["meta"]["mesh"] if h["name"].lower() not in CHECK_TAGS]
    rels, cols, ent = [], [], {}
    ptr = [0]
    for r in rows:
        found = scan(TOK.findall(nfc(f"{r['title']}. {r['text']}").lower()), gaz)
        rels += [(r["chunk_id"], e, "MENTIONS", {}) for e in found]
        cols += [ent.setdefault(e, len(ent)) for e in dict.fromkeys(list(found) + mesh_of.get(r["doc_id"], []))]
        ptr.append(len(cols))
    m = sparse.csr_matrix((np.ones(len(cols), np.float32), np.array(cols, np.int32), np.array(ptr, np.int64)), shape=(len(rows), len(ent)))
    sparse.save_npz(store.MENTION, m.tocsc())
    store.ENT_IDS.write_text(json.dumps(list(ent), ensure_ascii=False), encoding="utf-8")
    load_rels(rels, a="Chunk")


def step_clinical_nodes():
    by = defaultdict(list)
    for i, label, name, vi, aliases in O.CLINICAL:
        by[label].append({"id": i, "p": {"name": name, "name_vi": vi, "aliases": list(aliases), "origin": "dianexus", "degree": 50}})
    load_nodes(by)


def step_clinical():
    """Ngưỡng có bằng chứng (d) và cầu Feature (e)."""
    step_clinical_nodes()
    rows = store.thresholds_table()
    load_nodes({"Threshold": [{"id": r["id"], "p": {"name": f"{r['test']} {r['op']} {r['value']} {r['unit']}", **{k: r[k] for k in ("test", "op", "value", "unit", "meaning", "evidence", "url")}}} for r in rows]})
    load_rels([(r["test_id"], r["id"], "HAS_THRESHOLD", {}) for r in rows], b="Threshold")
    run("UNWIND $rows AS r MATCH (a:Threshold {id: r.s}) MATCH (b:Chunk {id: r.t}) MERGE (a)-[:EVIDENCED_BY]->(b)",
        [{"s": r["id"], "t": r["chunk_id"]} for r in rows])
    load_nodes({"Feature": [{"id": f"F:{k}", "p": {"name": k, "name_vi": v[0]}} for k, v in O.FEATURE_MAP.items()]})
    load_rels([(f"F:{k}", e, "MAPS_TO", {}) for k, v in O.FEATURE_MAP.items() for e in v[1]], a="Feature")


def step_llm():
    rows = [json.loads(x) for x in open(LLM_EDGES, encoding="utf-8")]
    nodes = defaultdict(dict)
    for e in rows:
        for side in ("s", "o"):
            nodes[e[f"{side}_label"]].setdefault(e[f"{side}_id"], {"id": e[f"{side}_id"], "p": {"name": e[f"{side}_name"], "origin": "llm", "degree": 1}})
    new = {k: [n for n in v.values() if n["id"].startswith("L:x:")] for k, v in nodes.items()}
    load_nodes({k: v for k, v in new.items() if v})
    load_rels([(e["s_id"], e["o_id"], e["rel"], {"origin": "llm", "model": "qwen2.5-7b", "evidence": e["evidence"], "chunk_id": e["chunk_id"], "url": e["url"]}) for e in rows])


def step_rules():
    """Nút Rule đọc từ dnx/rules.json qua load_rules của làn L3; thiếu tệp thì báo lỗi, không tạo nút giả."""
    from dnx.rules import load_rules
    nodes, rels = [], []
    feats = {f"F:{k}": k for k in O.FEATURE_MAP}
    tests = {"hba1c": "T:hba1c", "fpg": "T:fpg", "ogtt": "T:ogtt", "rpg": "T:rpg", "bmi": "T:bmi", "egfr": "T:egfr"}
    for r in load_rules():
        rid = r.get("id") or r["rule_id"]
        nodes.append({"id": f"RU:{rid}", "p": {"name": rid, "severity": r.get("severity", ""), "priority": r.get("priority", 0),
                                               "source": str(r.get("source", ""))[:200]}})
        blob = json.dumps(r, ensure_ascii=False, default=str)
        rels += [(f"RU:{rid}", f, "USES_FEATURE", {}) for f, k in feats.items() if re.search(rf"\b{k}\b", blob)]
        rels += [(f"RU:{rid}", t, "USES_FEATURE", {}) for k, t in tests.items() if re.search(rf"\b{k}\b", blob, re.I)]
    load_nodes({"Rule": nodes})
    load_rels([x for x in rels if x[1].startswith("F:")], a="Rule", b="Feature")
    load_rels([x for x in rels if not x[1].startswith("F:")], a="Rule")


SAMPLES = {
    "thuoc_dieu_tri_t2dm": "MATCH (c:Compound)-[:TREATS]->(d:Disease {name:'type 2 diabetes mellitus'}) RETURN c.name AS thuoc ORDER BY thuoc LIMIT 10",
    "ty_le_gen_lien_quan": "MATCH (d:Disease {name:'type 2 diabetes mellitus'})-[:ASSOCIATES]->(g:Gene) RETURN count(g) AS so_gen",
    "nguong_hba1c_co_bang_chung": "MATCH (t:Test {id:'T:hba1c'})-[:HAS_THRESHOLD]->(h)-[:EVIDENCED_BY]->(c:Chunk) RETURN h.op AS op, h.value AS value, h.unit AS unit, h.meaning AS y_nghia, c.source AS nguon LIMIT 6",
    "dac_trung_den_nguong": "MATCH (f:Feature {name:'hba1c'})-[:MAPS_TO]->(t)-[:HAS_THRESHOLD]->(h) RETURN f.name_vi AS dac_trung, t.name AS xet_nghiem, h.op AS op, h.value AS value, h.meaning AS y_nghia LIMIT 6",
    "mesh_dong_xuat_hien": "MATCH (a:MeshTerm {name:'Diabetes Mellitus, Type 2'})-[r:CO_OCCURS]-(b) RETURN b.name AS mesh, r.articles AS bai ORDER BY bai DESC LIMIT 8",
    "chunk_nhac_thuoc": "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name:'Metformin'}) RETURN c.source AS nguon, count(*) AS so_chunk ORDER BY so_chunk DESC LIMIT 6",
    "canh_llm_bang_chung": "MATCH (a)-[r {origin:'llm'}]->(b) RETURN a.name AS chu_the, type(r) AS quan_he, b.name AS doi_tuong, left(r.evidence, 120) AS bang_chung LIMIT 5",
    "duong_di_gen_thuoc": "MATCH p=(:Compound {name:'Metformin'})-[:BINDS]->(:Gene)<-[:ASSOCIATES]-(:Disease {name:'type 2 diabetes mellitus'}) RETURN [n IN nodes(p) | n.name] AS duong_di LIMIT 5",
}


def step_stats():
    from dnx.kb import search as K
    s = K.stats()
    samples = {}
    for k, q in SAMPLES.items():
        rows = store.cypher(q)
        samples[k] = {"cypher": q, "rows": rows[:8], "n": len(rows)}
    s["samples"] = samples
    prev = {}
    p = ROOT / "outputs" / "metrics" / "kg_hetionet.json"
    if p.exists():
        prev = json.loads(p.read_text(encoding="utf-8"))
    s["hetionet"] = prev
    s["n_labels"] = len(s["by_label"])   # gồm siêu nhãn Entity
    s["n_rel_types"] = len(s["by_type"])
    save_json("kg_stats", s)
    print(json.dumps({k: s[k] for k in ("nodes", "relationships", "n_labels", "n_rel_types")}), s["by_label"], s["by_type"], flush=True)
    return s


if __name__ == "__main__":
    limit_threads(8)
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+")
    a = ap.parse_args()
    steps = ["wipe", "schema", "hetionet", "pubmed", "mentions", "clinical", "llm", "rules", "stats"] if a.steps == ["all"] else a.steps
    with slot(2, "kb_graph"):
        for st in steps:
            t0 = time.time()
            print(f"== {st}", flush=True)
            res = globals()[f"step_{st}"]()
            if st == "hetionet":
                save_json("kg_hetionet", res)
            print(f"== {st} xong {time.time() - t0:.0f}s", flush=True)
            log_run("kg", "kb", f"neo4j_{st}", "neo4j", {}, time.time() - t0, "cpu")
