"""Đọc số thật từ Neo4j: quy mô đồ thị, thống kê bậc, mô tả ba dataset -> ch5_graph.json, ch5_datasets.json."""
import numpy as np

from tlf import ch5_neo as N
from tlf import ch5_schema as S
from tlf.ch5_data import DATA5, RAW, read_jsonl
from tlf.common import load_json, save_json

SIZE_MB = {"hetionet": ("hetionet-nodes.tsv", "hetionet-edges.sif.gz"), "vn30": ("vn30_prices.csv", "vn30_wiki.jsonl"),
           "hotpot": ("hotpot_val.jsonl",)}
META = {
    "hetionet": ("Hetionet v1.0, tiểu đồ thị đái tháo đường", "https://github.com/hetio/hetionet", "CC0 1.0", "Đồ thị y sinh"),
    "vn30": ("VN30: giá phiên, ngành, Wikipedia tiếng Việt", "https://finance.yahoo.com, https://vi.wikipedia.org",
             "Yahoo: chỉ học thuật; Wikipedia: CC BY-SA 4.0", "Đồ thị tài chính và kho văn bản"),
    "hotpot": ("HotpotQA, tập validation, 1.000 câu đầu", "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
               "CC BY-SA 4.0", "Hỏi đáp đa bước trên văn bản"),
}


def domain_counts(ds):
    nodes = N.cypher("MATCH (e:Entity {dataset: $ds}) RETURN count(e) AS n", {"ds": ds})[0]["n"]
    chunks = N.cypher("MATCH (c:Chunk {dataset: $ds}) RETURN count(c) AS n", {"ds": ds})[0]["n"]
    edges = N.cypher("MATCH (e:Entity {dataset: $ds})-[r]->(:Entity) RETURN count(r) AS n", {"ds": ds})[0]["n"]
    labels = N.cypher("""MATCH (e:Entity {dataset: $ds}) UNWIND labels(e) AS l WITH l, count(*) AS n WHERE l <> 'Entity'
                         RETURN l AS label, n ORDER BY n DESC""", {"ds": ds})
    types = N.cypher("""MATCH (:Entity {dataset: $ds})-[r]->(:Entity) RETURN type(r) AS type, count(*) AS n
                        ORDER BY n DESC""", {"ds": ds})
    deg = N.cypher(S.DEGREES, {"ds": ds})
    d = np.array([r["degree"] for r in deg])
    return {"nodes": nodes, "edges": edges, "chunks": chunks, "by_label": {"rows": labels}, "by_type": {"rows": types},
            "degree": {"mean": float(d.mean()), "median": float(np.median(d)), "p99": float(np.percentile(d, 99)),
                       "max": int(d.max()), "isolated": int((d == 0).sum())},
            "hubs": {"rows": [{"name": r["name"], "degree": r["degree"]} for r in deg[:6]]},
            "degree_series": {"series": [int(x) for x in d[d > 0]]}}


def sample_rows(ds):
    return [{"id": r["id"], "title": r["title"], "text": r["text"][:140]} for r in read_jsonl(DATA5 / f"chunks_{ds}.jsonl")[3:6]]


def dataset_row(ds, g):
    name, url, lic, task = META[ds]
    qa = load_json("ch5_questions")[ds]
    size = sum((RAW / f).stat().st_size for f in SIZE_MB[ds]) / 2 ** 20
    top = g["by_type"]["rows"][:8]
    return {"id": ds, "name": name, "source": url.split(",")[0], "url": url, "task": task, "n_samples": g["chunks"],
            "n_features": g["nodes"], "size_mb": round(size, 1), "license": lic,
            "note": f"{g['nodes']} nút, {g['edges']} cạnh, {g['chunks']} chunk, {qa['n']} câu hỏi đánh giá.",
            "sample_rows": sample_rows(ds),
            "distribution": {"labels": [t["type"] for t in top], "values": [t["n"] for t in top]}}


if __name__ == "__main__":
    domains = {ds: domain_counts(ds) for ds in S.DATASETS}
    total = N.stats()
    save_json("ch5_graph", {"total": {"nodes": total["nodes"], "relationships": total["relationships"]},
                            "by_label": {"rows": total["by_label"]}, "by_type": {"rows": total["by_type"]}} | domains)
    save_json("ch5_cfg", {"batch": 2000, "top_k": 4, "ctx_words": 170, "rrf_c": 60, "boost": 0.15, "fanout": 120, "max_steps": 4, "first_load_s": 106.7})
    save_json("ch5_datasets", {"datasets": [dataset_row(ds, domains[ds]) for ds in S.DATASETS]})
    print(total["nodes"], total["relationships"], {d: (g["nodes"], g["edges"], g["chunks"]) for d, g in domains.items()})
