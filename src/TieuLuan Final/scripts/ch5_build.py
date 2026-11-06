"""Nạp ba miền vào Neo4j và ghi chunk ra data/ch5/chunks_<miền>.jsonl.

    python scripts/ch5_build.py [hetionet vn30 hotpot]
Chạy lại an toàn: mọi lệnh ghi dùng MERGE, ràng buộc khai báo IF NOT EXISTS.
"""
import sys
import time

from tlf import ch5_kg as K
from tlf import ch5_neo as N
from tlf import ch5_schema as S
from tlf.ch5_data import DATA5, write_jsonl
from tlf.common import METRICS, load_json, save_json

BUILD = {"hetionet": K.build_hetionet, "vn30": K.build_vn30, "hotpot": K.build_hotpot}


def apply_schema():
    for q in S.CONSTRAINTS + S.INDEXES:
        N.cypher(q)
    for ds, label in S.DATASETS.items():
        N.cypher(S.VECTOR_INDEX.format(name=f"vec_{ds}", label=label, dim=S.EMBED_DIM))
    N.cypher("CALL db.awaitIndexes(120)")


def load(ds):
    t0 = time.time()
    d = BUILD[ds]()
    N.batched(S.LOAD_NODES, d["nodes"])
    N.batched(S.LOAD_EDGES, d["edges"], size=1000)
    N.batched(S.LOAD_STATS, d["stats"], size=500)
    N.batched(S.LOAD_CHUNKS, d["chunks"], size=500)
    N.batched(S.LOAD_MENTIONS, d["mentions"], size=2000)
    write_jsonl(DATA5 / f"chunks_{ds}.jsonl", [{"id": c["id"], **c["props"]} for c in d["chunks"]])
    meta = d["meta"] | {"node_rows": len(d["nodes"]), "edge_rows": len(d["edges"]), "chunks": len(d["chunks"]),
                        "mention_rows": len(d["mentions"]), "stat_rows": len(d["stats"]),
                        "seconds": round(time.time() - t0, 1)}
    print(ds, meta, flush=True)
    return meta


if __name__ == "__main__":
    apply_schema()
    metas = load_json("ch5_build") if (METRICS / "ch5_build.json").exists() else {}
    for name in sys.argv[1:] or BUILD:
        metas[name] = load(name)
    save_json("ch5_build", metas | {"graph": N.stats()})
    print(N.stats()["nodes"], N.stats()["relationships"])
