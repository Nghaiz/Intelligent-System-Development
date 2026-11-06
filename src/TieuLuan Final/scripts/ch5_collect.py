"""Gom kết quả sinh và tác tử thành ch5_compare.json (nguồn của bảng, hình, Web App)."""
import numpy as np

from tlf import ch5_schema as S
from tlf.ch5_data import DATA5, read_jsonl
from tlf.common import load_json, save_json
from tlf.llm import MODELS


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def summary(rows, dataset, model, kind, framework):
    secs = [r["seconds"] for r in rows]
    return {"dataset": dataset, "model": model, "framework": framework, "kind": kind, "n": len(rows), "em": mean(rows, "em"),
            "f1": mean(rows, "f1"), "faith": mean(rows, "faith"), "latency_p50": float(np.median(secs)),
            "latency_mean": float(np.mean(secs)), "seconds": float(np.sum(secs))}


def gen_rows():
    out = []
    for m in MODELS:
        path = DATA5 / f"gen_{m}.jsonl"
        if path.exists():
            rows = read_jsonl(path)
            out += [summary([r for r in rows if r["dataset"] == ds], ds, m, "llm", "RAG Hybrid top-4")
                    for ds in S.DATASETS if any(r["dataset"] == ds for r in rows)]
    return out


def agent_rows():
    path = DATA5 / "agent_compare.jsonl"
    if not path.exists():
        return [], {}
    rows = read_jsonl(path)
    out = []
    for ds in S.DATASETS:
        for mode in ("Naive RAG", "GraphRAG", "Agent"):
            g = [r for r in rows if r["dataset"] == ds and r["mode"] == mode]
            if g:
                out.append(summary(g, ds, mode, "system", rows[0]["model"]) | {"steps": mean(g, "steps"),
                                                                               "forced": mean(g, "forced")})
    return out, rows


def tools_used(rows):
    counts = {}
    for r in rows:
        if r["mode"] != "Agent":
            continue
        for s in r["trace"]:
            if s["kind"] == "action":
                t = s["text"].split("(")[0]
                counts[t] = counts.get(t, 0) + 1
    return counts


def tool_stats(rows):
    """Số lần gọi công cụ của tác tử theo kết cục, và EM theo loại câu hỏi của GraphRAG so với Agent."""
    calls = {}
    for r in rows:
        if r["mode"] != "Agent":
            continue
        acts = [s["text"].split("(")[0] for s in r["trace"] if s["kind"] == "action"]
        obs = [s["text"] for s in r["trace"] if s["kind"] == "observation"]
        for tool, o in zip(acts, obs):
            state = "err" if o.startswith("Lỗi") else "empty" if o.startswith("Không có") else "ok"
            calls.setdefault(r["dataset"], {}).setdefault(tool, {"ok": 0, "empty": 0, "err": 0})[state] += 1
    by_type = {}
    for r in rows:
        if r["mode"] in ("GraphRAG", "Agent"):
            by_type.setdefault(r["dataset"], {}).setdefault(r["type"], {}).setdefault(r["mode"], []).append(r["em"])
    em = {d: {t: {m: float(np.mean(v)) for m, v in ms.items()} | {"n": len(ms["Agent"])} for t, ms in ts.items()}
          for d, ts in by_type.items()}
    return {"calls": calls, "em": em}


def pick_traces(rows):
    """Một vết đúng và một vết sai của tác tử cho mỗi miền, lấy theo thứ tự xuất hiện, không chọn tay."""
    out = []
    for ds in S.DATASETS:
        g = [r for r in rows if r["mode"] == "Agent" and r["dataset"] == ds]
        for want in (1.0, 0.0):
            hit = next((r for r in g if r["em"] == want and r["trace"]), None)
            if hit:
                out.append({k: hit[k] for k in ("qid", "dataset", "question", "gold", "answer", "em", "trace", "sources", "seconds")})
    return out


def sl_keys(ret, llm, sysr):
    """Khóa ngắn cho macro SL trong lời văn: ch5_sl/<miền>/<mô hình>/<độ đo>."""
    out = {}
    for r in ret["rows"]:
        out.setdefault(r["dataset"], {})[r["model"]] = {k.replace("@", ""): r[k] for k in r if k.startswith("recall") or k in ("mrr", "ndcg", "ms_p50")}
    for r in ret["by_type"]["rows"]:
        out[r["dataset"]][r["model"]][r["type"]] = r["recall@5"]
    for r in llm + sysr:
        name = ("sys" if r["kind"] == "system" else "") + r["model"].replace(" ", "").replace(".", "")
        out[r["dataset"]][name] = {k: r[k] for k in ("em", "f1", "faith", "latency_p50", "n")}
    return out


if __name__ == "__main__":
    ret = load_json("ch5_retrieval")
    sysr, agent = agent_rows()
    llm = gen_rows()
    rows = [{k: v for k, v in r.items() if k != "seconds"} | {"kind": "retrieval", "seconds": r["seconds"]} for r in ret["rows"]]
    save_json("ch5_compare", {"rows": rows + llm + sysr, "primary": "em"})
    rows7 = read_jsonl(DATA5 / "gen_qwen2.5-7b.jsonl")
    save_json("ch5_sl", sl_keys(ret, llm, sysr) | {"gen": {"hit": mean(rows7, "hit"), "n": len(rows7),
                                                         "nodata": sum("Không đủ" in r["answer"] for r in rows7)}})
    if agent:
        save_json("ch5_traces", {"traces": pick_traces(agent), "tools": tools_used(agent)} | tool_stats(agent))
    print(len(rows), len(llm), len(sysr))
