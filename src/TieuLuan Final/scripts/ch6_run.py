"""Đánh giá kiến trúc tác tử theo từng lô ngắn: mỗi lô giữ khóa GPU vừa đủ cho --chunk câu trên một dataset
rồi trả
khóa cho làn khác. Kết quả ghi nối đuôi từng câu, nên tiến trình dừng giữa chừng thì chạy lại lệnh là tiếp tục."""
import argparse
import json
import time

from tlf.ch6_arch import ARCHS
from tlf.ch6_core import run_to_end
from tlf.ch6_domain import DOMAINS, is_correct, load_items
from tlf.ch6_llm import MODEL, gpu_session, open_llm
from tlf.common import data_dir
from tlf.mem import require_disk

RUNS = data_dir("ch6") / "runs"
RUNS.mkdir(exist_ok=True)


def result_path(ds, arch):
    return RUNS / f"{ds}_{arch}.jsonl"


def done_ids(ds, arch):
    path = result_path(ds, arch)
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").split(chr(10)) if line.strip()}


def record(run, item, ds, arch):
    """Một dòng cho mỗi câu: đáp án, sổ token và toàn bộ thông điệp đã trao đổi."""
    pred = run.result["answer"]
    return {"id": item["id"], "dataset": ds, "arch": arch, "model": MODEL, "question": item["question"],
            "gold": item["gold"], "pred": pred, "correct": is_correct(ds, pred, item["gold"]),
            "rounds": run.result["rounds"], "consensus": run.result.get("consensus"),
            "text": run.result["text"], "wall_s": round(run.wall_s, 3), "llm_s": round(run.llm_s, 3),
            **dict(run.meter), "messages": run.messages}


def play_chunk(ds, archs, items):
    """Một lượt GPU: mọi kiến trúc còn thiếu trên các câu này, xong trả khóa."""
    todo = [(a, it) for it in items for a in archs if it["id"] not in done_ids(ds, a)]
    if not todo:
        return 0
    with gpu_session(MODEL):
        llm = open_llm(MODEL)
        t0 = time.time()
        for arch, item in todo:
            run = run_to_end(ARCHS[arch](item, DOMAINS[ds]), llm)
            with open(result_path(ds, arch), "a", encoding="utf-8") as f:
                f.write(json.dumps(record(run, item, ds, arch), ensure_ascii=False) + "\n")
    print(f"{ds:9s} {len(todo)} lượt {time.time() - t0:.0f}s {time.strftime('%H:%M:%S')}", flush=True)
    return len(todo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--chunk", type=int, default=5)
    p.add_argument("--datasets", default="gsm8k,pubmed,clinical")
    p.add_argument("--archs", default="direct,single,supervisor,debate")
    a = p.parse_args()
    require_disk(1, "ch6 run")
    data = {ds: load_items(ds, a.n) for ds in a.datasets.split(",")}
    for start in range(0, a.n, a.chunk):
        for ds, items in data.items():
            play_chunk(ds, a.archs.split(","), items[start:start + a.chunk])


if __name__ == "__main__":
    main()
