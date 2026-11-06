"""Kịch bản hội chẩn minh họa: chọn một ca Metformin đã chạy thật trong data/ch6/runs, lưu vết và sinh hộp
vetsuyluan.
Không gọi LLM: mọi thông điệp lấy nguyên từ lượt đánh giá, chữ Hán do mô hình trôi ngôn ngữ được đánh dấu."""
import json
import re

from tlf.ch6_domain import DOMAINS, is_correct
from tlf.common import GEN, data_dir, save_json
from tlf.texfmt import tex_escape

RUNS = data_dir("ch6") / "runs"
CJK = re.compile(r"[\u3000-\u30ff\u4e00-\u9fff\uff00-\uffef]+")
COLOR = {"Supervisor": "cOrange", "Verifier": "cRed", "Trọng tài": "cGreen", "Tác tử A": "cBlue",
         "Tác tử B": "cPurple"}
SHORT = {"Dược lâm sàng": "Dược", "Thận-Nội tiết": "Thận", "Công cụ guideline": "Công cụ"}
MAX_CHARS = 125


def load(arch):
    path = RUNS / f"clinical_{arch}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split(chr(10)) if line.strip()]
    return {r["id"]: r for r in rows}


def pick(sup, deb):
    """Ca có nhiều thuốc chữ Metformin nhất ưu tiên, trong đó chọn vết Supervisor ngắn và ít chữ Hán."""
    ids = set(sup) & set(deb)
    pool = [i for i in ids if "Metformin" in sup[i]["question"]] or list(ids)

    def cost(i):
        msgs = sup[i]["messages"] + deb[i]["messages"]
        return sum(bool(CJK.search(m["content"])) for m in msgs) * 5 + len(msgs)
    return min(pool, key=cost)


def to_box(row):
    """Mỗi thông điệp một dòng; nội dung dài bị cắt và đánh dấu, bản đầy đủ nằm trong ch6_scenario.json."""
    lines, seen = [], set()
    for m in row["messages"]:
        if not m["tokens"]:
            if m["content"] in seen:     # công cụ trả lại đúng nội dung đã hiện: bỏ cho gọn hộp
                continue
            seen.add(m["content"])
        text = CJK.sub("[chữ Hán]", " ".join(m["content"].split()))
        cut = text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + " [...]"
        who = SHORT.get(m["from"], m["from"])
        color = COLOR.get(m["from"], "cTeal" if m["tokens"] else "cSlate")
        tail = f" \\textcolor{{black!55}}{{({m['tokens']} token)}}"
        lines.append(f"\\vetBuoc{{{color}}}{{{tex_escape(who)}}}{{{tex_escape(cut)}{tail}}}")
    return "\n".join(lines) + "\n"


def main():
    sup, deb, single = load("supervisor"), load("debate"), load("single")
    cid = pick(sup, deb)
    traces, summary = {}, {}
    for arch, table in (("single", single), ("supervisor", sup), ("debate", deb)):
        row = table[cid]
        pred = DOMAINS["clinical"].parse(row["text"])
        traces[arch] = row
        summary[arch] = {"answer": pred or "không đọc được", "correct": is_correct("clinical", pred, row["gold"]),
                         "rounds": row["rounds"], "tokens": row["tokens_in"] + row["tokens_out"],
                         "seconds": row["wall_s"], "messages": len(row["messages"])}
    save_json("ch6_scenario", {"case": sup[cid]["question"], "case_id": cid, "gold": sup[cid]["gold"],
                               "summary": summary, "traces": traces})
    (GEN / "ch6_case.tex").write_text(tex_escape(sup[cid]["question"]) + "\n", encoding="utf-8", newline="\n")
    for arch in ("supervisor", "debate"):
        (GEN / f"ch6_trace_{arch}.tex").write_text(to_box(traces[arch]), encoding="utf-8", newline="\n")
    print(cid, summary, flush=True)


if __name__ == "__main__":
    main()
