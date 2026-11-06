"""Kết luận: bảng mô hình tốt nhất từng bộ dữ liệu, đọc trực tiếp outputs/metrics/chN_compare.json của mọi chương.

Chạy lại sau khi các chương xong: bảng và ghi chú chương còn thiếu tự cập nhật, không điền tay con số nào.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlf.common import METRICS  # noqa: E402
from tlf.texfmt import fmt, table, tex_escape  # noqa: E402

CHAPTERS = (2, 3, 4, 5, 6, 7, 8)
LOWER_BETTER = ("rmse", "mae", "mape", "smape", "mse", "loss", "regret", "error", "latency", "seconds")
# Ch.8 có năm kịch bản của cùng một môi trường; Kết luận chỉ giữ kịch bản tham chiếu.
ONLY_DATASET = {8: "Tham chiếu"}
TIE = 1e-6
# Tên hiển thị; id không có ở đây được in nguyên văn.
NAMES = {"svhn": "SVHN", "eurosat": "EuroSAT", "pathmnist": "PathMNIST", "weather": "Nhiệt độ Hà Nội", "power": "Công suất điện",
         "hetionet": "Hetionet", "vn30": "VN30", "hotpot": "HotpotQA", "gsm8k": "GSM8K", "pubmed": "PubMedQA",
         "clinical": "Ca lâm sàng", "f1_macro": "F1 macro", "rmse": "RMSE", "em": "EM", "accuracy": "Độ đúng",
         "exact_value": "Giá trị đúng", "single": "Single ReAct", "debate": "Debate", "qwen2.5-7b": "Qwen2.5-7B"}


def label(text):
    return ", ".join(NAMES.get(part, part) for part in text.split(", "))


def lower_better(metric):
    return any(tok in metric.lower() for tok in LOWER_BETTER)


def best_rows(chapter):
    path = METRICS / f"ch{chapter}_compare.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("primary_by_dataset") or {}
    groups = {}
    for r in data["rows"]:
        if r.get(metrics.get(r["dataset"], data["primary"])) is not None and ONLY_DATASET.get(chapter, r["dataset"]) == r["dataset"]:
            groups.setdefault(r["dataset"], []).append(r)
    out = []
    for ds, rows in groups.items():
        metric = metrics.get(ds, data["primary"])
        top = (min if lower_better(metric) else max)(r[metric] for r in rows)
        winners = dict.fromkeys(r["model"] for r in rows if abs(r[metric] - top) <= TIE)   # hòa thì nêu đủ
        out.append((ds, ", ".join(winners), metric, top))
    return out


def main():
    rows, missing = [], []
    for ch in CHAPTERS:
        found = best_rows(ch)
        if not found:
            missing.append(ch)
            continue
        for ds, models, metric, value in found:
            rows.append([str(ch), tex_escape(label(ds)), tex_escape(label(models)), tex_escape(label(metric)), fmt(value)])
    note = None
    if missing:
        note = "Chưa có số liệu của chương: " + ", ".join(map(str, missing)) + "."
        print("!! thiếu chN_compare.json:", missing)
    table("ch10_best", "clllr", ["Ch.", "Bộ dữ liệu", "Mô hình tốt nhất", "Chỉ số chính", "Giá trị"], rows, note=note)
    print(f"{len(rows)} dòng")


if __name__ == "__main__":
    main()
