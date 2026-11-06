"""Chương 7: lấy lại số liệu Cora đã đo ở Tiểu luận 03 (cùng dữ liệu, cùng cấu hình), không chạy lại."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlf.common import ROOT, save_json  # noqa: E402

OLD = ROOT.parent / "TieuLuan 03" / "outputs" / "metrics"


def read(name):
    return json.loads((OLD / f"{name}.json").read_text(encoding="utf-8"))


def main():
    depth, extra = read("ch6_depth"), read("ch6_extra")
    tsne = extra["tsne"]
    save_json("ch7_reused", {
        "source": "src/TieuLuan 03/outputs/metrics (ch6_depth.json, ch6_extra.json)",
        "cora_depth": {"rows": depth["rows"], "propagation": {"rows": depth["propagation"]}},
        "cora_tsne": {"labels": tsne["labels"], "raw": tsne["raw"], "gcn": tsne["gcn"]},
        "cora_pyg": extra["pyg"], "cora_degree_acc": extra["degree_acc"]})
    print("đã ghi ch7_reused")


if __name__ == "__main__":
    main()
