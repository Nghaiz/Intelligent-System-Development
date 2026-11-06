"""Gộp số liệu, sinh bảng và hình, rồi tự kiểm tra ba dự đoán của serve_ch2 trong một slot bộ nhớ."""
import runpy
import sys

from tlf.common import limit_threads
from tlf.mem import require_disk, slot


def check_serve():
    from tlf import serve_ch2 as s
    from tlf.common import save_json
    samples = {"ch2_diabetes": {}, "ch2_housing": {}, "ch2_sentiment": {"text": "thầy giảng dễ hiểu và rất nhiệt tình"}}
    out = {"ch2_sentiment_neg": s.predict("ch2_sentiment", {"text": "môn học quá khó và giáo viên không nhiệt tình"})}
    for sid, payload in samples.items():
        out[sid] = s.predict(sid, payload)
        print(sid, out[sid], flush=True)
    print(out["ch2_sentiment_neg"], flush=True)
    save_json("ch2_serve_check", {"examples": out})


if __name__ == "__main__":
    require_disk(0.2, "ch2 hình và bảng")
    limit_threads(4)
    with slot(1.5, "ch2 finish"):
        for name in ("scripts/ch2_tables.py", "scripts/ch2_figs.py"):
            runpy.run_path(name, run_name="__main__")
        check_serve()
    sys.exit(0)
