"""Sinh bảng và khối kết quả của Chương 3 vào Report/TieuLuan Final/generated/ từ các tệp metrics."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlf.ch3_common import DS_IDS  # noqa: E402
from tlf.ch3_data import DATASETS  # noqa: E402
from tlf.common import GEN, LOGS, load_json  # noqa: E402
from tlf.texfmt import best, num, sci, table, tex_escape  # noqa: E402

SHORT = {d: DATASETS[d]["name"].replace(" RGB", "") for d in DS_IDS}


def t_overview():
    rows = []
    for d in load_json("ch3_datasets")["datasets"]:
        s = d["splits"]
        side = d["image_size"]
        rows.append([SHORT[d["id"]], f"{side}×{side}×3", d["n_classes"], num(s["train"], 0),
                     num(s["val"], 0) if "val" in s else "--", num(s["test"], 0), num(d["size_mb"], 1),
                     num(d["imbalance"], 2),
                     tex_escape(d["license"])])
    table("ch3_overview", "lcrrrrrrl",
          ["Bộ dữ liệu", "Cỡ ảnh", "Lớp", "Train", "Val", "Test", "Tải về (MB)", "Lệch lớp", "Giấy phép"],
          rows, note="Lệch lớp là tỉ số lớp đông nhất trên lớp ít nhất. Dung lượng là tệp gốc tải về.")


def t_compare():
    data = [r for r in load_json("ch3_compare")["rows"] if r["model"] != "CNN hai tầng"]
    rows = []
    for ds in DS_IDS:
        sub = [r for r in data if r["dataset"] == ds]
        acc = best([r["accuracy"] for r in sub], [num(r["accuracy"], 4) for r in sub])
        f1 = best([r["f1_macro"] for r in sub], [num(r["f1_macro"], 4) for r in sub])
        for i, r in enumerate(sub):
            tag = " (Keras)" if r["framework"] == "Keras" else ""
            rows.append([SHORT[ds] if i == 0 else "", tex_escape(r["model"]) + tag, acc[i], f1[i],
                         num(r["params"], 0) if r["params"] else "--",
                         num(r["seconds"], 0) if r["params"] else "--",
                         num(r["infer_ms"], 2) if r["infer_ms"] else "--",
                         "TL01" if r["reused_from"] else "mới"])
        rows.append("\\midrule")
    table("ch3_compare", "llrrrrrl",
          ["Bộ dữ liệu", "Mô hình", "Độ đúng", "F1 macro", "Tham số", "Huấn luyện (s)", "ms/ảnh CPU", "Nguồn"],
          rows[:-1], note="Số in đậm là tốt nhất trong từng bộ. Nguồn TL01 là kết quả dùng lại từ Tiểu luận 01, "
          "mới là chạy cho chương này.")


def t_platform():
    """So cùng CNN cơ bản qua nền tảng; dòng NumPy suy luận chạy lại trọng số PyTorch bằng hàm viết tay."""
    su = load_json("ch3_summary")
    rows = []
    for ds in ("svhn", "eurosat"):
        p = su["platform"][ds]
        rows.append([SHORT[ds], "PyTorch", "GPU", "huấn luyện", num(p["torch_acc"], 4), num(p["torch_s"], 0)])
        rows.append(["", "Keras", "CPU", "huấn luyện", num(p["keras_acc"], 4), num(p["keras_s"], 0)])
    f = su["numpy_forward"]
    rows.append("\\midrule")
    rows.append([SHORT["svhn"], "NumPy", "CPU", f"suy luận {num(f['n_img'], 0)} ảnh", num(f["acc_numpy"], 4),
                 num(f["numpy_net_ms_per_img"], 1) + " ms/ảnh"])
    rows.append(["", "PyTorch", "CPU", f"suy luận {num(f['n_img'], 0)} ảnh", num(f["acc_torch"], 4),
                 num(f["torch_cpu_ms_per_img"], 1) + " ms/ảnh"])
    short = su.get("short") or {}
    if short:
        rows.append("\\midrule")
        names = {"platform_numpy": ("NumPy", "CPU"), "platform_pytorch": ("PyTorch", "GPU")}
        for k, r in short.items():
            rows.append([SHORT["svhn"], *names[k], f"mạng hai tầng, {r['epochs']} epoch", num(r["acc"], 4),
                         num(r["sec_epoch"], 1) + " s/epoch"])
    table("ch3_platform", "llllrr",
          ["Bộ dữ liệu", "Nền tảng", "Thiết bị", "Việc", "Độ đúng", "Thời gian (s)"], rows)


def verbatim_block(name, lines):
    """Khối đầu ra thật để \\input trong hộp ketqua: mỗi dòng log một dòng in, ký tự đặc biệt được thoát."""
    body = "\\\\\n".join(tex_escape(s) for s in lines)
    (GEN / f"{name}.tex").write_text(body + "\n", encoding="utf-8", newline="\n")


def logs():
    lines = (LOGS / "ch3_train_torch.log").read_text(encoding="utf-8").splitlines()
    verbatim_block("ch3_log_runs", [s[3:] for s in lines if s.startswith("== pathmnist")])


def main():
    t_overview(), t_compare(), t_platform(), logs()


if __name__ == "__main__":
    main()
