"""Ch.8: sinh bảng LaTeX vào generated/ch8_*.tex từ outputs/metrics/ch8_*.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlf.common import load_json  # noqa: E402
from tlf.texfmt import best, num, table  # noqa: E402


def pm(s):
    return f"{num(s['eval_mean'], 2)} $\\pm$ {num(s['eval_sd'], 2)}"


def ci(s):
    return f"[{num(s['ci_lo'], 2)}; {num(s['ci_hi'], 2)}]"


def t_reward():
    m = load_json("ch8_model")
    rows = []
    for r20, r05 in zip(m["d20"]["rows"], m["d05"]["rows"]):
        rows.append([f"({r20['d']}, {r20['b']})", num(r20["p0"], 2), num(r20["p1"], 2), num(r20["keep"], 2),
                     num(r20["disc"], 2), "giảm" if r20["better"] else "giữ", num(r05["disc"], 2),
                     "giảm" if r05["better"] else "giữ"])
    table("ch8_reward", "crrrrlrl", ["$(d,b)$", "$p$ giữ", "$p$ giảm", "$\\mathbb{E}[r]$ giữ",
                                     "giảm, $\\delta$ 20\\%", "Tốt hơn", "giảm, $\\delta$ 5\\%", "Tốt hơn"], rows)


def t_sweeps():
    c, d = load_json("ch8_c"), load_json("ch8_d")
    conf = [("C", "$\\alpha=0{,}01$", c["0.01"]), ("A, E", "$\\alpha=0{,}10$ (cơ sở)", c["0.10"]),
            ("C", "$\\alpha=0{,}50$", c["0.50"]), ("D", "$\\varepsilon=0$", d["khong"]),
            ("D", "lịch $\\varepsilon$ thấp", d["thap"]), ("D", "lịch $\\varepsilon$ cao", d["cao"])]
    rows = [[n, lab, pm(s), f"{s['n_opt']}/{s['n']}", num(s["switches"], 1), ci(s)] for n, lab, s in conf]
    table("ch8_sweeps", "llcrrc", ["TN", "Cấu hình", "Thưởng đánh giá", "Seed tối ưu", "Đổi CS", "KTC 95\\% RL $-$ luật"], rows)


def t_env():
    f, g = load_json("ch8_f"), load_json("ch8_g")
    conf = [(f"$\\delta={int(k)}\\%$", f[k]) for k in sorted(f)]
    conf += [("Nhu cầu bền", g["ben_vung"]), ("Thưởng = doanh thu", g["doanh_thu"])]
    rows = [[lab, pm(s), num(s["exact_rule"], 2), num(s["optimal"], 2), f"{num(100 * s['eval_disc'], 1)}\\%",
             f"{s['n_opt']}/{s['n']}", ci(s)] for lab, s in conf]
    table("ch8_env", "lcrrrrc", ["Biến thể", "RL đánh giá", "Luật", "Tối ưu", "Giảm giá", "Seed tối ưu", "KTC 95\\% RL $-$ luật"],
          rows)


def t_q10():
    q = load_json("ch8_q10")
    names = {"tham_chieu": "Tham chiếu", "giam_5": "$\\delta=5\\%$", "an_mieng": "Đối thủ ăn miếng trả miếng",
             "luon_giam": "Đối thủ luôn giảm giá", "ben_vung": "Nhu cầu bền"}
    rows = []
    for key, lab in names.items():
        s = q[key]
        vals = [s[a]["exact_mean"] for a in ("q", "q_ext", "model")]
        cells = best([round(v, 2) for v in vals], [num(v, 2) for v in vals])
        rows.append([lab, num(s["q"]["optimal"], 2), num(s["q"]["exact_rule"], 2)]
                    + [f"{cell} ({s[a]['n_opt']})" for cell, a in zip(cells, ("q", "q_ext", "model"))])
    table("ch8_q10", "lrrrrr", ["Kịch bản", "Tối ưu", "Luật", "Q-learning", "Q mở rộng", "Mô hình"], rows,
          note="Giá trị chính xác trung bình mười seed; trong ngoặc là số seed đạt mức tối ưu trên chính sách dừng.")


def main():
    t_reward()
    t_sweeps()
    t_env()
    t_q10()


if __name__ == "__main__":
    main()
