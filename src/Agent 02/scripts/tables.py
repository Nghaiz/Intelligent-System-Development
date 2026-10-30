"""Sinh bảng LaTeX của Ch.4–Ch.7 vào Report/Agent 02/generated/ từ outputs/metrics/*.json."""
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ag2.agents import QLearningAgent  # noqa: E402
from ag2.common import load_json  # noqa: E402
from ag2.market import MarketConfig  # noqa: E402
from ag2.texfmt import best, fmt, num, table  # noqa: E402
from run_all import BASE_SEED, EVAL_EPISODES, EPISODES, EVAL_SEED_OFFSET, SCHEDULES, SEEDS  # noqa: E402

STATE = {"00": "(0, 0)", "01": "(0, 1)", "10": "(1, 0)", "11": "(1, 1)"}


def pm(s):
    return f"{num(s['eval_mean'], 2)} $\\pm$ {num(s['eval_sd'], 2)}"


def ci(s):
    return f"[{num(s['ci_lo'], 2)}; {num(s['ci_hi'], 2)}]"


def t_ch4():
    m = load_json("ch4_model")
    rows = []
    for r20, r05 in zip(m["d20"]["rows"], m["d05"]["rows"]):
        rows.append([f"({r20['d']}, {r20['b']})", num(r20["p0"], 2), num(r20["p1"], 2), num(r20["keep"], 2),
                     num(r20["disc"], 2), "giảm" if r20["better"] else "giữ", num(r05["disc"], 2),
                     "giảm" if r05["better"] else "giữ"])
    table("ch4_reward", "crrrrlrl", ["(d, b)", "p giữ", "p giảm", "E[r] giữ", "E[r] giảm (δ = 20\\%)", "Tốt hơn",
                                       "E[r] giảm (δ = 5\\%)", "Tốt hơn"], rows,
          note=r"b là hành động hiện tại của đối thủ; $\mathbb{E}[r] = N_d\, p\,$ × biên lợi nhuận (output của scripts/summarize.py).")


def t_ch5():
    cfg, q = MarketConfig(), QLearningAgent()
    names = {"base_price": "Giá gốc P", "unit_cost": "Giá vốn C", "discount_rate": "Mức giảm δ",
             "customers_low": "Khách khi d = 0", "customers_high": "Khách khi d = 1",
             "base_conversion": "Chuyển đổi nền", "demand_effect": "Hiệu ứng nhu cầu",
             "own_discount_effect": "Hiệu ứng tự giảm giá", "rival_discount_effect": "Hiệu ứng đối thủ giảm giá",
             "steps": "Số bước mỗi episode", "demand_persist": "P(d giữ nguyên)"}
    rows = [[names[f.name], fmt(getattr(cfg, f.name)), f"\\texttt{{{f.name}}}".replace("_", "\\_")]
            for f in fields(cfg) if f.name in names]
    rows += [["Tốc độ học α", fmt(q.alpha), "\\texttt{alpha}"], ["Hệ số chiết khấu γ", fmt(q.gamma), "\\texttt{gamma}"],
             [r"$\varepsilon_0$ / $\varepsilon_{\min}$ / $\rho$", f"{fmt(q.eps_start)} / {fmt(q.eps_min)} / {fmt(q.eps_decay)}", "\\texttt{eps\\_*}"],
             ["Episode huấn luyện / đánh giá", f"{num(EPISODES, 0)} / {num(EVAL_EPISODES, 0)}", "\\texttt{run\\_all}"],
             ["Seed tham chiếu / đánh giá", f"{BASE_SEED} / {BASE_SEED + EVAL_SEED_OFFSET}", "\\texttt{BASE\\_SEED}"],
             ["Mười seed (C–H, câu 10)", ", ".join(map(str, SEEDS)), "\\texttt{SEEDS}"]]
    table("ch5_params", "llL{4.2cm}", ["Tham số", "Giá trị", "Trong mã"], rows)
    lab = {"khong": "Không khám phá", "thap": "Thấp", "co_so": "Cơ sở", "cao": "Cao"}
    table("ch5_schedules", "lrrr", ["Lịch", r"$\varepsilon_0$", r"$\varepsilon_{\min}$", r"$\rho$"],
          [[lab[k], fmt(v[0]), fmt(v[1]), fmt(v[2])] for k, v in SCHEDULES.items()])


def t_ch6():
    a = load_json("ch6_a")
    table("ch6_a_q", "crrrrc", ["s = (d, c)", "Q(s, giữ)", "Q(s, giảm)", "lượt giữ", "lượt giảm", "Tham lam"],
          [[STATE[s], num(v[0], 2), num(v[1], 2), num(a["visits"][s][0], 0), num(a["visits"][s][1], 0),
            "giữ" if v[0] >= v[1] else "giảm"] for s, v in a["q"].items()])
    head = ["Cấu hình", "Đánh giá TB \\(\\pm\\) SD", "Chính xác", "Tối ưu", "Giảm giá", "HL TB", "Đổi CS", "RL − luật (KTC 95\\%)"]

    def sweep(name, data, keys, labels):
        rows = [[lab, pm(data[k]), num(data[k]["exact_mean"], 2), f"{data[k]['n_opt']}/{data[k]['n']}",
                 num(data[k]["eval_disc"], 3), num(data[k]["train_mean"], 2), num(data[k].get("switches", 0), 0),
                 ci(data[k])] for k, lab in zip(keys, labels)]
        col = best([data[k]["exact_mean"] for k in keys], [r[2] for r in rows])
        for r, c in zip(rows, col):
            r[2] = c
        table(name, "lrrcrrrr", head, rows)
    sweep("ch6_c", load_json("ch6_c"), ["0.01", "0.10", "0.50"], ["α = 0,01", "α = 0,10", "α = 0,50"])
    sweep("ch6_d", load_json("ch6_d"), ["khong", "thap", "co_so", "cao"], ["ε = 0", "Thấp", "Cơ sở", "Cao"])
    f = load_json("ch6_f")
    sweep("ch6_f", f, sorted(f), [f"δ = {int(k)}\\%" for k in sorted(f)])
    sweep("ch6_g", load_json("ch6_g"), ["ben_vung", "doanh_thu"], ["Nhu cầu bền", "Thưởng = doanh thu"])
    e = load_json("ch6_e")
    rows = [[s["seed"], num(s["eval_mean"], 3), num(s["rule_mean"], 3), num(s["paired_diff"], 3, sign=True),
             num(s["exact"], 2), f"\\texttt{{{s['policy']}}}", num(s["eval_disc"], 3)] for s in e["seeds"]]
    g = e["agg"]
    rows += ["\\midrule", ["TB", pm(g), num(g["rule_mean"], 3), num(g["diff_mean"], 3, sign=True),
                           num(g["exact_mean"], 2), f"{g['n_opt']}/{g['n']} tối ưu", num(g["eval_disc"], 3)]]
    table("ch6_e", "rrrrrcr", ["Seed", "RL", "Luật", "RL − luật", "Chính xác", "Chính sách", "Giảm giá"], rows)
    h = load_json("ch6_h")
    table("ch6_h", "rcc", ["Seed", "Chính sách Q-learning 1", "Chính sách Q-learning 2"],
          [[s, f"\\texttt{{{p}}}", f"\\texttt{{{o}}}"] for s, p, o in zip(SEEDS, h["policies"], h["opp_policies"])])


def t_ch7():
    q = load_json("ch7_q10")
    lab = {"tham_chieu": "Tham chiếu (luật, δ = 20\\%)", "giam_5": "Luật, δ = 5\\%", "an_mieng": "Ăn miếng trả miếng",
           "luon_giam": "Luôn giảm giá", "ben_vung": "Nhu cầu bền (0,8)"}
    rows = []
    for sk, name in lab.items():
        r = q[sk]
        vals = [r[a]["exact_mean"] for a in ("q", "q_ext", "model")]
        cells = best(vals, [num(v, 2) for v in vals])
        rows.append([name, num(r["q"]["optimal"], 2), *cells, num(r["q"]["exact_rule"], 2),
                     *(f"{r[a]['n_opt']}/10" for a in ("q", "q_ext", "model"))])
    table("ch7_q10", "lrrrrrccc", ["Kịch bản", "Tối ưu", "Q (d,c)", "Q mở rộng", "Mô hình", "Luật",
                                   "Tối ưu Q", "Tối ưu Q+", "Tối ưu MH"], rows)


if __name__ == "__main__":
    t_ch4()
    t_ch5()
    t_ch6()
    t_ch7()
    print("đã sinh bảng")
