"""Web demo Streamlit cho Bài tập Agent 02 — RL agent và rule-based agent trong hệ đa tác tử.

    .venv\\Scripts\\python.exe -m streamlit run "src/Agent 02/web/app.py" --server.port 8502

Mọi thuật toán gọi thẳng từ gói ag2/ (không viết lại). IntentNet chạy trên CPU.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

SRC = Path(__file__).resolve().parents[1]          # src/Agent 02
REPO = SRC.parents[1]
REPORT = REPO / "Report" / "Agent 02"
sys.path.insert(0, str(SRC))

from ag2.agents import QLearningAgent, RuleBasedAgent  # noqa: E402
from ag2.exact import exact_value, policy_of  # noqa: E402
from ag2.market import MarketConfig  # noqa: E402
from ag2.training import evaluate, train  # noqa: E402

METRICS = SRC / "outputs" / "metrics"
LOGS = SRC / "outputs" / "logs"
PDF = REPORT / "agent02_CT_07_nghiand.pdf"

st.set_page_config(page_title="Agent 02 — RL vs Luật & MAS mua sắm", layout="wide")


def nlines(p):
    try:
        return sum(1 for _ in open(p, encoding="utf-8", errors="ignore"))
    except OSError:
        return 0


def load_metric(name):
    p = METRICS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def pdf_page_count(path):
    try:
        import pymupdf
    except ImportError:
        return 0
    with pymupdf.open(path) as doc:
        return doc.page_count


@st.cache_data(show_spinner=False)
def render_pdf_page(path, index, dpi):
    """Ảnh PNG của một trang PDF (PyMuPDF) để hiện hình/báo cáo ngay trong trang."""
    import pymupdf
    with pymupdf.open(path) as doc:
        return doc[index].get_pixmap(dpi=dpi).tobytes("png")


ROLES = {
    "ag2/market.py": "Mô hình thị trường TMĐT hai cửa hàng (nguồn duy nhất của mô hình)",
    "ag2/agents.py": "QLearningAgent, RuleBasedAgent, đối thủ cố định",
    "ag2/training.py": "run_episode, train, evaluate",
    "ag2/exact.py": "Giá trị chính xác, chính sách tối ưu (quy hoạch động)",
    "ag2/model_agent.py": "Tác tử dựa trên mô hình (câu 10)",
    "ag2/common.py": "Đường dẫn, ghi/đọc outputs/metrics/*.json",
    "ag2/plot_style.py": "Kiểu hình matplotlib cho báo cáo",
    "ag2/texfmt.py": "Định dạng số, sinh bảng LaTeX",
    "ag2/shop_data.py": "Ch.8: catalog/tồn kho/câu truy vấn tổng hợp từ seed",
    "ag2/shop_intent.py": "Ch.8: IntentNet (PyTorch) phân loại ý định",
    "ag2/shop_agents.py": "Ch.8: Message + 4 tác tử chuyên trách",
    "ag2/shop_mas.py": "Ch.8: Coordinator và pipeline đơn khối để so sánh",
    "scripts/run_all.py": "Chạy thí nghiệm A–H + câu 10 song song -> outputs/raw",
    "scripts/summarize.py": "outputs/raw -> outputs/metrics/*.json",
    "scripts/figures.py": "Vẽ hình -> Report/Agent 02/figures/*.pdf",
    "scripts/tables.py": "Sinh bảng -> Report/Agent 02/generated/*.tex",
    "scripts/ch8_run.py": "Ch.8: huấn luyện IntentNet, demo, mô phỏng lỗi kho -> ch8_shop.json",
    "scripts/ch8_figures.py": "Ch.8: vẽ hình",
    "scripts/ch8_tables.py": "Ch.8: sinh bảng",
    "scripts/ch8_tests.py": "Ch.8: 8 unit test cho hợp đồng thông điệp",
}

tabs = st.tabs(["Tổng quan quy trình", "Mã nguồn", "Demo RL vs Luật", "Hệ đa tác tử mua sắm (Ch.8)",
                "Kết quả trong báo cáo"])

# ---------------------------------------------------------------- 1. Tổng quan
with tabs[0]:
    st.title("Bài tập Agent 02 — RL agent và rule-based agent trong hệ đa tác tử")
    st.caption("Nhóm 07 · Web demo trình bày toàn bộ quá trình từ mã nguồn đến báo cáo PDF")
    st.info("Dữ liệu là **mô phỏng / tổng hợp theo seed**: thị trường hai cửa hàng sinh ngẫu nhiên có seed "
            "(Binomial), catalog, tồn kho và câu truy vấn Chương 8 sinh tất định từ SHOP_SEED = 707. "
            "Không dùng dataset ngoài.")
    st.subheader("Pipeline")
    st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Arial"];
 ag2 [label="ag2/\\n(mô hình, tác tử)"]; run [label="scripts/run_all.py\\nch8_run.py"];
 raw [label="outputs/raw\\n*.json, *.npz"]; sum [label="summarize.py"]; met [label="outputs/metrics\\n*.json"];
 fig [label="figures.py / tables.py\\nch8_figures / ch8_tables"]; gen [label="Report/generated\\n+ figures/*.pdf"];
 build [label="build.py\\n(lualatex)"]; pdf [label="agent02_CT_07_nghiand.pdf", fillcolor="#dff0d8"];
 ag2 -> run -> raw -> sum -> met -> fig -> gen -> build -> pdf; met -> build [label="macro số liệu"]; }
""")
    rows = []
    for rel, role in ROLES.items():
        p = SRC / rel
        rows.append({"Tệp": f"src/Agent 02/{rel}", "Vai trò": role, "Số dòng": nlines(p) if p.exists() else "—"})
    for rel, role in [("build.py", "Sinh macro số liệu, trích mã, gọi latexmk -> PDF"),
                      ("main.tex", "Tệp LaTeX gốc của báo cáo")]:
        rows.append({"Tệp": f"Report/Agent 02/{rel}", "Vai trò": role, "Số dòng": nlines(REPORT / rel)})
    st.dataframe(rows, width="stretch", hide_index=True)
    st.subheader("Lệnh dựng lại từ đầu")
    st.code(r"""cd "src/Agent 02"
python scripts/run_all.py        # thí nghiệm A–H + câu 10 -> outputs/raw
python scripts/summarize.py      # -> outputs/metrics/*.json
python scripts/figures.py && python scripts/tables.py
python scripts/ch8_run.py && python scripts/ch8_figures.py && python scripts/ch8_tables.py
cd "../../Report/Agent 02"
python build.py                  # -> agent02_CT_07_nghiand.pdf""", language="bash")

# ---------------------------------------------------------------- 2. Mã nguồn
with tabs[1]:
    files = sorted([p for d in ("ag2", "scripts") for p in (SRC / d).glob("*.py")])
    names = [str(p.relative_to(SRC)).replace("\\", "/") for p in files]
    pick = st.selectbox("Chọn tệp", names, index=names.index("ag2/agents.py") if "ag2/agents.py" in names else 0)
    st.caption(ROLES.get(pick, "") + f" · {nlines(SRC / pick)} dòng")
    st.code((SRC / pick).read_text(encoding="utf-8"), language="python", line_numbers=True)

# ---------------------------------------------------------------- 3. Demo RL vs Luật
STATE_LABEL = {(0, 0): "Cầu thấp, đối thủ giữ giá", (0, 1): "Cầu thấp, đối thủ giảm giá",
               (1, 0): "Cầu cao, đối thủ giữ giá", (1, 1): "Cầu cao, đối thủ giảm giá"}
ACT = {0: "Giữ giá", 1: "Giảm giá"}


@st.cache_data(show_spinner=False)
def run_demo(episodes, alpha, gamma, eps, eval_eps, seed):
    cfg = MarketConfig()
    q = QLearningAgent(alpha=alpha, gamma=gamma, eps_start=eps, seed=seed + 10_000)
    hist = train(cfg, [q, RuleBasedAgent()], episodes=episodes, seed=seed)
    ev_q = evaluate(cfg, [q, RuleBasedAgent()], episodes=eval_eps, seed=seed + 500_000)
    ev_r = evaluate(cfg, [RuleBasedAgent(), RuleBasedAgent()], episodes=eval_eps, seed=seed + 500_000)
    ex_q = exact_value(cfg, policy_of(q, 0), policy_of(RuleBasedAgent(), 1))
    ex_r = exact_value(cfg, policy_of(RuleBasedAgent(), 0), policy_of(RuleBasedAgent(), 1))
    table = [{"Trạng thái (d, a_đối thủ(t−1))": f"{s} — {STATE_LABEL[s]}", "Q(giữ)": round(float(q.q[s][0]), 2),
              "Q(giảm)": round(float(q.q[s][1]), 2), "Hành động học được": ACT[int(np.argmax(q.q[s]))],
              "Luật đề bài": ACT[RuleBasedAgent().act(s)], "Số lần thăm": int(q.visits[s].sum())}
             for s in q.states]
    return {"reward": hist["reward"], "discount": hist["discount"], "epsilon": hist["epsilon"],
            "policy": q.policy_string(), "n_policies": len(set(hist["policy"])), "ev_q": {k: ev_q[k] for k in
            ("total", "mean", "discount_rate")}, "ev_r": {k: ev_r[k] for k in ("total", "mean", "discount_rate")},
            "ex_q": ex_q, "ex_r": ex_r, "table": table}


with tabs[2]:
    st.header("Q-learning (cửa hàng 1) đấu RuleBasedAgent (cửa hàng 2)")
    st.markdown("Môi trường `ECommerceMarket`: mỗi bước hai cửa hàng chọn **giữ giá / giảm giá 20%**; "
                "p_i = clip(0,30 + 0,25d + 0,15a_i − 0,10a_−i); S_i ~ Binomial(N_d, p_i); "
                "r_i = S_i[P(1−δa_i) − C]. Episode 20 bước. Mã gọi thẳng `ag2.training.train/evaluate`.")
    c1, c2, c3 = st.columns(3)
    episodes = c1.slider("Số episode huấn luyện", 50, 3000, 300, 50)
    eval_eps = c1.slider("Số episode đánh giá", 50, 1000, 200, 50)
    alpha = c2.slider("α (tốc độ học)", 0.01, 1.0, 0.10, 0.01)
    gamma = c2.slider("γ (chiết khấu)", 0.0, 0.99, 0.90, 0.01)
    eps = c3.slider("ε ban đầu (giảm ×0,995 mỗi episode, tối thiểu 0,02)", 0.0, 1.0, 0.20, 0.01)
    seed = c3.number_input("Seed", 0, 99999, 2026)
    if st.button("Huấn luyện và đánh giá", type="primary", key="train_btn") or "demo_res" not in st.session_state:
        with st.spinner("Đang huấn luyện..."):
            st.session_state["demo_res"] = run_demo(int(episodes), float(alpha), float(gamma), float(eps),
                                                    int(eval_eps), int(seed))
    R = st.session_state["demo_res"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lợi nhuận TB/bước — Q-learning", f"{R['ev_q']['mean'][0]:.2f}",
              f"{R['ev_q']['mean'][0] - R['ev_r']['mean'][0]:+.2f} so với Luật")
    m2.metric("Lợi nhuận TB/bước — đối thủ (Luật)", f"{R['ev_q']['mean'][1]:.2f}")
    m3.metric("Giá trị chính xác — Q / Luật", f"{R['ex_q'][0]:.2f} / {R['ex_r'][0]:.2f}")
    m4.metric("Chính sách học được", R["policy"], f"{R['n_policies']} chính sách khác nhau khi học",
              delta_color="off")

    w = max(1, min(50, len(R["reward"]) // 10))
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for i, lab in enumerate(("Q-learning", "Luật")):
        y = R["reward"][:, i]
        ax[0].plot(y, alpha=0.25, lw=0.8, color=f"C{i}")
        ax[0].plot(np.arange(w - 1, len(y)), np.convolve(y, np.ones(w) / w, "valid"), color=f"C{i}", label=lab)
    ax[0].set(title=f"Phần thưởng TB/bước theo episode (trung bình trượt {w})", xlabel="Episode",
              ylabel="Lợi nhuận/bước")
    ax[0].legend()
    ax[1].plot(R["discount"][:, 0], alpha=0.3, color="C0", label="Tỉ lệ giảm giá (Q)")
    ax[1].plot(R["epsilon"], color="C3", label="ε")
    ax[1].set(title="Tỉ lệ giảm giá của Q-learning và ε", xlabel="Episode")
    ax[1].legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("So sánh lợi nhuận khi đánh giá (bảng Q đóng băng, explore=False)")
    st.dataframe([
        {"Cặp": "Q-learning (cửa hàng 1) vs Luật (cửa hàng 2)", "LN TB/bước CH1": round(R["ev_q"]["mean"][0], 3),
         "LN TB/bước CH2": round(R["ev_q"]["mean"][1], 3), "Tổng LN CH1": round(R["ev_q"]["total"][0], 1),
         "Tổng LN CH2": round(R["ev_q"]["total"][1], 1), "Tỉ lệ giảm giá CH1": round(R["ev_q"]["discount_rate"][0], 3)},
        {"Cặp": "Luật vs Luật (đường cơ sở)", "LN TB/bước CH1": round(R["ev_r"]["mean"][0], 3),
         "LN TB/bước CH2": round(R["ev_r"]["mean"][1], 3), "Tổng LN CH1": round(R["ev_r"]["total"][0], 1),
         "Tổng LN CH2": round(R["ev_r"]["total"][1], 1), "Tỉ lệ giảm giá CH1": round(R["ev_r"]["discount_rate"][0], 3)},
    ], width="stretch", hide_index=True)
    st.subheader("Bảng Q và chính sách học được")
    st.dataframe(R["table"], width="stretch", hide_index=True)
    st.caption("Chuỗi chính sách theo thứ tự trạng thái (0,0),(0,1),(1,0),(1,1); '0000' = không bao giờ giảm giá, "
               "là chính sách tối ưu với cấu hình đề bài (giá trị chính xác 35,0 trong báo cáo).")

# ---------------------------------------------------------------- 4. MAS mua sắm
@st.cache_resource(show_spinner=False)
def load_shop():
    import torch
    from ag2.shop_data import build_catalog, build_stock
    from ag2.shop_intent import IntentNet, predict_proba
    torch.set_num_threads(2)
    model = IntentNet()
    model.load_state_dict(torch.load(SRC / "models" / "ch8" / "intent_net.pt", map_location="cpu"))
    model.eval()
    shop = load_metric("ch8_shop") or {}
    tau = float(shop.get("train", {}).get("tau", 0.9))
    catalog = build_catalog()
    now, old = build_stock(catalog)
    return model, predict_proba, tau, catalog, now, old


with tabs[3]:
    st.header("Hệ đa tác tử mua sắm — Coordinator + 4 tác tử")
    st.markdown("Coordinator gửi thông điệp lần lượt **Intent → Catalog → Recommend → Inventory**, kiểm hợp đồng "
                "từng phản hồi, thử lại tối đa 2 lần, không bao giờ khẳng định còn hàng khi chưa xác minh tồn kho. "
                "IntentNet (PyTorch) chạy trên **CPU**.")
    from ag2.shop_agents import CatalogAgent, IntentAgent, InventoryAgent, RecommendationAgent
    from ag2.shop_data import INTENTS
    from ag2.shop_mas import Coordinator
    model, predict_proba, tau, catalog, now, old = load_shop()
    examples = ["Tìm giúp mình laptop dưới 20 triệu", "so sanh tai nghe Sony voi JBL", "giày Adidas còn hàng không ạ",
                "điện thoại Samsung giá bao nhiêu", "Xin chào shop", "thời tiết hôm nay thế nào"]
    ex = st.selectbox("Câu mẫu", examples)
    query = st.text_input("Câu hỏi của khách", value=ex)
    c1, c2, c3 = st.columns(3)
    fail = c1.slider("Tỉ lệ lỗi dịch vụ kho (timeout / hỏng / dữ liệu cũ)", 0.0, 1.0, 0.0, 0.05)
    thr = c2.slider("Ngưỡng tin cậy ý định τ", 0.3, 0.95, tau, 0.05)
    mseed = c3.number_input("Seed mô phỏng", 0, 9999, 1)
    if st.button("Gửi cho Coordinator", type="primary", key="mas_btn") or "mas_res" not in st.session_state:
        intent_agent = IntentAgent(model, predict_proba, thr)
        inv = InventoryAgent(now, old, fail_rate=fail, seed=int(mseed))
        coord = Coordinator([intent_agent, CatalogAgent(catalog), RecommendationAgent(catalog), inv],
                            seed=int(mseed), verbose=True)
        res = coord.handle(query)
        proba = predict_proba(model, [query])[0].numpy()
        st.session_state["mas_res"] = (query, res, list(coord.log), proba, dict(inv.faults))
    q, res, log, proba, faults = st.session_state["mas_res"]
    st.write(f"**Câu:** {q}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ý định", res["intent"])
    m2.metric("Trạng thái", res["status"])
    m3.metric("Số thông điệp / thử lại", f"{res['messages']} / {res['retries']}")
    m4.metric("Độ trễ (đồng hồ ảo)", f"{res['latency_ms']:.1f} ms")
    left, right = st.columns([1, 2])
    with left:
        st.caption("Xác suất từng ý định (IntentNet)")
        fig, ax = plt.subplots(figsize=(4, 2.8))
        ax.barh(INTENTS, proba, color=["C2" if p == proba.max() else "C0" for p in proba])
        ax.axvline(thr, color="C3", ls="--", label=f"τ = {thr:.2f}")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.caption("Nhật ký thông điệp giữa các tác tử")
        st.code("\n".join(log) or "(không có thông điệp)", language="text")
        if res["events"]:
            st.warning("Sự kiện lỗi: " + "; ".join(res["events"]))
    by_id = {p["id"]: p for p in catalog}
    if res["items"]:
        st.subheader("Sản phẩm đề xuất")
        st.dataframe([{"ID": i, "Tên": by_id[i]["name"], "Thương hiệu": by_id[i]["brand"],
                       "Giá (triệu)": by_id[i]["price"], "KM %": by_id[i]["promo"],
                       "Giá sau KM": round(by_id[i]["price"] * (1 - by_id[i]["promo"] / 100), 2),
                       "Đánh giá": by_id[i]["rating"], "Tồn kho": res["stock"].get(i, "—")}
                      for i in res["items"]], width="stretch", hide_index=True)
    else:
        st.info({"chao": "Chào khách — không cần tra cứu.", "chuyen_nhan_vien": "Ngoài phạm vi — chuyển nhân viên.",
                 "can_lam_ro": "Chưa đủ tin cậy / không nhận ra loại hàng — hỏi lại khách.",
                 "khong_co": "Không có sản phẩm phù hợp.",
                 "chua_xac_minh": "Kho lỗi sau 2 lần thử lại — không khẳng định còn hàng."}.get(res["status"],
                                                                                              res["status"]))
    st.caption("Catalog 60 mặt hàng, tồn kho và câu truy vấn đều tổng hợp từ seed 707 (ag2/shop_data.py).")

# ---------------------------------------------------------------- 5. Kết quả
with tabs[4]:
    st.header("Số liệu chính trong báo cáo (outputs/metrics/*.json)")
    a, run, c, q10, s8 = (load_metric(n) for n in ("ch6_a", "ch5_run", "ch6_c", "ch7_q10", "ch8_shop"))
    k = st.columns(4)
    if a:
        k[0].metric("TN A: LN TB/bước Q vs Luật", f"{a['eval_mean']:.4f}", f"chênh {a['diff']:+.4f}")
        k[1].metric("Chính sách học / giá trị chính xác", a["policy"], f"{a['exact']:.1f}", delta_color="off")
    if run:
        k[2].metric("Số lượt chạy / thời gian", f"{run['runs']} lượt", f"{run['wall_s']} s, {run['workers']} tiến trình",
                    delta_color="off")
    if s8:
        k[3].metric("Ch.8: test acc IntentNet", f"{s8['test']['accuracy']:.3f}",
                    f"{s8['tests']['run']} test, {s8['tests']['failed']} lỗi", delta_color="off")

    figs = st.columns(2)
    if c:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        keys = list(c)
        ax.bar(keys, [c[x]["eval_mean"] for x in keys], yerr=[c[x]["eval_sd"] for x in keys], capsize=4,
               label="Q-learning")
        ax.axhline(c[keys[0]]["exact_rule"], color="C3", ls="--", label="Luật (chính xác)")
        ax.set(title="TN C: ảnh hưởng của α (10 seed)", xlabel="α", ylabel="LN TB/bước", ylim=(25, 37))
        ax.legend()
        fig.tight_layout()
        figs[0].pyplot(fig)
        plt.close(fig)
    if q10:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        opp = list(q10)
        kinds = ["q", "q_ext", "model"]
        x = np.arange(len(opp))
        for j, kd in enumerate(kinds):
            ax.bar(x + (j - 1) * 0.27, [q10[o][kd]["regret"] for o in opp], 0.27, label=kd)
        ax.set_xticks(x, opp, rotation=15)
        ax.set(title="Câu 10: regret so với tối ưu theo đối thủ", ylabel="Regret/bước")
        ax.legend()
        fig.tight_layout()
        figs[1].pyplot(fig)
        plt.close(fig)
    if s8:
        figs2 = st.columns(2)
        h = s8["train"]["history"]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot([e["train_acc"] for e in h], label="train acc")
        ax.plot([e["val_acc"] for e in h], label="val acc")
        ax.set(title="Ch.8: đường học IntentNet", xlabel="Epoch")
        ax.legend()
        fig.tight_layout()
        figs2[0].pyplot(fig)
        plt.close(fig)
        fr = s8["faults"]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        rates = [fr[r]["rate"] for r in fr]
        ax.plot(rates, [fr[r]["mas"]["completion"] for r in fr], "o-", label="MAS (Coordinator)")
        ax.plot(rates, [fr[r]["pipe"]["completion"] for r in fr], "s-", label="Pipeline đơn khối")
        ax.set(title="Ch.8: tỉ lệ hoàn thành theo tỉ lệ lỗi kho", xlabel="Tỉ lệ lỗi", ylabel="Hoàn thành")
        ax.legend()
        fig.tight_layout()
        figs2[1].pyplot(fig)
        plt.close(fig)
        st.dataframe([{"Tỉ lệ lỗi": fr[r]["rate"], "Hệ": s, "Hoàn thành": round(fr[r][s]["completion"], 4),
                       "Khẳng định sai": fr[r][s]["unsafe"], "Chưa xác minh": fr[r][s]["unverified"],
                       "Trễ TB (ms)": round(fr[r][s]["lat_mean"], 2)} for r in fr for s in ("mas", "pipe")],
                     width="stretch", hide_index=True)
    st.caption("Các biểu đồ trên vẽ lại từ JSON. Hình gốc của báo cáo (PDF) liệt kê bên dưới.")
    with st.expander("Hình gốc trong Report/Agent 02/figures", expanded=True):
        fig_files = sorted((REPORT / "figures").glob("*.pdf"))
        if fig_files:
            fname = st.selectbox("Hình", [p.name for p in fig_files])
            fp = REPORT / "figures" / fname
            if pdf_page_count(str(fp)):
                st.image(render_pdf_page(str(fp), 0, 150), width=760)
            st.download_button(f"Tải {fname}", fp.read_bytes(), file_name=fname, mime="application/pdf")
    with st.expander("Mọi tệp metrics (JSON)"):
        pick = st.selectbox("Tệp", sorted(p.name for p in METRICS.glob("*.json")))
        st.json(json.loads((METRICS / pick).read_text(encoding="utf-8")), expanded=False)
    st.subheader("Đuôi nhật ký chạy")
    for name in ("run_all.log", "ch8_run.log"):
        p = LOGS / name
        if p.exists():
            st.caption(f"outputs/logs/{name}")
            st.code("\n".join(p.read_text(encoding="utf-8", errors="ignore").splitlines()[-15:]), language="text")
    st.subheader("Báo cáo PDF")
    if PDF.exists():
        st.download_button("Tải báo cáo agent02_CT_07_nghiand.pdf", PDF.read_bytes(), file_name=PDF.name,
                           mime="application/pdf", type="primary")
        n_pages = pdf_page_count(str(PDF))
        if n_pages:
            page = st.number_input(f"Xem trang (1–{n_pages})", 1, n_pages, 1)
            st.image(render_pdf_page(str(PDF), int(page) - 1, 110), width=820)
        else:
            st.caption("Không dựng được ảnh trang PDF (thiếu PyMuPDF); dùng nút tải ở trên.")
    else:
        st.error("Chưa có PDF báo cáo — chạy Report/Agent 02/build.py.")
