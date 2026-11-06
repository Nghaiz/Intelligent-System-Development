"""Web demo Streamlit: trình bày quá trình làm báo cáo Bài tập Agent 01 (AI Agent + Q-Learning).

Chạy:  python -m streamlit run "src/Agent 01/web/app.py" --server.port 8501
App import lại mã thật trong ag1/ — không viết lại thuật toán.
"""
import base64
import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

AG_ROOT = Path(__file__).resolve().parents[1]          # src/Agent 01
REPO = AG_ROOT.parents[1]                               # repo gốc
REPORT = REPO / "Report" / "Agent 01"
WEB_DIR = Path(__file__).resolve().parent
for _p in (AG_ROOT, WEB_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ag1 import qlearning as ql  # noqa: E402
from ag1.maze import ACTIONS, LAYOUT, R_FIRE, R_GOAL, R_STEP, MazeEnv  # noqa: E402
from ag1.qlearning import QLearningAgent, greedy_path, train  # noqa: E402
from ag1.viz import draw_maze, draw_rewards, q_table_excerpt  # noqa: E402
from catmouse_tab import tab_catmouse  # noqa: E402

METRICS_FILE = AG_ROOT / "outputs" / "metrics" / "ch8_qlearning.json"
PDF_FILE = REPORT / "agent01_CT_07_nghiand.pdf"
NB_IMG_DIR = REPORT / "assets" / "nb"

# (đường dẫn tương đối từ repo, vai trò)
PIPELINE_FILES = [
    ("src/Agent 01/ag1/maze.py", "Môi trường mê cung 12x12 (MDP: S, A, P, R) + BFS kiểm chứng"),
    ("src/Agent 01/ag1/qlearning.py", "QLearningAgent (epsilon-greedy, cập nhật Bellman), train, greedy_path"),
    ("src/Agent 01/ag1/viz.py", "Vẽ mê cung + lộ trình, đường cong phần thưởng, trích Q-Table"),
    ("src/Agent 01/ag1/common.py", "Đường dẫn dùng chung, ghi/đọc JSON số liệu"),
    ("src/Agent 01/ag1/plot_style.py", "Kiểu vẽ matplotlib cho hình vector của báo cáo"),
    ("src/Agent 01/ag1/texfmt.py", "Định dạng số/bảng sang LaTeX"),
    ("src/Agent 01/scripts/make_notebook.py", "Sinh + chạy notebook từ mã ag1/ (ô mã = nguyên văn mã nguồn)"),
    ("src/Agent 01/notebooks/agent01_rl_qlearning.ipynb", "Notebook huấn luyện; ghi outputs/metrics/ch8_qlearning.json"),
    ("src/Agent 01/outputs/metrics/ch8_qlearning.json", "Nguồn duy nhất của mọi con số trong Chương 8"),
    ("src/Agent 01/scripts/report_assets.py", "Sinh hình vector figures/ch8_*.pdf và bảng generated/ch8_*.tex"),
    ("src/Agent 01/scripts/screenshots.py", "Chụp notebook trong JupyterLab -> assets/nb/*.png"),
    ("Report/Agent 01/generated/ch8_qtable.tex", "Bảng trích Q-Table (sinh tự động)"),
    ("Report/Agent 01/generated/ch8_log.tex", "Nhật ký huấn luyện (sinh tự động)"),
    ("Report/Agent 01/generated/so_lieu.tex", "Macro số liệu \\SL{...} (sinh tự động)"),
    ("Report/Agent 01/figures/ch8_route.pdf", "Hình lộ trình tối ưu trên mê cung"),
    ("Report/Agent 01/figures/ch8_rewards.pdf", "Hình đường cong hội tụ phần thưởng"),
    ("Report/Agent 01/main.tex", "Tệp LaTeX chính, gộp chapters/ch0..ch10"),
    ("Report/Agent 01/build.py", "Biên dịch LaTeX -> PDF"),
    ("Report/Agent 01/agent01_CT_07_nghiand.pdf", "Báo cáo cuối cùng (PDF)"),
]

REBUILD_CMDS = """cd "src/Agent 01"
python scripts/make_notebook.py      # sinh + chạy notebook, ~2 s huấn luyện trên CPU -> outputs/metrics/ch8_qlearning.json
python scripts/report_assets.py      # figures/ch8_*.pdf + generated/ch8_*.tex
# JupyterLab ở cổng 8899 trong notebooks/ (không token), rồi:
python scripts/screenshots.py        # assets/nb/*.png
cd "../../Report/Agent 01" && python build.py   # -> agent01_CT_07_nghiand.pdf"""

PIPELINE_DOT = """digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EFF6FF", color="#2563EB", fontname="Arial"];
  code [label="Mã nguồn ag1/\\nmaze.py · qlearning.py · viz.py"];
  nb [label="make_notebook.py\\n-> notebook .ipynb\\n(chạy huấn luyện)"];
  js [label="outputs/metrics/\\nch8_qlearning.json"];
  ra [label="report_assets.py"];
  gen [label="generated/ch8_*.tex\\nfigures/ch8_*.pdf"];
  sc [label="screenshots.py\\n-> assets/nb/*.png"];
  tex [label="main.tex + chapters/\\nbuild.py (LaTeX)"];
  pdf [label="agent01_CT_07_nghiand.pdf", fillcolor="#DCFCE7", color="#16A34A"];
  code -> nb -> js -> ra -> gen -> tex -> pdf;
  nb -> sc -> tex;
}"""


def count_lines(path):
    try:
        if path.suffix == ".ipynb":
            nb = json.loads(path.read_text(encoding="utf-8"))
            return sum(len(c.get("source", [])) if isinstance(c.get("source"), list)
                       else c.get("source", "").count("\n") + 1 for c in nb.get("cells", []))
        if path.suffix == ".pdf":
            return None
        return len(path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return None


def file_table():
    rows = []
    for rel, role in PIPELINE_FILES:
        p = REPO / rel
        n = count_lines(p) if p.exists() else None
        rows.append({"Tệp": rel, "Vai trò": role, "Tồn tại": "có" if p.exists() else "KHÔNG",
                     "Số dòng": "" if n is None else str(n)})
    return rows


def run_training(alpha, gamma, eps_start, episodes, seed):
    """Huấn luyện bằng đúng mã ag1 (train, greedy_path); trả về env, agent, lịch sử, lộ trình, log."""
    env = MazeEnv()
    agent = QLearningAgent(env.n_states, env.n_actions, alpha=alpha, gamma=gamma, epsilon=eps_start, seed=seed)
    buf = io.StringIO()
    log_every = max(1, episodes // 10)
    with contextlib.redirect_stdout(buf):
        hist = train(env, agent, episodes=episodes, log_every=log_every)
    path, total = greedy_path(env, agent)
    return env, agent, hist, path, total, buf.getvalue()


def fig_maze(env, path):
    fig, ax = plt.subplots(figsize=(6.2, 6.6))
    draw_maze(ax, env, path)
    fig.tight_layout()
    return fig


def fig_rewards(hist):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    draw_rewards(ax, hist, window=max(1, min(100, len(hist["reward"]) // 5)))
    fig.tight_layout()
    return fig


def tab_overview():
    st.subheader("Quy trình từ mã nguồn đến báo cáo PDF")
    st.info("**Không có dataset ngoài.** Dữ liệu của bài là môi trường mê cung 12x12 tự định nghĩa trong "
            "`ag1/maze.py` (hằng `LAYOUT`). Tác tử tự sinh kinh nghiệm bằng cách tương tác với môi trường "
            "qua hàng nghìn episode, nên không cần tải hay tiền xử lý dữ liệu nào.")
    st.graphviz_chart(PIPELINE_DOT)
    st.markdown("**Các tệp đã dùng**")
    st.dataframe(file_table(), width="stretch", hide_index=True)
    st.markdown("**Lệnh dựng lại toàn bộ** (từ thư mục gốc repo)")
    st.code(REBUILD_CMDS, language="bash")
    st.markdown("**Mô hình MDP của mê cung**")
    st.markdown(f"- Trạng thái: tọa độ ô (r, c), 144 trạng thái\n"
                f"- Hành động: {', '.join(a[0] for a in ACTIONS)}\n"
                f"- Phần thưởng: bước hợp lệ {R_STEP:+.0f}, đâm tường lửa/ra ngoài {R_FIRE:+.0f} (đứng yên), "
                f"về đích {R_GOAL:+.0f} (kết thúc)")
    st.code("\n".join(LAYOUT), language="text")


def tab_source():
    st.subheader("Mã nguồn")
    files = (sorted((AG_ROOT / "ag1").glob("*.py")) + sorted((AG_ROOT / "scripts").glob("*.py"))
             + sorted(WEB_DIR.glob("catmouse*.py")))
    labels = [str(f.relative_to(AG_ROOT)).replace("\\", "/") for f in files]
    choice = st.selectbox("Chọn tệp", labels, index=labels.index("ag1/qlearning.py") if "ag1/qlearning.py" in labels else 0)
    text = (AG_ROOT / choice).read_text(encoding="utf-8")
    st.caption(f"{choice} — {len(text.splitlines())} dòng")
    st.code(text, language="python", line_numbers=True)


def tab_demo():
    st.subheader("Demo Q-Learning trực tiếp")
    st.caption("Huấn luyện bằng đúng `QLearningAgent`, `train`, `greedy_path` trong `ag1/qlearning.py`. "
               f"Epsilon giảm theo hệ số {ql.EPS_DECAY} mỗi episode (tối thiểu {ql.EPS_MIN}); "
               f"giới hạn {ql.MAX_STEPS} bước/episode.")
    c1, c2, c3 = st.columns(3)
    alpha = c1.slider("alpha (tốc độ học)", 0.01, 1.0, float(ql.ALPHA), 0.01)
    gamma = c2.slider("gamma (chiết khấu)", 0.50, 0.99, float(ql.GAMMA), 0.01)
    eps = c3.slider("epsilon ban đầu", 0.0, 1.0, float(ql.EPS_START), 0.05)
    c4, c5 = st.columns(2)
    episodes = c4.slider("Số episode", 100, 5000, int(ql.EPISODES), 100)
    seed = c5.number_input("Seed", 0, 10_000, int(ql.SEED))
    if st.button("Huấn luyện", type="primary"):
        with st.spinner("Đang huấn luyện..."):
            st.session_state["demo"] = run_training(alpha, gamma, eps, episodes, int(seed))
    if "demo" not in st.session_state:
        st.write("Chỉnh siêu tham số rồi bấm **Huấn luyện** (mặc định = cấu hình trong báo cáo).")
        return
    env, agent, hist, path, total, log = st.session_state["demo"]
    bfs = env.shortest_path_length()
    reached = path[-1] == env.goal
    m = st.columns(4)
    m[0].metric("Thời gian", f"{hist['seconds']:.2f} s")
    m[1].metric("Về đích (tham lam)", "Có" if reached else "Không")
    m[2].metric("Số bước lộ trình", f"{len(path) - 1}", help=f"BFS ngắn nhất = {bfs}")
    m[3].metric("Phần thưởng lộ trình", f"{total:.0f}")
    if reached and len(path) - 1 == bfs:
        st.success(f"Lộ trình học được bằng đường ngắn nhất BFS ({bfs} bước).")
    elif not reached:
        st.warning("Chính sách tham lam chưa về đích — thử tăng số episode hoặc alpha.")
    a, b = st.columns([1, 1.3])
    with a:
        st.pyplot(fig_maze(env, path))
    with b:
        st.pyplot(fig_rewards(hist))
        last = slice(-min(100, len(hist["reward"])), None)
        st.write(f"100 episode cuối: thưởng TB **{np.mean(hist['reward'][last]):.2f}**, "
                 f"về đích **{100 * np.mean(hist['success'][last]):.0f}%**.")
    if len(path) >= 2:
        st.markdown("**Trích Q-Table** tại ô xuất phát, các điểm rẽ và ô trước đích")
        st.dataframe(q_table_excerpt(env, agent, path), width="stretch", hide_index=True)
    with st.expander("Nhật ký huấn luyện"):
        st.code(log or "(trống)", language="text")


def tab_results():
    st.subheader("Kết quả trong báo cáo")
    if METRICS_FILE.exists():
        data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        r = data.get("result", {})
        m = st.columns(4)
        m[0].metric("Lộ trình học được", f"{r.get('path_len')} bước", help=f"BFS = {r.get('bfs_len')}")
        m[1].metric("Phần thưởng tham lam", r.get("greedy_reward"))
        m[2].metric("Về đích 100 ep cuối", f"{100 * r.get('success_last100', 0):.0f}%")
        m[3].metric("Thời gian huấn luyện", f"{r.get('seconds', 0):.2f} s")
        c1, c2 = st.columns(2)
        c1.markdown("**Siêu tham số**")
        c1.json(data.get("params", {}))
        c2.markdown("**Kết quả**")
        c2.json(r)
        if data.get("qtable"):
            st.markdown("**Trích Q-Table (báo cáo)**")
            st.dataframe(data["qtable"], width="stretch", hide_index=True)
    else:
        st.error(f"Không thấy {METRICS_FILE}")
    imgs = sorted(NB_IMG_DIR.glob("*.png"))
    st.markdown(f"**Ảnh chụp notebook trong báo cáo** ({len(imgs)} ảnh, `Report/Agent 01/assets/nb/`)")
    if imgs:
        pick = st.multiselect("Chọn ảnh", [p.name for p in imgs],
                              default=[n for n in ("out_path.png", "out_plot.png") if (NB_IMG_DIR / n).exists()])
        for name in pick:
            st.image(str(NB_IMG_DIR / name), caption=name, width="stretch")
    st.markdown("**Báo cáo PDF**")
    if PDF_FILE.exists():
        pdf = PDF_FILE.read_bytes()
        st.download_button("Tải báo cáo PDF", pdf, file_name=PDF_FILE.name, mime="application/pdf")
        if st.checkbox("Nhúng PDF vào trang (một số trình duyệt chặn iframe base64, khi đó dùng nút tải)"):
            b64 = base64.b64encode(pdf).decode()
            st.markdown(f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="900"></iframe>',
                        unsafe_allow_html=True)
    else:
        st.error(f"Không thấy {PDF_FILE}")


def main():
    st.set_page_config(page_title="Agent 01 — AI Agent & Q-Learning", layout="wide")
    st.title("Bài tập Agent 01 — AI Agent và Q-Learning")
    st.caption("Nhóm 07 · Demo quy trình làm báo cáo, mã nguồn và huấn luyện Q-Learning trực tiếp")
    t1, t2, t3, t4, t5 = st.tabs(["Tổng quan quy trình", "Mã nguồn", "Demo Q-Learning trực tiếp",
                                  "Kết quả trong báo cáo", "🐱 Mèo bắt chuột"])
    with t1:
        tab_overview()
    with t2:
        tab_source()
    with t3:
        tab_demo()
    with t4:
        tab_results()
    with t5:
        tab_catmouse()


main()
