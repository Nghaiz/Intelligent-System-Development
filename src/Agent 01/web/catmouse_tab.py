"""Tab "Mèo bắt chuột" của web demo: chọn kịch bản, huấn luyện chuột Q-Learning, xem ván chơi, bản đồ nhiệt Q."""
import matplotlib.pyplot as plt
import streamlit as st

import catmouse as cm
import catmouse_viz as cv

POLICIES = {"Chuột đã học": "learned", "Chuột mới học (10% episode)": "snapshot", "Chuột ngẫu nhiên": "random"}
SPEEDS = {"Mỗi bước (cùng tốc độ chuột)": 1, "Cách một bước (chậm hơn chuột)": 2}


def _show(fig):
    st.pyplot(fig)
    plt.close(fig)


def _settings():
    c1, c2, c3 = st.columns([1.3, 1, 1.2])
    scenario = c1.selectbox("Kịch bản", list(cm.SCENARIOS), key="cm_scenario")
    mode_label = c2.selectbox("Mức khó của mèo", list(cm.CAT_MODES.values()), index=1, key="cm_mode")
    speed_label = c3.radio("Tốc độ mèo", list(SPEEDS), key="cm_speed")
    cfg = cm.SCENARIOS[scenario]
    mode = next(k for k, v in cm.CAT_MODES.items() if v == mode_label)
    a, b = st.columns([1, 1.6])
    with a:
        env = cm.CatMouseEnv(cfg["layout"], mode, SPEEDS[speed_label], cfg["max_steps"])
        st.image(cv.render_frame(env, env.m, env.c, env.mask, "Vị trí xuất phát"), width=min(420, 70 * env.n_cols))
    with b:
        st.markdown(f"**{scenario}** · {cfg['note']}")
        st.markdown(f"- Trạng thái = (ô chuột, ô mèo, phô mai còn lại): **{env.n_states:,}** trạng thái × 4 hành động\n"
                    f"- Phần thưởng: ăn phô mai **{cm.R_CHEESE:+.0f}**, ăn hết **{cm.R_WIN:+.0f}** (thắng), "
                    f"bị bắt **{cm.R_CAUGHT:+.0f}**, mỗi bước **{cm.R_STEP:+.1f}**, đâm tường **{cm.R_BUMP:+.0f}** "
                    f"(đứng yên); tối đa **{cfg['max_steps']}** bước/ván\n"
                    "- Mèo: *Ngẫu nhiên* đi bừa; *Đuổi tham lam* bước giảm khoảng cách Manhattan "
                    f"({100 * cm.GREEDY_NOISE:.0f}% đi bừa); *Săn BFS* đi đường ngắn nhất tới chuột. "
                    "Mèo không vào được hang 🕳.\n"
                    "- Ký hiệu: 🐭 chuột · 🐱 mèo · 🧀 phô mai · ô xám đậm = tường · ô nâu = hang")
        p1, p2, p3 = st.columns(3)
        alpha = p1.slider("alpha", 0.01, 1.0, 0.2, 0.01, key="cm_alpha")
        gamma = p2.slider("gamma", 0.50, 0.99, 0.95, 0.01, key="cm_gamma")
        eps = p3.slider("epsilon ban đầu", 0.0, 1.0, 1.0, 0.05, key="cm_eps")
        p4, p5 = st.columns([2, 1])
        episodes = p4.slider("Số episode", 200, 30000, int(cfg["episodes"]), 100, key=f"cm_ep_{scenario}")
        seed = p5.number_input("Seed", 0, 10_000, 42, key="cm_seed")
    return scenario, mode, SPEEDS[speed_label], alpha, gamma, eps, episodes, int(seed)


def _train(scenario, mode, every, alpha, gamma, eps, episodes, seed):
    bar = st.progress(0.0, text="Đang huấn luyện chuột...")
    res = cm.run_experiment(scenario, mode, every, alpha, gamma, eps, episodes, seed, n_eval=100,
                            progress=lambda f: bar.progress(min(1.0, f), text=f"Đang huấn luyện chuột... {100 * f:.0f}%"))
    bar.progress(1.0, text=f"Xong: {episodes} episode trong {res['hist']['seconds']:.2f} s")
    st.session_state["cm"] = res
    st.session_state["cm_game_seed"] = 1000
    st.session_state.pop("cm_game", None)


def _results(res):
    ev, hist = res["eval"], res["hist"]
    speed = next(k for k, v in SPEEDS.items() if v == res["cat_every"])
    st.markdown(f"#### Kết quả: {res['scenario']} · mèo *{cm.CAT_MODES[res['cat_mode']]}* · {speed.lower()}")
    learned, rnd = ev["Chuột đã học"], ev["Chuột ngẫu nhiên"]
    m = st.columns(4)
    m[0].metric("Thời gian huấn luyện", f"{hist['seconds']:.2f} s", help=f"{res['episodes']} episode")
    m[1].metric("Chuột đã học thắng", f"{100 * learned['Thắng']:.0f}%",
                delta=f"{100 * (learned['Thắng'] - rnd['Thắng']):+.0f} điểm so với ngẫu nhiên")
    m[2].metric("Chuột đã học bị bắt", f"{100 * learned['Bị bắt']:.0f}%")
    m[3].metric("Số bước TB mỗi ván", f"{learned['Số bước TB']:.1f}")
    st.markdown("**Quá trình học**")
    _show(cv.fig_learning(hist, window=max(20, min(300, len(hist["reward"]) // 15))))
    st.markdown("**So sánh qua 100 ván đánh giá** (chuột chơi tham lam theo Q, không thăm dò)")
    a, b = st.columns([1.1, 1])
    with a:
        rows = [{"Loại chuột": k, "Thắng": f"{100 * v['Thắng']:.0f}%", "Bị bắt": f"{100 * v['Bị bắt']:.0f}%",
                 "Hết giờ": f"{100 * v['Hết giờ']:.0f}%", "Thưởng TB": round(v["Thưởng TB"], 1),
                 "Số bước TB": round(v["Số bước TB"], 1)} for k, v in ev.items()]
        st.dataframe(rows, width="stretch", hide_index=True)
    with b:
        _show(cv.fig_eval(ev))


def _game(res):
    st.markdown("**Xem ván chơi**")
    c1, c2, c3 = st.columns([1.4, 1, 1])
    kind_label = c1.radio("Ai chơi?", list(POLICIES), horizontal=True, key="cm_who")
    ms = c2.slider("Tốc độ phát (ms/bước)", 100, 1000, 350, 50, key="cm_ms")
    if c3.button("🎲 Ván mới", key="cm_new"):
        st.session_state["cm_game_seed"] = st.session_state.get("cm_game_seed", 1000) + 1
    gseed = st.session_state.get("cm_game_seed", 1000)
    key = (id(res["agent"]), POLICIES[kind_label], gseed, ms)
    cached = st.session_state.get("cm_game")
    if not cached or cached["key"] != key:
        env = res["make_env"](gseed)
        pol = cm.make_policy(POLICIES[kind_label], res["agent"], res["snapshot"], seed=gseed)
        outcome, total, steps, frames = cm.play(env, pol, record=True)
        imgs = cv.episode_frames(env, frames, outcome)
        cached = {"key": key, "outcome": outcome, "total": total, "steps": steps, "imgs": imgs,
                  "gif": cv.to_gif(imgs, ms)}
        st.session_state["cm_game"] = cached
    label = {"win": "🧀 Thắng", "caught": "🐱 Bị bắt", "timeout": "⏱ Hết giờ"}[cached["outcome"]]
    st.caption(f"Ván số {gseed} · {kind_label}: **{label}** sau {cached['steps']} bước, "
               f"tổng thưởng {cached['total']:.1f}")
    a, b = st.columns(2)
    width = min(460, 70 * res["make_env"]().n_cols)
    with a:
        st.markdown("Hoạt hình (GIF lặp lại)")
        st.image(cached["gif"], width=width)
    with b:
        n = len(cached["imgs"]) - 1
        i = st.slider("Tua từng bước", 0, n, n, key=f"cm_step_{gseed}_{POLICIES[kind_label]}") if n > 0 else 0
        st.image(cached["imgs"][i], width=width)


def _heatmap(res):
    st.markdown("**Bản đồ nhiệt giá trị: chuột “sợ” vùng quanh mèo**")
    env = res["make_env"]()
    cat_cells = [i for i in range(env.n) if not env.is_hole[i]]
    labels = [f"hàng {env.cells[i][0]}, cột {env.cells[i][1]}" for i in cat_cells]
    a, b = st.columns([1, 1.4])
    with a:
        pick = st.selectbox("Đặt mèo ở ô", labels, index=cat_cells.index(env.cat_start), key="cm_heat_cat")
        mask = env.full_mask
        if len(env.cheeses) > 1:
            opts = {f"Còn {bin(mk).count('1')} miếng (mã {mk:0{len(env.cheeses)}b})": mk
                    for mk in range(env.n_mask - 1, 0, -1)}
            mask = opts[st.selectbox("Phô mai còn lại", list(opts), key="cm_heat_mask")]
        st.caption("Mỗi ô = max_a Q(s, a) khi chuột đứng ở ô đó, mèo ở ô đã chọn, phô mai như trên. "
                   "Mũi tên = hành động tham lam π*(s): chuột vòng tránh mèo để tới phô mai. "
                   "Ô càng xanh: kỳ vọng thắng càng cao; ô đỏ: kỳ vọng bị bắt. "
                   "Ô viền nâu = hang; ô xám “?” = trạng thái chuột chưa từng gặp (Q còn 0).")
    with b:
        _show(cv.fig_heatmap(env, res["agent"].q, cat_cells[labels.index(pick)], mask))


def tab_catmouse():
    st.subheader("🐱 Mèo bắt chuột — chuột học bằng Q-Learning")
    st.caption("Chuột là tác tử Q-Learning dạng bảng (epsilon-greedy, cập nhật Bellman như `ag1/qlearning.py`); "
               "mèo đi theo luật. Mã: `web/catmouse.py` (môi trường + tác tử), `web/catmouse_viz.py` (vẽ).")
    params = _settings()
    if st.button("Huấn luyện chuột", type="primary", key="cm_train"):
        _train(*params)
    res = st.session_state.get("cm")
    if res is None:
        st.info("Chọn kịch bản, mức khó của mèo rồi bấm **Huấn luyện chuột**.")
        return
    _results(res)
    st.divider()
    _game(res)
    st.divider()
    _heatmap(res)
