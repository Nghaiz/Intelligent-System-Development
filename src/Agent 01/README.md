# Bài tập Agent 01 — Discovering AI Agents + Q-Learning

Mã nguồn của báo cáo [`Report/Agent 01/agent01_CT_07_nghiand.pdf`](../../Report/Agent%2001/agent01_CT_07_nghiand.pdf).

```
ag1/maze.py        môi trường mê cung 12x12 (MDP: S, A, P, R) + BFS kiểm chứng
ag1/qlearning.py   QLearningAgent (epsilon-greedy, cập nhật Bellman), train, greedy_path
ag1/viz.py         vẽ mê cung, đường cong phần thưởng, trích Q-Table
notebooks/agent01_rl_qlearning.ipynb   notebook sinh từ ba tệp trên (ô mã = nguyên văn mã nguồn)
scripts/make_notebook.py   sinh + chạy notebook; notebook ghi outputs/metrics/ch8_qlearning.json
scripts/screenshots.py     chụp notebook trong JupyterLab (copy screen code / output)
scripts/report_assets.py   hình vector và bảng LaTeX cho Chương 8
```

## Dựng lại

```bash
python scripts/make_notebook.py      # ~2 s huấn luyện trên CPU
python scripts/report_assets.py
# JupyterLab ở cổng 8899 trong notebooks/, không token, windowingMode "none", rồi:
python scripts/screenshots.py
cd "../../Report/Agent 01" && python build.py
```
