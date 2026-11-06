# Kế hoạch Project 01 — DiaNexus

**Trạng thái:** đang thực hiện (phiên 2026-11-06). Đầu ra: `Report/Project 01/project01_CT_07_nghiand.pdf`, **≤ 55 trang**, mục tiêu 50–55.
Hợp đồng: [contract.md](contract.md). Thiết kế: `plans/reports/brainstorm-261106-project01.md`.

## Giai đoạn 1 — Hệ thống (4 làn song song)
| Làn | Phase | Sở hữu | GPU |
|---|---|---|---|
| L1 | [phase-1-data-ml.md](phase-1-data-ml.md) | `dnx/db.py`, `dnx/ml/`, `scripts/ml_*`, `scripts/data_*` | MLP, FT-Transformer, CNN |
| L2 | [phase-2-kb-rag.md](phase-2-kb-rag.md) | `dnx/kb/`, `scripts/kb_*` | embedding, LLM trích xuất |
| L3 | [phase-3-agents-rl.md](phase-3-agents-rl.md) | `dnx/rules.py`, `dnx/rl/`, `dnx/agents/`, `scripts/rl_*`, `scripts/agent_*`, `tests/` | DQN, ablation |
| L4 | [phase-4-web.md](phase-4-web.md) | `web/` | không |

L3 viết agent trên hợp đồng (stub tới khi L1/L2 xong); ablation chạy sau khi L1, L2 xong.

## Giai đoạn 2 — Báo cáo (sau khi có số)
[phase-5-report.md](phase-5-report.md): lead dựng khung, 4 làn theo nhóm chương, lead ghép và kiểm trang.

## Tiêu chí xong
- [ ] `outputs/metrics/` có đủ `data_* ml_* dl_* retina_* kb_* kg_* rag_* rl_* agent_*`
- [ ] Neo4j có ≥ 30k nút, ≥ 200k cạnh; truy vấn thật trên web
- [ ] Web chạy `127.0.0.1:8097`, mọi trang có dữ liệu thật, ảnh chụp thật trong báo cáo
- [ ] PDF ≤ 55 trang, mở được, số liệu qua `\SL{}`, không lỗi LaTeX
