# Hợp đồng tích hợp — DiaNexus (SSOT, mọi làn phải tuân)

Thư mục gốc mã: `src/Project 01/` (gói `dnx`). Python: `D:/Python/Intelligent-System-Development/.venv/Scripts/python`.
Chạy từ `src/Project 01`. Thư viện dùng chung có sẵn: `dnx/common.py gpu.py mem.py llm.py plot_style.py runlog.py texfmt.py`.

## Môi trường bắt buộc
- Chỉ ghi trên ổ D, ổ D giữ sàn 25 GB (`dnx.mem.require_disk`). Tổng dữ liệu mới của Project 01 ≤ 8 GB.
- Commit memory: mọi tiến trình nặng bọc `dnx.mem.slot(gb, ...)`; GPU nặng bọc `dnx.gpu.heavy(...)`; LLM bọc `dnx.llm.gpu_session`.
- CPU: mỗi tiến trình `limit_threads(8)`, XGBoost `n_jobs=12`. Không tự dừng tiến trình của làn khác.
- Neo4j: `bolt://127.0.0.1:7687`, `neo4j/dianexus2026`, container `dnx-neo4j`. Ollama: `powershell -File scripts/ollama_serve.ps1`, tag `qwen2.5:7b`.
- HF cục bộ: `hf_path("e5-base")`, `hf_path("bge-reranker")`. Mô-đun dùng lại được: `src/TieuLuan Final/tlf/ch5_*.py`, `ch6_rules.py`, `ch6_tools.py`, `web/`.
- Mỗi tệp ≤ ~200 dòng khi hợp lý. Comment tiếng Việt ngắn. Không fallback âm thầm.
- **Không `git add/commit/push`.** Chỉ sửa tệp thuộc làn mình. `pip install` được khi thật cần, ghi vào `requirements.txt` cuối làn.
- Không chạy lâu ở tiền cảnh quá 9 phút: chạy nền, ghi log vào `outputs/logs/`, chờ bằng lệnh `Monitor` hoặc vòng until.
- Kết quả số: `save_json("<tiền tố>_<tên>", obj)` vào `outputs/metrics/`. Tiền tố: `data_`, `ml_`, `dl_`, `retina_`, `kb_`, `kg_`, `rag_`, `rl_`, `agent_`, `web_`. Mọi lượt huấn luyện gọi `log_run(...)`.
- Hình: PDF vector ở `Report/Project 01/figures/<tiền tố>_*.pdf`, rộng 6,3 in, chữ ≥ 12pt (`dnx/plot_style.py`), kèm ảnh mẫu thật khi có.

## Đặc trưng hồ sơ (Profile)
Khóa tiếng Anh snake_case; thiếu thì `None`:
`age, sex("male"|"female"), bmi, hba1c, glucose, hypertension(0/1), heart_disease(0/1), smoking("never"|"former"|"current"|"unknown"), high_bp, high_chol, phys_activity, gen_hlth(1-5), ...` (các cột CDC giữ tên gốc đổi snake_case).

## API Python
```python
# dnx/ml/serve.py  (làn L1)
predict(profile: dict, mode="auto") -> {
  "mode": "full"|"screening", "prob": float, "label": 0|1, "threshold": float, "model": str, "calibrated": True,
  "shap": [{"feature": str, "label_vi": str, "value": any, "contribution": float}],  # giảm dần theo |contribution|
}
what_if(profile: dict, changes: dict) -> {"before": float, "after": float, "delta": float}
readmission(record: dict) -> {"prob": float, "label": 0|1, "threshold": float}
model_card() -> dict
# dnx/ml/serve_retina.py  (làn L1)
grade(image_bytes: bytes) -> {"grade": 0-4, "label_vi": str, "probs": [float]*5, "gradcam_png_b64": str}

# dnx/kb/search.py  (làn L2)
search(query: str, k=8, mode="hybrid"|"bm25"|"dense"|"graph"|"hybrid_rerank") -> [
  {"chunk_id": str, "text": str, "title": str, "source": str, "url": str, "lang": "vi"|"en", "score": float, "retrievers": [str]}]
entities(text: str) -> [{"id": str, "name": str, "label": str}]
subgraph(entity_ids: list[str], hops=1, limit=120) -> {"nodes":[{"id","name","label"}], "links":[{"source","target","type"}]}
thresholds(test: str) -> [{"test","op","value","unit","meaning","evidence","url"}]
stats() -> dict

# dnx/rules.py  (làn L3)
evaluate(facts: dict) -> [{"rule_id": str, "severity": "info"|"warn"|"urgent", "message": str, "source": str, "fired_on": dict}]
# dnx/rl/policy.py  (làn L3)
next_question(answers: dict) -> {"stop": bool, "feature": str|None, "question_vi": str, "options": list, "prob": float, "q_values": dict}
# dnx/agents/graph.py  (làn L3)
run(query: str, profile: dict|None=None, image: bytes|None=None, config="full"|"no_critic"|"vector_only"|"single_agent"|"no_rules",
    on_event: callable|None=None, session_id: str|None=None) -> {
  "answer": str, "citations": [{"n": int, "title": str, "url": str, "quote": str}],
  "risk": dict|None, "rules": list, "retina": dict|None, "trace": list[Event], "path": list[str], "seconds": float}
# Event: {"t": float, "agent": str, "kind": "start"|"tool"|"llm"|"decision"|"end"|"error", "detail": str, "data": dict}

# dnx/db.py  (làn L1 viết, mọi làn dùng) — SQLite `data/dianexus.db`, WAL
# bảng dữ liệu: patients_lab, brfss, hospital, pima, frankfurt
# bảng vận hành: sessions, messages, predictions, agent_events, screenings, feedback
connect() -> sqlite3.Connection;  log_event(session_id, event);  save_message(session_id, role, content);  save_prediction(session_id, payload)
```

## HTTP (làn L4, máy chủ `127.0.0.1:8097`, `web/server.py`)
`GET /api/status` · `POST /api/chat` (trả `session_id`) + `GET /api/chat/{id}/stream` (SSE Event) · `POST /api/predict` · `POST /api/whatif` ·
`POST /api/screen/next` · `POST /api/retina` (multipart) · `GET /api/search?q=&mode=` · `GET /api/graph?q=&hops=` · `GET /api/graph/overview` ·
`GET /api/db/{table}?limit=&offset=` · `GET /api/runs` · `GET /api/metrics/{name}` · `GET /api/logs/stream` (SSE) · `GET /api/sessions` · `POST /api/feedback`.
Thành công: JSON trực tiếp. Lỗi: HTTP 4xx/5xx + `{"error": str}`. Khởi động không nạp torch/LLM (nạp lười ở yêu cầu đầu).

## Ý đồ thiết kế chung
- Tên sản phẩm **DiaNexus**. Giọng: y tế, cẩn trọng, tiếng Việt. Câu miễn trừ: "Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ."
- An toàn: không nêu liều thuốc cụ thể, không khẳng định người dùng đã mắc bệnh; mọi ý y khoa trong câu trả lời có trích dẫn `[n]`.
- Tất cả tên hiển thị tiếng Việt; không nhắc "bài mẫu" hay tên bạn học trong sản phẩm và báo cáo.
- Ngưỡng chẩn đoán lấy từ KB có nguồn, không viết tay trong câu trả lời.
