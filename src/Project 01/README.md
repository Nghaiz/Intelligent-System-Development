# DiaNexus

Hệ đa tác tử sàng lọc, tư vấn và tra cứu đái tháo đường, chạy hoàn toàn cục bộ. Mã nguồn của báo cáo `Report/Project 01/project01_CT_07_nghiand.pdf`.

Hệ thống chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `dnx/` | CSDL, phục vụ mô hình, luật, GPU và bộ nhớ |
| `dnx/ml/` | Mô hình bảng, võng mạc |
| `dnx/kb/` | Ontology, kho Neo4j, tìm kiếm lai |
| `dnx/rl/` | MaskedMLP, môi trường, Double DQN |
| `dnx/agents/` | Đồ thị trạng thái, tám vai trò, Critic |
| `web/` | FastAPI, giao diện tĩnh, script chụp màn hình |
| `scripts/` | Nạp dữ liệu, huấn luyện, đánh giá, vẽ hình |
| `tests/` | Kiểm thử luật, Critic, RL, tác tử |

Dữ liệu, mô hình và kết quả nằm trong `data/`, `models/`, `outputs/`, không đưa vào git.

## Chạy website

Từ `src/Project 01`, dùng Python của `.venv` ở gốc repo:

```bash
docker compose up -d
powershell -File scripts/ollama_serve.ps1
python -m uvicorn web.server:app --host 127.0.0.1 --port 8097
```

Mở <http://127.0.0.1:8097>. Máy chủ chỉ nghe ở địa chỉ cục bộ. Mô hình nạp ở yêu cầu đầu tiên cần đến nó; khi thiếu bộ nhớ, API trả 503 kèm thông báo.

Cần Neo4j ở cổng 7687, Ollama với `qwen2.5:7b`, mô hình trong `models/` và `src/TieuLuan Final/models/hf`.

## Kiểm thử

```bash
python -m pytest tests -q
```

## Dựng lại báo cáo

Từ `Report/Project 01`:

```bash
python scripts/report_tables.py
python build.py
```

`build.py` dừng nếu PDF vượt 55 trang. Mọi con số lấy từ `outputs/metrics`, đoạn mã được trích từ mã nguồn lúc dựng.
