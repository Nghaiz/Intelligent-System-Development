# Tiểu luận Final: Phát triển các hệ thống thông minh

Mã nguồn thực nghiệm và Web App Đồng Tâm AI Lab. Nhóm lớp CT, nhóm 07, Nguyễn Duy Nghĩa.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `tlf/` | Thư viện dùng chung và mã từng chương (`chN_*.py`), lớp phục vụ mô hình (`serve_chN.py`) |
| `scripts/` | Tải dữ liệu, huấn luyện, đánh giá, vẽ hình, xuất bảng theo chương |
| `web/` | Máy chủ FastAPI (`server.py`, `api/`) và giao diện tĩnh (`static/`) |
| `outputs/metrics/` | Kết quả dạng JSON, báo cáo và Web App đều đọc từ đây |
| `outputs/runs.jsonl` | Nhật ký mọi lượt chạy |
| `docker-compose.yml` | Neo4j 5.26 với volume có tên |

`data/`, `models/` và `outputs/logs/` không nằm trong git.

## Chạy lại

Từ thư mục này, với môi trường ảo của repo:

```powershell
docker compose up -d                          # Neo4j ở bolt://127.0.0.1:7687
powershell -File scripts/ollama_serve.ps1     # LLM cục bộ ở 127.0.0.1:11434
python scripts/ollama_pull.py qwen2.5:7b
python scripts/ollama_pull.py gemma3:4b
python scripts/ollama_pull.py qwen2.5:1.5b
```

Web App:

```powershell
cd web; npm install; npm run vendor; cd ..
python -m uvicorn web.server:app --host 127.0.0.1 --port 8090
```

Mở <http://127.0.0.1:8090>. Máy chủ chỉ nghe ở địa chỉ cục bộ và không nạp mô hình lúc khởi động.

## Lưu ý tài nguyên

Giới hạn thực trên Windows là commit memory, không phải RAM trống. Mọi việc nặng đi qua `tlf.mem.slot`, mọi việc GPU đi qua
`tlf.gpu.heavy`, và mỗi lúc chỉ một LLM trên GPU. Bộ nhớ đệm của thư viện được ép về `D:/DevCache` trong `tlf/common.py`.

Nội dung y tế và đầu tư chỉ mang tính học thuật.
