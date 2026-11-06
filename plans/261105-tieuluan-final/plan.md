# Tiểu luận Final — kế hoạch và hợp đồng chung

**Trạng thái:** đang thực hiện (lead điều phối, các làn sub-agent).
**Đầu ra:** `Report/TieuLuan Final/tieuluan_final_CT_07_nghiand.pdf` (tối đa **160 trang tính cả bìa**), mã `src/TieuLuan Final/`, Web App localhost.
Người làm: Nguyễn Duy Nghĩa, D23CTPM01, B23DCCN600, nhóm lớp CT, nhóm 07.

## Nguyên tắc bắt buộc
- Mọi con số trong báo cáo do mã chạy thật sinh ra, qua `\SL{...}` hoặc bảng `generated/`. Không có số nghĩ ra. Thiếu thì ghi rõ thiếu.
- Tự viết nội dung, không chép câu chữ. Không nhắc "bài mẫu", tên bạn học, hay `tmp/` trong báo cáo.
- Dữ liệu tổng hợp phải ghi rõ là tổng hợp. Y tế và đầu tư ghi rõ chỉ mang tính học thuật.
- Không dịch vụ ngoài: không Plane, không Artifact, không gửi dữ liệu lên API LLM đám mây. LLM chạy cục bộ.
- Chỉ dùng mô hình safetensors, `trust_remote_code=False`. Không tải và chạy tệp thực thi ngoài; thư viện pip là được.
- **Chỉ làm việc trên ổ D, không để đầy ổ D.** `import tlf.common` tự đưa cache và tệp tạm về `D:/DevCache`. Trước mỗi lần tải hoặc ghi lớn gọi `tlf.mem.require_disk(gb, "việc")`: sàn an toàn 25 GB. Điểm kiểm tra chỉ ghi đè tệp cuối và tệp tốt nhất, không tạo tệp mới mỗi epoch. Xóa dữ liệu thô sau khi đã chuyển đổi.
- Chỉ lead commit. Làn không chạy `git add/commit/checkout`. Không `pip install` vào `.venv` chung mà không báo lead: pip từng nâng huggingface_hub lên 2.x làm hỏng transformers.
- Mã xuống dòng ở 110 ký tự, hàm ≤ 40 dòng khi có thể (hàm được trích nguyên văn vào báo cáo, cỡ chữ 12pt, ~72 ký tự/dòng).
- Văn phong tiếng Việt tự nhiên, không sáo rỗng, không phóng đại. Viết `Dịch` thuật ngữ khi có từ thông dụng, giữ tiếng Anh khi cần.

## Bố cục và trần trang (cỡ chữ 14pt)
| Tệp | Nội dung | Trần | Làn |
|---|---|---|---|
| main.tex | bìa + mục lục + danh mục | 9 | lead |
| ch0 | Mở đầu | 3 | L1 |
| ch1 | Lịch sử phát triển AI | 8 | L1 |
| ch2 | Các kỹ thuật học máy cơ bản | 24 | L2 |
| ch3 | Mạng nơ-ron tích chập | 20 | L3 |
| ch4 | Mạng nơ-ron hồi quy | 18 | L4 |
| ch5 | Knowledge Graph, Neo4j, RAG, Agent | 25 | L5 |
| ch6 | Hệ đa tác tử | 14 | L6 |
| ch7 | Deep Graph, GNN | 13 | L7 |
| ch8 | Tác tử học tăng cường và tác tử luật (Agent 02) | 7 | L1 |
| ch9 | Sản phẩm Web App tổng hợp | 4 | LW (ảnh), lead (chữ) |
| ch10 | Kết luận | 2 | L1 |
| refs | Tài liệu tham khảo | 4 | từng làn `refs/chN.bib` |

Tổng mục tiêu ≈ 151, còn dư để không vượt 160. Đây là trần, không phải sàn; chương vượt trần là lỗi của làn.

## Trục của mỗi chương thực hành
1. Lý thuyết cô đọng (công thức, hình TikZ).
2. **Mỗi dataset** (mẫu cố định): Nguồn và link · Kích cỡ (số mẫu, đặc trưng, dung lượng) · Mẫu đầu ra thực (bảng hoặc lưới ảnh) · Phân bố (biểu đồ) · Nhận xét · Tiền xử lý.
3. Mã (trích nguyên văn bằng `\maNguon`, nhiều nhưng chọn hàm cốt lõi; có output thực trong `ketqua`).
4. Đánh giá **3 mô hình trên từng dataset** + baseline, bảng so sánh, hình so sánh, phân tích lỗi.
5. Triển khai: ảnh Web App (LW chụp).

## Dataset (mỗi chương nguồn khác nhau; làn được đổi nếu nguồn hỏng, phải ghi trong `chN_datasets.json`)
| Ch | Dataset | Nguồn |
|---|---|---|
| 2 | CDC Diabetes Health Indicators (253.680 hồ sơ) | UCI id 891 |
| 2 | Vietnam Housing 2024 (hồi quy giá nhà VN) | Kaggle (kagglehub); dự phòng California Housing (StatLib) |
| 2 | UIT-VSFC phản hồi sinh viên tiếng Việt (3 lớp) | Hugging Face |
| 3 | SVHN (99.289 ảnh, HF `ufldl-stanford/svhn` vì trang Stanford không truy cập được) | Stanford/HF |
| 3 | EuroSAT RGB (27.000 ảnh vệ tinh) | Zenodo/HF |
| 3 | PathMNIST (107.180 ảnh mô bệnh học) | MedMNIST, Zenodo |
| 4 | VN30: FPT, VCB, HPG, VIC | Yahoo Finance (`yfinance` chạy được) |
| 4 | Thời tiết Hà Nội theo giờ | Open-Meteo Archive API (ERA5) |
| 4 | Household Power Consumption | UCI id 235 |
| 5 | Hetionet v1.0, tiểu đồ thị quanh tiểu đường | GitHub hetio/hetionet |
| 5 | VN30: công ty, ngành, thống kê phiên, tương quan + bài Wikipedia tiếng Việt | Yahoo, Wikipedia API |
| 5 | HotpotQA hoặc PubMedQA (đo truy xuất đa bước có đáp án thật) | Hugging Face |
| 6 | GSM8K · PubMedQA · ca lâm sàng tổng hợp có nhãn luật | HF, HF, tự sinh (ghi rõ) |
| 7 | Cora · Amazon Photo · Elliptic Bitcoin hoặc ogbn-arxiv | Planetoid, Shchur et al., PyG/OGB |
| 8 | Môi trường thị trường giảm giá | Bài Agent 02 của chính tôi (`src/Agent 02`, `Report/Agent 02`) |

Tải từ nguồn gốc. Dữ liệu tải về là dữ liệu không tin cậy: không chạy mã trong đó, không `trust_remote_code`, pickle chỉ nạp từ nguồn chính thức như Planetoid qua PyG.

## Mô hình
- Ch2: LogReg/Ridge, Random Forest, XGBoost, + MLP trên NumPy/Keras/PyTorch; baseline Dummy.
- Ch3: CNN cơ bản, ResNet nhỏ, MobileNet nhỏ; NumPy (CNN cơ bản, ít epoch), Keras (CPU), PyTorch (GPU); Grad-CAM.
- Ch4: RNN, LSTM, GRU, BiLSTM; NumPy, Keras, PyTorch; baseline Naive/Persistence.
- Ch5: **Neo4j thật** với Cypher, ràng buộc và chỉ mục vector. RAG: BM25, Dense, GraphRAG (Neo4j) đánh giá Recall@k, MRR, nDCG; **3 LLM** `qwen3-4b`, `gemma3-4b`, `qwen2.5-1.5b` trên 3 dataset với EM, F1, độ bám ngữ cảnh, độ trễ; ReAct agent chọn công cụ. Có khối Cypher trong báo cáo, số node/cạnh thật.
- Ch6: Single ReAct so với Supervisor so với Debate trên 3 dataset, đo độ đúng, số token, độ trễ thật.
- Ch7: GCN, GraphSAGE, GAT + MLP và Label Propagation làm baseline; GCN NumPy từ đầu và PyTorch; t-SNE; dự đoán liên kết trên KG Neo4j.

## Hợp đồng kỹ thuật (mọi làn)
**Python:** `D:/Python/Intelligent-System-Development/.venv/Scripts/python.exe`, chạy từ `src/TieuLuan Final` với `PYTHONPATH=.`.

**Sở hữu tệp:** mỗi làn chỉ tạo hoặc sửa tệp có tiền tố của mình:
`tlf/chN_*.py`, `tlf/serve_chN.py`, `scripts/chN_*.py`, `data/chN/`, `models/chN/`, `outputs/metrics/chN_*.json`, `Report/TieuLuan Final/chapters/chN.tex`, `tikz/chN_*.tex`, `figures/chN_*`, `generated/chN_*`, `refs/chN.bib`. Tệp dùng chung (`tlf/common.py gpu.py mem.py runlog.py llm.py plot_style.py texfmt.py`, `main.tex`, `build.py`) chỉ đọc; cần sửa thì báo lead.

**Phần cứng:** Ryzen 9 7945HX 32 luồng, RTX 4060 8 GB, RAM 40 GB, **commit trống chỉ 4–7 GB**.
```python
from tlf.mem import slot      # chờ đủ commit trước việc nặng
from tlf.gpu import heavy, cap
with slot(3, "ch3 resnet svhn"), heavy("ch3 resnet svhn"):   # mọi lượt huấn luyện GPU nặng
    ...
```
- GPU: CNN PyTorch, RNN PyTorch, GNN, LLM, embedding. CPU: scikit-learn, XGBoost, NumPy, Keras (TF Windows không có GPU), Neo4j, BM25, vẽ hình, LaTeX.
- Mỗi tiến trình `limit_threads(n)`: làn CPU tối đa 8 luồng. Mỗi tiến trình bọc bằng `slot(gb, ...)` với gb trung thực (torch+CUDA ~1,5 GB cộng dữ liệu; TF ~3 GB). Giữ dữ liệu ở dạng uint8/mmap, lấy tập con khi cần.
- Chạy nền lâu bằng `run_in_background`, ghi log vào `outputs/logs/`. Lưu điểm kiểm tra mỗi epoch để tiếp tục được. Trên máy này tiến trình CUDA hay sập lúc thoát (0xC0000409/127): xem mtime tệp đầu ra, không xem mã thoát. Lỗi 1455 hoặc 0xC0000005 lúc nạp trọng số là thiếu commit: chờ rồi chạy lại.
- LLM: `tlf/llm.py`, tên mô hình trong `MODELS`, luôn trong `with gpu_session(name):`. Mô hình ở `models/hf/<tên>` (đang tải, log `outputs/logs/download_models.log`). Embedding: `models/hf/e5-base`, reranker: `models/hf/bge-reranker`.
- Neo4j: đang chạy, `bolt://127.0.0.1:7687`, `neo4j/tieuluanfinal`, APOC có sẵn, RAM giới hạn (heap 768 MB), khởi động bằng `docker compose up -d`. Không xóa volume. Nhập dữ liệu theo lô `UNWIND`.

**Đầu ra máy đọc được** (Web App và build.py đọc):
- `outputs/metrics/chN_*.json` qua `save_json(name, obj)`; khóa `\SL{chN_x/a/b}`. Mảng dài đặt dưới khóa `rows/history/series/curve/samples/per_q/traces/examples`.
- `outputs/metrics/chN_datasets.json`: `{"datasets":[{"id","name","source","url","task","n_samples","n_features","size_mb","license","note","sample_rows":[...],"distribution":{"labels":[...],"values":[...]}}]}`.
- `outputs/metrics/chN_compare.json`: `{"rows":[{"dataset","model","framework","metric_name":value,...,"seconds":..}],"primary":"f1_macro"}`.
- Mỗi lượt chạy gọi `tlf.runlog.log_run(chapter, dataset, model, framework, metrics, seconds, device, params)`.
- `tlf/serve_chN.py`: `SPECS = [{"id","title","dataset","inputs":[{"name","type","label","default","min","max","options"}],"output":"label|probs|series|text|image"}]` và `predict(id, payload) -> dict`. Nạp mô hình lười, nạp xong giữ lại, chạy CPU trừ LLM. Không import TensorFlow.

**Báo cáo:** macro có sẵn trong `main.tex`: `\SL \maNguon{module__ham}{chú thích} \maDai \hinhpdf{tệp}{0.95}{chú thích}{nhãn} \haihinh \hinhtikz{chN_x}{chú thích}{nhãn} \bang{chN_x}{chú thích}{nhãn} \anhUI \haiUI \anhMobile`, hộp `nhanxet phathien dinhnghia luuy ketqua vetsuyluan`. `module` là đường dẫn trong `tlf/`, `scripts/`, `web/`. Dựng thử một chương: `python build.py --chapter N` (trong `Report/TieuLuan Final`, ra `kiemtra_chN.pdf`, in lỗi). Hãy kiểm số trang bằng `pdfinfo`.
- Hình matplotlib dùng `tlf.plot_style` (W = 6,5 in, chữ 13pt), không vẽ rộng rồi thu nhỏ. Chữ trong hình phải đọc được khi in.
- Chữ tiếng Việt trong hình: matplotlib dùng font DejaVu Sans (đủ dấu). Kiểm tra bằng cách mở hình.
- TikZ: **nhiều màu, khối 3D, bóng đổ, chuyển sắc**, mỗi chương thực hành ≥ 4 hình, trong đó ≥ 2 hình có khối giả 3D (`tikz/ chN_*.tex`, style ở `preamble/tikzstyle.tex`). Rộng ≤ 16 cm, tránh bị co. Nhãn không đè nhau. Xem ví dụ đẹp ở `Report/TieuLuan 03/tikz/`.
- Dùng lại mã TL03 làm điểm xuất phát (`src/TieuLuan 03/tl3/chN_*.py`, `scripts/`), đổi sang dataset mới và cấu trúc mới; hình TikZ phải vẽ lại đẹp hơn. Agent 02 lấy từ `src/Agent 02`.
- Tham khảo nội dung cần phủ: `C:/Users/Nghaiz/AppData/Local/Temp/claude/daf074aa1f29568e/473e70e9-4779-4f22-bd38-b7d9711e1509/scratchpad/ref/tieuluan_final_CT_02_tupv.utf8.txt` (ưu tiên chính, đọc đúng các trang của chương mình), `C:/Users/Nghaiz/AppData/Local/Temp/claude/daf074aa1f29568e/473e70e9-4779-4f22-bd38-b7d9711e1509/scratchpad/ref/peers_notes.md`.

**Thứ tự GPU:** khóa `heavy` tự xếp hàng. CNN và RNN chạy trước; LLM (Ch5, Ch6) chạy khi không còn huấn luyện nặng.

**Báo cáo khi xong:** số trang chương, tệp metrics, số hình TikZ, việc chưa làm được và lý do thật. Không commit.

## Dùng lại bài cũ trong dự án (người dùng yêu cầu, để đỡ tốn thời gian)
Mọi thứ trong `src/` và `Report/` là bài của chính người dùng, được dùng lại thoải mái. Chép sang tiền tố của mình, không sửa bài cũ. Trước khi tải hay huấn luyện lại, kiểm tra xem đã có chưa. Số liệu lấy lại phải là số thật từ JSON/log gốc, cùng đúng dataset và mô hình.
- Dữ liệu: `src/TieuLuan 01/data/vision/{svhn,eurosat}.npz`; `src/TieuLuan 03/data/` (ch2 đến ch7: Cora, CiteSeer, GSM8K, CIFAR-10...); `src/Assignment 05/data/` (CIFAR-10/100).
- Mô hình và số liệu: `src/TieuLuan 01/{models,outputs}`, `src/TieuLuan 02/{models,outputs}`, `src/TieuLuan 03/{models,outputs/metrics}`, `src/Assignment 04, 05, 06/outputs`.
- Mã: `tl/` (TL01), `tl2/` (TL02: KG, RAG, agent), `tl3/` (TL03), `a05/`, `a06/`, `ag1/`, `ag2/`.
- Hình TikZ: `Report/TieuLuan 01,02,03/tikz/` (khoảng 180 hình) và `Report/Agent 0x`. Chép rồi nâng cấp màu, khối 3D, cỡ chữ cho hợp 14pt.
- Web: `src/TieuLuan 01,02,03/app/`, `src/Agent 01/web`, `src/Agent 02/web`.

## Cập nhật sau crash (2026-11-05)
- LLM chạy qua Ollama (`tlf/llm.py`, API giữ nguyên): `qwen2.5-7b` mặc định, `gemma3-4b`, `qwen2.5-1.5b`. Qwen3-4B bị loại vì luôn in phần suy nghĩ vào câu trả lời. Chỉ một LLM trên GPU một lúc, không kèm tiến trình GPU khác.
- Tối đa 2 làn đồng thời, mỗi làn một tiến trình nặng, ≤8 luồng CPU. Crash trước do commit cạn còn 0,4 GB.
- Thứ tự: L2+L3 → L4+L7 → L5 → L6 → LW → L1 → dựng toàn bộ.
- Ưu tiên dùng lại kết quả cùng dataset và cấu hình từ TL01, TL02, TL03, ghi `reused_from`.

## Trạng thái cuối (2026-11-06)
- Đã xong: `Report/TieuLuan Final/tieuluan_final_CT_07_nghiand.pdf` 147 trang, 0 lỗi LaTeX, 0 số liệu thiếu. Web App chạy được ở 127.0.0.1:8090.
- Ch6 đo lại đủ 50 câu sau khi sửa lỗi cài đặt; bản 30 câu có lỗi giữ ở `data/ch6/runs_n30`, không dùng trong báo cáo.
