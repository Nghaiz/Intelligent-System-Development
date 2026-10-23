# Brainstorm — Tiểu luận 03: Phân tích các hệ thống thông minh (AI → ML → CNN → RNN → KG/RAG → GNN → đa tác tử)

**Trạng thái:** đang thực hiện. Người dùng giao toàn quyền chọn phương án, làm từ đầu đến cuối trong một phiên, **được phép** dùng sub-agent (ghi đè ghi nhớ "đơn luồng" cho riêng bài này).
**Đầu ra:** `Report/TieuLuan 03/tieuluan03_CT_07_nghiand.pdf`; mã nguồn `src/TieuLuan 03/`. Kế hoạch thực thi: `plans/261023-tieuluan03/plan.md`.

## Đề bài (suy từ 4 bài mẫu trong `tmp/`)
- **dungvt (chính, 182 tr):** Mở đầu → I. Lịch sử AI (1.1–1.15) → II. ML cơ bản (lý thuyết 2.1–2.12; TN1 Diabetes 100k phân loại; TN2 House Price 2000 dòng hồi quy; mỗi TN: mô tả, dữ liệu, khảo sát, làm sạch, biểu diễn, EDA, chia tập, mô hình, baseline, đánh giá, chọn mô hình, triển khai web + mobile) → III. CNN (lý thuyết, LeNet…EfficientNet; TN MNIST, Fashion-MNIST, Rock/Paper/Scissors; mỗi TN Basic CNN from scratch / PyTorch / TensorFlow + Improved CNN PyTorch / TensorFlow, so sánh, triển khai) → IV. RNN (lý thuyết 4.1–4.21; TN e-commerce doanh thu ngày, AAPL; RNN/LSTM/GRU PyTorch + Keras, so sánh, triển khai) → V. KG & RAG (lý thuyết; TN diabetes, cổ phiếu Google, giá vàng FINAL_USO: KB, bộ ba, NetworkX, TF-IDF, embedding, truy xuất đồ thị, hybrid; chatbot deploy) → TLTK.
- **lucnvm:** thêm TN customer comments (phân loại văn bản), CIFAR-10, agent chọn công cụ (graph / tính CSV / RAG + Qwen3 cục bộ), "dữ liệu, phép đo, tái lập".
- **linhvtm:** lý thuyết ML chi tiết hơn (PLA, Softmax, SVR, KNN, K-means), Dog/Cat, RNN giá vàng, lịch sử 2025–2026 (agentic, reasoning, SLM), RAG naive→agentic, RAG vs fine-tune.
- **AnhNT:** thêm Ch.6 Deep Graph (GCN, GraphSAGE, GAT trên Cora/CiteSeer) và Ch.7 Hệ đa tác tử (GSM8K, so sánh 3 hệ), phần triển khai & tái lập.

## Điểm yếu của mẫu cần làm tốt hơn
- dungvt dán ảnh notebook, số liệu không đồng bộ; KG/RAG chỉ truy xuất, "chưa hợp nhất thành thứ hạng chung hoặc sinh câu trả lời". → Mọi số trong báo cáo sinh từ JSON kết quả (`\SL{}`), mã trong báo cáo trích nguyên văn từ mã đã chạy; RAG có bộ hỏi đáp đánh giá định lượng (Recall@k, MRR, độ chính xác đáp án) và LLM cục bộ sinh câu trả lời có trích dẫn.
- Mobile của dungvt là Expo Go chụp tay. → Web App Flask đáp ứng (responsive) + ảnh chụp Playwright ở khung desktop và khung điện thoại, đặt trong khung điện thoại vẽ TikZ.

## Phương án chọn (khuyến nghị, không hỏi lại)
| Hạng mục | Chọn | Lý do |
|---|---|---|
| Bố cục | Theo dungvt (5 chương) + Ch.6 GNN, Ch.7 đa tác tử (AnhNT) + Kết luận | "Chủ yếu dungvt, tham khảo đủ các bài khác" |
| Dữ liệu | Đúng bộ Kaggle của dungvt (kagglehub tải ẩn danh được, đã thử), thêm CIFAR-10, comments, gold RNN, Cora/CiteSeer, GSM8K | Bám mẫu, có link nguồn |
| ML | LogReg, KNN, NB, DT, RF, XGBoost, SVM/SVR, MLP; baseline; ROC/PR; chọn ngưỡng; K-means/PCA | Phủ lý thuyết 2.3–2.11 |
| CNN | NumPy from scratch (im2col) / PyTorch / TF-Keras; Basic + Improved (BN, Dropout, augmentation, residual); Grad-CAM | Đúng 3 nhánh của dungvt |
| RNN | SimpleRNN, LSTM, GRU (+BiLSTM) PyTorch & Keras; baseline naive; RNN NumPy BPTT minh hoạ | Đúng mẫu |
| KG/RAG | NetworkX + rdflib, TF-IDF, BM25, e5 dense, đồ thị, hybrid RRF, rerank, Qwen2.5-1.5B cục bộ, agent định tuyến | Vượt mẫu, nội bộ |
| GNN | GCN, GraphSAGE, GAT tự cài PyTorch (GPU), Cora/CiteSeer, độ sâu/over-smoothing, t-SNE | AnhNT |
| Đa tác tử | Điều phối + giải + kiểm chứng + công cụ Python; GSM8K tập con; so sánh LLM trực tiếp / 1 tác tử / đa tác tử | AnhNT |
| Web | Flask, phân hệ theo từng thực nghiệm, ảnh desktop + mobile | Như TL01/02 |
| Báo cáo | Khuôn LaTeX TL02: extreport 14pt, minted ≥12pt, TikZ nhiều màu/3D, matplotlib rộng 6,3 in | Phản hồi "mọi thứ phải to" |

## Phân chia CPU / GPU (RTX 4060 8 GB, Ryzen 9 7945HX 32 luồng, RAM 40 GB)
- GPU nặng (khoá `tl3.gpu.heavy()`): CNN PyTorch, LLM sinh, embedding hàng loạt. GPU nhẹ (≤1,5 GB, không khoá): MLP, RNN, GNN.
- CPU song song: NumPy scratch, TF/Keras (TF ≥2.11 trên Windows không GPU), scikit-learn, BM25/TF-IDF, vẽ hình, LaTeX. Mỗi tiến trình giới hạn luồng.

## Rủi ro
- 6 làn song song chung cây làm việc → mỗi làn sở hữu tệp theo tiền tố chương; chỉ lead commit.
- VRAM 8 GB → khoá GPU nặng; LLM 1,5B bf16 ~3,5 GB.
- CUDA hay sập lúc thoát (0xC0000409) → kiểm tra mtime đầu ra, không tin mã thoát.
- Hệ thống mang tính học thuật, không thay lời khuyên y tế/đầu tư.
