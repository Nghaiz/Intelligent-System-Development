# Phase 0 — Khởi tạo dự án

**Mục tiêu:** dựng bộ khung thư mục, môi trường chạy được, và kho mã trên GitHub.

## Việc phải làm

1. Tạo cây thư mục: `data/{raw,processed}`, `src/kg`, `notebooks`, `models`,
   `figures`, `outputs`, `report/chapters`, `app/pages`, `setup-guides`, `plans`.
2. Chuyển hai tệp CSV vào `data/raw/`.
3. Viết `.gitignore` — chặn `.venv/`, `.env`, `report/build/`, `TAILIEU/`.
4. Viết `requirements.txt` và `.env.example`.
5. Cài gói còn thiếu: `xgboost`, `neo4j`, `pyvis`, `networkx`, `jupyter`, `nbconvert`.
6. Viết `setup-guides/` — hướng dẫn từng bước cho tài khoản bên ngoài.
7. `git init`, commit đầu tiên, `gh repo create ... --public`, push.

## Tiêu chí hoàn thành

- [x] `python -c "import xgboost, neo4j, pyvis, networkx"` chạy không lỗi
- [x] `git status --short` không chứa tệp bí mật nào
- [x] Repo GitHub công khai tồn tại và truy cập được

Kho mã: https://github.com/Nghaiz/Assignment-01-Intelligent-System

## Rủi ro đã kiểm chứng

**pandas 3.0.5 có tương thích seaborn 0.13.2 không?** — Đã kiểm tra bằng smoke
test vẽ đồng thời KDE, boxplot và heatmap: chạy sạch. Không cần hạ phiên bản.

**SVR có treo trên 21.764 dòng không?** — Đo thời gian trên mẫu nhỏ: 2.000 dòng
mất 0.3 giây, 5.000 dòng mất 1.2 giây. Ngoại suy bậc hai ra khoảng 23 giây cho
toàn tập. Thực tế cả 5 mô hình hồi quy huấn luyện xong trong 45 giây. Không cần
lấy mẫu con.

**Thư mục models có đẩy lên GitHub được không?** — Lần đóng gói đầu, tệp
`housing_random_forest.joblib` nặng 149 MB, vượt giới hạn cứng 100 MB mỗi tệp
của GitHub, nên lệnh push sẽ bị từ chối thẳng. Đã xử lý bằng hai thay đổi:
đặt `min_samples_leaf=5` cho rừng hồi quy và bật nén mức 3 khi lưu. Toàn bộ
thư mục còn 17 MB, đổi lại R² chỉ giảm từ 0.5976 xuống 0.5925.
