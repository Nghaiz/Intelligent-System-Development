# %% [markdown]
# # Notebook 01 — Mạng Nơ-ron Cơ Sở Viết Tay Bằng NumPy (Pima Indians Diabetes, 768 mẫu)
#
# **Học phần:** Phát triển các Hệ thống Thông minh — Học viện Công nghệ Bưu chính Viễn thông
# **Sinh viên:** Nguyễn Duy Nghĩa · B23DCCN600 · D23CTPM01
# **Giảng viên hướng dẫn:** PGS.TS Trần Đình Quế
#
# ---
#
# ## Mục tiêu của notebook này
#
# Notebook mở đầu chuỗi thực nghiệm Assignment 03. Ở đây tôi dựng **một mạng nơ-ron
# nhiều tầng hoàn toàn bằng NumPy** — không TensorFlow, không PyTorch, không
# `sklearn.neural_network`. Mọi phép nhân ma trận, mọi đạo hàm riêng, mọi bước cập
# nhật trọng số đều được viết tay.
#
# Nhưng mục tiêu thật sự của notebook này **không phải là đạt điểm số cao**. Mục tiêu
# là dựng một **mô hình cơ sở có khuyết tật cố ý**, để notebook 02 có cái mà sửa. Cụ
# thể, tôi sẽ lặp lại đúng bốn lỗi phương pháp luận mà một người mới học rất hay mắc:
#
# | # | Khuyết tật cố ý | Hệ quả |
# |---|---|---|
# | 1 | Chuẩn hoá Z-Score trên **toàn bộ** dữ liệu trước khi chia train/test | Rò rỉ dữ liệu (Data Leakage) |
# | 2 | Chia train/test **ngẫu nhiên thuần**, không phân tầng | Tỷ lệ nhãn bệnh lệch giữa hai tập |
# | 3 | Giữ nguyên các giá trị **0 phi lý sinh học** (Glucose = 0, BMI = 0…) | Mạng phải học từ nhiễu cực đoan |
# | 4 | Áp ngưỡng quyết định **cứng 0.50** | Bỏ sót rất nhiều ca bệnh (False Negative) |
#
# Kết quả của notebook này sẽ trở thành cột "Mô hình Cơ sở" trong Bảng đối chứng ở
# Chương 2 của báo cáo.
#
# ## Bộ dữ liệu
#
# `Pima Indians Diabetes Database` — 768 hồ sơ bệnh nhân nữ gốc Pima, 8 đặc trưng y
# sinh, nhãn nhị phân `Outcome` (1 = mắc tiểu đường).
# Nguồn: <https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database>

# %% [markdown]
# ## 1. Nạp thư viện và cố định seed
#
# Toàn bộ chuỗi thực nghiệm dùng `random_state = 42` để mọi kết quả công bố trong báo
# cáo có thể tái lập chính xác. Ba thư viện duy nhất được phép dùng ở notebook này:
#
# - `numpy` — toàn bộ phần tính toán mạng nơ-ron;
# - `pandas` — chỉ để đọc CSV và thống kê mô tả;
# - `matplotlib` — chỉ để vẽ hình.
#
# Không có thư viện học sâu nào được nạp. Đây là điều kiện bắt buộc của Assignment 03.

# %%
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # backend không cần màn hình, an toàn khi chạy tự động
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

# Cấu hình hiển thị chung cho mọi hình trong notebook
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.family": "DejaVu Sans",     # bộ font có đầy đủ dấu tiếng Việt
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
})

NB_DIR = Path.cwd()
ROOT = NB_DIR.parent if NB_DIR.name == "notebook" else NB_DIR
DATA = ROOT / "data" / "pima_diabetes.csv"
FIG = ROOT / "reports" / "figures"
REP = ROOT / "reports"
FIG.mkdir(parents=True, exist_ok=True)

print("NumPy   :", np.__version__)
print("pandas  :", pd.__version__)
print("Thư mục dữ liệu :", DATA)
print("Thư mục hình vẽ :", FIG)

# %% [markdown]
# ## 2. Nền tảng toán học của mạng nơ-ron nhiều tầng
#
# Trước khi viết một dòng code huấn luyện nào, phần này ghi lại đầy đủ hệ phương trình
# mà đoạn code phía dưới sẽ hiện thực hoá. Đây là phần cốt lõi của Assignment 03: mạng
# nơ-ron không phải hộp đen, nó là một hàm toán học có thể viết ra giấy.
#
# ### 2.1 Mạng nơ-ron là phép hợp hàm tham số hoá
#
# Một mạng $L$ tầng đơn thuần là phép hợp của $L$ ánh xạ liên tiếp:
#
# $$\hat{y} = f_\theta(\mathbf{x}) = \big(f_L \circ f_{L-1} \circ \cdots \circ f_2 \circ f_1\big)(\mathbf{x})$$
#
# Mỗi tầng $l$ thực hiện đúng hai thao tác nối tiếp:
#
# 1. **Biến đổi affine** — chiếu vector từ $\mathbb{R}^{d_{l-1}}$ sang $\mathbb{R}^{d_l}$:
#
# $$\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]}\mathbf{W}^{[l]} + \mathbf{b}^{[l]}, \qquad \mathbf{W}^{[l]} \in \mathbb{R}^{d_{l-1}\times d_l},\ \ \mathbf{b}^{[l]} \in \mathbb{R}^{1\times d_l}$$
#
# 2. **Kích hoạt phi tuyến** — áp hàm $\phi(\cdot)$ lên từng phần tử:
#
# $$\mathbf{A}^{[l]} = \phi\big(\mathbf{Z}^{[l]}\big)$$
#
# ### 2.2 Vì sao bắt buộc phải có phi tuyến?
#
# Giả sử ta bỏ hàm kích hoạt (tức chọn $\phi(z) = z$). Với mạng hai tầng ẩn:
#
# $$\mathbf{h}_1 = \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1$$
# $$\mathbf{h}_2 = \mathbf{h}_1\mathbf{W}_2 + \mathbf{b}_2 = \mathbf{x}(\mathbf{W}_1\mathbf{W}_2) + (\mathbf{b}_1\mathbf{W}_2 + \mathbf{b}_2)$$
#
# Đặt $\mathbf{W}_{\text{composite}} = \mathbf{W}_1\mathbf{W}_2$ và $\mathbf{b}_{\text{composite}} = \mathbf{b}_1\mathbf{W}_2 + \mathbf{b}_2$, ta thu được:
#
# $$\mathbf{h}_2 = \mathbf{x}\mathbf{W}_{\text{composite}} + \mathbf{b}_{\text{composite}}$$
#
# **Kết luận:** xếp chồng bao nhiêu tầng tuyến tính cũng chỉ tương đương **một** phép
# biến đổi affine duy nhất. Chiều sâu không mang lại thêm chút năng lực biểu diễn nào.
# Tính phi tuyến chính là điều kiện tiên quyết tạo nên sức mạnh của Deep Learning —
# và ở notebook 02 tôi sẽ **kiểm chứng thực nghiệm** mệnh đề này bằng một ablation
# study bỏ hẳn ReLU.
#
# ### 2.3 Hàm kích hoạt ReLU
#
# $$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & z \ge 0 \\ 0 & z < 0\end{cases}, \qquad \frac{d}{dz}\text{ReLU}(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0\end{cases}$$
#
# Hai ưu thế quyết định:
#
# - **Không bão hoà đạo hàm.** Với $z > 0$ đạo hàm luôn bằng 1, nên gradient truyền
#   ngược qua nhiều tầng mà không bị suy giảm theo cấp số nhân như Sigmoid/Tanh
#   (hiện tượng Vanishing Gradient).
# - **Tạo tính thưa thớt.** Các nơ-ron có $z \le 0$ bị khoá về đúng 0, giúp mạng học
#   được biểu diễn phân tách rời rạc và tăng tốc phép nhân ma trận.
#
# ### 2.4 Tầng ra và hàm mất mát
#
# Bài toán phân loại nhị phân dùng Sigmoid ở tầng cuối:
#
# $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$
#
# và hàm mất mát Binary Cross-Entropy trên batch $m$ mẫu:
#
# $$\mathcal{L}_{\text{BCE}} = -\frac{1}{m}\sum_{i=1}^{m}\Big[y^{(i)}\log(\hat{y}^{(i)} + \epsilon) + (1-y^{(i)})\log(1 - \hat{y}^{(i)} + \epsilon)\Big]$$
#
# ### 2.5 Lan truyền ngược (Backpropagation)
#
# Tại tầng ra $L$, đạo hàm của BCE theo $z$ rút gọn cực đẹp nhờ sự triệt tiêu giữa đạo
# hàm hàm mất mát và đạo hàm Sigmoid:
#
# $$\mathbf{dZ}^{[L]} = \frac{1}{m}(\hat{\mathbf{y}} - \mathbf{y})$$
#
# Gradient của trọng số và độ chệch tại tầng $L$:
#
# $$\mathbf{dW}^{[L]} = \big(\mathbf{A}^{[L-1]}\big)^{T}\mathbf{dZ}^{[L]}, \qquad \mathbf{db}^{[L]} = \sum_{i=1}^{m}\mathbf{dZ}^{[L]}_{i,:}$$
#
# Truyền ngược về các tầng ẩn $l = L-1, L-2, \ldots, 1$:
#
# $$\mathbf{dA}^{[l]} = \mathbf{dZ}^{[l+1]}\big(\mathbf{W}^{[l+1]}\big)^{T}$$
# $$\mathbf{dZ}^{[l]} = \mathbf{dA}^{[l]} \odot \mathbb{I}\big(\mathbf{Z}^{[l]} > 0\big)$$
# $$\mathbf{dW}^{[l]} = \big(\mathbf{A}^{[l-1]}\big)^{T}\mathbf{dZ}^{[l]}, \qquad \mathbf{db}^{[l]} = \sum_{i=1}^{m}\mathbf{dZ}^{[l]}_{i,:}$$
#
# ### 2.6 Cập nhật trọng số
#
# $$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta\, \mathbf{dW}^{[l]}, \qquad \mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta\, \mathbf{db}^{[l]}$$
#
# Toàn bộ hệ phương trình trên được hiện thực hoá **nguyên văn** trong lớp
# `BaselineNeuralNetwork` ở mục 5.

# %% [markdown]
# ## 3. Nạp dữ liệu và khảo sát nhanh

# %%
df = pd.read_csv(DATA)
print(f"Kích thước bảng dữ liệu: {df.shape[0]} hàng × {df.shape[1]} cột\n")
print("Năm dòng đầu tiên:")
display(df.head())

print("\nThống kê mô tả:")
display(df.describe().T[["mean", "std", "min", "50%", "max"]].round(3))

FEATURES = [c for c in df.columns if c != "Outcome"]
TARGET = "Outcome"
print(f"\n{len(FEATURES)} đặc trưng đầu vào: {FEATURES}")

n_pos = int(df[TARGET].sum())
n_neg = len(df) - n_pos
print(f"\nPhân bố nhãn: {n_neg} âm tính ({n_neg/len(df):.1%}) · "
      f"{n_pos} dương tính ({n_pos/len(df):.1%})")

# %% [markdown]
# ### 3.1 Ghi nhận vấn đề chất lượng dữ liệu — nhưng CỐ Ý chưa xử lý
#
# Trong y học, một người còn sống **không thể** có nồng độ glucose huyết bằng 0, huyết
# áp tâm trương bằng 0 hay chỉ số BMI bằng 0. Những con số 0 này thực chất là **giá trị
# thiếu** được mã hoá nhầm thành 0 trong quá trình ghi chép bệnh án.
#
# Ô dưới đây định lượng chính xác mức độ nghiêm trọng. Mô hình cơ sở sẽ **giữ nguyên**
# các giá trị này — đó là khuyết tật số 3 trong bảng ở đầu notebook.

# %%
ZERO_INVALID = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

audit = pd.DataFrame({
    "Số bản ghi bằng 0": [int((df[c] == 0).sum()) for c in ZERO_INVALID],
    "Tỷ lệ (%)": [round((df[c] == 0).mean() * 100, 2) for c in ZERO_INVALID],
}, index=ZERO_INVALID).sort_values("Số bản ghi bằng 0", ascending=False)

print("Kiểm toán các giá trị 0 phi lý sinh học:")
display(audit)
print("\n→ Mô hình cơ sở CỐ Ý giữ nguyên toàn bộ các giá trị này để bộc lộ khuyết tật.")

# %% [markdown]
# ## 4. Tiền xử lý theo cách SAI (cố ý) — Rò rỉ dữ liệu và chia ngẫu nhiên
#
# Đây là khuyết tật số 1 và số 2. Phép chuẩn hoá Z-Score được tính trên **toàn bộ** 768
# mẫu **trước khi** chia train/test:
#
# $$\mu_{\text{global}} = \frac{1}{N}\sum_{i=1}^{N}x_i, \qquad \sigma_{\text{global}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu_{\text{global}})^2}$$
#
# Vì $\mu_{\text{global}}$ và $\sigma_{\text{global}}$ có chứa thông tin của tập kiểm
# thử, mô hình đã "nhìn trộm" phân phối của dữ liệu mà lẽ ra nó chưa từng thấy. Đây là
# vi phạm nguyên tắc khách quan khoa học — điểm số thu được sẽ **lạc quan giả tạo**.
#
# Đồng thời, phép chia 80/20 dùng hoán vị ngẫu nhiên thuần, không giữ tỷ lệ nhãn bệnh.

# %%
X_raw = df[FEATURES].to_numpy(dtype=np.float64)
y_all = df[TARGET].to_numpy(dtype=np.float64).reshape(-1, 1)

# ✗ SAI: chuẩn hoá trên toàn bộ dữ liệu → rò rỉ thống kê của tập test vào tập train
mu_global = X_raw.mean(axis=0)
sigma_global = X_raw.std(axis=0)
X_all = (X_raw - mu_global) / sigma_global

# ✗ SAI: chia ngẫu nhiên thuần, không phân tầng theo nhãn
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(X_all))
n_train = int(0.8 * len(X_all))
idx_tr, idx_te = perm[:n_train], perm[n_train:]

X_tr, X_te = X_all[idx_tr], X_all[idx_te]
y_tr, y_te = y_all[idx_tr], y_all[idx_te]

print(f"Tập huấn luyện : {X_tr.shape[0]} mẫu — tỷ lệ dương tính {y_tr.mean():.2%}")
print(f"Tập kiểm thử   : {X_te.shape[0]} mẫu — tỷ lệ dương tính {y_te.mean():.2%}")
print(f"Toàn bộ dữ liệu: {len(y_all)} mẫu — tỷ lệ dương tính {y_all.mean():.2%}")
print(f"\n→ Độ lệch tỷ lệ nhãn giữa tập kiểm thử và quần thể: "
      f"{abs(y_te.mean() - y_all.mean()) * 100:.2f} điểm phần trăm "
      f"(hệ quả trực tiếp của việc chia ngẫu nhiên không phân tầng).")

# %% [markdown]
# ## 5. Hiện thực hoá mạng nơ-ron thuần NumPy
#
# Lớp `BaselineNeuralNetwork` dưới đây là bản dịch trực tiếp từ hệ phương trình ở mục
# 2 sang code. Cấu hình kiến trúc:
#
# $$8 \xrightarrow{\ \mathbf{W}_1,\mathbf{b}_1\ } 16 \xrightarrow{\text{ReLU}} 8 \xrightarrow{\text{ReLU}} 1 \xrightarrow{\ \sigma\ } \hat{y}$$
#
# **Đếm số tham số học được:**
#
# $$\theta = \underbrace{(8 \times 16 + 16)}_{\text{tầng 1}} + \underbrace{(16 \times 8 + 8)}_{\text{tầng 2}} + \underbrace{(8 \times 1 + 1)}_{\text{tầng ra}} = 144 + 136 + 9 = \mathbf{289}$$
#
# **Cấu hình tối ưu hoá của mô hình cơ sở** (cố ý để ở mức "ngây thơ"):
#
# - Khởi tạo trọng số: $\mathcal{N}(0, 1)$ ngẫu nhiên chuẩn — chưa dùng He Normal;
# - Tốc độ học $\eta = 0.01$;
# - Số chu kỳ huấn luyện: 1,000 epochs;
# - Phương pháp cập nhật: Full-Batch Gradient Descent.

# %%
class BaselineNeuralNetwork:
    """Mạng nơ-ron 3 tầng (2 tầng ẩn ReLU + 1 tầng ra Sigmoid) viết hoàn toàn bằng NumPy.

    Đây là phiên bản CƠ SỞ, giữ nguyên các lựa chọn "ngây thơ" để notebook 02 có đối
    tượng cải tiến: khởi tạo N(0,1) thay vì He Normal, và không có bất kỳ cơ chế điều
    chuẩn nào.
    """

    def __init__(self, n_input: int, n_hidden1: int = 16, n_hidden2: int = 8,
                 lr: float = 0.01, seed: int = SEED):
        rng = np.random.default_rng(seed)
        # ✗ Khởi tạo chuẩn N(0,1): phương sai không phụ thuộc số nút vào
        self.W1 = rng.standard_normal((n_input, n_hidden1))
        self.b1 = np.zeros((1, n_hidden1))
        self.W2 = rng.standard_normal((n_hidden1, n_hidden2))
        self.b2 = np.zeros((1, n_hidden2))
        self.W3 = rng.standard_normal((n_hidden2, 1))
        self.b3 = np.zeros((1, 1))
        self.lr = lr
        self.history: list[float] = []

    # ---- các hàm kích hoạt -------------------------------------------------
    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(np.float64)

    @staticmethod
    def sigmoid(z):
        # Chặn miền giá trị trước khi mũ hoá để tránh tràn số exp(-z)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

    # ---- lan truyền tiến ---------------------------------------------------
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.relu(self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.sigmoid(self.Z3)
        return self.A3

    # ---- hàm mất mát -------------------------------------------------------
    @staticmethod
    def bce(y_true, y_pred, eps: float = 1e-9):
        return float(-np.mean(y_true * np.log(y_pred + eps)
                              + (1 - y_true) * np.log(1 - y_pred + eps)))

    # ---- lan truyền ngược --------------------------------------------------
    def backward(self, X, y):
        m = X.shape[0]

        # Tầng ra: đạo hàm BCE ∘ Sigmoid rút gọn thành (ŷ − y)/m
        dZ3 = (self.A3 - y) / m
        dW3 = self.A2.T @ dZ3
        db3 = dZ3.sum(axis=0, keepdims=True)

        # Tầng ẩn 2
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.relu_grad(self.Z2)
        dW2 = self.A1.T @ dZ2
        db2 = dZ2.sum(axis=0, keepdims=True)

        # Tầng ẩn 1
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_grad(self.Z1)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0, keepdims=True)

        # Cập nhật theo luật Gradient Descent
        for param, grad in ((self.W3, dW3), (self.b3, db3),
                            (self.W2, dW2), (self.b2, db2),
                            (self.W1, dW1), (self.b1, db1)):
            param -= self.lr * grad

    # ---- vòng huấn luyện ---------------------------------------------------
    def fit(self, X, y, epochs: int = 1000, verbose_every: int = 200):
        for ep in range(1, epochs + 1):
            y_hat = self.forward(X)
            loss = self.bce(y, y_hat)
            self.history.append(loss)
            self.backward(X, y)
            if verbose_every and (ep == 1 or ep % verbose_every == 0):
                print(f"  epoch {ep:>5} / {epochs}   BCE loss = {loss:.4f}")
        return self

    def predict_proba(self, X):
        return self.forward(X)

    def n_params(self) -> int:
        return sum(p.size for p in (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3))

# %% [markdown]
# ## 6. Huấn luyện mô hình cơ sở

# %%
t0 = time.perf_counter()

baseline = BaselineNeuralNetwork(n_input=X_tr.shape[1], n_hidden1=16, n_hidden2=8, lr=0.01)
print(f"Kiến trúc: 8 → 16 → 8 → 1   ·   {baseline.n_params()} tham số học được")
print(f"Tốc độ học η = {baseline.lr}   ·   1,000 epochs   ·   Full-Batch Gradient Descent\n")

baseline.fit(X_tr, y_tr, epochs=1000, verbose_every=200)

train_time = time.perf_counter() - t0
loss_start, loss_end = baseline.history[0], baseline.history[-1]
print(f"\nThời gian huấn luyện: {train_time:.2f}s")
print(f"Hàm mất mát: {loss_start:.4f} → {loss_end:.4f}  "
      f"(giảm {(1 - loss_end / loss_start) * 100:.1f}%)")

# %% [markdown]
# ### Hình 1 — Đường cong suy giảm hàm mất mát Binary Cross-Entropy

# %%
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.plot(baseline.history, lw=1.6, color="#1f77b4", label="Train Loss (BCE)")
ax.set_xlabel("Epochs")
ax.set_ylabel("Binary Cross-Entropy Loss")
ax.set_title("Đường cong Mất mát Huấn luyện (Baseline 3-Layer NN)")
ax.legend()
ax.annotate(f"Loss cuối: {loss_end:.4f}",
            xy=(len(baseline.history) - 1, loss_end),
            xytext=(-130, 32), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig01_baseline_loss_curve.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Đường cong giảm đều và trơn, không dao động — nhìn bề ngoài thì "mô hình học được".
# Nhưng loss huấn luyện giảm **không** đồng nghĩa với năng lực sàng lọc lâm sàng. Mục
# tiếp theo sẽ cho thấy khoảng cách giữa hai điều đó.

# %% [markdown]
# ## 7. Đánh giá trên tập kiểm thử độc lập
#
# Bốn chỉ số được tính trực tiếp từ ma trận nhầm lẫn, không dùng thư viện:
#
# $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \qquad \text{Precision} = \frac{TP}{TP + FP}$$
# $$\text{Recall} = \frac{TP}{TP + FN}, \qquad \text{F1} = 2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision} + \text{Recall}}$$
#
# Trong bài toán y tế, **Recall là chỉ số quan trọng nhất**: nó trả lời câu hỏi "trong
# tất cả bệnh nhân thực sự mắc bệnh, mô hình phát hiện được bao nhiêu phần trăm?".
# Mỗi ca False Negative là một bệnh nhân bị bỏ sót, mất cơ hội can thiệp sớm.

# %%
def confusion(y_true, y_pred_label):
    """Trả về (TN, FP, FN, TP) tính thủ công."""
    yt = y_true.ravel().astype(int)
    yp = y_pred_label.ravel().astype(int)
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tp = int(((yt == 1) & (yp == 1)).sum())
    return tn, fp, fn, tp


def classification_scores(y_true, y_prob, threshold: float = 0.50) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion(y_true, y_pred)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"threshold": float(threshold), "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "TN": tn, "FP": fp, "FN": fn, "TP": tp}


prob_te = baseline.predict_proba(X_te)
base_scores = classification_scores(y_te, prob_te, threshold=0.50)

print(f"KẾT QUẢ MÔ HÌNH CƠ SỞ trên {len(y_te)} mẫu kiểm thử (ngưỡng cứng 0.50)")
print("=" * 62)
print(f"  Accuracy  (Độ chính xác)      : {base_scores['accuracy']:.4f}  ({base_scores['accuracy']:.2%})")
print(f"  Precision (Độ chuẩn xác)      : {base_scores['precision']:.4f}  ({base_scores['precision']:.2%})")
print(f"  Recall    (Độ nhạy phát hiện) : {base_scores['recall']:.4f}  ({base_scores['recall']:.2%})")
print(f"  F1-Score                      : {base_scores['f1']:.4f}  ({base_scores['f1']:.2%})")
print("-" * 62)
print(f"  TN = {base_scores['TN']:>3}   FP = {base_scores['FP']:>3}")
print(f"  FN = {base_scores['FN']:>3}   TP = {base_scores['TP']:>3}")
print("=" * 62)
print(f"\n⚠ CÓ {base_scores['FN']} CA BỆNH BỊ BỎ SÓT (False Negative) "
      f"trên tổng {base_scores['FN'] + base_scores['TP']} ca dương tính thực tế.")

# %% [markdown]
# ## 8. Chẩn đoán bốn khuyết tật phương pháp luận
#
# Kết quả trên xác nhận đúng dự đoán ban đầu. Cụ thể từng khuyết tật đã gây ra hậu quả gì:
#
# **1. Rò rỉ dữ liệu (Data Leakage).** `mu_global` và `sigma_global` được tính trên cả
# 768 mẫu. Tập kiểm thử vì thế không còn "chưa từng thấy" — mọi điểm số ở mục 7 đều
# lạc quan hơn thực tế. Bản thân con số Accuracy đã mất tính khách quan khoa học.
#
# **2. Chia ngẫu nhiên không phân tầng.** Tỷ lệ nhãn bệnh giữa tập kiểm thử và quần thể
# lệch nhau, khiến điểm số dao động mạnh chỉ vì đổi seed — không phải vì mô hình tốt lên.
#
# **3. Giữ nguyên giá trị 0 phi lý sinh học.** 374 bản ghi Insulin = 0 và 227 bản ghi
# SkinThickness = 0 sau khi chuẩn hoá Z-Score trở thành các điểm outlier cực đoan. Mạng
# buộc phải dành năng lực biểu diễn để "học" các tín hiệu nhiễu này thay vì học quy luật
# sinh học thật.
#
# **4. Ngưỡng quyết định cứng 0.50.** Đây là khuyết tật tốn kém nhất về mặt lâm sàng.
# Ngưỡng 0.50 chỉ tối ưu khi hai lớp cân bằng 50:50 và chi phí của False Positive bằng
# chi phí của False Negative. Trong sàng lọc tiểu đường, hai giả định này **đều sai**:
# lớp bệnh chỉ chiếm ~35%, và một ca bỏ sót nguy hiểm hơn nhiều một lần xét nghiệm lại.
#
# Notebook 02 sẽ sửa cả bốn điểm và đo lại bằng đúng bộ chỉ số này.

# %%
prob_flat = prob_te.ravel()
print("Phân bố xác suất dự đoán trên tập kiểm thử:")
print(f"  Nhỏ nhất : {prob_flat.min():.4f}")
print(f"  Trung vị : {np.median(prob_flat):.4f}")
print(f"  Lớn nhất : {prob_flat.max():.4f}")

missed = prob_flat[(y_te.ravel() == 1) & (prob_flat < 0.50)]
if missed.size:
    print(f"\n{missed.size} ca bệnh bị bỏ sót có xác suất dự đoán trong khoảng "
          f"[{missed.min():.3f}, {missed.max():.3f}]")
    near = int((missed >= 0.40).sum())
    print(f"→ Trong đó {near} ca nằm ngay sát dưới ngưỡng (≥ 0.40): "
          f"chỉ cần hạ ngưỡng một chút là cứu được. Đây chính là lý do "
          f"notebook 02 phải dò tìm ngưỡng tối ưu.")

# %% [markdown]
# ## 9. Lưu kết quả để dựng báo cáo
#
# Mọi con số trong Chương 2 của báo cáo được đọc trực tiếp từ file JSON này, không gõ
# tay. Sửa notebook rồi chạy lại là báo cáo tự cập nhật theo.

# %%
payload = {
    "dataset": {
        "name": "Pima Indians Diabetes Database",
        "n_samples": int(len(df)),
        "n_features": len(FEATURES),
        "features": FEATURES,
        "positive_rate": float(y_all.mean()),
    },
    "zero_audit": {
        c: {"count": int((df[c] == 0).sum()), "pct": float((df[c] == 0).mean() * 100)}
        for c in ZERO_INVALID
    },
    "architecture": {
        "layers": "8 -> 16 -> 8 -> 1",
        "n_params": baseline.n_params(),
        "activation_hidden": "ReLU",
        "activation_output": "Sigmoid",
        "init": "N(0,1) standard normal",
        "lr": baseline.lr,
        "epochs": 1000,
        "optimizer": "Full-Batch Gradient Descent",
    },
    "split": {
        "strategy": "random 80/20 (KHONG phan tang)",
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "train_pos_rate": float(y_tr.mean()),
        "test_pos_rate": float(y_te.mean()),
    },
    "scaling": "Z-Score tren TOAN BO du lieu (RO RI DU LIEU - co y)",
    "loss": {"start": loss_start, "end": loss_end,
             "reduction_pct": (1 - loss_end / loss_start) * 100},
    "train_time_sec": train_time,
    "test_scores": base_scores,
    "loss_history": [float(v) for v in baseline.history],
    "flaws": [
        "Data Leakage: chuan hoa Z-Score tren toan bo du lieu truoc khi chia train/test",
        "Random split khong phan tang: ty le nhan benh lech giua hai tap",
        "Giu nguyen gia tri 0 phi ly sinh hoc o 5 chi so y sinh",
        "Nguong quyet dinh cung 0.50: bo sot rat nhieu ca benh",
    ],
}

out = REP / "metrics_baseline.json"
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✓ Đã lưu {out}")
print(f"✓ Đã lưu {FIG / 'fig01_baseline_loss_curve.png'}")

# %% [markdown]
# ## 10. Tổng kết notebook 01
#
# | Hạng mục | Kết quả |
# |---|---|
# | Kiến trúc | 8 → 16 → 8 → 1 (289 tham số, 100% NumPy) |
# | Hàm mất mát cuối | xem ô output mục 6 |
# | Chỉ số kiểm thử | xem bảng mục 7 |
# | Số ca bệnh bỏ sót | xem cảnh báo mục 7 |
#
# Mạng nơ-ron viết tay đã **chạy đúng về mặt toán học**: hàm mất mát hội tụ đơn điệu,
# gradient truyền ngược chính xác qua ba tầng. Nhưng quy trình thực nghiệm bao quanh nó
# thì sai ở bốn điểm, và cái giá phải trả nằm ở con số False Negative.
#
# **→ Notebook 02** sẽ sửa cả bốn khuyết tật (làm sạch y sinh bằng trung vị, triệt tiêu
# rò rỉ, phân chia phân tầng, dò ngưỡng tối ưu), sau đó tiến hành **6 nghiên cứu bóc
# tách kiến trúc (Ablation Studies)** để định lượng vai trò của từng thành phần cấu tạo
# nên mạng nơ-ron.
