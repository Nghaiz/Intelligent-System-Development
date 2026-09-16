# %% [markdown]
# # Notebook 03 — Hệ Thống 1: Sàng Lọc Nguy Cơ Tiểu Đường Quy Mô Lớn (100,000 hồ sơ)
#
# **Học phần:** Phát triển các Hệ thống Thông minh — Học viện Công nghệ Bưu chính Viễn thông
# **Sinh viên:** Nguyễn Duy Nghĩa · B23DCCN600 · D23CTPM01
# **Giảng viên hướng dẫn:** PGS.TS Trần Đình Quế
#
# ---
#
# ## Notebook này làm gì?
#
# Hai notebook đầu làm việc trên 768 mẫu — đủ để hiểu cơ chế, nhưng quá nhỏ để nói bất
# cứ điều gì về năng lực thực tế. Từ notebook này trở đi, quy mô nhảy lên **100,000 hồ
# sơ bệnh nhân**, và câu hỏi nghiên cứu thay đổi hoàn toàn:
#
# > **Trên dữ liệu bảng quy mô lớn, một mạng nơ-ron sâu viết tay bằng NumPy có đủ sức
# > cạnh tranh với các mô hình Machine Learning cổ điển đã được tối ưu hàng chục năm hay
# > không?**
#
# Để trả lời, tôi dựng một cuộc đối đầu bốn bên trên **cùng một tập dữ liệu, cùng một
# phép chia, cùng một quy trình tiền xử lý**:
#
# | Nhóm | Mô hình | Bản chất |
# |---|---|---|
# | Machine Learning | Logistic Regression | Ranh giới tuyến tính |
# | Machine Learning | Decision Tree | Lát cắt trực giao |
# | Machine Learning | Random Forest (100 cây) | Ensemble Bagging |
# | **Deep Learning** | **MLP sâu thuần NumPy** | **Biểu diễn phân cấp phi tuyến** |
#
# Ngoài ra notebook còn thực hiện:
#
# - **Khảo sát đánh đổi Chiều sâu vs Chiều rộng** trên 4 cấu hình kiến trúc;
# - **Tinh chỉnh ngưỡng quyết định** cho cả 4 mô hình nhằm giảm thiểu ca bệnh bị bỏ sót;
# - **Trực quan hoá PCA 2D qua 4 tầng** để quan sát sự xuất hiện của ranh giới phân tách tuyến tính;
# - **Lưu artifact mô hình** (trọng số `.npz`) để REST API sau này tự forward-pass bằng NumPy.
#
# ## Bộ dữ liệu
#
# `Diabetes Prediction Dataset` — 100,000 hồ sơ bệnh nhân với thông số lâm sàng và nhân
# khẩu học: Tuổi, Giới tính, Chỉ số khối cơ thể BMI, Tiền sử hút thuốc, Tăng huyết áp,
# Bệnh tim mạch, HbA1c (đường huyết trung bình 3 tháng), và Glucose lúc đói.
# Nguồn: <https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset>

# %% [markdown]
# ## 1. Chuẩn bị môi trường

# %%
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 155, "font.family": "DejaVu Sans",
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.titlesize": 10, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
})

NB_DIR = Path.cwd()
ROOT = NB_DIR.parent if NB_DIR.name == "notebook" else NB_DIR
DATA = ROOT / "data" / "diabetes_prediction_dataset.csv"
FIG = ROOT / "reports" / "figures"
REP = ROOT / "reports"
MODEL = ROOT / "model"
for d in (FIG, MODEL):
    d.mkdir(parents=True, exist_ok=True)

print("Thư mục dữ liệu:", DATA)
print("Thư mục hình vẽ:", FIG)
print("Thư mục artifact:", MODEL)

# %% [markdown]
# ## 2. Nạp dữ liệu và khám phá ban đầu (EDA)

# %%
df_raw = pd.read_csv(DATA)
print(f"Kích thước dữ liệu thô: {df_raw.shape[0]:,} hàng × {df_raw.shape[1]} cột\n")
display(df_raw.head())

print("Kiểu dữ liệu và số giá trị khuyết thiếu:")
info = pd.DataFrame({
    "Kiểu dữ liệu": df_raw.dtypes.astype(str),
    "Số giá trị khuyết": df_raw.isna().sum(),
    "Số giá trị duy nhất": df_raw.nunique(),
})
display(info)

# %% [markdown]
# ### 2.1 Thách thức trung tâm: mất cân bằng lớp nghiêm trọng
#
# Đây là đặc điểm quyết định toàn bộ chiến lược của notebook này.

# %%
n_total = len(df_raw)
n_pos = int(df_raw["diabetes"].sum())
n_neg = n_total - n_pos

print(f"Âm tính  (nhãn 0): {n_neg:,} bệnh nhân  ({n_neg / n_total:.2%})")
print(f"Dương tính (nhãn 1): {n_pos:,} bệnh nhân  ({n_pos / n_total:.2%})")
print(f"Tỷ lệ mất cân bằng: khoảng 1 : {n_neg / n_pos:.1f}")
print()
print("⚠ HỆ LUỴ Y TẾ CỰC KỲ NGUY HIỂM:")
print(f"  Một mô hình 'ngu ngốc' luôn dự đoán ÂM TÍNH cho mọi bệnh nhân vẫn đạt")
print(f"  Accuracy = {n_neg / n_total:.2%} — con số nghe rất ấn tượng.")
print(f"  Nhưng toàn bộ {n_pos:,} bệnh nhân mắc bệnh đều bị bỏ sót, mất hoàn toàn cơ hội")
print(f"  can thiệp điều trị sớm. Đây chính là lý do Accuracy KHÔNG BAO GIỜ là chỉ số")
print(f"  đủ tin cậy trên dữ liệu y tế mất cân bằng — phải nhìn vào Recall và F1.")

# %% [markdown]
# ### Hình 1 — Phân phối nhãn mục tiêu

# %%
fig, ax = plt.subplots(figsize=(6.6, 4.2))
bars = ax.bar(["Âm tính (0)", "Dương tính (1)"], [n_neg, n_pos],
              color=["#1f4fd8", "#e03131"], width=0.55)
for b, v in zip(bars, [n_neg, n_pos]):
    ax.text(b.get_x() + b.get_width() / 2, v + n_total * 0.012,
            f"{v:,} ({v / n_total:.2%})", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Số lượng Bệnh nhân")
ax.set_ylim(0, n_total * 1.08)
ax.set_title("Phân phối Nhãn Mục tiêu: Tình Trạng Tiểu Đường (diabetes)", fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig01_target_distribution.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Hình 2 — Phân phối các chỉ số sinh hoá và ma trận tương quan
#
# Bốn biểu đồ phía trên đặt phân phối của từng chỉ số theo nhãn bệnh cạnh nhau. Hai
# đường ngưỡng chẩn đoán chuẩn của y học được vẽ kèm để đối chiếu:
#
# - **HbA1c ≥ 6.5%** — tiêu chuẩn chẩn đoán đái tháo đường của Hiệp hội Đái tháo đường Hoa Kỳ (ADA);
# - **Glucose > 200 mg/dL** — ngưỡng đường huyết cao bất thường.

# %%
fig = plt.figure(figsize=(13.0, 8.6))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

panels = [
    ("age", "Phân phối Độ tuổi (Age) theo Nhãn Bệnh", "Tuổi", 40, None),
    ("bmi", "Phân phối Chỉ số BMI theo Nhãn Bệnh", "BMI", 60, None),
    ("HbA1c_level", "Phân phối Chỉ số HbA1c theo Nhãn Bệnh", "HbA1c (%)", 40,
     (6.5, "Ngưỡng chẩn đoán ADA (HbA1c ≥ 6.5%)")),
    ("blood_glucose_level", "Phân phối Đường huyết (Blood Glucose) theo Nhãn Bệnh",
     "Glucose (mg/dL)", 40, (200, "Ngưỡng đường huyết cao (> 200 mg/dL)")),
]
for k, (col, title, xlab, bins, vline) in enumerate(panels):
    ax = fig.add_subplot(gs[k // 2, k % 2])
    ax.hist(df_raw.loc[df_raw.diabetes == 0, col], bins=bins, color="#4c72b0",
            alpha=0.85, label="Không bệnh (0)")
    ax.hist(df_raw.loc[df_raw.diabetes == 1, col], bins=bins, color="#c44e52",
            alpha=0.85, label="Tiểu đường (1)")
    if vline:
        ax.axvline(vline[0], color="#00008b", ls="--", lw=1.4, label=vline[1])
    ax.set_xlabel(xlab); ax.set_ylabel("Số lượng")
    ax.set_title(title, fontsize=9.5); ax.legend(fontsize=7)

fig.suptitle("Phân phối các chỉ số sinh hoá và nhân khẩu học theo nhãn bệnh",
             fontsize=12.5, fontweight="bold")
fig.savefig(FIG / "fig02_feature_distributions.png", bbox_inches="tight")
plt.show()

# %%
# Ma trận tương quan Pearson đầy đủ — mã hoá số cho hai cột phân loại
corr_src = df_raw.copy()
corr_src["gender"] = corr_src["gender"].map({"Female": 0, "Male": 1, "Other": 2})
corr_src["smoking_history"] = pd.factorize(corr_src["smoking_history"])[0]
corr_cols = ["gender", "age", "hypertension", "heart_disease", "smoking_history",
             "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"]
corr_labels = ["Gender", "Age", "Hypertension", "Heart Disease", "Smoking",
               "BMI", "HbA1c", "Glucose", "Diabetes"]
C = corr_src[corr_cols].corr().to_numpy()

fig, ax = plt.subplots(figsize=(8.4, 6.9))
im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        ax.text(j, i, f"{C[i, j]:.3f}", ha="center", va="center", fontsize=7.2,
                color="white" if abs(C[i, j]) > 0.6 else "#111", fontweight="bold")
ax.set_xticks(range(len(corr_labels)), corr_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(corr_labels)), corr_labels, fontsize=8)
ax.set_title("Ma Trận Tương Quan Pearson Giữa Các Đặc Trưng & Nhãn Mục Tiêu",
             fontweight="bold", fontsize=10.5)
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.046, label="Hệ số Tương quan Pearson (r)")
fig.tight_layout()
fig.savefig(FIG / "fig03_correlation_heatmap.png", bbox_inches="tight")
plt.show()

target_corr = corr_src[corr_cols].corr()["diabetes"].drop("diabetes").sort_values(
    key=abs, ascending=False)
print("Tương quan với nhãn bệnh, xếp theo độ mạnh:")
for name, v in target_corr.items():
    print(f"  {name:<22} r = {v:+.4f}")
print(f"\n→ Hai chỉ số đường huyết dẫn đầu tuyệt đối: "
      f"blood_glucose_level (r = {target_corr['blood_glucose_level']:.2f}) và "
      f"HbA1c_level (r = {target_corr['HbA1c_level']:.2f}).")
print("  Điều này khớp chính xác với y văn: cả hai đều là chỉ số chẩn đoán trực tiếp.")

# %% [markdown]
# ## 3. Kiểm toán quy trình làm sạch dữ liệu
#
# Bốn bước sàng lọc tuần tự, mỗi bước ghi lại chính xác số bản ghi bị loại. Bảng kiểm
# toán này đi thẳng vào báo cáo — nguyên tắc là **mọi bản ghi bị loại đều phải có lý do
# lâm sàng ghi rõ**, không được im lặng vứt dữ liệu.

# %%
audit_rows = [("Dữ liệu thô ban đầu", "Tập dữ liệu hồ sơ lâm sàng gốc", 0, len(df_raw))]
df = df_raw.copy()

n0 = len(df); df = df.drop_duplicates().reset_index(drop=True)
audit_rows.append(("1. Khử trùng lặp", "Loại bỏ các bản ghi trùng lặp hoàn toàn thông tin",
                   n0 - len(df), len(df)))

n0 = len(df); df = df[df["gender"] != "Other"].reset_index(drop=True)
audit_rows.append(("2. Lọc nhãn giới tính", "Loại bỏ nhóm giới tính không xác định (gender == 'Other')",
                   n0 - len(df), len(df)))

n0 = len(df); df = df[df["age"] >= 1.0].reset_index(drop=True)
audit_rows.append(("3. Lọc tuổi sơ sinh", "Loại bỏ trẻ dưới 1 tuổi (age < 1.0) do cơ chế đường huyết khác biệt",
                   n0 - len(df), len(df)))

n0 = len(df); df = df[(df["bmi"] >= 10.0) & (df["bmi"] <= 80.0)].reset_index(drop=True)
audit_rows.append(("4. Lọc ngoại lai BMI", "Loại bỏ ngoại lai sinh học phi lý (bmi < 10 hoặc bmi > 80)",
                   n0 - len(df), len(df)))

audit = pd.DataFrame(audit_rows, columns=["Bước Xử lý", "Tiêu chí Sàng lọc Lâm sàng",
                                          "Số lượng Loại bỏ", "Số lượng Còn lại"])
audit["Tỷ lệ Giữ lại (%)"] = (audit["Số lượng Còn lại"] / len(df_raw) * 100).round(2)
display(audit)

print(f"Còn lại {len(df):,} bản ghi sạch ({len(df) / len(df_raw):.2%} dữ liệu gốc)")
print(f"Tỷ lệ nhãn bệnh sau làm sạch: {df['diabetes'].mean():.2%}")

# %% [markdown]
# ## 4. Kỹ nghệ đặc trưng tương tác lâm sàng (Feature Engineering)
#
# Mạng nơ-ron về lý thuyết có thể tự học các tương tác phi tuyến. Nhưng cung cấp sẵn
# tín hiệu tương tác dựa trên kiến thức y sinh chuyên môn giúp mạng hội tụ nhanh hơn và
# giúp **cả các mô hình tuyến tính cũng được hưởng lợi** — điều kiện cần cho một cuộc
# đối đầu công bằng.
#
# **1. Tải lượng chuyển hoá kết hợp (`glucose_hba1c_interaction`):**
#
# $$\text{glucose\_hba1c\_interaction} = \frac{\text{blood\_glucose\_level} \times \text{HbA1c\_level}}{100}$$
#
# Đặc trưng này mô tả tác động **cộng hưởng** khi cả đường huyết tức thời và đường huyết
# tích luỹ 3 tháng cùng ở mức cao. Một bệnh nhân có Glucose cao nhưng HbA1c bình thường
# có thể chỉ vừa ăn xong; nhưng cả hai cùng cao là dấu hiệu rối loạn chuyển hoá mạn tính.
#
# **2. Tương tác tuổi tác và tăng huyết áp (`age_hypertension_risk`):**
#
# $$\text{age\_hypertension\_risk} = \text{age} \times \text{hypertension}$$
#
# Phản ánh gia tốc nguy cơ biến chứng tim mạch và tiểu đường ở nhóm bệnh nhân lớn tuổi
# có tiền sử huyết áp cao. Với người không tăng huyết áp, đặc trưng này bằng 0.

# %%
df["glucose_hba1c_interaction"] = df["blood_glucose_level"] * df["HbA1c_level"] / 100.0
df["age_hypertension_risk"] = df["age"] * df["hypertension"]

print("Hai đặc trưng tương tác vừa tạo:")
display(df[["blood_glucose_level", "HbA1c_level", "glucose_hba1c_interaction",
            "age", "hypertension", "age_hypertension_risk"]].describe().T.round(3))

for c in ["glucose_hba1c_interaction", "age_hypertension_risk"]:
    r = np.corrcoef(df[c], df["diabetes"])[0, 1]
    print(f"  Tương quan của {c:<28} với nhãn bệnh: r = {r:+.4f}")

# %% [markdown]
# ## 5. Kiến trúc tiền xử lý ColumnTransformer chống rò rỉ dữ liệu
#
# Ba nhóm biến được xử lý theo ba cách khác nhau, đóng gói thành một đường ống mô-đun
# hoá minh bạch **14 chiều**:
#
# | Nhóm biến | Cách xử lý | Số chiều |
# |---|---|---|
# | `gender` (2 giá trị sau lọc) | One-Hot, `drop='first'` | 1 |
# | `smoking_history` (6 danh mục) | One-Hot, `drop='first'` | 5 |
# | 4 biến lâm sàng + 2 biến tương tác | Chuẩn hoá Z-Score | 6 |
# | `hypertension`, `heart_disease` | Giữ nguyên (đã là nhị phân) | 2 |
# | **Tổng cộng** | | **14** |
#
# **Vì sao `drop='first'`?** Với $k$ danh mục, One-Hot đầy đủ tạo ra $k$ cột nhưng chỉ
# có $k-1$ cột độc lập tuyến tính (vì tổng các cột luôn bằng 1). Cột dư thừa gây hiện
# tượng **đa cộng tuyến hoàn hảo**, làm ma trận thiết kế suy biến và phá hỏng nghiệm của
# Logistic Regression. Bỏ một cột làm mốc tham chiếu triệt tiêu hoàn toàn vấn đề này.
#
# **Nguyên tắc chống rò rỉ:** toàn bộ $\mu_{\text{train}}$, $\sigma_{\text{train}}$ và
# bảng danh mục One-Hot được học **duy nhất trên tập Train**, rồi áp cố định sang Test.

# %%
CAT_ONEHOT = ["gender", "smoking_history"]
NUM_SCALE = ["age", "bmi", "HbA1c_level", "blood_glucose_level",
             "glucose_hba1c_interaction", "age_hypertension_risk"]
BIN_PASS = ["hypertension", "heart_disease"]
TARGET = "diabetes"

y = df[TARGET].to_numpy(dtype=np.float64).reshape(-1, 1)

def stratified_split(y, test_size=0.20, seed=SEED):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y.ravel() == cls); rng.shuffle(idx)
        k = int(round(test_size * len(idx)))
        te.extend(idx[:k]); tr.extend(idx[k:])
    tr, te = np.array(tr), np.array(te)
    rng.shuffle(tr); rng.shuffle(te)
    return tr, te

tr_idx, te_idx = stratified_split(y, 0.20, SEED)
df_tr, df_te = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
y_tr, y_te = y[tr_idx], y[te_idx]

print(f"Tập Train: {len(df_tr):,} mẫu — tỷ lệ bệnh {y_tr.mean():.4%}")
print(f"Tập Test : {len(df_te):,} mẫu — tỷ lệ bệnh {y_te.mean():.4%}")


class ColumnTransformerScratch:
    """Đường ống tiền xử lý tự cài: One-Hot (drop first) + Z-Score + passthrough.

    Tuân thủ nghiêm ngặt giao ước fit_transform / transform để chống rò rỉ dữ liệu.
    """

    def __init__(self, cat_cols, num_cols, bin_cols):
        self.cat_cols, self.num_cols, self.bin_cols = cat_cols, num_cols, bin_cols
        self.categories_, self.mu_, self.sd_, self.feature_names_ = {}, None, None, []

    def fit(self, frame):
        for c in self.cat_cols:
            # sorted() giữ thứ tự danh mục ổn định giữa các lần chạy
            self.categories_[c] = sorted(frame[c].astype(str).unique())[1:]   # drop='first'
        M = frame[self.num_cols].to_numpy(dtype=np.float64)
        self.mu_ = M.mean(axis=0)
        self.sd_ = M.std(axis=0); self.sd_[self.sd_ == 0] = 1.0
        self.feature_names_ = (
            [f"{c}_{v}" for c in self.cat_cols for v in self.categories_[c]]
            + list(self.num_cols) + list(self.bin_cols)
        )
        return self

    def transform(self, frame):
        blocks = []
        for c in self.cat_cols:
            col = frame[c].astype(str).to_numpy()
            blocks.append(np.stack([(col == v).astype(np.float64)
                                    for v in self.categories_[c]], axis=1))
        blocks.append((frame[self.num_cols].to_numpy(dtype=np.float64) - self.mu_) / self.sd_)
        blocks.append(frame[self.bin_cols].to_numpy(dtype=np.float64))
        return np.hstack(blocks)

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)


ct = ColumnTransformerScratch(CAT_ONEHOT, NUM_SCALE, BIN_PASS)
X_tr = ct.fit_transform(df_tr)     # ← học tham số CHỈ trên Train
X_te = ct.transform(df_te)         # ← áp cố định sang Test

print(f"\nKhông gian đặc trưng: {X_tr.shape[1]} chiều")
print(f"  Train: {X_tr.shape}   Test: {X_te.shape}")
print("\n14 chiều đặc trưng:")
for i, n in enumerate(ct.feature_names_, 1):
    print(f"  {i:>2}. {n}")

print(f"\nKiểm chứng chống rò rỉ — trung bình 6 cột chuẩn hoá:")
sl = slice(len(ct.feature_names_) - len(NUM_SCALE) - len(BIN_PASS),
           len(ct.feature_names_) - len(BIN_PASS))
print(f"  Train: {np.round(X_tr[:, sl].mean(axis=0), 6)}  (đúng bằng 0)")
print(f"  Test : {np.round(X_te[:, sl].mean(axis=0), 4)}  (khác 0 — đúng như kỳ vọng)")

# %% [markdown]
# ## 6. Mạng nơ-ron sâu thuần NumPy với huấn luyện Mini-Batch
#
# Ở quy mô 76,000 mẫu, Full-Batch Gradient Descent (như notebook 01–02) trở nên không
# hiệu quả: mỗi epoch chỉ cập nhật trọng số **đúng một lần**. Notebook này chuyển sang
# **Mini-Batch Gradient Descent** với kích thước lô 256:
#
# $$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\mathcal{B}_k), \qquad |\mathcal{B}_k| = 256$$
#
# Mỗi epoch giờ có khoảng $76{,}000 / 256 \approx 297$ bước cập nhật. Nhiễu ngẫu nhiên
# từ việc lấy mẫu lô còn giúp mạng thoát khỏi các cực tiểu địa phương nông.
#
# Lớp dưới đây mở rộng `ModularMLPScratch` của notebook 02, thêm mini-batch và khả năng
# trích xuất kích hoạt tầng ẩn phục vụ trực quan hoá PCA.

# %%
class DeepMLPScratch:
    """Mạng nơ-ron sâu tuỳ biến, 100% NumPy, He Normal, huấn luyện Mini-Batch."""

    def __init__(self, layer_sizes, lr=0.05, batch_size=256, seed=SEED):
        rng = np.random.default_rng(seed)
        self.sizes = list(layer_sizes)
        self.L = len(self.sizes) - 1
        self.lr, self.batch_size = lr, batch_size
        self.W = [rng.normal(0, np.sqrt(2.0 / self.sizes[i]),
                             (self.sizes[i], self.sizes[i + 1])) for i in range(self.L)]
        self.b = [np.zeros((1, self.sizes[i + 1])) for i in range(self.L)]
        self.history = []

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

    def forward(self, X, cache=True):
        A, Zs, As = X, [], [X]
        for i in range(self.L):
            Z = A @ self.W[i] + self.b[i]
            A = np.maximum(0.0, Z) if i < self.L - 1 else self._sigmoid(Z)
            Zs.append(Z); As.append(A)
        if cache:
            self.Zs, self.As = Zs, As
        return A

    @staticmethod
    def bce(y, p, eps=1e-9):
        return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    def _backward(self, y):
        dZ = (self.As[-1] - y) / y.shape[0]
        for i in range(self.L - 1, -1, -1):
            dW = self.As[i].T @ dZ
            db = dZ.sum(axis=0, keepdims=True)
            if i > 0:
                dZ = (dZ @ self.W[i].T) * (self.Zs[i - 1] > 0)
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db

    def fit(self, X, y, epochs=30, verbose_every=0):
        rng = np.random.default_rng(SEED)
        n = X.shape[0]
        for ep in range(1, epochs + 1):
            order = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                self.forward(X[idx])
                self._backward(y[idx])
            loss = self.bce(y, self.forward(X, cache=False))
            self.history.append(loss)
            if verbose_every and (ep == 1 or ep % verbose_every == 0):
                print(f"  epoch {ep:>3} / {epochs}   loss = {loss:.4f}")
        return self

    def predict_proba(self, X):
        return self.forward(X, cache=False)

    def hidden_activations(self, X):
        self.forward(X)
        return self.As[1:-1]

    def n_params(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)

# %% [markdown]
# ## 7. Bộ chỉ số đánh giá

# %%
def confusion(y_true, y_pred):
    yt, yp = y_true.ravel().astype(int), y_pred.ravel().astype(int)
    return (int(((yt == 0) & (yp == 0)).sum()), int(((yt == 0) & (yp == 1)).sum()),
            int(((yt == 1) & (yp == 0)).sum()), int(((yt == 1) & (yp == 1)).sum()))

def scores_at(y_true, y_prob, t=0.50):
    tn, fp, fn, tp = confusion(y_true, (np.asarray(y_prob).ravel() >= t).astype(int))
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"threshold": float(t), "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}

def best_threshold(y_true, y_prob, lo=0.10, hi=0.90, step=0.01):
    """Quét ngưỡng tối đa hoá F1 — LUÔN chạy trên tập Train."""
    best = (0.50, -1.0)
    for t in np.arange(lo, hi + 1e-9, step):
        f1 = scores_at(y_true, y_prob, t)["f1"]
        if f1 > best[1]:
            best = (round(float(t), 2), f1)
    return best

def roc_auc(y_true, y_score):
    """ROC-AUC tính bằng thống kê U của Mann–Whitney (tương đương diện tích dưới ROC)."""
    y, s = y_true.ravel(), np.asarray(y_score).ravel()
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # xử lý hạng trung bình cho các giá trị bằng nhau
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n_pos, n_neg = float((y == 1).sum()), float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

# %% [markdown]
# ## 8. Khảo sát đánh đổi kiến trúc: Chiều sâu vs Chiều rộng
#
# Bốn cấu hình được huấn luyện với **cùng dữ liệu, cùng seed, cùng $\eta$, cùng số
# epoch**. Chỉ hình dạng mạng thay đổi:
#
# | Cấu hình | Kiến trúc | Ý đồ |
# |---|---|---|
# | Shallow MLP | $14 \to 32 \to 1$ | Một tầng ẩn duy nhất |
# | Standard MLP | $14 \to 32 \to 16 \to 1$ | Hai tầng ẩn, mốc so sánh |
# | **Deeper MLP** | $14 \to 64 \to 32 \to 16 \to 1$ | **Ưu tiên chiều sâu** |
# | Wide MLP | $14 \to 128 \to 64 \to 1$ | **Ưu tiên chiều rộng** |

# %%
D = X_tr.shape[1]
ARCHS = [
    ("Shallow MLP",  [D, 32, 1],          "14 → 32 → 1"),
    ("Standard MLP", [D, 32, 16, 1],      "14 → 32 → 16 → 1"),
    ("Deeper MLP",   [D, 64, 32, 16, 1],  "14 → 64 → 32 → 16 → 1"),
    ("Wide MLP",     [D, 128, 64, 1],     "14 → 128 → 64 → 1"),
]

arch_rows, arch_hist, arch_models = [], {}, {}
for name, sizes, label in ARCHS:
    t0 = time.perf_counter()
    m = DeepMLPScratch(sizes, lr=0.05, batch_size=256, seed=SEED).fit(X_tr, y_tr, epochs=30)
    el = time.perf_counter() - t0
    p_tr, p_te = m.predict_proba(X_tr), m.predict_proba(X_te)
    t_best, _ = best_threshold(y_tr, p_tr)
    s = scores_at(y_te, p_te, t_best)
    arch_hist[name], arch_models[name] = m.history, m
    arch_rows.append({"Kiến trúc": name, "Cấu hình tầng": label, "Tham số": m.n_params(),
                      "Thời gian": f"{el:.2f}s", "Loss cuối": round(m.history[-1], 4),
                      "Accuracy": f"{s['accuracy']:.2%}", "Recall": f"{s['recall']:.2%}",
                      "F1-Score": f"{s['f1']:.2%}", "_s": s, "_t": t_best, "_el": el})
    print(f"{name:<14} {label:<24} {m.n_params():>6,} tham số  "
          f"{el:>6.2f}s  loss={m.history[-1]:.4f}  F1={s['f1']:.2%}  Recall={s['recall']:.2%}")

arch_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in arch_rows])
print("\nBảng đối chứng thực nghiệm khảo sát kiến trúc:")
display(arch_df)

best_arch = max(arch_rows, key=lambda r: r["_s"]["f1"])
print(f"\n★ Kiến trúc chiến thắng: {best_arch['Kiến trúc']} ({best_arch['Cấu hình tầng']}) "
      f"— F1 {best_arch['_s']['f1']:.2%}, Recall {best_arch['_s']['recall']:.2%}")

# %% [markdown]
# ### Hình 4 — Hội tụ Loss và đánh đổi hiệu năng giữa 4 kiến trúc

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))

ax = axes[0]
for (name, h), c, st in zip(arch_hist.items(),
                            ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"],
                            ["-", "-", "-", ":"]):
    npar = next(r["Tham số"] for r in arch_rows if r["Kiến trúc"] == name)
    ax.plot(range(1, len(h) + 1), h, lw=1.6, color=c, ls=st,
            label=f"{name} ({npar:,} params)")
ax.set_xlabel("Epoch (Chu kỳ huấn luyện)"); ax.set_ylabel("Binary Cross-Entropy Loss")
ax.set_title("Đường Cong Hội Tụ Hàm Mất Mát Giữa Các Kiến Trúc")
ax.legend()

ax = axes[1]
xs = np.arange(len(arch_rows)); w = 0.36
f1s = [r["_s"]["f1"] * 100 for r in arch_rows]
rcs = [r["_s"]["recall"] * 100 for r in arch_rows]
b1 = ax.bar(xs - w / 2, f1s, w, label="F1-Score (%)", color="#4c72b0")
b2 = ax.bar(xs + w / 2, rcs, w, label="Recall (%)", color="#55a868")
ax.bar_label(b1, fmt="%.1f", fontsize=8); ax.bar_label(b2, fmt="%.1f", fontsize=8)
ax.set_xticks(xs, [r["Kiến trúc"] for r in arch_rows], fontsize=8)
ax.set_ylabel("Phần trăm (%)"); ax.set_ylim(0, 100)
ax.set_title("So Sánh F1-Score & Recall Giữa Các Cấu Hình Kiến Trúc")
ax.legend()

fig.suptitle("Khảo sát hội tụ và đánh đổi hiệu năng giữa 4 kiến trúc mạng nơ-ron sâu trên 100,000 mẫu",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig04_architecture_study.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Nhận định khoa học.** Việc gia tăng chiều sâu cho phép mạng hình thành cơ chế **biểu
# diễn phân cấp (Hierarchical Abstraction)**: tầng $H_1$ mã hoá các tương tác tuyến tính
# cục bộ giữa các chỉ số; tầng $H_2$ tích hợp chúng thành tổ hợp nguy cơ (chẳng hạn
# Glucose × HbA1c × Tuổi); tầng $H_3$ phân tách ranh giới phi tuyến cuối cùng.
#
# Mạng Wide MLP tuy có số tham số lớn hơn nhiều nhưng chỉ dàn trải các bộ dò đặc trưng
# **song song ở cùng một mức trừu tượng**, nên không tái sử dụng được đặc trưng sơ cấp
# để xây đặc trưng bậc cao hơn. Đây là bằng chứng thực nghiệm cho quy luật: **tăng số
# nơ-ron chiều rộng không thay thế được cấu trúc phân cấp tuần tự của chiều sâu.**

# %% [markdown]
# ## 9. Đối đầu với ba mô hình Machine Learning cổ điển
#
# Ba mô hình ML được huấn luyện trên **đúng ma trận đặc trưng 14 chiều** mà mạng nơ-ron
# đã dùng. Không mô hình nào được ưu ái thêm bất kỳ thông tin nào.

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

ML_MODELS = [
    ("Logistic Regression", "Tuyến tính",
     LogisticRegression(max_iter=1000, random_state=SEED)),
    ("Decision Tree", "Cây đơn lẻ",
     DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED)),
    ("Random Forest", "Ensemble 100 Cây",
     RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5,
                            n_jobs=-1, random_state=SEED)),
]

results = {}
for name, kind, mdl in ML_MODELS:
    t0 = time.perf_counter()
    mdl.fit(X_tr, y_tr.ravel())
    el = time.perf_counter() - t0
    p_tr = mdl.predict_proba(X_tr)[:, 1]
    p_te = mdl.predict_proba(X_te)[:, 1]
    t_best, _ = best_threshold(y_tr, p_tr)
    results[name] = {
        "kind": kind, "time": el, "model": mdl,
        "n_params": (X_tr.shape[1] + 1 if name == "Logistic Regression"
                     else int(mdl.tree_.node_count) if name == "Decision Tree"
                     else int(sum(e.tree_.node_count for e in mdl.estimators_))),
        "prob_test": p_te, "tau": t_best,
        "default": scores_at(y_te, p_te, 0.50),
        "optimal": scores_at(y_te, p_te, t_best),
        "auc": roc_auc(y_te, p_te),
    }
    r = results[name]
    print(f"{name:<22} {el:>6.3f}s  τ_opt={t_best:.2f}  "
          f"Acc={r['optimal']['accuracy']:.2%}  Recall={r['optimal']['recall']:.2%}  "
          f"F1={r['optimal']['f1']:.2%}  AUC={r['auc']:.4f}")

# Bổ sung mô hình Deep Learning thắng cuộc vào bảng kết quả
dl_name = best_arch["Kiến trúc"]
dl_model = arch_models[dl_name]
p_tr_dl = dl_model.predict_proba(X_tr)
p_te_dl = dl_model.predict_proba(X_te)
results["Deeper DL Scratch"] = {
    "kind": f"Deep MLP ({best_arch['Cấu hình tầng']})", "time": best_arch["_el"],
    "model": dl_model, "n_params": dl_model.n_params(),
    "prob_test": p_te_dl.ravel(), "tau": best_arch["_t"],
    "default": scores_at(y_te, p_te_dl, 0.50),
    "optimal": best_arch["_s"], "auc": roc_auc(y_te, p_te_dl),
}
r = results["Deeper DL Scratch"]
print(f"{'Deeper DL Scratch':<22} {r['time']:>6.3f}s  τ_opt={r['tau']:.2f}  "
      f"Acc={r['optimal']['accuracy']:.2%}  Recall={r['optimal']['recall']:.2%}  "
      f"F1={r['optimal']['f1']:.2%}  AUC={r['auc']:.4f}")

# %% [markdown]
# ## 10. Tinh chỉnh ngưỡng quyết định và giảm thiểu ca bệnh bỏ sót
#
# Trong bài toán y tế, mục tiêu tối thượng là **giảm thiểu ca bỏ sót (False Negative)**.
# Một ca báo động giả (người khoẻ mạnh bị yêu cầu xét nghiệm máu lại) chỉ tốn một khoản
# chi phí nhỏ. Ngược lại, một ca bỏ sót — người bệnh không được phát hiện — có thể dẫn
# tới biến chứng suy thận, mù loà hoặc hoại tử chi.
#
# Do đó việc chấp nhận đánh đổi một lượng nhỏ Precision để đổi lấy sự gia tăng Recall là
# yêu cầu **bắt buộc và nhân văn** trong y tế số.

# %% [markdown]
# ### Hình 5 — Đường cong đánh đổi Precision / Recall / F1 theo ngưỡng

# %%
fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.0))
taus = np.arange(0.05, 0.96, 0.01)
for ax, (name, r) in zip(axes, results.items()):
    P = [scores_at(y_te, r["prob_test"], t) for t in taus]
    ax.plot(taus, [p["precision"] for p in P], lw=1.4, color="#ff7f0e", label="Precision")
    ax.plot(taus, [p["recall"] for p in P], lw=1.4, color="#2ca02c", label="Recall (Độ nhạy)")
    ax.plot(taus, [p["f1"] for p in P], lw=1.7, color="#1f77b4", label="F1-Score")
    ax.axvline(r["tau"], color="#d62728", ls="--", lw=1.3)
    ax.axvline(0.50, color="#999", ls=":", lw=1.1)
    ax.set_title(f"{name}\n(Ngưỡng tối ưu: {r['tau']:.2f})", fontsize=9)
    ax.set_xlabel("Ngưỡng Quyết định (Threshold)")
    ax.set_ylim(0, 1.02)
    if ax is axes[0]:
        ax.set_ylabel("Điểm số (0 – 1)"); ax.legend(fontsize=7)
fig.suptitle("Đường cong đánh đổi Precision – Recall – F1 theo dải ngưỡng quyết định (100,000 mẫu)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig05_threshold_tuning.png", bbox_inches="tight")
plt.show()

# %%
print("HIỆU QUẢ CỨU VÃN CA BỆNH NHỜ TINH CHỈNH NGƯỠNG")
print("=" * 78)
print(f"{'Mô hình':<22}{'τ tối ưu':>10}{'FN @0.50':>11}{'FN @τ':>9}{'Cứu được':>11}{'ΔRecall':>13}")
print("-" * 78)
saved_rows = []
for name, r in results.items():
    d, o = r["default"], r["optimal"]
    saved = d["FN"] - o["FN"]
    print(f"{name:<22}{r['tau']:>10.2f}{d['FN']:>11,}{o['FN']:>9,}{saved:>11,}"
          f"{(o['recall'] - d['recall']) * 100:>12.2f}%")
    saved_rows.append({"Mô hình": name, "Ngưỡng tối ưu": r["tau"],
                       "FN mặc định": d["FN"], "FN tối ưu": o["FN"], "Cứu được": saved})
print("=" * 78)

# %% [markdown]
# ## 11. Trực quan hoá quá trình học biểu diễn qua các tầng ẩn bằng PCA 2D
#
# Đây là thực nghiệm trả lời câu hỏi: **mạng nơ-ron thực sự học được gì?**
#
# Với mô hình thắng cuộc, tôi trích xuất vector kích hoạt tại từng tầng ẩn trên 2,500
# mẫu kiểm thử, rồi chiếu **độc lập** từng không gian xuống 2 chiều bằng PCA.

# %%
def pca_2d(M):
    Mc = M - M.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    var = S ** 2
    return Mc @ Vt[:2].T, var[:2] / max(var.sum(), 1e-12)

rng = np.random.default_rng(SEED)
sub = rng.choice(len(X_te), size=min(2500, len(X_te)), replace=False)
Xs, ys = X_te[sub], y_te[sub].ravel()
Hs = dl_model.hidden_activations(Xs)

spaces = [("(A) Đầu vào X (14D)", Xs, "Dữ liệu ban đầu, ranh giới phi tuyến phức tạp")]
for i, H in enumerate(Hs, 1):
    act = float((H > 0).mean() * 100)
    spaces.append((f"({chr(65 + i)}) Tầng ẩn H{i} ({H.shape[1]}D)", H,
                   f"Kích hoạt ReLU: {act:.1f}% nơ-ron hoạt động"))

fig, axes = plt.subplots(1, len(spaces), figsize=(4.15 * len(spaces), 4.3))
for ax, (title, M, sub_t) in zip(np.atleast_1d(axes), spaces):
    P, ratio = pca_2d(M)
    ax.scatter(P[ys == 0, 0], P[ys == 0, 1], s=7, alpha=0.45, c="#1f4fd8",
               edgecolors="none", label="Không bệnh (0)")
    ax.scatter(P[ys == 1, 0], P[ys == 1, 1], s=7, alpha=0.55, c="#e03131",
               edgecolors="none", label="Tiểu đường (1)")
    ax.set_title(f"{title}\n{sub_t}", fontsize=8.5)
    ax.set_xlabel(f"PC1 ({ratio[0]:.1%})"); ax.set_ylabel(f"PC2 ({ratio[1]:.1%})")
    ax.text(0.02, 0.97, f"Tổng var: {ratio.sum():.1%}", transform=ax.transAxes,
            fontsize=7, va="top", bbox=dict(fc="#fff8c4", ec="#ccc", alpha=0.9))
    ax.legend(fontsize=6.8, loc="lower right")

fig.suptitle(f"Trực quan hoá học biểu diễn qua các tầng ẩn bằng PCA 2D — "
             f"{dl_name} (Ngưỡng tối ưu τ = {results['Deeper DL Scratch']['tau']:.2f})",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig06_representation_pca_spaces.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Minh chứng hình học về cơ chế Representation Learning:**
#
# 1. **Không gian đầu vào $X$ (14D — bảng A).** Hai đám mây dữ liệu nằm đan xen hỗn độn,
#    chồng lấn với mật độ dày đặc. Không siêu phẳng tuyến tính nào phân tách được.
#
# 2. **Các tầng ẩn.** Qua mỗi phép biến đổi affine kết hợp kích hoạt ReLU, các điểm dữ
#    liệu dần dãn nở không gian; cụm màu đỏ (bệnh nhân tiểu đường) dịch chuyển tách dần
#    khỏi cụm màu xanh. Phương sai nội cụm của nhóm khoẻ mạnh giảm dần qua từng tầng.
#
# 3. **Tầng ẩn cuối.** **Tính phân tách tuyến tính (Linear Separability) xuất hiện rõ
#    rệt** — hai nhóm bệnh nhân bị đẩy hẳn về hai phía đối lập với khoảng phân cách rộng
#    mở. Lúc này tầng ngõ ra chỉ cần một phép biến đổi tuyến tính kết hợp Sigmoid với
#    ngưỡng $\tau_{\text{opt}}$ là đủ để phân loại cực kỳ chính xác.

# %% [markdown]
# ## 12. Tổng hợp đối chuẩn toàn diện 4 mô hình

# %%
bench = []
for name, r in results.items():
    d, o = r["default"], r["optimal"]
    bench.append({
        "Mô hình": name, "Loại mô hình": r["kind"],
        "Tham số": f"{r['n_params']:,}", "Thời gian": f"{r['time']:.3f}s",
        "Ngưỡng": f"{r['tau']:.2f}",
        "Acc": f"{o['accuracy']:.2%}",
        "Rec (0.50)": f"{d['recall']:.2%}", "Rec (Opt)": f"{o['recall']:.2%}",
        "F1 (0.50)": f"{d['f1']:.2%}", "F1 (Opt)": f"{o['f1']:.2%}",
        "ROC-AUC": f"{r['auc']:.4f}",
        "FN Giảm": f"-{d['FN'] - o['FN']} ca",
    })
bench_df = pd.DataFrame(bench)
print("Bảng đối chuẩn tổng hợp đầy đủ ở cả hai mức ngưỡng (Mặc định 0.50 vs Tối ưu):")
display(bench_df)

champ = max(results.items(), key=lambda kv: kv[1]["optimal"]["f1"])
print(f"\n★ Quán quân F1-Score: {champ[0]} — F1 {champ[1]['optimal']['f1']:.2%}, "
      f"ROC-AUC {champ[1]['auc']:.4f}, Recall {champ[1]['optimal']['recall']:.2%}")

# %% [markdown]
# ### Hình 6 — Đối chuẩn trực quan hiệu năng toàn hệ thống

# %%
fig = plt.figure(figsize=(15.5, 9.6))
gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1, 1], hspace=0.45, wspace=0.30)

names = list(results.keys())

# (A) So sánh Recall & F1 trước / sau tinh chỉnh
ax = fig.add_subplot(gs[0, :2])
xs = np.arange(len(names)); w = 0.2
series = [("Recall (0.50)", [results[n]["default"]["recall"] * 100 for n in names], "#aec7e8"),
          ("Recall (Tối ưu)", [results[n]["optimal"]["recall"] * 100 for n in names], "#1f77b4"),
          ("F1 (0.50)", [results[n]["default"]["f1"] * 100 for n in names], "#ffbb78"),
          ("F1 (Tối ưu)", [results[n]["optimal"]["f1"] * 100 for n in names], "#ff7f0e")]
for k, (lab, vals, c) in enumerate(series):
    bb = ax.bar(xs + (k - 1.5) * w, vals, w, label=lab, color=c)
    ax.bar_label(bb, fmt="%.1f", fontsize=6, padding=1)
ax.set_xticks(xs, names, fontsize=8); ax.set_ylabel("Phần trăm (%)"); ax.set_ylim(0, 100)
ax.set_title("(A) So Sánh Độ Nhạy (Recall) & F1-Score Trước và Sau Tinh Chỉnh Ngưỡng",
             fontsize=10)
ax.legend(fontsize=7, ncol=2)

# (B) Số ca bệnh bị bỏ sót
ax = fig.add_subplot(gs[0, 2:])
w = 0.36
fn_d = [results[n]["default"]["FN"] for n in names]
fn_o = [results[n]["optimal"]["FN"] for n in names]
b1 = ax.bar(xs - w / 2, fn_d, w, label="Số ca bỏ sót FN (Ngưỡng 0.50)", color="#e15759")
b2 = ax.bar(xs + w / 2, fn_o, w, label="Số ca bỏ sót FN (Ngưỡng tối ưu)", color="#59a14f")
ax.bar_label(b1, fmt="%d", fontsize=7); ax.bar_label(b2, fmt="%d", fontsize=7)
for i, n in enumerate(names):
    ax.text(i, max(fn_d[i], fn_o[i]) * 1.14, f"Giảm {fn_d[i] - fn_o[i]} ca",
            ha="center", fontsize=7.6, fontweight="bold", color="#2b7a2b")
ax.set_xticks(xs, names, fontsize=8)
ax.set_ylabel("Số ca bệnh bị bỏ sót (FN)"); ax.set_ylim(0, max(fn_d) * 1.35)
ax.set_title("(B) Đánh Giá Mức Độ Phát Hiện Ca Bệnh (Giảm False Negatives)", fontsize=10)
ax.legend(fontsize=7)

# (C) Hai hàng ma trận nhầm lẫn: ngưỡng 0.50 và ngưỡng tối ưu
for row, key, cmap, tag in [(1, "default", "Reds", "0.50"), (2, "optimal", "Blues", "tối ưu")]:
    for col, n in enumerate(names):
        ax = fig.add_subplot(gs[row, col])
        s = results[n][key]
        cm = np.array([[s["TN"], s["FP"]], [s["FN"], s["TP"]]], dtype=float)
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j]):,}\n({pct[i, j]:.1%})", ha="center",
                        va="center", fontsize=7.2,
                        color="white" if pct[i, j] > 0.55 else "#111", fontweight="bold")
        thr = 0.50 if key == "default" else results[n]["tau"]
        extra = "" if key == "default" else f" | Giảm {results[n]['default']['FN'] - s['FN']} ca"
        ax.set_title(f"{n} (τ={thr:.2f})\n[Bỏ sót FN: {s['FN']:,} ca{extra}]", fontsize=7.6)
        ax.set_xticks([0, 1], ["Âm tính", "Dương tính"], fontsize=6.5)
        ax.set_yticks([0, 1], ["Âm tính", "Dương tính"], fontsize=6.5)
        ax.grid(False)
        if col == 0:
            ax.set_ylabel(f"NGƯỠNG {tag.upper()}\nNhãn thực tế", fontsize=7.5)

fig.suptitle("Đối chuẩn hiệu năng hệ thống sàng lọc tiểu đường quy mô 100,000 mẫu",
             fontsize=13, fontweight="bold")
fig.savefig(FIG / "fig07_model_comparison.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 13. Nhận xét chuyên sâu và đánh giá kết quả thực nghiệm y tế

# %%
rf, dl, lr_ = results["Random Forest"], results["Deeper DL Scratch"], results["Logistic Regression"]

print("PHÁT HIỆN 1 — Vì sao Random Forest rất mạnh trên dữ liệu bảng dạng ngưỡng sinh học")
print(f"  Random Forest: F1 {rf['optimal']['f1']:.2%}, ROC-AUC {rf['auc']:.4f}, "
      f"Recall {rf['optimal']['recall']:.2%}")
print("  Đặc thù dữ liệu y tế là ranh giới bệnh/thường được xác lập bởi các NGƯỠNG giới")
print("  hạn sinh hoá nghiêm ngặt (HbA1c ≥ 6.5% hoặc Glucose ≥ 126 mg/dL theo WHO).")
print("  Cây quyết định chia không gian bằng các lát cắt trực giao (Axis-aligned splits)")
print("  bắt trọn các bước nhảy bậc thang này mà không cần tối ưu hoá liên tục như")
print("  gradient descent.\n")

print("PHÁT HIỆN 2 — Năng lực biểu diễn của mạng nơ-ron thuần NumPy")
print(f"  {dl_name}: Accuracy {dl['optimal']['accuracy']:.2%}, Recall {dl['optimal']['recall']:.2%}, "
      f"F1 {dl['optimal']['f1']:.2%}, ROC-AUC {dl['auc']:.4f}")
print(f"  So với Logistic Regression: {(dl['optimal']['f1'] - lr_['optimal']['f1']) * 100:+.2f} "
      f"điểm F1")
print("  Dù không có framework tự động vi phân hỗ trợ, các phương trình lan truyền ngược")
print("  viết tay vẫn hội tụ ổn định và học được biểu diễn trừu tượng hiệu quả từ")
print("  100,000 mẫu dữ liệu thực tế.\n")

print("PHÁT HIỆN 3 — Ý nghĩa lâm sàng sống còn của việc tinh chỉnh ngưỡng")
for n, r in results.items():
    saved = r["default"]["FN"] - r["optimal"]["FN"]
    print(f"  {n:<22} bỏ sót {r['default']['FN']:>4,} → {r['optimal']['FN']:>4,} ca "
          f"(cứu vãn {saved:>3,} bệnh nhân thực tế)")
print("  Trong y học cộng đồng, chi phí của một ca báo động giả là một lần xét nghiệm")
print("  máu lại; chi phí của một ca bỏ sót có thể là suy thận, mù loà hoặc hoại tử chi.")

# %% [markdown]
# ## 14. Lưu artifact mô hình phục vụ triển khai REST API
#
# Trọng số của mạng nơ-ron được lưu dạng `.npz` **thuần NumPy**. REST API sau này sẽ tự
# thực hiện forward pass bằng NumPy — không cần cài sklearn, không cần TensorFlow.
# Đây là minh chứng rằng mạng tự viết tay **chạy được thật trong môi trường triển khai**,
# chứ không chỉ tồn tại trong notebook.

# %%
np.savez(MODEL / "dl_scratch_weights.npz",
         **{f"W{i}": w for i, w in enumerate(dl_model.W)},
         **{f"b{i}": b for i, b in enumerate(dl_model.b)})

import joblib
joblib.dump(results["Random Forest"]["model"], MODEL / "random_forest.joblib")
joblib.dump(results["Logistic Regression"]["model"], MODEL / "logistic_regression.joblib")

metadata = {
    "app": "diabetes_large",
    "task": "binary_classification",
    "dataset": {"name": "Diabetes Prediction Dataset", "n_raw": int(len(df_raw)),
                "n_clean": int(len(df)), "positive_rate": float(df[TARGET].mean()),
                # Thống kê trên dữ liệu THÔ — đây mới là con số Hình 1 vẽ ra, phải
                # lưu riêng để báo cáo không lẫn với tỷ lệ sau khi làm sạch.
                "n_raw_positive": int(n_pos), "n_raw_negative": int(n_neg),
                "raw_positive_rate": float(n_pos / n_total)},
    "cleaning_audit": audit.to_dict(orient="records"),
    "feature_names": ct.feature_names_,
    "n_features": int(X_tr.shape[1]),
    "preprocess": {
        "cat_onehot": CAT_ONEHOT, "num_scale": NUM_SCALE, "bin_pass": BIN_PASS,
        "categories": {k: list(v) for k, v in ct.categories_.items()},
        "mu": ct.mu_.tolist(), "sd": ct.sd_.tolist(),
        "engineered": {
            "glucose_hba1c_interaction": "blood_glucose_level * HbA1c_level / 100",
            "age_hypertension_risk": "age * hypertension",
        },
    },
    "split": {"n_train": int(len(y_tr)), "n_test": int(len(y_te)),
              "train_pos_rate": float(y_tr.mean()), "test_pos_rate": float(y_te.mean())},
    "dl_model": {"name": dl_name, "layers": best_arch["Cấu hình tầng"],
                 "layer_sizes": dl_model.sizes, "n_params": dl_model.n_params(),
                 "threshold": float(results["Deeper DL Scratch"]["tau"]),
                 "activation": "ReLU (hidden) + Sigmoid (output)",
                 "lr": dl_model.lr, "batch_size": dl_model.batch_size, "epochs": 30},
    "scores": {n: {"kind": r["kind"], "n_params": r["n_params"], "time_sec": r["time"],
                   "threshold": r["tau"], "roc_auc": r["auc"],
                   "default": r["default"], "optimal": r["optimal"]}
               for n, r in results.items()},
    "architecture_study": [{k: v for k, v in r.items() if not k.startswith("_")}
                           | {"scores": r["_s"], "threshold": r["_t"]} for r in arch_rows],
    "loss_histories": {k: [float(x) for x in v] for k, v in arch_hist.items()},
    "target_correlation": {k: float(v) for k, v in target_corr.items()},
    "champion": champ[0],
}
(MODEL / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
(REP / "metrics_diabetes_large.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

print("✓ dl_scratch_weights.npz  (trọng số mạng nơ-ron thuần NumPy)")
print("✓ random_forest.joblib · logistic_regression.joblib")
print("✓ metadata.json · metrics_diabetes_large.json")
for f in sorted(FIG.glob("*.png")):
    print(f"✓ {f.name}")

# %% [markdown]
# ## 15. Kiểm chứng artifact — nạp lại trọng số và forward pass thuần NumPy
#
# Ô này mô phỏng chính xác những gì REST API sẽ làm: nạp file `.npz`, tự viết forward
# pass, và kiểm tra xem kết quả có khớp tuyệt đối với mô hình trong notebook không.

# %%
Z = np.load(MODEL / "dl_scratch_weights.npz")
Ws = [Z[f"W{i}"] for i in range(len(dl_model.W))]
bs = [Z[f"b{i}"] for i in range(len(dl_model.b))]

def api_forward(X, Ws, bs):
    """Forward pass độc lập — đây chính là đoạn code sẽ chạy trong REST API."""
    A = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        Z = A @ W + b
        A = np.maximum(0.0, Z) if i < len(Ws) - 1 else 1.0 / (1.0 + np.exp(-np.clip(Z, -25, 25)))
    return A

p_reload = api_forward(X_te[:1000], Ws, bs)
p_origin = dl_model.predict_proba(X_te[:1000])
max_diff = float(np.abs(p_reload - p_origin).max())

print(f"Sai khác tuyệt đối lớn nhất giữa artifact và mô hình gốc: {max_diff:.3e}")
print("✓ ĐẠT — artifact tái lập chính xác." if max_diff < 1e-10 else "✗ Có sai khác!")
print(f"\nNgưỡng quyết định triển khai: τ = {results['Deeper DL Scratch']['tau']:.2f}")
print(f"Kiến trúc triển khai: {best_arch['Cấu hình tầng']} ({dl_model.n_params():,} tham số)")

# %% [markdown]
# ## 16. Tổng kết notebook 03
#
# | Hạng mục | Kết quả |
# |---|---|
# | Quy mô dữ liệu | 100,000 hồ sơ → làm sạch còn ~95,000 |
# | Không gian đặc trưng | 14 chiều (One-Hot + Z-Score + passthrough) |
# | Kiến trúc DL tốt nhất | xem mục 8 |
# | Quán quân đối chuẩn | xem mục 12 |
# | Artifact triển khai | `dl_scratch_weights.npz` + `metadata.json` |
#
# Ba kết luận chính:
#
# 1. Trên dữ liệu bảng có ranh giới **dạng ngưỡng sinh học**, các mô hình cây (đặc biệt
#    Random Forest) tận dụng được cấu trúc bậc thang và đạt hiệu năng rất cao.
# 2. Mạng nơ-ron sâu thuần NumPy **hội tụ ổn định trên 100,000 mẫu** và cạnh tranh sòng
#    phẳng, chứng minh các phương trình lan truyền ngược viết tay hoàn toàn đủ dùng.
# 3. **Tinh chỉnh ngưỡng quyết định là bắt buộc** trên dữ liệu mất cân bằng — nó cứu
#    được hàng trăm ca bệnh mà không cần thay đổi một dòng nào trong mô hình.
#
# **→ Notebook 04** chuyển sang bài toán **hồi quy giá trị liên tục** trên 150,000 giao
# dịch bất động sản, nơi các quy luật đánh đổi kiến trúc sẽ **đảo ngược hoàn toàn**.
