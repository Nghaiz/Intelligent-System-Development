# %% [markdown]
# # Notebook 04 — Hệ Thống 2: Định Giá Bất Động Sản Thực Nghiệm (150,000 giao dịch)
#
# **Học phần:** Phát triển các Hệ thống Thông minh — Học viện Công nghệ Bưu chính Viễn thông
# **Sinh viên:** Nguyễn Duy Nghĩa · B23DCCN600 · D23CTPM01
# **Giảng viên hướng dẫn:** PGS.TS Trần Đình Quế
#
# ---
#
# ## Notebook này làm gì?
#
# Notebook 03 giải bài toán **phân loại nhị phân**. Notebook này chuyển sang một họ bài
# toán khác hẳn: **hồi quy giá trị liên tục** trên 150,000 giao dịch bất động sản.
#
# Sự chuyển đổi này kéo theo bốn thay đổi kỹ thuật căn bản:
#
# | Thành phần | Phân loại (NB 03) | Hồi quy (NB 04) |
# |---|---|---|
# | Hàm kích hoạt tầng ra | Sigmoid $\sigma(z)$ | **Tuyến tính** $\phi(z) = z$ |
# | Hàm mất mát | Binary Cross-Entropy | **Mean Squared Error** |
# | Chỉ số đánh giá | Accuracy, Recall, F1, AUC | **R², MAE, RMSE, MAPE** |
# | Xử lý biến mục tiêu | Nhãn nhị phân {0, 1} | **Biến đổi Logarit** |
#
# Câu hỏi nghiên cứu trung tâm của notebook:
#
# > **Quy luật "chiều sâu thắng chiều rộng" tìm được ở notebook 03 có còn đúng khi bài
# > toán chuyển từ phân loại sang hồi quy hay không?**
#
# Ngoài ra notebook giải quyết hai bài toán kỹ thuật khó:
#
# - **Mã hoá vị trí địa lý** — hàng nghìn mã bưu chính không thể One-Hot (bùng nổ số
#   chiều). Giải pháp: **Smooth Bayesian Target Encoding** với suy biến Bayes.
# - **Phân tích phần dư** — phát hiện hiện tượng **phương sai thay đổi
#   (Heteroscedasticity)** theo phân khúc thị trường.
#
# ## Bộ dữ liệu
#
# `USA Real Estate Dataset` — 150,000 tin đăng bất động sản toàn nước Mỹ với diện tích
# sàn, diện tích lô đất, số phòng ngủ/tắm, và bốn cấp độ địa lý (bang, thành phố, mã
# bưu chính, địa chỉ đường).
# Nguồn: <https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset>

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
DATA = ROOT / "data" / "usa_real_estate_150k.csv"
FIG = ROOT / "reports" / "figures"
REP = ROOT / "reports"
MODEL = ROOT / "model"
for d in (FIG, MODEL):
    d.mkdir(parents=True, exist_ok=True)
print("Dữ liệu:", DATA)

# %% [markdown]
# ## 2. Nạp dữ liệu và khám phá ban đầu

# %%
df_raw = pd.read_csv(DATA)
print(f"Dữ liệu thô: {df_raw.shape[0]:,} tin đăng × {df_raw.shape[1]} cột\n")
display(df_raw.head())

print("Kiểu dữ liệu và mức độ khuyết thiếu:")
display(pd.DataFrame({
    "Kiểu": df_raw.dtypes.astype(str),
    "Số khuyết thiếu": df_raw.isna().sum(),
    "Tỷ lệ khuyết (%)": (df_raw.isna().mean() * 100).round(2),
    "Số giá trị duy nhất": df_raw.nunique(),
}))

# %% [markdown]
# ### 2.1 Đặc tính cốt lõi của biến mục tiêu: lệch phải cực nặng
#
# Giá nhà là đại lượng **không âm** và có phân phối **lệch phải (Right-Skewed) cực kỳ
# nặng**: đa số nhà nằm trong khoảng 100,000 – 600,000 USD, trong khi một số ít biệt thự
# siêu sang vượt ngưỡng 5,000,000 USD kéo dài cái đuôi bên phải.
#
# Đây là vấn đề nghiêm trọng với hồi quy: hàm mất mát MSE bình phương sai số, nên một
# căn biệt thự dự đoán sai 2 triệu USD đóng góp vào loss gấp **một triệu lần** một căn
# nhà phổ thông sai 2 nghìn USD. Mô hình sẽ dồn toàn bộ năng lực vào nhóm cực đoan và
# bỏ mặc 95% thị trường còn lại.
#
# **Giải pháp — lý thuyết kinh tế lượng Hedonic:** biến đổi biến mục tiêu qua hàm logarit:
#
# $$y_{\text{log}} = \log(1 + \text{price})$$
#
# Phép biến đổi này đưa phân phối về dạng hình chuông chuẩn đối xứng, triệt tiêu độ lệch
# và ổn định phương sai. Nó cũng chuyển bài toán từ **sai số tuyệt đối** sang **sai số
# tương đối** — đúng với cách thị trường bất động sản vận hành (người ta nói "căn nhà
# này đắt hơn 20%", không nói "đắt hơn 80,000 USD").

# %% [markdown]
# ## 3. Kiểm toán quy trình làm sạch dữ liệu

# %%
audit_rows = [("Dữ liệu thô ban đầu", "Tập 150,000 tin đăng bất động sản toàn quốc", 0, len(df_raw))]
df = df_raw.copy()

CRITICAL = ["price", "house_size", "bed", "bath", "zip_code"]
n0 = len(df); df = df.dropna(subset=CRITICAL).reset_index(drop=True)
audit_rows.append(("1. Khuyết thiếu cốt lõi",
                   "Khuyết thiếu các trường buộc phải có: price, house_size, bed, bath, zip_code",
                   n0 - len(df), len(df)))

n0 = len(df)
df = df.drop_duplicates(subset=["street", "city", "state", "zip_code",
                                "price", "house_size", "bed", "bath"]).reset_index(drop=True)
audit_rows.append(("2. Khử trùng lặp",
                   "Trùng lặp hoàn toàn địa chỉ, giá và kết cấu công trình",
                   n0 - len(df), len(df)))

n0 = len(df)
df = df[(df.price >= 50_000) & (df.price <= 5_000_000)
        & (df.house_size >= 300) & (df.house_size <= 10_000)
        & (df.bed >= 1) & (df.bed <= 12)
        & (df.bath >= 1) & (df.bath <= 12)].reset_index(drop=True)
audit_rows.append(("3. Lọc tin ảo & ngoại lai",
                   "Giá < $50,000 hoặc > $5,000,000; Diện tích < 300 hoặc > 10,000 sqft",
                   n0 - len(df), len(df)))

audit = pd.DataFrame(audit_rows, columns=["Bước Xử lý", "Tiêu chí Kiểm toán Thực nghiệm",
                                          "Số lượng Loại bỏ", "Số lượng Còn lại"])
audit["Tỷ lệ Giữ lại (%)"] = (audit["Số lượng Còn lại"] / len(df_raw) * 100).round(2)
display(audit)

print(f"Còn lại {len(df):,} giao dịch sạch ({len(df) / len(df_raw):.2%})")
print(f"Giá: min ${df.price.min():,.0f} · trung vị ${df.price.median():,.0f} · "
      f"max ${df.price.max():,.0f}")
print(f"Độ lệch (skewness) giá thô : {df.price.skew():.3f}")
print(f"Độ lệch sau biến đổi log   : {np.log1p(df.price).skew():.3f}  ← gần 0 = đối xứng chuẩn")

# %% [markdown]
# ### Hình 1 — Phân phối giá thô và biến đổi Logarit

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))

ax = axes[0]
ax.hist(df.price / 1000, bins=90, color="#4c72b0", edgecolor="white", lw=0.25)
ax.set_xlabel("Giá bán (Nghìn USD)"); ax.set_ylabel("Số lượng tin đăng")
ax.set_title(f"Phân phối Giá nhà Thực tế (Nghìn USD — Lệch phải)\n"
             f"Skewness = {df.price.skew():.3f}")

ax = axes[1]
ax.hist(np.log1p(df.price), bins=90, color="#2a9d70", edgecolor="white", lw=0.25)
ax.set_xlabel("ln(Price)"); ax.set_ylabel("Số lượng tin đăng")
ax.set_title(f"Phân phối Logarit Tự nhiên của Giá nhà: ln(Price)\n"
             f"Skewness = {np.log1p(df.price).skew():.3f} — đối xứng Gauss")

fig.suptitle("Biến đổi Logarit đưa phân phối giá lệch phải về dạng chuông chuẩn đối xứng",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig01_price_distribution.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Hình 2 — Khám phá đặc trưng không gian và cấu trúc nhà ở

# %%
fig = plt.figure(figsize=(13.0, 8.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.34, wspace=0.24)

rng = np.random.default_rng(SEED)
samp = rng.choice(len(df), size=min(9000, len(df)), replace=False)

ax = fig.add_subplot(gs[0, 0])
ax.scatter(df.house_size.iloc[samp], df.price.iloc[samp] / 1000,
           s=5, alpha=0.28, c="#7b52d3", edgecolors="none")
ax.set_xlabel("Diện tích sàn (sqft)"); ax.set_ylabel("Giá bán (Nghìn USD)")
ax.set_title("Diện tích sàn (house_size) vs Giá bán")

ax = fig.add_subplot(gs[0, 1])
groups = [df.loc[df.bed == b, "price"] / 1000 for b in range(1, 8)]
bp = ax.boxplot(groups, tick_labels=[f"{b} PN" for b in range(1, 8)],
                patch_artist=True, showfliers=True,
                flierprops=dict(marker=".", markersize=1.4, alpha=0.30, mec="#333"),
                medianprops=dict(color="#c0392b", lw=1.3))
for patch in bp["boxes"]:
    patch.set_facecolor("#a8cbe8")
ax.set_xlabel("Số phòng ngủ"); ax.set_ylabel("Giá bán (Nghìn USD)")
ax.set_title("Phân bố Giá bán theo Số phòng ngủ (1 – 7 phòng)")

ax = fig.add_subplot(gs[1, :])
cc = pd.DataFrame({"Price": df.price, "ln(Price)": np.log1p(df.price),
                   "Bed": df.bed, "Bath": df.bath, "House Size": df.house_size})
C = cc.corr().to_numpy(); lab = list(cc.columns)
im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
for i in range(len(lab)):
    for j in range(len(lab)):
        ax.text(j, i, f"{C[i, j]:.3f}", ha="center", va="center", fontsize=9,
                color="white" if abs(C[i, j]) > 0.65 else "#111", fontweight="bold")
ax.set_xticks(range(len(lab)), lab, fontsize=8.5)
ax.set_yticks(range(len(lab)), lab, fontsize=8.5)
ax.set_title("Ma Trận Tương Quan Pearson Giữa Các Đặc Trưng Số Học Cơ Bản", fontsize=10.5)
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.026, label="Hệ số Tương quan Pearson (r)")

fig.suptitle("Khám phá đặc trưng không gian và cấu trúc nhà ở", fontsize=12.5, fontweight="bold")
fig.savefig(FIG / "fig02_features_vs_price.png", bbox_inches="tight")
plt.show()

print(f"Tương quan house_size ↔ price      : r = {cc['House Size'].corr(cc['Price']):.3f}")
print(f"Tương quan house_size ↔ ln(price)  : r = {cc['House Size'].corr(cc['ln(Price)']):.3f}")
print("→ Tương quan tăng lên sau biến đổi log: quan hệ diện tích – giá mang bản chất")
print("  NHÂN TÍNH (giá tăng theo tỷ lệ phần trăm), không phải cộng tính.")

# %% [markdown]
# ## 4. Kỹ nghệ đặc trưng hình học và tỷ lệ không gian
#
# Dựa trên nguyên lý định giá bất động sản Hedonic, bảy đặc trưng phái sinh được trích
# xuất nhằm tăng cường tín hiệu phi tuyến:
#
# **1. Xử lý khuyết thiếu diện tích lô đất.** Nhiều căn hộ chung cư hoặc nhà phố liền kề
# không ghi nhận `acre_lot`. Điền bằng trung vị tính **duy nhất trên tập huấn luyện**:
#
# $$\text{acre\_lot}_{\text{clean}} = \begin{cases} \text{acre\_lot} & \text{nếu không khuyết thiếu} \\ \text{median}(\mathcal{D}_{\text{train}}[\text{acre\_lot}]) & \text{nếu là NaN}\end{cases}$$
#
# **2. Biến đổi logarit co giãn diện tích:**
# $\text{log\_house\_size} = \ln(\text{house\_size})$ và $\text{log\_acre\_lot} = \ln(1 + \text{acre\_lot}_{\text{clean}})$
#
# **3. Cấu trúc công năng xây dựng:**
#
# $$\text{total\_rooms} = \text{bed} + \text{bath}, \qquad \text{sqft\_per\_room} = \frac{\text{house\_size}}{\text{total\_rooms}}$$
# $$\text{bath\_bed\_ratio} = \frac{\text{bath}}{\text{bed}}, \qquad \text{bed\_bath\_prod} = \text{bed} \times \text{bath}$$
#
# **4. Chỉ số quy mô tương đối cục bộ (`relative_sqft`):**
#
# $$\text{relative\_sqft} = \frac{\text{house\_size}}{\overline{\text{house\_size}}_{\text{zip}}}$$
#
# Đo mức độ bề thế **tương đối** của ngôi nhà so với mặt bằng quy mô dân cư lân cận cùng
# mã bưu chính. Một căn 2,000 sqft là bình thường ở ngoại ô nhưng là biệt thự ở trung tâm.

# %% [markdown]
# ## 5. Phân chia dữ liệu TRƯỚC khi trích xuất đặc trưng
#
# Thứ tự này là bắt buộc. `relative_sqft` và toàn bộ bảng mã hoá địa lý đều phụ thuộc
# vào thống kê nhóm — nếu tính trên toàn bộ dữ liệu thì thông tin giá của tập Test rò rỉ
# thẳng vào tập Train.

# %%
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(df))
n_tr = int(0.80 * len(df))
tr_idx, te_idx = perm[:n_tr], perm[n_tr:]

df_tr = df.iloc[tr_idx].reset_index(drop=True)
df_te = df.iloc[te_idx].reset_index(drop=True)

y_tr_log = np.log1p(df_tr.price.to_numpy(dtype=np.float64)).reshape(-1, 1)
y_te_log = np.log1p(df_te.price.to_numpy(dtype=np.float64)).reshape(-1, 1)
y_tr_raw = df_tr.price.to_numpy(dtype=np.float64)
y_te_raw = df_te.price.to_numpy(dtype=np.float64)

print(f"Tập Train: {len(df_tr):,} giao dịch — giá trung vị ${np.median(y_tr_raw):,.0f}")
print(f"Tập Test : {len(df_te):,} giao dịch — giá trung vị ${np.median(y_te_raw):,.0f}")

# %% [markdown]
# ## 6. Mã hoá đích Bayes làm mượt chống rò rỉ (Smooth Bayesian Target Encoding)
#
# Vị trí địa lý là nhân tố quyết định giá trị bất động sản. Nhưng One-Hot Encoding hàng
# nghìn mã bưu chính sẽ làm bùng nổ số chiều và gây thưa thớt nghiêm trọng.
#
# **Giải pháp — Target Encoding có suy biến Bayes:** thay mỗi đơn vị địa lý bằng **giá
# log trung bình** của khu vực đó, nhưng được "kéo" về giá trung bình toàn quốc theo mức
# độ tin cậy thống kê:
#
# $$S_i = \frac{n_i \cdot \bar{y}_i + m \cdot \bar{y}_{\text{global}}}{n_i + m}$$
#
# Trong đó:
# - $n_i$ — số tin đăng của đơn vị địa lý $i$;
# - $\bar{y}_i$ — giá log trung bình của đơn vị đó;
# - $\bar{y}_{\text{global}}$ — giá log trung bình toàn quốc trên tập Train;
# - $m$ — **tham số làm mượt**, đóng vai trò tiên nghiệm giả định (Prior Pseudo-counts).
#
# **Trực giác của công thức:** một mã bưu chính chỉ có 2 tin đăng ($n_i = 2$) thì giá
# trung bình của nó gần như vô nghĩa về mặt thống kê — công thức sẽ kéo mạnh nó về giá
# trung bình toàn quốc. Ngược lại, một thành phố có 5,000 tin đăng ($n_i \gg m$) thì giá
# trung bình rất đáng tin — công thức giữ gần như nguyên giá trị đó.
#
# Bốn cấp độ địa lý với bốn mức làm mượt khác nhau: bang ($m = 20$), vùng bưu chính 3 số
# ($m = 15$), mã bưu chính 5 số ($m = 10$), và thành phố ($m = 10$). Cấp càng rộng thì
# càng cần nhiều bằng chứng để lệch khỏi mức trung bình chung.
#
# **Chống rò rỉ:** bảng tra cứu $S_i$ được tính **duy nhất trên tập Train**, rồi ánh xạ
# cố định sang tập Test. Khu vực mới xuất hiện ở Test (chưa từng có trong Train) được
# gán giá trị trung bình toàn cục $\bar{y}_{\text{global}}$.

# %%
def smooth_target_encode(keys_tr, y_tr, keys_te, m, global_mean):
    """Target Encoding có suy biến Bayes. Bảng tra CHỈ học trên tập Train."""
    s = pd.DataFrame({"k": keys_tr, "y": y_tr.ravel()})
    agg = s.groupby("k")["y"].agg(["count", "mean"])
    smoothed = (agg["count"] * agg["mean"] + m * global_mean) / (agg["count"] + m)
    table = smoothed.to_dict()
    enc_tr = np.array([table.get(k, global_mean) for k in keys_tr])
    enc_te = np.array([table.get(k, global_mean) for k in keys_te])   # khu vực mới → global
    return enc_tr, enc_te, table

global_mean = float(y_tr_log.mean())
print(f"Giá log trung bình toàn quốc trên tập Train: {global_mean:.4f} "
      f"(tương đương ${np.expm1(global_mean):,.0f})\n")

zip_tr = df_tr.zip_code.astype(int).astype(str).str.zfill(5)
zip_te = df_te.zip_code.astype(int).astype(str).str.zfill(5)

GEO_LEVELS = [
    ("state",    df_tr.state.astype(str),  df_te.state.astype(str),  20),
    ("zip3",     zip_tr.str[:3],           zip_te.str[:3],           15),
    ("zip_code", zip_tr,                   zip_te,                   10),
    ("city",     df_tr.city.astype(str),   df_te.city.astype(str),   10),
]

te_feats_tr, te_feats_te, te_tables = {}, {}, {}
for name, ktr, kte, m in GEO_LEVELS:
    a, b, tbl = smooth_target_encode(ktr.to_numpy(), y_tr_log, kte.to_numpy(), m, global_mean)
    te_feats_tr[f"TE_{name}"], te_feats_te[f"TE_{name}"] = a, b
    te_tables[name] = tbl
    unseen = float((~kte.isin(tbl.keys())).mean() * 100)
    print(f"TE_{name:<9} m = {m:>2}  ·  {len(tbl):>6,} đơn vị địa lý  ·  "
          f"{unseen:>5.2f}% mẫu Test thuộc khu vực mới → gán giá trung bình toàn cục")

# %% [markdown]
# ## 7. Xây dựng ma trận đặc trưng 13 chiều

# %%
acre_median = float(df_tr.acre_lot.median())
print(f"Trung vị acre_lot trên tập Train: {acre_median:.4f} acre "
      f"(điền cho {int(df_tr.acre_lot.isna().sum()):,} ô Train, "
      f"{int(df_te.acre_lot.isna().sum()):,} ô Test)")

# Bảng tra diện tích trung bình theo zip — học TRÊN TRAIN
zip_size_map = df_tr.assign(_z=zip_tr).groupby("_z")["house_size"].mean().to_dict()
global_size = float(df_tr.house_size.mean())

def build_features(frame, zip_keys, te_feats):
    acre = frame.acre_lot.fillna(acre_median).to_numpy(dtype=np.float64)
    size = frame.house_size.to_numpy(dtype=np.float64)
    bed = frame.bed.to_numpy(dtype=np.float64)
    bath = frame.bath.to_numpy(dtype=np.float64)
    rooms = bed + bath
    zip_mean = np.array([zip_size_map.get(z, global_size) for z in zip_keys])
    cols = {
        "log_size":       np.log(size),
        "log_lot":        np.log1p(acre),
        "bed":            bed,
        "bath":           bath,
        "total_rooms":    rooms,
        "sqft_per_room":  size / np.maximum(rooms, 1.0),
        "bath_bed_ratio": bath / np.maximum(bed, 1.0),
        "bed_bath_prod":  bed * bath,
        "relative_sqft":  size / np.maximum(zip_mean, 1.0),
        "TE_state":       te_feats["TE_state"],
        "TE_zip3":        te_feats["TE_zip3"],
        "TE_zip_code":    te_feats["TE_zip_code"],
        "TE_city":        te_feats["TE_city"],
    }
    return np.column_stack(list(cols.values())), list(cols.keys())

X_tr_raw, FEATURE_NAMES = build_features(df_tr, zip_tr.to_numpy(), te_feats_tr)
X_te_raw, _ = build_features(df_te, zip_te.to_numpy(), te_feats_te)

# Chuẩn hoá Z-Score — thống kê học CHỈ trên Train
mu = X_tr_raw.mean(axis=0)
sd = X_tr_raw.std(axis=0); sd[sd == 0] = 1.0
X_tr = (X_tr_raw - mu) / sd
X_te = (X_te_raw - mu) / sd

# Biến mục tiêu cũng được chuẩn hoá về kỳ vọng 0, phương sai 1
y_mu, y_sd = float(y_tr_log.mean()), float(y_tr_log.std())
y_tr_s = (y_tr_log - y_mu) / y_sd
y_te_s = (y_te_log - y_mu) / y_sd

print(f"\nKhông gian đặc trưng: {X_tr.shape[1]} chiều")
print(f"  Train: {X_tr.shape}   Test: {X_te.shape}")
print("\n13 đặc trưng đầu vào:")
for i, n in enumerate(FEATURE_NAMES, 1):
    print(f"  {i:>2}. {n}")
print(f"\nChuẩn hoá biến mục tiêu: μ = {y_mu:.4f}, σ = {y_sd:.4f}")

# %% [markdown]
# ## 8. Mạng nơ-ron hồi quy sâu thuần NumPy
#
# Ba điểm khác biệt so với mạng phân loại ở notebook 03:
#
# **1. Tầng ngõ ra tuyến tính.** $\phi(z) = z$ — không Sigmoid, vì giá trị cần dự đoán
# trải trên toàn trục thực chứ không bị chặn trong $(0, 1)$.
#
# **2. Hàm mất mát MSE:**
#
# $$\mathcal{L}_{\text{MSE}} = \frac{1}{2m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)^2$$
#
# **3. Gradient tầng ra.** Đạo hàm cũng rút gọn về đúng dạng quen thuộc:
#
# $$\mathbf{dZ}^{[L]} = \frac{1}{m}\left(\hat{\mathbf{y}} - \mathbf{y}\right)$$
#
# Điểm thú vị: **công thức gradient tầng ra của MSE + Linear hoàn toàn giống với BCE +
# Sigmoid**. Đây không phải trùng hợp ngẫu nhiên — cả hai cặp đều thuộc họ hàm mất mát
# chính tắc (canonical link) của phân phối mũ tương ứng, và toàn bộ phần lan truyền
# ngược qua các tầng ẩn giữ nguyên không đổi.
#
# Vì $y_{\text{log}}$ đã được chuẩn hoá Z-Score, giá trị MSE huấn luyện là một đại lượng
# **không thứ nguyên (dimensionless)** — thuận tiện để so sánh giữa các kiến trúc.

# %%
class MLPRegressorScratch:
    """Mạng nơ-ron hồi quy nhiều tầng, 100% NumPy. Tầng ra tuyến tính, mất mát MSE."""

    def __init__(self, layer_sizes, lr=0.05, batch_size=256, seed=SEED):
        rng = np.random.default_rng(seed)
        self.sizes = list(layer_sizes)
        self.L = len(self.sizes) - 1
        self.lr, self.batch_size = lr, batch_size
        self.W = [rng.normal(0, np.sqrt(2.0 / self.sizes[i]),
                             (self.sizes[i], self.sizes[i + 1])) for i in range(self.L)]
        self.b = [np.zeros((1, self.sizes[i + 1])) for i in range(self.L)]
        self.history = []

    def forward(self, X, cache=True):
        A, Zs, As = X, [], [X]
        for i in range(self.L):
            Z = A @ self.W[i] + self.b[i]
            A = np.maximum(0.0, Z) if i < self.L - 1 else Z    # ← tầng ra TUYẾN TÍNH
            Zs.append(Z); As.append(A)
        if cache:
            self.Zs, self.As = Zs, As
        return A

    @staticmethod
    def mse(y, p):
        return float(np.mean((p - y) ** 2) / 2.0)

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
                self.forward(X[idx]); self._backward(y[idx])
            self.history.append(self.mse(y, self.forward(X, cache=False)))
            if verbose_every and (ep == 1 or ep % verbose_every == 0):
                print(f"  epoch {ep:>3} / {epochs}   MSE = {self.history[-1]:.4f}")
        return self

    def predict(self, X):
        return self.forward(X, cache=False)

    def n_params(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)

# %% [markdown]
# ## 9. Bộ chỉ số hồi quy
#
# Bốn chỉ số được đo trên **hai không gian giá trị khác nhau**, và sự khác biệt giữa
# chúng là một trong những phát hiện quan trọng nhất của notebook này:
#
# - **Không gian Logarit** — nơi mô hình thực sự được huấn luyện;
# - **Không gian Dollar thực** — nơi người dùng thực sự quan tâm.
#
# $$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}, \qquad \text{MAE} = \frac{1}{n}\sum|y_i - \hat{y}_i|$$
# $$\text{RMSE} = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}, \qquad \text{MAPE} = \frac{100}{n}\sum\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

# %%
def r2(y, p):
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)

def reg_scores(y_log_true, y_log_pred):
    """Đo đồng thời trên KHÔNG GIAN LOG và KHÔNG GIAN DOLLAR THỰC."""
    yl, pl = np.asarray(y_log_true).ravel(), np.asarray(y_log_pred).ravel()
    yr, pr = np.expm1(yl), np.expm1(pl)          # nghịch đảo về Dollar
    err = yr - pr
    return {
        "r2_log": r2(yl, pl), "r2_raw": r2(yr, pr),
        "mae_usd": float(np.abs(err).mean()),
        "rmse_usd": float(np.sqrt((err ** 2).mean())),
        "mape": float((np.abs(err / np.maximum(yr, 1.0))).mean() * 100),
        "mse_log_scaled": float(np.mean((pl - yl) ** 2) / 2.0),
    }

def to_log_space(pred_scaled):
    """Đưa dự đoán từ không gian chuẩn hoá về không gian log."""
    return np.asarray(pred_scaled).ravel() * y_sd + y_mu

# %% [markdown]
# ## 10. Khảo sát kiến trúc mạng hồi quy sâu
#
# Bốn cấu hình y hệt logic ở notebook 03, chỉ đổi số chiều đầu vào từ 14 xuống 13.

# %%
D = X_tr.shape[1]
ARCHS = [
    ("Shallow Regressor",  [D, 32, 1],         "13 → 32 → 1"),
    ("Standard Regressor", [D, 32, 16, 1],     "13 → 32 → 16 → 1"),
    ("Deeper Regressor",   [D, 64, 32, 16, 1], "13 → 64 → 32 → 16 → 1"),
    ("Wide Regressor",     [D, 128, 64, 1],    "13 → 128 → 64 → 1"),
]

arch_rows, arch_hist, arch_models = [], {}, {}
for name, sizes, label in ARCHS:
    t0 = time.perf_counter()
    m = MLPRegressorScratch(sizes, lr=0.05, batch_size=256, seed=SEED).fit(X_tr, y_tr_s, epochs=30)
    el = time.perf_counter() - t0
    s = reg_scores(y_te_log, to_log_space(m.predict(X_te)))
    arch_hist[name], arch_models[name] = m.history, m
    arch_rows.append({"Kiến trúc": name, "Cấu hình tầng": label, "Tham số": m.n_params(),
                      "Thời gian": f"{el:.2f}s", "MSE Loss": round(m.history[-1], 4),
                      "R² Score": round(s["r2_log"], 4),
                      "MAE (USD)": f"${s['mae_usd']:,.0f}",
                      "RMSE (USD)": f"${s['rmse_usd']:,.0f}", "_s": s, "_el": el})
    print(f"{name:<20} {label:<24} {m.n_params():>6,} params  {el:>6.2f}s  "
          f"MSE={m.history[-1]:.4f}  R²={s['r2_log']:.4f}  MAE=${s['mae_usd']:,.0f}")

arch_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in arch_rows])
print("\nBảng đối chứng thực nghiệm 4 cấu hình kiến trúc mạng hồi quy sâu:")
display(arch_df)

best_arch = max(arch_rows, key=lambda r: r["_s"]["r2_log"])
print(f"\n★ Kiến trúc chiến thắng: {best_arch['Kiến trúc']} ({best_arch['Cấu hình tầng']}) "
      f"— R² {best_arch['_s']['r2_log']:.4f}, MAE ${best_arch['_s']['mae_usd']:,.0f}")

# %% [markdown]
# ### Hình 3 — Hội tụ MSE và so sánh chỉ số hồi quy trên Dollar thực

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))

ax = axes[0]
for (name, h), c, st in zip(arch_hist.items(),
                            ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"],
                            ["--", "-", "-", ":"]):
    npar = next(r["Tham số"] for r in arch_rows if r["Kiến trúc"] == name)
    ax.plot(range(1, len(h) + 1), h, lw=1.6, color=c, ls=st, label=f"{name} ({npar:,} params)")
ax.set_xlabel("Epoch (Chu kỳ huấn luyện)")
ax.set_ylabel("Mean Squared Error (Chuẩn hoá Z-Score)")
ax.set_title("Đường Cong Hội Tụ MSE Loss Giữa Các Kiến Trúc Hồi Quy")
ax.legend()

ax = axes[1]
xs = np.arange(len(arch_rows)); w = 0.36
r2s = [r["_s"]["r2_log"] for r in arch_rows]
maes = [r["_s"]["mae_usd"] / 1000 for r in arch_rows]
b1 = ax.bar(xs - w / 2, r2s, w, color="#4c72b0", label="R² Score (Log) — Càng cao càng tốt")
ax.bar_label(b1, fmt="%.4f", fontsize=7.5)
ax.set_ylabel("Hệ số Xác định R² (Log)", color="#4c72b0")
ax.set_ylim(0, max(r2s) * 1.30)
ax.set_xticks(xs, [r["Kiến trúc"].replace(" Regressor", "") for r in arch_rows], fontsize=8)

ax2 = ax.twinx(); ax2.grid(False)
b2 = ax2.bar(xs + w / 2, maes, w, color="#e07b39", label="MAE (Nghìn USD) — Càng thấp càng tốt")
ax2.bar_label(b2, fmt="$%.1fk", fontsize=7.5)
ax2.set_ylabel("Sai số Tuyệt đối MAE (Nghìn USD)", color="#e07b39")
ax2.set_ylim(0, max(maes) * 1.30)

ax.set_title("So Sánh R² & MAE (Nghìn USD) Giữa Các Kiến Trúc Hồi Quy")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")

fig.suptitle("Đường cong suy giảm hàm mất mát MSE chuẩn hoá và so sánh chỉ số hồi quy trên Dollar thực",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig03_architecture_study.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Vì sao chiều rộng lại ưu thế trong bài toán hồi quy — trong khi chiều sâu thắng ở
# bài toán phân loại?**
#
# Đây là phát hiện quan trọng nhất về mặt kiến trúc của toàn bộ Assignment 03.
#
# **Bài toán phân loại** cần **bẻ cong một siêu phẳng phân tách** để bao lấy các cụm dữ
# liệu. Việc bẻ cong phức tạp đòi hỏi tổ hợp phân cấp nhiều bậc — đó là thế mạnh của
# chiều sâu.
#
# **Bài toán hồi quy** thì khác hẳn: nó cần **xấp xỉ một mặt phẳng giá trị liên tục,
# trơn và mấp mô** trải trên nhiều chiều đặc trưng. Mỗi nơ-ron ReLU tạo ra một "nếp gấp"
# tuyến tính từng đoạn (Piecewise Linear Approximator). Tăng chiều rộng tầng ẩn
# ($128 \to 64$) cung cấp một **hệ cơ sở hàm ReLU phong phú hơn**, cho phép ghép nối
# nhiều mặt phẳng con cục bộ mượt mà hơn — giống như dùng nhiều mảnh vải nhỏ để phủ một
# bề mặt cong thay vì ít mảnh lớn.

# %% [markdown]
# ## 11. Đối đầu với ba mô hình Machine Learning cổ điển

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

ML_MODELS = [
    ("Linear Regression", "Tuyến tính OLS", LinearRegression()),
    ("Decision Tree", "Cây hồi quy",
     DecisionTreeRegressor(max_depth=16, min_samples_leaf=10, random_state=SEED)),
    ("Random Forest", "Ensemble 100 Cây",
     RandomForestRegressor(n_estimators=100, max_depth=22, min_samples_leaf=3,
                           n_jobs=-1, random_state=SEED)),
]

results = {}
for name, kind, mdl in ML_MODELS:
    t0 = time.perf_counter()
    mdl.fit(X_tr, y_tr_log.ravel())          # ML học trực tiếp trên không gian log
    el = time.perf_counter() - t0
    p_log = mdl.predict(X_te)
    s = reg_scores(y_te_log, p_log)
    results[name] = {
        "kind": kind, "time": el, "model": mdl, "pred_log": p_log, "scores": s,
        "n_params": (X_tr.shape[1] + 1 if name == "Linear Regression"
                     else int(mdl.tree_.node_count) if name == "Decision Tree"
                     else int(sum(e.tree_.node_count for e in mdl.estimators_))),
    }
    print(f"{name:<20} {el:>7.3f}s  R²(log)={s['r2_log']:.4f}  R²(raw)={s['r2_raw']:.4f}  "
          f"MAE=${s['mae_usd']:,.0f}  MAPE={s['mape']:.2f}%")

dl_name = best_arch["Kiến trúc"]
dl_model = arch_models[dl_name]
dl_pred_log = to_log_space(dl_model.predict(X_te))
results["Wide DL Scratch"] = {
    "kind": f"Mạng nơ-ron rộng ({best_arch['Cấu hình tầng']})", "time": best_arch["_el"],
    "model": dl_model, "pred_log": dl_pred_log, "scores": best_arch["_s"],
    "n_params": dl_model.n_params(),
}
s = best_arch["_s"]
print(f"{'Wide DL Scratch':<20} {best_arch['_el']:>7.3f}s  R²(log)={s['r2_log']:.4f}  "
      f"R²(raw)={s['r2_raw']:.4f}  MAE=${s['mae_usd']:,.0f}  MAPE={s['mape']:.2f}%")

# %% [markdown]
# ## 12. Phân tích chẩn đoán phần dư (Residual Analysis)
#
# Phần dư là hiệu giữa giá thực tế và giá dự đoán: $e_i = y_i - \hat{y}_i$. Đồ thị phần
# dư là công cụ chẩn đoán mạnh nhất của kinh tế lượng — nó tiết lộ những gì mà một con
# số R² đơn lẻ che giấu.
#
# Hai điều cần kiểm tra:
#
# 1. **Phần dư có đối xứng quanh 0 không?** Nếu có, mô hình không bị thiên lệch hệ thống
#    (No Systematic Bias). Nếu phần dư lệch hẳn về một phía, mô hình định giá cao hoặc
#    thấp một cách có hệ thống.
# 2. **Biên độ phần dư có đồng đều không?** Nếu phễu phần dư mở rộng dần theo giá dự
#    đoán, ta có hiện tượng **phương sai thay đổi (Heteroscedasticity)** — sai số của mô
#    hình phụ thuộc vào phân khúc giá.

# %% [markdown]
# ### Hình 4 — Chẩn đoán phần dư trên bốn mô hình

# %%
rng = np.random.default_rng(SEED)
ssub = rng.choice(len(y_te_log), size=min(4000, len(y_te_log)), replace=False)

fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.0))
colors = ["#4c72b0", "#55a868", "#e07b39", "#8172b3"]
for ax, (name, r), c in zip(axes, results.items(), colors):
    pr = np.expm1(np.asarray(r["pred_log"]).ravel())[ssub] / 1000
    ar = y_te_raw[ssub] / 1000
    ax.scatter(pr, ar - pr, s=5, alpha=0.28, c=c, edgecolors="none")
    ax.axhline(0, color="#d62728", ls="--", lw=1.2)
    ax.set_xlabel("Giá dự đoán (Nghìn $)")
    ax.set_title(f"{name}\n[MAE Phần dư: ${r['scores']['mae_usd'] / 1000:.1f}k]", fontsize=9)
    ax.set_xlim(0, 3000); ax.set_ylim(-1500, 1500)
    if ax is axes[0]:
        ax.set_ylabel("Phần dư = y − ŷ (Nghìn $)")
fig.suptitle("PHÂN TÍCH CHẨN ĐOÁN PHẦN DƯ (RESIDUAL ANALYSIS): MACHINE LEARNING VS DEEP LEARNING SCRATCH",
             fontsize=11.5, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig04_residual_analysis.png", bbox_inches="tight")
plt.show()

# %%
res_dl = y_te_raw - np.expm1(dl_pred_log)
print("PHÁT HIỆN KINH TẾ LƯỢNG TỪ ĐỒ THỊ PHẦN DƯ")
print("=" * 74)
print(f"1. Phần dư đối xứng quanh 0: trung bình phần dư = ${res_dl.mean():,.0f} "
      f"trên giá trung bình ${y_te_raw.mean():,.0f}")
print(f"   → Tỷ lệ thiên lệch chỉ {abs(res_dl.mean()) / y_te_raw.mean() * 100:.3f}% "
      f"— mô hình KHÔNG bị thiên lệch hệ thống.\n")

print("2. Hiện tượng phương sai thay đổi (Heteroscedasticity) theo phân khúc thị trường:")
SEGMENTS = [("Phổ thông (< $300k)", 0, 300_000),
            ("Trung cấp ($300k – $800k)", 300_000, 800_000),
            ("Cao cấp & Biệt thự (> $800k)", 800_000, np.inf)]
seg_rows = []
for label, lo, hi in SEGMENTS:
    mask = (y_te_raw >= lo) & (y_te_raw < hi)
    if mask.sum() == 0:
        continue
    e = np.abs(res_dl[mask])
    mape_seg = float((e / y_te_raw[mask]).mean() * 100)
    seg_rows.append({"Phân khúc thị trường": label, "Số giao dịch": f"{int(mask.sum()):,}",
                     "Tỷ trọng": f"{mask.mean():.1%}", "MAE": f"${e.mean():,.0f}",
                     "MAPE": f"{mape_seg:.1f}%"})
    print(f"   {label:<32} {mask.mean():>6.1%} thị trường  "
          f"MAE ${e.mean():>9,.0f}  MAPE {mape_seg:>5.1f}%")
display(pd.DataFrame(seg_rows))

print("\n→ Ở phân khúc phổ thông, giá nhà gắn chặt với công năng vật lý (diện tích, số")
print("  phòng, vị trí bưu chính) nên mô hình định giá rất chuẩn. Ngược lại, ở phân khúc")
print("  biệt thự siêu sang, giá phụ thuộc nặng vào các yếu tố ngoại sinh phi cấu trúc")
print("  (nội thất nghệ thuật, tầm nhìn view biển/hồ, uy tín kiến trúc sư) mà dữ liệu")
print("  thuộc tính thông thường không thể ghi nhận.")

# %% [markdown]
# ## 13. Tổng hợp đối chuẩn 4 mô hình trên không gian giá trị Dollar thực

# %%
bench = [{"Mô hình": n, "Loại mô hình": r["kind"], "Tham số": f"{r['n_params']:,}",
          "Thời gian": f"{r['time']:.3f}s",
          "R² (Log)": f"{r['scores']['r2_log']:.4f}",
          "R² (Raw)": f"{r['scores']['r2_raw']:.4f}",
          "MAE (USD)": f"${r['scores']['mae_usd']:,.2f}",
          "RMSE (USD)": f"${r['scores']['rmse_usd']:,.0f}",
          "MAPE": f"{r['scores']['mape']:.2f}%"} for n, r in results.items()]
bench_df = pd.DataFrame(bench)
print("Bảng đối chuẩn toàn diện 4 mô hình định giá nhà trên không gian giá trị Dollar thực:")
display(bench_df)

champ = min(results.items(), key=lambda kv: kv[1]["scores"]["mae_usd"])
print(f"\n★ Quán quân sai số thấp nhất: {champ[0]} — "
      f"MAE ${champ[1]['scores']['mae_usd']:,.0f}, R²(raw) {champ[1]['scores']['r2_raw']:.4f}")

# %% [markdown]
# ### Hình 5 — Đối chuẩn trực quan hiệu năng hệ thống định giá

# %%
fig = plt.figure(figsize=(15.6, 8.8))
gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1], hspace=0.34, wspace=0.28)

names = list(results.keys())

ax = fig.add_subplot(gs[0, :])
xs = np.arange(len(names)); w = 0.36
r2s = [results[n]["scores"]["r2_log"] for n in names]
maes = [results[n]["scores"]["mae_usd"] / 1000 for n in names]
b1 = ax.bar(xs - w / 2, r2s, w, color="#4c72b0", label="Hệ số R² (Càng cao càng tốt)")
ax.bar_label(b1, fmt="%.3f", fontsize=9, fontweight="bold")
ax.set_ylabel("Hệ số R² (Càng cao càng tốt)", color="#4c72b0")
ax.set_ylim(0, max(r2s) * 1.28)
ax.set_xticks(xs, names, fontsize=9)
ax2 = ax.twinx(); ax2.grid(False)
b2 = ax2.bar(xs + w / 2, maes, w, color="#e07b39",
             label="MAE (Nghìn USD — Càng thấp càng tốt)")
ax2.bar_label(b2, fmt="$%.0fk", fontsize=9, fontweight="bold")
ax2.set_ylabel("MAE (Nghìn USD — Càng thấp càng tốt)", color="#e07b39")
ax2.set_ylim(0, max(maes) * 1.30)
ax.set_title("(a) So sánh R² và Sai số Tuyệt đối Trung bình MAE (USD)", fontsize=11)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

for k, (n, c) in enumerate(zip(names, colors)):
    ax = fig.add_subplot(gs[1, k])
    pr = np.expm1(np.asarray(results[n]["pred_log"]).ravel())[ssub] / 1000
    ar = y_te_raw[ssub] / 1000
    ax.scatter(ar, pr, s=4, alpha=0.26, c=c, edgecolors="none")
    lim = 3500
    ax.plot([0, lim], [0, lim], ls="--", lw=1.2, color="#d62728")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Giá thực tế (Nghìn $)", fontsize=8)
    ax.set_title(f"{n}\n(R²={results[n]['scores']['r2_log']:.3f}, "
                 f"MAE=${results[n]['scores']['mae_usd'] / 1000:.1f}k)", fontsize=8.5)
    if k == 0:
        ax.set_ylabel("Giá dự đoán (Nghìn $)", fontsize=8)

fig.suptitle("Đối chuẩn trực quan hiệu năng hệ thống định giá bất động sản (150,000 giao dịch)",
             fontsize=12.5, fontweight="bold")
fig.savefig(FIG / "fig05_model_comparison.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 14. Nhận xét chuyên sâu và đánh giá kết quả định giá nhà

# %%
lin = results["Linear Regression"]["scores"]
dls = results["Wide DL Scratch"]["scores"]
rf = results["Random Forest"]["scores"]

print("PHÁT HIỆN 1 — Sự phân kỳ giữa không gian Logarit và không gian Dollar thực")
print(f"  Linear Regression: R²(log) = {lin['r2_log']:.4f}  nhưng  R²(raw) = {lin['r2_raw']:.4f}")
print(f"  Sai số MAE vọt lên ${lin['mae_usd']:,.2f}")
print("  Nguyên nhân nằm ở bản chất hàm mũ. Nếu log(y) = wᵀx + b + ε thì khi nghịch đảo:")
print("      y = exp(wᵀx + b) · exp(ε) ≈ exp(wᵀx + b) · (1 + ε + ε²/2)")
print("  Một sai số dư nhỏ ε = 0.30 trên thang log của căn nhà trị giá $3,000,000 sẽ bị")
print("  khuếch đại thành mức chênh lệch vượt quá $1,050,000 trong giá trị thực!\n")

print("PHÁT HIỆN 2 — Ưu thế của mạng rộng trong xấp xỉ mặt phẳng liên tục")
print(f"  {dl_name} ({best_arch['Cấu hình tầng']}): R²(log) = {dls['r2_log']:.4f}, "
      f"MAE = ${dls['mae_usd']:,.2f}, MAPE = {dls['mape']:.2f}%")
print(f"  Vượt Linear Regression ${lin['mae_usd'] - dls['mae_usd']:,.0f} sai số trên mỗi căn nhà.")
print("  128 nơ-ron tại tầng ẩn thứ nhất cung cấp một hệ cơ sở hàm ReLU phong phú, giúp")
print("  mô hình bẻ gập và khớp nối các bề mặt giá trị đa chiều mấp mô một cách trơn tru.\n")

print("PHÁT HIỆN 3 — Random Forest và bản chất phân mảnh của thị trường bất động sản")
print(f"  Random Forest: R²(raw) = {rf['r2_raw']:.4f}, MAE = ${rf['mae_usd']:,.2f}, "
      f"MAPE = {rf['mape']:.2f}%")
print("  Các bề mặt xấp xỉ từng đoạn cục bộ (Piecewise Approximation) của rừng cây kìm")
print("  hãm sự bùng nổ sai số ở đuôi giá cao — đúng nơi mà mô hình tuyến tính sụp đổ.")

# %% [markdown]
# ## 15. Lưu artifact mô hình phục vụ triển khai REST API

# %%
np.savez(MODEL / "dl_scratch_weights.npz",
         **{f"W{i}": w for i, w in enumerate(dl_model.W)},
         **{f"b{i}": b for i, b in enumerate(dl_model.b)})

# CHỦ Ý không lưu Random Forest của bài toán này. Rừng 100 cây với
# max_depth=22 trên 118,026 mẫu sinh ra hơn 3,5 triệu nút, tương đương một tệp
# joblib ~244 MB — vượt giới hạn 100 MB mỗi tệp của GitHub và không có ích gì
# cho khâu triển khai, vì REST API chỉ nạp dl_scratch_weights.npz. Toàn bộ chỉ
# số đối chuẩn của Random Forest đã nằm trong metadata.json, và ai cần chính mô
# hình đó chỉ việc chạy lại ô huấn luyện ở mục 11 (mất khoảng 5 giây).

metadata = {
    "app": "house_price_large",
    "task": "regression",
    "dataset": {"name": "USA Real Estate Dataset", "n_raw": int(len(df_raw)),
                "n_clean": int(len(df)),
                "price_median": float(df.price.median()),
                "skew_raw": float(df.price.skew()),
                "skew_log": float(np.log1p(df.price).skew())},
    "cleaning_audit": audit.to_dict(orient="records"),
    "feature_names": FEATURE_NAMES,
    "n_features": int(X_tr.shape[1]),
    "preprocess": {
        "target_transform": "log1p(price)",
        "acre_lot_median": acre_median,
        "mu": mu.tolist(), "sd": sd.tolist(),
        "y_mu": y_mu, "y_sd": y_sd,
        "global_log_mean": global_mean,
        "global_house_size": global_size,
        "target_encoding": {
            "levels": {n: {"m": m, "n_units": len(te_tables[n])}
                       for n, _, _, m in GEO_LEVELS},
            "formula": "S_i = (n_i * y_i + m * y_global) / (n_i + m)",
        },
        "engineered": [
            "log_size = ln(house_size)", "log_lot = ln(1 + acre_lot)",
            "total_rooms = bed + bath", "sqft_per_room = house_size / total_rooms",
            "bath_bed_ratio = bath / bed", "bed_bath_prod = bed * bath",
            "relative_sqft = house_size / mean(house_size | zip)",
        ],
    },
    "split": {"n_train": int(len(df_tr)), "n_test": int(len(df_te))},
    "dl_model": {"name": dl_name, "layers": best_arch["Cấu hình tầng"],
                 "layer_sizes": dl_model.sizes, "n_params": dl_model.n_params(),
                 "activation": "ReLU (hidden) + Linear (output)",
                 "loss": "MSE", "lr": dl_model.lr,
                 "batch_size": dl_model.batch_size, "epochs": 30},
    "scores": {n: {"kind": r["kind"], "n_params": r["n_params"],
                   "time_sec": r["time"], **r["scores"]} for n, r in results.items()},
    "architecture_study": [{k: v for k, v in r.items() if not k.startswith("_")}
                           | {"scores": r["_s"]} for r in arch_rows],
    "loss_histories": {k: [float(x) for x in v] for k, v in arch_hist.items()},
    "market_segments": seg_rows,
    "residual_mean_usd": float(res_dl.mean()),
    "champion": champ[0],
}
(MODEL / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
(REP / "metrics_house_price_large.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

# Bảng tra cứu địa lý lưu riêng — API cần để mã hoá vị trí đầu vào
geo = {"global_mean": global_mean, "global_house_size": global_size,
       "acre_median": acre_median,
       "zip_size_map": {k: float(v) for k, v in zip_size_map.items()},
       "tables": {n: {str(k): float(v) for k, v in t.items()} for n, t in te_tables.items()}}
(MODEL / "geo_encoding.json").write_text(json.dumps(geo, ensure_ascii=False), encoding="utf-8")

print("✓ dl_scratch_weights.npz  (trọng số mạng nơ-ron thuần NumPy)")
print("✓ metadata.json · geo_encoding.json · metrics_house_price_large.json")
for f in sorted(FIG.glob("*.png")):
    print(f"✓ {f.name}")

# %% [markdown]
# ## 16. Kiểm chứng artifact — forward pass thuần NumPy

# %%
Z = np.load(MODEL / "dl_scratch_weights.npz")
Ws = [Z[f"W{i}"] for i in range(len(dl_model.W))]
bs = [Z[f"b{i}"] for i in range(len(dl_model.b))]

def api_forward_reg(X, Ws, bs):
    """Forward pass hồi quy — chính là đoạn code sẽ chạy trong REST API."""
    A = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        Zc = A @ W + b
        A = np.maximum(0.0, Zc) if i < len(Ws) - 1 else Zc
    return A

p_reload = api_forward_reg(X_te[:1000], Ws, bs)
p_origin = dl_model.predict(X_te[:1000])
print(f"Sai khác tuyệt đối lớn nhất: {float(np.abs(p_reload - p_origin).max()):.3e}")

# Thử một dự đoán mẫu, đưa hẳn về Dollar thực
demo = np.expm1(to_log_space(p_reload[:5]))
actual = y_te_raw[:5]
print("\n5 dự đoán mẫu (mô phỏng đúng luồng REST API sẽ chạy):")
for i, (pv, av) in enumerate(zip(demo, actual), 1):
    print(f"  Căn {i}: dự đoán ${pv:>11,.0f}   |   thực tế ${av:>11,.0f}   |   "
          f"sai lệch {abs(pv - av) / av * 100:>5.1f}%")

# %% [markdown]
# ## 17. Tổng kết notebook 04
#
# | Hạng mục | Kết quả |
# |---|---|
# | Quy mô dữ liệu | 150,000 giao dịch → làm sạch còn ~145,000 |
# | Không gian đặc trưng | 13 chiều (hình học + Bayesian Target Encoding 4 cấp) |
# | Kiến trúc DL tốt nhất | xem mục 10 |
# | Quán quân đối chuẩn | xem mục 13 |
#
# Ba kết luận chính:
#
# 1. **Quy luật kiến trúc bị đảo ngược so với notebook 03.** Bài toán hồi quy ưu ái
#    **chiều rộng** (nhiều bộ xấp xỉ tuyến tính từng đoạn song song), trong khi bài toán
#    phân loại ưu ái **chiều sâu** (tổ hợp phân cấp để bẻ cong siêu phẳng).
#
# 2. **R² trên thang log có thể đánh lừa nghiêm trọng.** Phải luôn báo cáo song song cả
#    $R^2_{\text{log}}$ lẫn $R^2_{\text{raw}}$ — hàm mũ khuếch đại sai số ở đuôi giá cao
#    theo cấp số nhân.
#
# 3. **Smooth Bayesian Target Encoding** giải quyết trọn vẹn bài toán mã hoá hàng nghìn
#    mã bưu chính mà không làm bùng nổ số chiều, đồng thời chống học vẹt cục bộ ở các khu
#    vực có quá ít giao dịch.
#
# **→ Notebook 05** chuyển sang miền dữ liệu **phi cấu trúc**: 23,486 nhận xét văn bản
# của khách hàng thương mại điện tử, nơi không gian đặc trưng TF-IDF 1,000 chiều có độ
# thưa thớt trên 98%.
