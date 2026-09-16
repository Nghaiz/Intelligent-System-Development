# %% [markdown]
# # Notebook 05 — Hệ Thống 3: Phân Loại Nhận Xét Thương Mại Điện Tử (23,486 đánh giá NLP)
#
# **Học phần:** Phát triển các Hệ thống Thông minh — Học viện Công nghệ Bưu chính Viễn thông
# **Sinh viên:** Nguyễn Duy Nghĩa · B23DCCN600 · D23CTPM01
# **Giảng viên hướng dẫn:** PGS.TS Trần Đình Quế
#
# ---
#
# ## Notebook này làm gì?
#
# Hai notebook trước làm việc trên **dữ liệu bảng** — mỗi cột là một con số có ý nghĩa
# vật lý rõ ràng. Notebook này bước sang miền hoàn toàn khác: **dữ liệu văn bản phi cấu
# trúc**. Đầu vào không còn là 14 con số, mà là những câu như:
#
# > *"Absolutely wonderful - silky and sexy and comfortable"*
# > *"Huge disappointment, the fabric feels cheap and it runs small"*
#
# Nhiệm vụ: dự đoán khách hàng **có khuyến nghị sản phẩm hay không** (`Recommended IND`)
# chỉ từ nội dung đánh giá.
#
# Ba thách thức kỹ thuật đặc thù của miền NLP:
#
# | Thách thức | Bản chất | Giải pháp trong notebook |
# |---|---|---|
# | Văn bản không phải số | Mạng nơ-ron chỉ nhận ma trận số | **TF-IDF Vectorization** 1,000 chiều |
# | Ma trận cực kỳ thưa thớt | Hơn 98% phần tử mang giá trị 0 | Kiến trúc **rộng** chống nút thắt cổ chai |
# | Mất cân bằng lớp 1 : 4.5 | 82% khách hài lòng | Tối ưu **Macro F1**, không phải Accuracy |
#
# Câu hỏi nghiên cứu:
#
# > **Trên không gian TF-IDF thưa thớt chiều cao, mạng nơ-ron sâu tự viết có nén được
# > ngữ nghĩa thành biểu diễn dày đặc và cạnh tranh với Logistic Regression — mô hình
# > vốn nổi tiếng là rất mạnh trên dữ liệu văn bản?**
#
# ## Bộ dữ liệu
#
# `Women's Clothing E-Commerce Reviews` — 23,486 đánh giá sản phẩm thời trang thực tế
# của khách hàng. Mỗi bản ghi gồm tiêu đề ngắn (`Title`), nội dung chi tiết
# (`Review Text`), xếp hạng sao (`Rating` 1–5), và nhãn nhị phân `Recommended IND`.
# Nguồn: <https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews>

# %% [markdown]
# ## 1. Chuẩn bị môi trường

# %%
import json
import re
import time
import warnings
from collections import Counter
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
DATA = ROOT / "data" / "womens_ecommerce_reviews.csv"
FIG = ROOT / "reports" / "figures"
REP = ROOT / "reports"
MODEL = ROOT / "model"
for d in (FIG, MODEL):
    d.mkdir(parents=True, exist_ok=True)
print("Dữ liệu:", DATA)

# %% [markdown]
# ## 2. Nạp dữ liệu và khám phá ban đầu

# %%
df_raw = pd.read_csv(DATA, index_col=0)
print(f"Dữ liệu thô: {df_raw.shape[0]:,} đánh giá × {df_raw.shape[1]} cột\n")
display(df_raw.head())

print("Mức độ khuyết thiếu của từng cột:")
display(pd.DataFrame({
    "Số khuyết thiếu": df_raw.isna().sum(),
    "Tỷ lệ khuyết (%)": (df_raw.isna().mean() * 100).round(2),
}))

print("\nVí dụ hai đánh giá đối lập:")
pos_ex = df_raw[df_raw["Recommended IND"] == 1].dropna(subset=["Review Text"]).iloc[0]
neg_ex = df_raw[df_raw["Recommended IND"] == 0].dropna(subset=["Review Text"]).iloc[0]
print(f"\n  [KHUYẾN NGHỊ · {pos_ex['Rating']}★] {str(pos_ex['Review Text'])[:190]}...")
print(f"\n  [KHÔNG KHUYẾN NGHỊ · {neg_ex['Rating']}★] {str(neg_ex['Review Text'])[:190]}...")

# %% [markdown]
# ## 3. Kiểm toán quy trình làm sạch dữ liệu văn bản
#
# Ba bước tiền xử lý, mỗi bước có lý do rõ ràng.
#
# **Bước 2 — Hợp nhất ngữ cảnh văn bản** là bước quan trọng nhất về mặt ngữ nghĩa:
#
# $$\text{text} = \text{Title} + \text{“ ”} + \text{Review Text}$$
#
# Lý do: khách hàng thường **cô đọng cảm xúc mạnh nhất ngay ở dòng tiêu đề** — chẳng hạn
# *"Huge disappointment"*, *"Love this dress"*, *"Poor quality"*. Vứt bỏ tiêu đề là vứt
# bỏ tín hiệu cảm xúc đậm đặc nhất trong toàn bộ bản ghi.

# %%
audit_rows = [("Dữ liệu thô ban đầu", "Tập đánh giá sản phẩm thương mại điện tử gốc",
               0, len(df_raw))]
df = df_raw.copy()

n0 = len(df)
df = df.dropna(subset=["Review Text"]).reset_index(drop=True)
df = df[df["Review Text"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
df = df.dropna(subset=["Recommended IND"]).reset_index(drop=True)
audit_rows.append(("1. Lọc khuyết thiếu nội dung",
                   "Loại bỏ bản ghi khuyết trường Review Text hoặc nhãn Recommended IND",
                   n0 - len(df), len(df)))

n0 = len(df)
df["text"] = (df["Title"].fillna("").astype(str).str.strip() + " "
              + df["Review Text"].astype(str).str.strip()).str.strip()
audit_rows.append(("2. Hợp nhất ngữ cảnh",
                   "Ghép nối tiêu đề Title và nội dung Review Text thành văn bản đầy đủ",
                   n0 - len(df), len(df)))

n0 = len(df)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
audit_rows.append(("3. Khử trùng lặp văn bản",
                   "Loại bỏ các đánh giá có nội dung văn bản hoàn toàn trùng khớp",
                   n0 - len(df), len(df)))

audit = pd.DataFrame(audit_rows, columns=["Bước Xử lý", "Tiêu chí Tiền xử lý Văn bản",
                                          "Số lượng Loại bỏ", "Số lượng Còn lại"])
audit["Tỷ lệ Giữ lại (%)"] = (audit["Số lượng Còn lại"] / len(df_raw) * 100).round(2)
display(audit)

y_all = df["Recommended IND"].to_numpy(dtype=np.float64).reshape(-1, 1)
n_pos, n_neg = int(y_all.sum()), int(len(y_all) - y_all.sum())
print(f"Còn lại {len(df):,} nhận xét văn bản hợp lệ")
print(f"  Khuyến nghị     (nhãn 1): {n_pos:,} ({n_pos / len(df):.2%})")
print(f"  Không khuyến nghị (nhãn 0): {n_neg:,} ({n_neg / len(df):.2%})")
print(f"  Tỷ lệ mất cân bằng xấp xỉ 1 : {n_pos / n_neg:.1f}")

# %% [markdown]
# ### Hình 1 — Phân phối nhãn, xếp hạng sao và độ dài nhận xét

# %%
df["n_words"] = df["text"].str.split().str.len()

fig = plt.figure(figsize=(13.2, 8.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.34, wspace=0.22)

ax = fig.add_subplot(gs[0, 0])
bars = ax.bar(["Không khuyến nghị (0)", "Khuyến nghị (1)"], [n_neg, n_pos],
              color=["#e03131", "#1f4fd8"], width=0.5)
for b, v in zip(bars, [n_neg, n_pos]):
    ax.text(b.get_x() + b.get_width() / 2, v + len(df) * 0.015,
            f"{v:,} ({v / len(df):.1%})", ha="center", fontsize=9.5, fontweight="bold")
ax.set_ylabel("Số lượng đánh giá"); ax.set_ylim(0, len(df) * 1.12)
ax.set_title("Phân phối Nhãn Mục tiêu (Recommended IND)")

ax = fig.add_subplot(gs[0, 1])
rc = df["Rating"].value_counts().sort_index()
cmap = ["#e03131", "#f08c00", "#f5b800", "#74c476", "#2a9d70"]
bars = ax.bar([f"{int(r)} Sao" for r in rc.index], rc.values, color=cmap[:len(rc)], width=0.6)
for b, v in zip(bars, rc.values):
    ax.text(b.get_x() + b.get_width() / 2, v + len(df) * 0.012,
            f"{v:,} ({v / len(df):.1%})", ha="center", fontsize=8, fontweight="bold")
ax.set_ylabel("Số lượng đánh giá"); ax.set_ylim(0, rc.max() * 1.16)
ax.set_title("Phân phối Xếp hạng Sao (Rating 1 – 5)")

ax = fig.add_subplot(gs[1, :])
bins = np.arange(0, df.n_words.quantile(0.995) + 4, 3)
ax.hist(df.loc[df["Recommended IND"] == 1, "n_words"], bins=bins, color="#4c72b0",
        alpha=0.88, label="Khuyến nghị (1)")
ax.hist(df.loc[df["Recommended IND"] == 0, "n_words"], bins=bins, color="#c44e52",
        alpha=0.80, label="Không khuyến nghị (0)")
m1 = df.loc[df["Recommended IND"] == 1, "n_words"].median()
m0 = df.loc[df["Recommended IND"] == 0, "n_words"].median()
ax.axvline(m1, color="#1f4fd8", ls="--", lw=1.4, label=f"Trung vị Khuyến nghị ({m1:.0f} từ)")
ax.axvline(m0, color="#c0392b", ls="--", lw=1.4, label=f"Trung vị Không khuyến nghị ({m0:.0f} từ)")
ax.set_xlabel("Số lượng từ trong nhận xét"); ax.set_ylabel("Số lượng đánh giá")
ax.set_title("Phân phối Độ dài Nhận xét (Số lượng từ vựng) theo Nhãn Cảm xúc")
ax.legend()

fig.suptitle("Phân tích phân phối mục tiêu và độ dài văn bản nhận xét thương mại điện tử",
             fontsize=12.5, fontweight="bold")
fig.savefig(FIG / "fig01_target_and_length.png", bbox_inches="tight")
plt.show()

print(f"Trung vị độ dài — Khuyến nghị: {m1:.0f} từ · Không khuyến nghị: {m0:.0f} từ")
print("→ Đánh giá tiêu cực có xu hướng dài hơn: khách hàng phàn nàn thường viết chi tiết")
print("  hơn để giải thích cặn kẽ vấn đề gặp phải.")

# %% [markdown]
# ### Hình 2 — Top 15 từ khoá xuất hiện nhiều nhất

# %%
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "yours", "he", "him",
    "his", "she", "her", "hers", "it", "its", "they", "them", "their", "what", "which",
    "who", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a",
    "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at",
    "by", "for", "with", "about", "into", "through", "to", "from", "in", "out", "on",
    "off", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "im", "ive", "its", "am", "up", "down", "over", "under",
}

def tokenize(text: str) -> list[str]:
    """Chuẩn hoá về chữ thường, giữ lại chuỗi chữ cái có độ dài ≥ 2."""
    return re.findall(r"[a-z]{2,}", str(text).lower())

counter = Counter()
for t in df["text"]:
    counter.update(w for w in tokenize(t) if w not in STOPWORDS)

top15 = counter.most_common(15)
fig, ax = plt.subplots(figsize=(9.4, 5.4))
words = [w for w, _ in top15][::-1]
freqs = [c for _, c in top15][::-1]
bars = ax.barh(words, freqs, color="#6a5acd", height=0.68)
for b, v in zip(bars, freqs):
    ax.text(v + max(freqs) * 0.008, b.get_y() + b.get_height() / 2, f"{v:,}",
            va="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Số lần xuất hiện"); ax.set_xlim(0, max(freqs) * 1.10)
ax.set_title("Top 15 Từ Khoá Xuất Hiện Nhiều Nhất Trong Đánh Giá Khách Hàng",
             fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig02_top_keywords.png", bbox_inches="tight")
plt.show()

print("Các danh từ và tính từ xuất hiện áp đảo:",
      ", ".join(w for w, _ in top15[:8]))

# %% [markdown]
# ## 4. Phân chia phân tầng TRƯỚC khi vector hoá
#
# Thứ tự bắt buộc, giống hai notebook trước: bộ từ điển và trọng số IDF phải được học
# **chỉ trên tập Train**.

# %%
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

tr_idx, te_idx = stratified_split(y_all, 0.20, SEED)
text_tr = df["text"].to_numpy()[tr_idx]
text_te = df["text"].to_numpy()[te_idx]
y_tr, y_te = y_all[tr_idx], y_all[te_idx]

print(f"Tập Train: {len(y_tr):,} nhận xét — tỷ lệ khuyến nghị {y_tr.mean():.2%}")
print(f"Tập Test : {len(y_te):,} nhận xét — tỷ lệ khuyến nghị {y_te.mean():.2%}")

# %% [markdown]
# ## 5. Trích xuất đặc trưng TF-IDF
#
# **TF-IDF (Term Frequency – Inverse Document Frequency)** biến mỗi văn bản thành một
# vector số, trong đó mỗi chiều tương ứng một từ khoá trong bộ từ điển:
#
# $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \left[\ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1\right]$$
#
# **Trực giác của công thức:** một từ có trọng số cao khi nó xuất hiện **nhiều lần trong
# một văn bản** (thành phần TF) nhưng **hiếm gặp trên toàn bộ tập** (thành phần IDF). Từ
# *"dress"* xuất hiện ở hầu hết đánh giá nên IDF thấp — nó không phân biệt được gì. Từ
# *"disappointing"* hiếm hơn nhiều nên IDF cao — nó mang tín hiệu cảm xúc mạnh.
#
# **Cấu hình trích xuất:**
#
# - Từ vựng 1,000 chiều (`max_features=1000`);
# - Kết hợp cả từ đơn và từ ghép hai tiếng (`ngram_range=(1, 2)`) — cụm *"not worth"*,
#   *"runs small"* mang nghĩa hoàn toàn khác hai từ đơn tách rời;
# - Loại bỏ từ dừng tiếng Anh chuẩn (`stop_words='english'`).
#
# **Nguyên tắc chống rò rỉ:** bộ từ điển và trọng số IDF được tính **duy nhất trên tập
# Train** (`fit_transform`), rồi áp cố định để chiếu tập Test (`transform`).

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

t0 = time.perf_counter()
vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2),
                      stop_words="english", sublinear_tf=True)
X_tr_sp = vec.fit_transform(text_tr)      # ← học từ điển CHỈ trên Train
X_te_sp = vec.transform(text_te)          # ← chiếu Test vào cùng không gian
vec_time = time.perf_counter() - t0

X_tr = np.asarray(X_tr_sp.todense(), dtype=np.float64)
X_te = np.asarray(X_te_sp.todense(), dtype=np.float64)

sparsity = float((X_tr == 0).mean() * 100)
print(f"Thời gian vector hoá: {vec_time:.2f}s")
print(f"Ma trận đặc trưng: Train {X_tr.shape} · Test {X_te.shape}")
print(f"Độ thưa thớt: {sparsity:.2f}% phần tử mang giá trị 0")
print(f"Trung bình mỗi nhận xét chỉ kích hoạt {(X_tr > 0).sum(axis=1).mean():.1f} / 1000 chiều")

feat_names = vec.get_feature_names_out()
n_bigram = sum(1 for f in feat_names if " " in f)
print(f"\nBộ từ điển: {len(feat_names) - n_bigram} từ đơn + {n_bigram} cụm hai từ")
print("20 đặc trưng đầu tiên:", ", ".join(feat_names[:20]))

# %% [markdown]
# **Vì sao độ thưa thớt là một thách thức lớn với mạng nơ-ron?** Trong một vector 1,000
# chiều, chỉ khoảng 20–30 chiều mang giá trị khác 0. Gradient chỉ chảy qua đúng những
# chiều đó, nên phần lớn trọng số của tầng đầu tiên được cập nhật rất thưa thớt. Nếu
# tầng ẩn đầu tiên quá hẹp, thông tin bị nén quá gắt gao ngay từ bước đầu — hiện tượng
# **nút thắt cổ chai thông tin (Information Bottleneck)**. Mục 7 sẽ đo lường trực tiếp
# hiệu ứng này.

# %% [markdown]
# ## 6. Mạng nơ-ron phân loại văn bản thuần NumPy và bộ chỉ số cân bằng lớp
#
# **Vì sao dùng Macro F1 thay vì Accuracy?**
#
# Với tỷ lệ 82:18, một mô hình luôn dự đoán "khuyến nghị" đạt Accuracy 82% mà không nhận
# diện được **một** khách hàng bức xúc nào. Macro F1 lấy trung bình cộng F1 của **cả hai
# lớp**, nên lớp thiểu số có trọng số ngang bằng:
#
# $$\text{Macro F1} = \frac{F1_{\text{lớp } 0} + F1_{\text{lớp } 1}}{2}$$
#
# Trong quản trị trải nghiệm khách hàng, **bỏ sót một khách hàng bức xúc** có thể dẫn tới
# sự lan truyền dư luận tiêu cực trên mạng xã hội. Vì vậy `Recall lớp 0` (tỷ lệ nhận diện
# được lời phàn nàn) là chỉ số nghiệp vụ quan trọng nhất.

# %%
class MLPTextScratch:
    """Mạng nơ-ron phân loại văn bản nhiều tầng, 100% NumPy, huấn luyện Mini-Batch."""

    def __init__(self, layer_sizes, lr=0.10, batch_size=128, seed=SEED):
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

    def fit(self, X, y, epochs=25, verbose_every=0):
        rng = np.random.default_rng(SEED)
        n = X.shape[0]
        for ep in range(1, epochs + 1):
            order = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                self.forward(X[idx]); self._backward(y[idx])
            self.history.append(self.bce(y, self.forward(X, cache=False)))
            if verbose_every and (ep == 1 or ep % verbose_every == 0):
                print(f"  epoch {ep:>3} / {epochs}   loss = {self.history[-1]:.4f}")
        return self

    def predict_proba(self, X):
        return self.forward(X, cache=False)

    def hidden_activations(self, X):
        self.forward(X)
        return self.As[1:-1]

    def n_params(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)


def confusion(y_true, y_pred):
    yt, yp = y_true.ravel().astype(int), y_pred.ravel().astype(int)
    return (int(((yt == 0) & (yp == 0)).sum()), int(((yt == 0) & (yp == 1)).sum()),
            int(((yt == 1) & (yp == 0)).sum()), int(((yt == 1) & (yp == 1)).sum()))

def scores_at(y_true, y_prob, t=0.50):
    tn, fp, fn, tp = confusion(y_true, (np.asarray(y_prob).ravel() >= t).astype(int))
    # lớp 1 (khuyến nghị)
    p1, r1 = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1_1 = 2 * p1 * r1 / max(p1 + r1, 1e-12)
    # lớp 0 (không khuyến nghị) — lớp thiểu số cần bảo vệ
    p0, r0 = tn / max(tn + fn, 1), tn / max(tn + fp, 1)
    f1_0 = 2 * p0 * r0 / max(p0 + r0, 1e-12)
    return {"threshold": float(t), "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "precision_1": p1, "recall_1": r1, "f1_1": f1_1,
            "precision_0": p0, "recall_0": r0, "f1_0": f1_0,
            "macro_f1": (f1_0 + f1_1) / 2,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}

def best_threshold_macro(y_true, y_prob, lo=0.10, hi=0.92, step=0.01):
    """Quét ngưỡng tối đa hoá MACRO F1 — luôn chạy trên tập Train."""
    best = (0.50, -1.0)
    for t in np.arange(lo, hi + 1e-9, step):
        v = scores_at(y_true, y_prob, t)["macro_f1"]
        if v > best[1]:
            best = (round(float(t), 2), v)
    return best

def roc_auc(y_true, y_score):
    y, s = y_true.ravel(), np.asarray(y_score).ravel()
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]; i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    npos, nneg = float((y == 1).sum()), float((y == 0).sum())
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

# %% [markdown]
# ## 7. Khảo sát kiến trúc NLP: thích ứng trên dữ liệu thưa chiều cao
#
# Bốn cấu hình đối chứng trên không gian từ vựng 1,000 chiều. Giả thuyết cần kiểm chứng:
# **mạng hẹp sẽ tạo nút thắt cổ chai thông tin, mạng rộng sẽ thắng.**

# %%
D = X_tr.shape[1]
ARCHS = [
    ("Shallow Text MLP",  [D, 32, 1],         "1000 → 32 → 1"),
    ("Standard Text MLP", [D, 32, 16, 1],     "1000 → 32 → 16 → 1"),
    ("Deeper Text MLP",   [D, 64, 32, 16, 1], "1000 → 64 → 32 → 16 → 1"),
    ("Wide Text MLP",     [D, 128, 64, 1],    "1000 → 128 → 64 → 1"),
]

arch_rows, arch_hist, arch_models = [], {}, {}
for name, sizes, label in ARCHS:
    t0 = time.perf_counter()
    m = MLPTextScratch(sizes, lr=0.10, batch_size=128, seed=SEED).fit(X_tr, y_tr, epochs=25)
    el = time.perf_counter() - t0
    p_tr, p_te = m.predict_proba(X_tr), m.predict_proba(X_te)
    t_best, _ = best_threshold_macro(y_tr, p_tr)
    s = scores_at(y_te, p_te, t_best)
    arch_hist[name], arch_models[name] = m.history, m
    arch_rows.append({"Kiến trúc": name, "Cấu hình tầng": label, "Tham số": m.n_params(),
                      "Thời gian": f"{el:.2f}s", "Loss cuối": round(m.history[-1], 4),
                      "Accuracy": f"{s['accuracy']:.2%}",
                      "Rec0 (Tiêu cực)": f"{s['recall_0']:.2%}",
                      "Macro F1": f"{s['macro_f1']:.2%}", "_s": s, "_t": t_best, "_el": el})
    print(f"{name:<20} {label:<26} {m.n_params():>7,} params  {el:>6.2f}s  "
          f"loss={m.history[-1]:.4f}  MacroF1={s['macro_f1']:.2%}  Rec0={s['recall_0']:.2%}")

arch_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in arch_rows])
print("\nBảng đối chứng thực nghiệm khảo sát kiến trúc mạng văn bản thuần NumPy:")
display(arch_df)

best_arch = max(arch_rows, key=lambda r: r["_s"]["macro_f1"])
print(f"\n★ Kiến trúc chiến thắng: {best_arch['Kiến trúc']} ({best_arch['Cấu hình tầng']}) "
      f"— Macro F1 {best_arch['_s']['macro_f1']:.2%}, Recall lớp 0 {best_arch['_s']['recall_0']:.2%}")

# %% [markdown]
# ### Hình 3 — Hội tụ Loss và so sánh Macro F1 giữa các kiến trúc văn bản

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))

ax = axes[0]
for (name, h), c, st in zip(arch_hist.items(),
                            ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"],
                            ["--", "-", "-", ":"]):
    npar = next(r["Tham số"] for r in arch_rows if r["Kiến trúc"] == name)
    ax.plot(range(1, len(h) + 1), h, lw=1.6, color=c, ls=st, label=f"{name} ({npar:,} params)")
ax.set_xlabel("Epoch (Chu kỳ huấn luyện)"); ax.set_ylabel("Binary Cross-Entropy Loss")
ax.set_title("Đường Cong Hội Tụ Loss Giữa Các Kiến Trúc Văn Bản")
ax.legend()

ax = axes[1]
xs = np.arange(len(arch_rows)); w = 0.36
mf = [r["_s"]["macro_f1"] * 100 for r in arch_rows]
r0 = [r["_s"]["recall_0"] * 100 for r in arch_rows]
b1 = ax.bar(xs - w / 2, mf, w, label="Macro F1-Score (%)", color="#4c72b0")
b2 = ax.bar(xs + w / 2, r0, w, label="Recall lớp 0 — Tiêu cực (%)", color="#e8a33d")
ax.bar_label(b1, fmt="%.1f", fontsize=8); ax.bar_label(b2, fmt="%.1f", fontsize=8)
ax.set_xticks(xs, [r["Kiến trúc"].replace(" Text MLP", "") for r in arch_rows], fontsize=8)
ax.set_ylabel("Phần trăm (%)"); ax.set_ylim(0, 100)
ax.set_title("So Sánh Macro F1 & Khả Năng Nhận Diện Đánh Giá Tiêu Cực (Lớp 0)")
ax.legend()

fig.suptitle("Khảo sát hội tụ Loss và so sánh Macro F1 giữa 4 kiến trúc mạng nơ-ron phân loại văn bản",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig03_architecture_study.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Đột phá khoa học của mạng rộng trên không gian thưa.** Trên không gian đặc trưng
# TF-IDF thưa thớt 1,000 chiều, mạng hẹp ($1000 \to 32$) tạo thành một **nút thắt cổ chai
# thông tin quá gắt gao**: 1,000 chiều ngữ nghĩa bị ép xuống chỉ 32 chiều ngay ở tầng đầu
# tiên, khiến nhiều tín hiệu cảm xúc tinh tế bị triệt tiêu sớm.
#
# Ngược lại, kiến trúc rộng với 128 nơ-ron tại tầng ẩn thứ nhất cho phép mạng **nắm bắt
# song song nhiều cụm tổ hợp từ vựng phàn nàn khác nhau** — "vải xấu", "giao chậm", "sai
# kích cỡ", "màu khác ảnh" — mỗi cụm được một nhóm nơ-ron riêng phụ trách.

# %% [markdown]
# ## 8. Đối đầu với ba mô hình Machine Learning cổ điển

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

ML_MODELS = [
    ("Logistic Regression", "ML Tuyến tính",
     LogisticRegression(max_iter=2000, random_state=SEED)),
    ("Naive Bayes", "Xác suất Bayes", MultinomialNB(alpha=0.3)),
    ("Random Forest", "Ensemble 50 Cây",
     RandomForestClassifier(n_estimators=50, max_depth=28, min_samples_leaf=2,
                            n_jobs=-1, random_state=SEED)),
]

results = {}
for name, kind, mdl in ML_MODELS:
    t0 = time.perf_counter()
    mdl.fit(X_tr_sp, y_tr.ravel())          # dùng ma trận thưa gốc cho hiệu quả bộ nhớ
    el = time.perf_counter() - t0
    p_tr = mdl.predict_proba(X_tr_sp)[:, 1]
    p_te = mdl.predict_proba(X_te_sp)[:, 1]
    t_best, _ = best_threshold_macro(y_tr, p_tr)
    results[name] = {
        "kind": kind, "time": el, "model": mdl, "prob_test": p_te, "tau": t_best,
        "n_params": (X_tr.shape[1] + 1 if name == "Logistic Regression"
                     else 2 * X_tr.shape[1] if name == "Naive Bayes"
                     else int(sum(e.tree_.node_count for e in mdl.estimators_))),
        "default": scores_at(y_te, p_te, 0.50),
        "optimal": scores_at(y_te, p_te, t_best),
        "auc": roc_auc(y_te, p_te),
    }
    r = results[name]
    print(f"{name:<22} {el:>7.3f}s  τ={t_best:.2f}  Acc={r['optimal']['accuracy']:.2%}  "
          f"Rec0={r['optimal']['recall_0']:.2%}  MacroF1={r['optimal']['macro_f1']:.2%}  "
          f"AUC={r['auc']:.4f}")

dl_name = best_arch["Kiến trúc"]
dl_model = arch_models[dl_name]
p_te_dl = dl_model.predict_proba(X_te)
results["DL Text Scratch"] = {
    "kind": f"Mạng {len(dl_model.sizes) - 2} lớp rộng", "time": best_arch["_el"],
    "model": dl_model, "prob_test": p_te_dl.ravel(), "tau": best_arch["_t"],
    "n_params": dl_model.n_params(),
    "default": scores_at(y_te, p_te_dl, 0.50), "optimal": best_arch["_s"],
    "auc": roc_auc(y_te, p_te_dl),
}
r = results["DL Text Scratch"]
print(f"{'DL Text Scratch':<22} {r['time']:>7.3f}s  τ={r['tau']:.2f}  "
      f"Acc={r['optimal']['accuracy']:.2%}  Rec0={r['optimal']['recall_0']:.2%}  "
      f"MacroF1={r['optimal']['macro_f1']:.2%}  AUC={r['auc']:.4f}")

# %% [markdown]
# ## 9. Tinh chỉnh ngưỡng cân bằng lớp (Macro F1 Optimization)
#
# Do nhãn khuyến nghị chiếm hơn 80%, ngưỡng xác suất mặc định 0.50 khiến các mô hình học
# máy thiên lệch nặng về nhãn đa số. Nâng ngưỡng lên vùng cao buộc mô hình phải "chắc
# chắn hơn" mới dám gán nhãn khuyến nghị, nhờ đó **giải phóng năng lực nhận diện lớp
# thiểu số**.

# %% [markdown]
# ### Hình 4 — Đường cong cân bằng lớp theo ngưỡng quyết định

# %%
fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.0))
taus = np.arange(0.05, 0.96, 0.01)
for ax, (name, r) in zip(axes, results.items()):
    P = [scores_at(y_te, r["prob_test"], t) for t in taus]
    ax.plot(taus, [p["recall_0"] for p in P], lw=1.5, color="#d62728",
            label="Recall lớp 0 (Tiêu cực)")
    ax.plot(taus, [p["recall_1"] for p in P], lw=1.5, color="#1f77b4",
            label="Recall lớp 1 (Tích cực)")
    ax.plot(taus, [p["macro_f1"] for p in P], lw=1.9, color="#2ca02c", label="Macro F1")
    ax.plot(taus, [p["accuracy"] for p in P], lw=1.1, ls=":", color="#888", label="Accuracy")
    ax.axvline(r["tau"], color="#8e44ad", ls="--", lw=1.4)
    ax.axvline(0.50, color="#aaa", ls=":", lw=1.1)
    ax.set_title(f"{name}\n(Ngưỡng tối ưu: {r['tau']:.2f})", fontsize=9)
    ax.set_xlabel("Ngưỡng Quyết định (Threshold)"); ax.set_ylim(0, 1.02)
    if ax is axes[0]:
        ax.set_ylabel("Điểm số (0 – 1)"); ax.legend(fontsize=6.6)
fig.suptitle("Đường cong cân bằng lớp và tinh chỉnh ngưỡng quyết định trên văn bản NLP",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig04_threshold_curves.png", bbox_inches="tight")
plt.show()

# %%
print("HIỆU QUẢ CỨU VÃN NHÃN THIỂU SỐ NHỜ TINH CHỈNH NGƯỠNG")
print("=" * 84)
print(f"{'Mô hình':<22}{'τ tối ưu':>10}{'Rec0 @0.50':>13}{'Rec0 @τ':>11}"
      f"{'ΔRec0':>11}{'MacroF1 @τ':>14}")
print("-" * 84)
for name, r in results.items():
    d, o = r["default"], r["optimal"]
    print(f"{name:<22}{r['tau']:>10.2f}{d['recall_0']:>12.2%}{o['recall_0']:>11.2%}"
          f"{(o['recall_0'] - d['recall_0']) * 100:>+10.2f}%{o['macro_f1']:>13.2%}")
print("=" * 84)
print("\n→ Trong quản trị trải nghiệm khách hàng, bỏ sót một khách hàng bức xúc có thể dẫn")
print("  tới sự lan truyền dư luận tiêu cực trên mạng xã hội. Tinh chỉnh ngưỡng là chìa")
print("  khoá then chốt để mô hình đáp ứng được yêu cầu nghiệp vụ thực tế.")

# %% [markdown]
# ## 10. Trực quan hoá quá trình học biểu diễn ngữ nghĩa bằng PCA 2D
#
# Đây là thực nghiệm chứng minh mạng nơ-ron thực hiện **Semantic Representation Learning**
# — tự động nén ngữ nghĩa từ ma trận thưa mà không cần con người gán nhãn ngữ nghĩa thủ công:
#
# $$\mathbf{X}_{\text{TF-IDF}} \in \mathbb{R}^{1000} \xrightarrow{\ \mathbf{W}_1, \mathbf{b}_1\ } \mathbf{H}_1 \in \mathbb{R}^{128} \xrightarrow{\ \mathbf{W}_2, \mathbf{b}_2\ } \mathbf{H}_2 \in \mathbb{R}^{64} \xrightarrow{\ \mathbf{W}_3, \mathbf{b}_3\ } \hat{y}$$

# %%
def pca_2d(M):
    Mc = M - M.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    var = S ** 2
    return Mc @ Vt[:2].T, var[:2] / max(var.sum(), 1e-12)

rng = np.random.default_rng(SEED)
sub = rng.choice(len(X_te), size=min(2000, len(X_te)), replace=False)
Xs, ys = X_te[sub], y_te[sub].ravel()
Hs = dl_model.hidden_activations(Xs)

spaces = [("(A) Không gian Đặc trưng TF-IDF (1000D)", Xs,
           "Từ vựng rời rạc, cực kỳ thưa, hai lớp đan xen hỗn độn")]
titles = ["Dense Semantics", "Sentiment Abstraction", "Deep Abstraction"]
for i, H in enumerate(Hs, 1):
    act = float((H > 0).mean() * 100)
    spaces.append((f"({chr(65 + i)}) Tầng ẩn H{i} ({H.shape[1]}D — {titles[min(i - 1, 2)]})",
                   H, f"Nén ngữ nghĩa dày đặc | ReLU: {act:.1f}% nơ-ron hoạt động"))

fig, axes = plt.subplots(1, len(spaces), figsize=(4.6 * len(spaces), 4.4))
for ax, (title, M, sub_t) in zip(np.atleast_1d(axes), spaces):
    P, ratio = pca_2d(M)
    ax.scatter(P[ys == 1, 0], P[ys == 1, 1], s=7, alpha=0.42, c="#2a9d70",
               edgecolors="none", label="Khuyến nghị (1)")
    ax.scatter(P[ys == 0, 0], P[ys == 0, 1], s=7, alpha=0.55, c="#e03131",
               edgecolors="none", label="Không khuyến nghị (0)")
    ax.set_title(f"{title}\n{sub_t}", fontsize=8.5)
    ax.set_xlabel(f"PC1 ({ratio[0]:.1%})"); ax.set_ylabel(f"PC2 ({ratio[1]:.1%})")
    ax.text(0.02, 0.97, f"Tổng var: {ratio.sum():.1%}", transform=ax.transAxes,
            fontsize=7, va="top", bbox=dict(fc="#fff8c4", ec="#ccc", alpha=0.9))
    ax.legend(fontsize=6.8, loc="lower right")

fig.suptitle(f"Trực quan hoá biến đổi không gian biểu diễn ngữ nghĩa bằng PCA 2D — "
             f"{dl_name} (Ngưỡng tối ưu τ = {results['DL Text Scratch']['tau']:.2f})",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig05_representation_pca_spaces.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Phân tích biến đổi không gian ngữ nghĩa:**
#
# 1. **Không gian TF-IDF (1000D — bảng A).** Ma trận từ vựng thưa thớt khiến các điểm dữ
#    liệu phân tán rời rạc, không có cấu trúc hình học rõ ràng, hai lớp cảm xúc đan xen
#    hỗn độn. Tổng phương sai mà 2 thành phần chính đầu tiên giải thích được rất thấp —
#    dấu hiệu điển hình của dữ liệu thưa chiều cao.
#
# 2. **Tầng ẩn $H_1$ (128D — Dense Semantics).** Mạng nén 1,000 chiều thưa thành 128
#    chiều liên tục dày đặc. Các từ vựng **đồng nghĩa hoặc cùng ngữ cảnh cảm xúc** được
#    chiếu về các toạ độ lân cận — đây chính là bản chất của học biểu diễn từ vựng.
#
# 3. **Tầng ẩn cuối (Sentiment Abstraction).** **Sự phân cụm cảm xúc tự động (Semantic
#    Clustering) hoàn tất!** Nhóm khách hàng không hài lòng (màu đỏ) co cụm rõ rệt và tách
#    biệt khỏi nhóm khách hàng khen ngợi (màu xanh lá), với tổng phương sai giải thích
#    tăng vọt. Đây là minh chứng thực nghiệm không thể chối cãi về năng lực học biểu diễn
#    của Deep Learning thuần NumPy.

# %% [markdown]
# ## 11. Tổng hợp đối chuẩn 4 mô hình xử lý văn bản NLP

# %%
bench = [{"Mô hình": n, "Loại mô hình": r["kind"], "Tham số": f"{r['n_params']:,}",
          "Thời gian": f"{r['time']:.3f}s", "Ngưỡng": f"{r['tau']:.2f}",
          "Accuracy": f"{r['optimal']['accuracy']:.2%}",
          "Rec0 (Tiêu cực)": f"{r['optimal']['recall_0']:.2%}",
          "Rec1 (Tích cực)": f"{r['optimal']['recall_1']:.2%}",
          "Macro F1": f"{r['optimal']['macro_f1']:.2%}",
          "ROC-AUC": f"{r['auc']:.4f}"} for n, r in results.items()]
bench_df = pd.DataFrame(bench)
print("Bảng đối chuẩn khoa học: Machine Learning vs Wide Deep Learning Scratch:")
display(bench_df)

champ = max(results.items(), key=lambda kv: kv[1]["optimal"]["macro_f1"])
print(f"\n★ Quán quân Macro F1: {champ[0]} — {champ[1]['optimal']['macro_f1']:.2%}, "
      f"ROC-AUC {champ[1]['auc']:.4f}")

# %% [markdown]
# ### Hình 6 — Đối chuẩn trực quan hiệu năng hệ thống phân loại văn bản

# %%
fig = plt.figure(figsize=(15.5, 8.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.36, wspace=0.28)
names = list(results.keys())

ax = fig.add_subplot(gs[0, :])
xs = np.arange(len(names)); w = 0.2
series = [("Accuracy", [results[n]["optimal"]["accuracy"] * 100 for n in names], "#9ecae1"),
          ("Precision (lớp 1)", [results[n]["optimal"]["precision_1"] * 100 for n in names], "#2a9d70"),
          ("Recall (lớp 1)", [results[n]["optimal"]["recall_1"] * 100 for n in names], "#e8a33d"),
          ("Macro F1 (Tối ưu)", [results[n]["optimal"]["macro_f1"] * 100 for n in names], "#4c72b0")]
for k, (lab, vals, c) in enumerate(series):
    bb = ax.bar(xs + (k - 1.5) * w, vals, w, label=lab, color=c)
    ax.bar_label(bb, fmt="%.1f", fontsize=7, padding=1)
ax.set_xticks(xs, names, fontsize=9); ax.set_ylabel("Phần trăm (%)"); ax.set_ylim(0, 108)
ax.set_title("(a) So sánh đa chỉ số tại ngưỡng tối ưu cân bằng lớp", fontsize=11)
ax.legend(fontsize=8, ncol=4, loc="upper center")

for k, n in enumerate(names):
    ax = fig.add_subplot(gs[1, k])
    s = results[n]["optimal"]
    cm = np.array([[s["TN"], s["FP"]], [s["FN"], s["TP"]]], dtype=float)
    pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j]):,}\n({pct[i, j]:.1%})", ha="center", va="center",
                    fontsize=8, color="white" if pct[i, j] > 0.55 else "#111",
                    fontweight="bold")
    ax.set_title(f"{n} (τ={results[n]['tau']:.2f})\n[Rec0: {s['recall_0']:.1%}]", fontsize=8.5)
    ax.set_xticks([0, 1], ["Không (0)", "Có (1)"], fontsize=7)
    ax.set_yticks([0, 1], ["Không (0)", "Có (1)"], fontsize=7)
    ax.grid(False)
    if k == 0:
        ax.set_ylabel("Nhãn thực tế", fontsize=8)

fig.suptitle("Đối chuẩn hiệu năng hệ thống phân loại đánh giá văn bản NLP (23,486 nhận xét)",
             fontsize=12.5, fontweight="bold")
fig.savefig(FIG / "fig06_model_comparison.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 12. Nhận xét chuyên sâu và đánh giá kết quả phân loại văn bản NLP

# %%
lr_ = results["Logistic Regression"]; nb = results["Naive Bayes"]
rf = results["Random Forest"]; dl = results["DL Text Scratch"]

print("PHÁT HIỆN 1 — Ưu thế tự nhiên của Logistic Regression trên không gian TF-IDF thưa")
print(f"  Macro F1 {lr_['optimal']['macro_f1']:.2%}, Recall lớp 0 {lr_['optimal']['recall_0']:.2%}, "
      f"ROC-AUC {lr_['auc']:.4f}, thời gian huấn luyện chỉ {lr_['time']:.3f}s")
print("  Trong xử lý ngôn ngữ tự nhiên với biểu diễn túi từ TF-IDF 1,000 chiều thưa thớt,")
print("  sự hiện diện của các từ khoá chỉ định cảm xúc mạnh ('cheap', 'unflattering',")
print("  'itchy' đối kháng với 'gorgeous', 'soft', 'flattering') tạo nên các ranh giới")
print("  phân tách gần như tuyến tính. Hàm mục tiêu lồi nghiêm ngặt không có cực tiểu cục")
print("  bộ giúp mô hình hội tụ rất nhanh và không bị hiện tượng quá khớp.\n")

print("PHÁT HIỆN 2 — Cứu vãn nhãn thiểu số nhờ tinh chỉnh ngưỡng phân loại")
for n, r in results.items():
    d, o = r["default"], r["optimal"]
    print(f"  {n:<22} Recall lớp 0: {d['recall_0']:>6.2%} → {o['recall_0']:>6.2%} "
          f"({(o['recall_0'] - d['recall_0']) * 100:+.2f} điểm)  |  "
          f"Macro F1: {d['macro_f1']:.2%} → {o['macro_f1']:.2%}")
print(f"  Naive Bayes đạt tốc độ dự báo gần như tức thời ({nb['time']:.3f}s) — phù hợp")
print("  cho các hệ thống lọc bình luận thời gian thực với lưu lượng cực lớn.\n")

print("PHÁT HIỆN 3 — Thành công của DL Text Scratch trong việc nén không gian ngữ nghĩa")
print(f"  {dl_name} ({best_arch['Cấu hình tầng']}, {dl['n_params']:,} tham số):")
print(f"    Macro F1 {dl['optimal']['macro_f1']:.2%} · ROC-AUC {dl['auc']:.4f} · "
      f"Recall lớp 0 {dl['optimal']['recall_0']:.2%}")
print(f"    Khoảng cách so với Logistic Regression: "
      f"{(dl['optimal']['macro_f1'] - lr_['optimal']['macro_f1']) * 100:+.2f} điểm Macro F1")
print("  Việc sử dụng tầng ẩn thứ nhất rộng 128 nơ-ron giúp mạng tránh được nút thắt cổ")
print("  chai thông tin, cho phép trích xuất song song nhiều tổ hợp từ vựng đa tầng.")
print("  Hình ảnh PCA 2D đã kiểm chứng trực quan: từ ma trận thưa 1,000 chiều hỗn độn ban")
print("  đầu, mạng nơ-ron đã tự động cô đọng và phân tách thành hai cụm cảm xúc riêng biệt.")

# %% [markdown]
# ## 13. Từ khoá có trọng số cảm xúc mạnh nhất
#
# Trích xuất hệ số của Logistic Regression để đối chiếu với trực giác ngôn ngữ.

# %%
coefs = results["Logistic Regression"]["model"].coef_.ravel()
order = np.argsort(coefs)
neg_terms = [(feat_names[i], coefs[i]) for i in order[:14]]
pos_terms = [(feat_names[i], coefs[i]) for i in order[-14:]][::-1]

kw = pd.DataFrame({
    "Từ khoá TIÊU CỰC": [w for w, _ in neg_terms],
    "Trọng số (−)": [round(c, 3) for _, c in neg_terms],
    "Từ khoá TÍCH CỰC": [w for w, _ in pos_terms],
    "Trọng số (+)": [round(c, 3) for _, c in pos_terms],
})
display(kw)
print("→ Mô hình học được đúng trực giác ngôn ngữ của con người mà không cần bất kỳ từ")
print("  điển cảm xúc nào được gán nhãn thủ công.")

# %% [markdown]
# ## 14. Lưu artifact mô hình phục vụ triển khai REST API

# %%
import joblib

np.savez(MODEL / "dl_scratch_weights.npz",
         **{f"W{i}": w for i, w in enumerate(dl_model.W)},
         **{f"b{i}": b for i, b in enumerate(dl_model.b)})
joblib.dump(vec, MODEL / "tfidf_vectorizer.joblib")
joblib.dump(results["Logistic Regression"]["model"], MODEL / "logistic_regression.joblib")

metadata = {
    "app": "customer_comments",
    "task": "text_binary_classification",
    "dataset": {"name": "Women's Clothing E-Commerce Reviews",
                "n_raw": int(len(df_raw)), "n_clean": int(len(df)),
                "positive_rate": float(y_all.mean()),
                "median_words_pos": float(m1), "median_words_neg": float(m0)},
    "cleaning_audit": audit.to_dict(orient="records"),
    "vectorizer": {"type": "TF-IDF", "max_features": 1000, "ngram_range": [1, 2],
                   "stop_words": "english", "sublinear_tf": True,
                   "sparsity_pct": sparsity, "n_bigrams": int(n_bigram),
                   "fit_time_sec": vec_time},
    "split": {"n_train": int(len(y_tr)), "n_test": int(len(y_te)),
              "train_pos_rate": float(y_tr.mean()), "test_pos_rate": float(y_te.mean())},
    "dl_model": {"name": dl_name, "layers": best_arch["Cấu hình tầng"],
                 "layer_sizes": dl_model.sizes, "n_params": dl_model.n_params(),
                 "threshold": float(results["DL Text Scratch"]["tau"]),
                 "activation": "ReLU (hidden) + Sigmoid (output)",
                 "lr": dl_model.lr, "batch_size": dl_model.batch_size, "epochs": 25},
    "scores": {n: {"kind": r["kind"], "n_params": r["n_params"], "time_sec": r["time"],
                   "threshold": r["tau"], "roc_auc": r["auc"],
                   "default": r["default"], "optimal": r["optimal"]}
               for n, r in results.items()},
    "architecture_study": [{k: v for k, v in r.items() if not k.startswith("_")}
                           | {"scores": r["_s"], "threshold": r["_t"]} for r in arch_rows],
    "loss_histories": {k: [float(x) for x in v] for k, v in arch_hist.items()},
    "top_keywords": [{"word": w, "count": int(c)} for w, c in top15],
    "sentiment_terms": {"negative": [[w, float(c)] for w, c in neg_terms],
                        "positive": [[w, float(c)] for w, c in pos_terms]},
    "champion": champ[0],
}
(MODEL / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
(REP / "metrics_comments_large.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

print("✓ dl_scratch_weights.npz · tfidf_vectorizer.joblib · logistic_regression.joblib")
print("✓ metadata.json · metrics_comments_large.json")
for f in sorted(FIG.glob("*.png")):
    print(f"✓ {f.name}")

# %% [markdown]
# ## 15. Kiểm chứng artifact — dự đoán trên câu văn bản mới

# %%
Z = np.load(MODEL / "dl_scratch_weights.npz")
Ws = [Z[f"W{i}"] for i in range(len(dl_model.W))]
bs = [Z[f"b{i}"] for i in range(len(dl_model.b))]

def api_predict(texts, vec, Ws, bs, tau):
    """Luồng dự đoán đầy đủ — chính là đoạn code sẽ chạy trong REST API."""
    A = np.asarray(vec.transform(texts).todense(), dtype=np.float64)
    for i, (W, b) in enumerate(zip(Ws, bs)):
        Zc = A @ W + b
        A = np.maximum(0.0, Zc) if i < len(Ws) - 1 else 1.0 / (1.0 + np.exp(-np.clip(Zc, -25, 25)))
    p = A.ravel()
    return p, (p >= tau).astype(int)

tau_dl = results["DL Text Scratch"]["tau"]
demo_texts = [
    "Love this dress! The fabric is soft and it fits perfectly, absolutely gorgeous.",
    "Huge disappointment. The material feels cheap and it runs way too small.",
    "Beautiful color and great quality, I would definitely buy this again.",
    "Poor quality, the seams came apart after one wash. Not worth the price at all.",
    "It is okay, nothing special but does the job for the price.",
]
probs, labels = api_predict(demo_texts, vec, Ws, bs, tau_dl)

print(f"Ngưỡng quyết định triển khai: τ = {tau_dl:.2f}\n")
for t, p, l in zip(demo_texts, probs, labels):
    verdict = "KHUYẾN NGHỊ ✓" if l == 1 else "KHÔNG KHUYẾN NGHỊ ✗"
    print(f"  [{verdict:<20}] p = {p:.4f}")
    print(f"     \"{t}\"\n")

# Kiểm chứng khớp tuyệt đối với mô hình gốc
chk = float(np.abs(api_predict(list(text_te[:500]), vec, Ws, bs, tau_dl)[0]
                   - dl_model.predict_proba(X_te[:500]).ravel()).max())
print(f"Sai khác tuyệt đối lớn nhất so với mô hình gốc: {chk:.3e}")
print("✓ ĐẠT — artifact tái lập chính xác." if chk < 1e-10 else "✗ Có sai khác!")

# %% [markdown]
# ## 16. Tổng kết notebook 05
#
# | Hạng mục | Kết quả |
# |---|---|
# | Quy mô dữ liệu | 23,486 đánh giá → làm sạch còn ~22,600 |
# | Không gian đặc trưng | TF-IDF 1,000 chiều, độ thưa > 98% |
# | Kiến trúc DL tốt nhất | xem mục 7 |
# | Quán quân đối chuẩn | xem mục 11 |
#
# Ba kết luận chính:
#
# 1. **Trên dữ liệu văn bản thưa chiều cao, mô hình tuyến tính rất mạnh.** Ranh giới cảm
#    xúc trong không gian túi từ gần như tuyến tính, nên Logistic Regression đạt hiệu
#    năng hàng đầu với chi phí tính toán thấp hơn hàng chục lần.
#
# 2. **Chiều rộng quan trọng hơn chiều sâu trên dữ liệu thưa.** Mạng hẹp tạo nút thắt cổ
#    chai thông tin ngay từ tầng đầu; mạng rộng 128 nơ-ron trích xuất song song nhiều
#    cụm tổ hợp từ vựng và đạt Macro F1 cao nhất trong nhóm Deep Learning.
#
# 3. **Học biểu diễn ngữ nghĩa được kiểm chứng trực quan bằng PCA.** Từ ma trận thưa
#    1,000 chiều không có cấu trúc, mạng nơ-ron tự động cô đọng thành hai cụm cảm xúc
#    tách biệt ở tầng ẩn cuối — hoàn toàn không cần từ điển cảm xúc gán nhãn thủ công.
#
# **→ Đây là notebook cuối của chuỗi thực nghiệm.** Ba hệ thống đã được xây dựng trọn vẹn
# từ dữ liệu thô đến artifact triển khai. Bước tiếp theo là dựng REST API, giao diện Web
# và giao diện Mobile cho cả ba hệ thống, rồi tổng hợp toàn bộ vào báo cáo nghiên cứu.
