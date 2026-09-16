"""Sinh toàn bộ biểu đồ dùng cho notebook và báo cáo LaTeX.

Mỗi hàm nhận dữ liệu, vẽ một hình và ghi xuống ``figures/`` dưới dạng PNG.
Notebook và LaTeX cùng đọc từ thư mục đó, nên hình trong báo cáo luôn khớp
tuyệt đối với hình trong notebook (yêu cầu R14 — tính tái lập).
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from . import config as cfg

cfg.apply_plot_style()


# ==========================================================================
#  SƠ ĐỒ KIẾN TRÚC HỆ THỐNG
# ==========================================================================

def plot_system_diagram(name: str = "fig_system_diagram") -> None:
    """Vẽ sơ đồ luồng thông tin của hệ thống thông minh.

    Trực quan hoá chuỗi Environment → Input → Representation → Model →
    Prediction → Application mà đề bài yêu cầu mô tả trước khi huấn luyện.
    """
    stages = [
        ("ENVIRONMENT", "Thế giới thực: bệnh nhân / thị trường nhà đất", "#D8F3DC"),
        ("INPUT DATA", "Chỉ số sinh tồn · Thuộc tính căn nhà", "#DDEBF7"),
        ("REPRESENTATION", r"Vector đặc trưng $x = [x_1, \ldots, x_d]$", "#FFF3CD"),
        ("LEARNING MODEL", "5 thuật toán Học máy truyền thống", "#E7DFF5"),
        ("PREDICTION", "Nhãn bệnh / Giá nhà dự báo", "#FADBD8"),
        ("APPLICATION", "Bác sĩ · Nhà đầu tư · Đồ thị Tri thức", "#D6EAF8"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 8.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(stages) * 2)
    ax.axis("off")

    for i, (title, subtitle, colour) in enumerate(stages):
        y = (len(stages) - i - 1) * 2 + 0.35
        box = mpatches.FancyBboxPatch(
            (1.0, y), 8.0, 1.25,
            boxstyle="round,pad=0.08",
            facecolor=colour, edgecolor="#34495E", linewidth=1.6,
        )
        ax.add_patch(box)
        ax.text(5.0, y + 0.82, title, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color="#1B4F72")
        ax.text(5.0, y + 0.34, subtitle, ha="center", va="center",
                fontsize=9.5, color="#2C3E50")

        if i < len(stages) - 1:
            ax.annotate("", xy=(5.0, y - 0.32), xytext=(5.0, y - 0.02),
                        arrowprops=dict(arrowstyle="-|>", lw=1.9, color="#34495E"))

    cfg.save_fig(fig, name)


# ==========================================================================
#  PHÂN TÍCH KHÁM PHÁ — BIỂU ĐỒ ĐƠN BIẾN
# ==========================================================================

def plot_kde(df: pd.DataFrame, column: str, name: str, unit: str = "") -> None:
    """Ước lượng mật độ hạt nhân (KDE) cho một biến liên tục.

    Khác biểu đồ cột tần suất, KDE dùng đường cong liên tục nên thể hiện hình
    dáng phân phối mượt hơn và không phụ thuộc vào cách chọn độ rộng khoảng.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    sns.kdeplot(data=df, x=column, fill=True, color=cfg.PALETTE[0],
                alpha=0.55, linewidth=2.0, ax=ax)
    ax.set_title(f"Ước lượng mật độ (KDE) của {column}")
    ax.set_xlabel(f"{column} {unit}".strip())
    ax.set_ylabel("Mật độ")
    cfg.save_fig(fig, name)


def plot_boxplot(df: pd.DataFrame, column: str, name: str, unit: str = "") -> None:
    """Biểu đồ hộp — tóm tắt năm số thống kê và phát hiện giá trị ngoại lai.

    Hộp bao phủ khoảng tứ phân vị (Q1 đến Q3), đường giữa là trung vị, râu kéo
    dài tối đa 1.5·IQR, và mọi điểm nằm ngoài râu được đánh dấu là ngoại lai.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    sns.boxplot(data=df, y=column, color=cfg.PALETTE[1], width=0.35,
                flierprops=dict(marker="o", markersize=4, markerfacecolor="none",
                                markeredgecolor="#666"), ax=ax)
    ax.set_title(f"Biểu đồ hộp (Boxplot) của {column}")
    ax.set_ylabel(f"{column} {unit}".strip())
    cfg.save_fig(fig, name)


def plot_boxplot_grid(df: pd.DataFrame, columns: list[str], name: str,
                      ncols: int = 4) -> None:
    """Lưới biểu đồ hộp cho nhiều đặc trưng, dùng để so sánh mức độ phân tán."""
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, columns):
        sns.boxplot(data=df, y=col, ax=ax, width=0.4,
                    color=cfg.PALETTE[i_colour(col, columns)],
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor="none", markeredgecolor="#777"))
        ax.set_title(col, fontsize=11)
        ax.set_ylabel("")

    for ax in axes[len(columns):]:
        ax.axis("off")

    fig.suptitle("Biểu đồ hộp của các đặc trưng định lượng",
                 fontsize=14, fontweight="bold", y=1.005)
    fig.tight_layout()
    cfg.save_fig(fig, name)


def i_colour(col: str, columns: list[str]) -> int:
    """Gán màu ổn định theo vị trí cột để các hình dùng chung bảng màu."""
    return columns.index(col) % len(cfg.PALETTE)


def plot_histogram(df: pd.DataFrame, column: str, name: str, bins: int = 30) -> None:
    """Biểu đồ tần suất — đếm số quan sát rơi vào từng khoảng giá trị."""
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    sns.histplot(data=df, x=column, bins=bins, color=cfg.PALETTE[0],
                 edgecolor="white", ax=ax)
    ax.set_title(f"Phân phối tần suất của {column}")
    ax.set_ylabel("Số quan sát")
    cfg.save_fig(fig, name)


def plot_target_pie(y: pd.Series, name: str, labels: list[str]) -> None:
    """Biểu đồ tròn thể hiện tỷ trọng các lớp của biến mục tiêu."""
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90,
           colors=[cfg.PALETTE[0], cfg.PALETTE[1]],
           wedgeprops=dict(edgecolor="white", linewidth=2),
           textprops=dict(fontsize=11))
    ax.set_title("Phân phối biến mục tiêu Outcome")
    cfg.save_fig(fig, name)


# ==========================================================================
#  PHÂN TÍCH KHÁM PHÁ — BIỂU ĐỒ ĐA BIẾN
# ==========================================================================

def plot_correlation_heatmap(df: pd.DataFrame, name: str,
                             title: str = "Ma trận tương quan Pearson") -> None:
    """Bản đồ nhiệt tương quan Pearson giữa mọi cặp biến định lượng.

    Hệ số chỉ đo quan hệ **tuyến tính**: giá trị gần 0 không loại trừ khả năng
    tồn tại quan hệ phi tuyến mạnh, nên không dùng riêng nó để loại đặc trưng.
    """
    corr = df.select_dtypes(include="number").corr()
    size = max(7.0, 0.62 * len(corr))
    fig, ax = plt.subplots(figsize=(size, size * 0.82))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(title, pad=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    cfg.save_fig(fig, name)


def plot_kde_by_class(df: pd.DataFrame, columns: list[str], target: str,
                      name: str, labels: list[str]) -> None:
    """So sánh phân phối từng đặc trưng giữa hai nhóm của biến mục tiêu.

    Đặc trưng nào có hai đường cong tách xa nhau thì mang nhiều tín hiệu phân
    biệt; đặc trưng nào có hai đường chồng khít thì gần như vô dụng với mô hình.
    """
    ncols = min(3, len(columns))
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, columns):
        for value, label, colour in zip(sorted(df[target].unique()), labels, cfg.PALETTE):
            sns.kdeplot(data=df[df[target] == value], x=col, fill=True, alpha=0.4,
                        linewidth=1.8, color=colour, label=label, ax=ax)
        ax.set_title(col)
        ax.set_ylabel("Mật độ")
        ax.legend(fontsize=9)

    for ax in axes[len(columns):]:
        ax.axis("off")

    fig.suptitle("Phân phối đặc trưng theo tình trạng bệnh",
                 fontsize=14, fontweight="bold", y=1.005)
    fig.tight_layout()
    cfg.save_fig(fig, name)


def plot_scatter(df: pd.DataFrame, x: str, y: str, name: str,
                 hue: str | None = None, sample: int = 4000) -> None:
    """Biểu đồ phân tán giữa hai biến định lượng.

    Lấy mẫu ngẫu nhiên khi dữ liệu quá lớn: vẽ đủ 30 nghìn điểm sẽ tạo một mảng
    mực đặc, che mất chính cấu trúc quan hệ mà biểu đồ cần thể hiện.
    """
    data = df.sample(min(sample, len(df)), random_state=cfg.RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sns.scatterplot(data=data, x=x, y=y, hue=hue, alpha=0.35, s=18,
                    color=cfg.PALETTE[0], edgecolor=None, ax=ax)
    ax.set_title(f"Quan hệ giữa {x} và {y}")
    cfg.save_fig(fig, name)


def plot_category_bar(df: pd.DataFrame, column: str, target: str, name: str,
                      top: int = 12, ylabel: str = "Giá trung bình (tỷ VNĐ)") -> None:
    """Giá trị trung bình của biến mục tiêu theo từng hạng mục định danh."""
    agg = (df.groupby(column)[target].agg(["mean", "count"])
             .query("count >= 30").sort_values("mean", ascending=False).head(top))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    sns.barplot(x=agg.index, y=agg["mean"], hue=agg.index, legend=False,
                palette="viridis", ax=ax)
    ax.set_title(f"{ylabel} theo {column}")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    cfg.save_fig(fig, name)


def plot_missing_bar(df: pd.DataFrame, name: str) -> None:
    """Tỷ lệ thiếu dữ liệu của từng cột, sắp xếp giảm dần."""
    ratio = (df.isna().mean() * 100).sort_values(ascending=False)
    ratio = ratio[ratio > 0]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    sns.barplot(x=ratio.values, y=ratio.index, hue=ratio.index, legend=False,
                palette="rocket_r", ax=ax)
    ax.set_title("Tỷ lệ dữ liệu khuyết theo từng thuộc tính")
    ax.set_xlabel("Tỷ lệ thiếu (%)")
    for i, v in enumerate(ratio.values):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9)
    cfg.save_fig(fig, name)


# ==========================================================================
#  BIỂU ĐỒ KẾT QUẢ MÔ HÌNH
# ==========================================================================

def plot_model_comparison(results: pd.DataFrame, metrics: list[str], name: str,
                          title: str) -> None:
    """Biểu đồ cột nhóm so sánh nhiều mô hình trên nhiều độ đo cùng lúc."""
    melted = results.melt(id_vars="Model", value_vars=metrics,
                          var_name="Độ đo", value_name="Giá trị")
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    sns.barplot(data=melted, x="Model", y="Giá trị", hue="Độ đo",
                palette=cfg.PALETTE[: len(metrics)], ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.legend(title="", ncols=len(metrics), loc="lower right", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7.5, padding=1.5)
    cfg.save_fig(fig, name)


def plot_confusion_matrix(pipeline, X_test, y_test, name: str,
                          labels: list[str], threshold: float | None = None) -> None:
    """Ma trận nhầm lẫn — bốn ô TP/TN/FP/FN của một mô hình phân lớp."""
    if threshold is None:
        y_pred = pipeline.predict(X_test)
        subtitle = "ngưỡng mặc định 0.50"
    else:
        y_pred = (pipeline.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        subtitle = f"ngưỡng tối ưu {threshold:.2f}"

    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=labels, cmap="Blues",
        colorbar=False, values_format="d", ax=ax
    )
    ax.set_title(f"Ma trận nhầm lẫn ({subtitle})")
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    cfg.save_fig(fig, name)


def plot_threshold_curve(sweep: pd.DataFrame, name: str) -> None:
    """Đường cong đánh đổi Precision–Recall theo ngưỡng quyết định."""
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for metric, colour in zip(["Accuracy", "Precision", "Recall", "F1-Score"], cfg.PALETTE):
        ax.plot(sweep["Threshold"], sweep[metric], marker="o", linewidth=2,
                color=colour, label=metric)

    best = sweep.loc[sweep["F1-Score"].idxmax(), "Threshold"]
    ax.axvline(best, linestyle="--", color="#C73E1D", linewidth=1.4)

    at_edge = best in (sweep["Threshold"].min(), sweep["Threshold"].max())
    note = f"  θ tối ưu = {best:.2f}" + ("  (chạm biên dải quét)" if at_edge else "")
    # Toạ độ theo hệ trục (0..1) chứ không theo giá trị dữ liệu: đặt theo giá
    # trị tuyệt đối sẽ ghim nhãn ngoài vùng vẽ và kéo giãn khung hình.
    ax.text(best, 0.02, note, transform=ax.get_xaxis_transform(),
            color="#C73E1D", fontsize=10, fontweight="bold", va="bottom")

    ax.set_title("Ảnh hưởng của ngưỡng quyết định tới các độ đo")
    ax.set_xlabel("Ngưỡng quyết định θ")
    ax.set_ylabel("Giá trị độ đo")
    ax.legend(ncols=4, fontsize=9)
    cfg.save_fig(fig, name)


def _feature_names(pipeline, expected: int) -> list[str]:
    """Lấy tên đặc trưng ở đầu ra bước tiền xử lý.

    ``FunctionTransformer`` bọc hàm thiết kế đặc trưng không khai báo
    ``feature_names_out``, nên ``get_feature_names_out()`` có thể ném lỗi. Khi
    đó lùi về đọc tên cột mà bước trước đó đã ghi lại, và cuối cùng mới dùng
    tên đánh số — để biểu đồ vẫn vẽ được thay vì làm hỏng cả pipeline.
    """
    prep_step = pipeline.named_steps["prep"]
    for getter in (
        lambda: list(prep_step.get_feature_names_out()),
        lambda: list(prep_step[-1].feature_names_in_),
    ):
        try:
            names = getter()
            if len(names) == expected:
                return names
        except Exception:
            continue
    return [f"feature_{i}" for i in range(expected)]


def plot_hyperparameter_sweep(sweep: pd.DataFrame, param: str, metrics: list[str],
                              name: str, title: str, higher_is_better: bool = True) -> None:
    """Đường cong hiệu năng theo từng giá trị của một siêu tham số.

    Nếu giá trị tối ưu rơi đúng vào đầu mút của dải khảo sát thì chưa thể kết
    luận đó là cực trị — hàm sẽ ghi chú cảnh báo ngay trên hình.
    """
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for metric, colour in zip(metrics, cfg.PALETTE):
        ax.plot(sweep[param], sweep[metric], marker="o", linewidth=2,
                color=colour, label=metric)

    primary = metrics[0]
    idx = sweep[primary].idxmax() if higher_is_better else sweep[primary].idxmin()
    best = sweep.loc[idx, param]
    ax.axvline(best, linestyle="--", color="#C73E1D", linewidth=1.4)

    at_edge = best in (sweep[param].iloc[0], sweep[param].iloc[-1])
    note = f"  {param} tối ưu = {best:g}" + ("  (chạm biên dải khảo sát)" if at_edge else "")
    ax.text(best, ax.get_ylim()[0], note, color="#C73E1D", fontsize=9.5,
            fontweight="bold", va="bottom")

    ax.set_title(title)
    ax.set_xlabel(f"Giá trị siêu tham số {param}")
    ax.set_ylabel("Giá trị độ đo")
    ax.legend(ncols=len(metrics), fontsize=9)
    cfg.save_fig(fig, name)


def plot_feature_importance(pipeline, name: str, top: int = 15,
                            title: str = "Mức độ quan trọng của đặc trưng") -> None:
    """Xếp hạng đóng góp của từng đặc trưng trong mô hình dạng cây."""
    model = pipeline.named_steps["model"]
    features = _feature_names(pipeline, len(model.feature_importances_))
    importance = pd.Series(model.feature_importances_, index=features)
    importance = importance.sort_values(ascending=False).head(top)[::-1]

    # Bỏ tiền tố "num__" / "cat__" do ColumnTransformer sinh ra cho dễ đọc.
    importance.index = [str(i).split("__", 1)[-1] for i in importance.index]

    fig, ax = plt.subplots(figsize=(8.4, 0.38 * len(importance) + 1.6))
    sns.barplot(x=importance.values, y=importance.index, hue=importance.index,
                legend=False, palette="mako_r", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Mức độ quan trọng")
    cfg.save_fig(fig, name)


def plot_prediction_scatter(y_true, y_pred, name: str,
                            unit: str = "tỷ VNĐ", sample: int = 3000) -> None:
    """Đối chiếu giá trị dự đoán với giá trị thực tế của mô hình hồi quy.

    Điểm càng bám sát đường chéo y = ŷ thì dự báo càng chính xác; độ phân tán
    quanh đường chéo chính là sai số mô hình được thể hiện bằng hình ảnh.
    """
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    idx = rng.choice(len(y_true), size=min(sample, len(y_true)), replace=False)
    yt, yp = np.asarray(y_true)[idx], np.asarray(y_pred)[idx]

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(yt, yp, alpha=0.3, s=16, color=cfg.PALETTE[0], edgecolor=None)
    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="#C73E1D", linewidth=1.8,
            label="Dự báo hoàn hảo")
    ax.set_xlabel(f"Giá thực tế ({unit})")
    ax.set_ylabel(f"Giá dự đoán ({unit})")
    ax.set_title("Giá dự đoán so với giá thực tế")
    ax.legend()
    cfg.save_fig(fig, name)


def plot_representation_comparison(results: pd.DataFrame, metric: str, name: str,
                                   title: str) -> None:
    """So sánh hiệu năng giữa các phương án biểu diễn dữ liệu."""
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    sns.barplot(data=results, x=metric, y="Representation", hue="Representation",
                legend=False, palette="crest", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("")
    for i, v in enumerate(results[metric]):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=9.5, fontweight="bold")
    cfg.save_fig(fig, name)


def plot_ood_comparison(test_df: pd.DataFrame, ood_df: pd.DataFrame,
                        metric: str, name: str, title: str) -> None:
    """Đối chiếu hiệu năng trên tập kiểm tra và tập ngoại kiểm.

    Chênh lệch lớn giữa hai cột là dấu hiệu mô hình đã khớp quá mức vào đặc thù
    của tập kiểm tra thay vì học được quy luật tổng quát.
    """
    merged = pd.DataFrame({
        "Model": test_df["Model"],
        "Tập kiểm tra": test_df[metric].values,
        "Tập ngoại kiểm": ood_df.set_index("Model").loc[test_df["Model"], metric].values,
    }).melt(id_vars="Model", var_name="Tập dữ liệu", value_name=metric)

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    sns.barplot(data=merged, x="Model", y=metric, hue="Tập dữ liệu",
                palette=[cfg.PALETTE[0], cfg.PALETTE[2]], ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    ax.legend(title="")
    cfg.save_fig(fig, name)


# ==========================================================================
#  BIỂU ĐỒ BỔ SUNG — ROC, LƯỚI MA TRẬN NHẦM LẪN, PHẦN DƯ, EDA LIÊN HỢP
# ==========================================================================

def plot_roc_curves(models: dict, X_test, y_test, name: str, title: str) -> None:
    """Đường cong ROC của nhiều mô hình phân lớp trên cùng một trục.

    ROC vẽ Recall (tỷ lệ Dương tính thật) theo tỷ lệ Dương tính giả khi ngưỡng
    quyết định quét từ 1 về 0, nên nó đo **khả năng xếp hạng** của mô hình chứ
    không đo chất lượng tại một ngưỡng cụ thể. Đây là lý do ROC-AUC là độ đo duy
    nhất trong nhóm không đổi khi ta chỉnh ngưỡng ở Thực nghiệm 1 — điều mà biểu
    đồ cột so sánh mô hình không thể hiện được.

    Mô hình không có ``predict_proba`` bị bỏ qua có chủ đích: mô hình cơ sở đoán
    lớp đa số không sinh ra xác suất liên tục nào để xếp hạng, nên vẽ nó lên đây
    là vô nghĩa chứ không phải là thiếu sót.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(7.4, 6.0))

    duong_cong = []
    for ten, pipeline in models.items():
        if not hasattr(pipeline, "predict_proba"):
            continue
        diem = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, diem)
        duong_cong.append((roc_auc_score(y_test, diem), ten, fpr, tpr))

    # Xếp theo AUC giảm dần để chú giải đọc được như một bảng xếp hạng.
    duong_cong.sort(reverse=True, key=lambda t: t[0])
    for (auc, ten, fpr, tpr), colour in zip(duong_cong, cfg.PALETTE * 3):
        ax.plot(fpr, tpr, linewidth=2.1, color=colour,
                label=f"{ten} — AUC = {auc:.4f}")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.3, color="#8A8A8A",
            label="Đoán ngẫu nhiên — AUC = 0.5000")

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_title(title)
    ax.set_xlabel("Tỷ lệ Dương tính giả (1 − Độ đặc hiệu)")
    ax.set_ylabel("Tỷ lệ Dương tính thật (Recall)")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    cfg.save_fig(fig, name)


def plot_confusion_grid(models: dict, X_test, y_test, name: str,
                        labels: list[str], title: str) -> None:
    """Lưới ma trận nhầm lẫn — mọi mô hình cạnh nhau trên cùng một thang màu.

    Bảng so sánh độ đo cho biết mô hình nào tốt hơn; lưới này cho biết chúng
    **sai khác nhau kiểu gì**. Hai mô hình cùng Accuracy có thể một bên thiên về
    bỏ sót người bệnh (FN cao) và bên kia thiên về báo động nhầm (FP cao) — khác
    biệt quyết định trong bài toán sàng lọc, mà con số Accuracy gộp lại che mất.

    Mọi ô dùng chung ``vmax``; nếu để mỗi ma trận tự chuẩn hoá theo giá trị lớn
    nhất của riêng nó thì màu sắc giữa các ô không so sánh được với nhau.
    """
    from sklearn.metrics import confusion_matrix

    ten_mo_hinh = list(models)
    cols = 3
    rows = int(np.ceil(len(ten_mo_hinh) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.9 * rows))
    axes = np.atleast_1d(axes).ravel()

    ma_tran = {ten: confusion_matrix(y_test, mo_hinh.predict(X_test))
               for ten, mo_hinh in models.items()}
    vmax = max(m.max() for m in ma_tran.values())

    for ax, ten in zip(axes, ten_mo_hinh):
        cm = ma_tran[ten]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    vmin=0, vmax=vmax, square=True,
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    annot_kws={"fontsize": 12, "fontweight": "bold"})
        fn, fp = int(cm[1, 0]), int(cm[0, 1])
        ax.set_title(f"{ten}\nFN = {fn} · FP = {fp}", fontsize=10.5)
        ax.set_xlabel("Nhãn dự đoán", fontsize=9)
        ax.set_ylabel("Nhãn thực tế", fontsize=9)
        ax.tick_params(labelsize=8.5)

    for ax in axes[len(ten_mo_hinh):]:
        ax.axis("off")

    # Tiêu đề hai dòng của mỗi ô cao gần gấp đôi tiêu đề thường, nên bố cục mặc
    # định để hàng dưới đè lên nhãn trục của hàng trên. ``tight_layout`` với
    # ``rect`` chừa sẵn dải trên cùng cho tiêu đề chung.
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.4)
    cfg.save_fig(fig, name)


def plot_residual_analysis(y_true, y_pred, name: str, title: str,
                           unit: str = "tỷ VNĐ") -> None:
    """Phân tích phần dư của mô hình hồi quy — bốn bảng.

    Bài toán hồi quy không có đường cong ROC, nhưng câu hỏi mà ROC trả lời cho
    bài phân lớp — *mô hình sai theo kiểu gì, chứ không phải sai bao nhiêu* —
    thì vẫn nguyên giá trị. Bốn bảng ở đây trả lời câu ấy:

    1. Phần dư theo giá trị dự báo — lộ ra độ chệch có hệ thống nếu đám mây điểm
       không nằm cân quanh trục 0.
    2. Phân phối phần dư — kiểm tra tính đối xứng và độ dày đuôi.
    3. Q–Q chuẩn — phần dư có theo phân phối chuẩn hay không.
    4. Độ chệch trung bình theo ngũ phân vị giá thật — bảng quan trọng nhất, vì
       nó phát hiện hiện tượng hồi quy về trung bình: định giá vượt ở phân khúc
       rẻ và định giá hụt ở phân khúc đắt.
    """
    import scipy.stats as stats

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    phan_du = y_pred - y_true

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.4))

    # --- (1) Phần dư theo giá trị dự báo ---
    ax = axes[0, 0]
    ax.scatter(y_pred, phan_du, s=9, alpha=0.22, color=cfg.PALETTE[0],
               edgecolors="none")
    ax.axhline(0, linestyle="--", linewidth=1.4, color="#C73E1D")
    ax.set_title("Phần dư theo giá trị dự báo")
    ax.set_xlabel(f"Giá dự báo ({unit})")
    ax.set_ylabel(f"Phần dư = Dự báo − Thực tế ({unit})")

    # --- (2) Phân phối phần dư ---
    ax = axes[0, 1]
    sns.histplot(phan_du, bins=45, kde=True, color=cfg.PALETTE[0], ax=ax)
    ax.axvline(0, linestyle="--", linewidth=1.4, color="#C73E1D")
    ax.set_title(f"Phân phối phần dư (độ chệch trung bình = {phan_du.mean():+.4f})")
    ax.set_xlabel(f"Phần dư ({unit})")
    ax.set_ylabel("Số quan sát")

    # --- (3) Q–Q chuẩn ---
    ax = axes[1, 0]
    stats.probplot(phan_du, dist="norm", plot=ax)
    ax.get_lines()[0].set(markersize=2.6, alpha=0.35, color=cfg.PALETTE[0])
    ax.get_lines()[1].set(linewidth=1.6, color="#C73E1D")
    ax.set_title("Đồ thị Q–Q so với phân phối chuẩn")
    ax.set_xlabel("Phân vị lý thuyết")
    ax.set_ylabel("Phân vị mẫu")

    # --- (4) Độ chệch theo ngũ phân vị giá thật ---
    ax = axes[1, 1]
    khung = pd.DataFrame({"that": y_true, "chech": phan_du})
    khung["nhom"] = pd.qcut(khung["that"], 5,
                            labels=["Q1\nrẻ nhất", "Q2", "Q3", "Q4", "Q5\nđắt nhất"])
    gom = khung.groupby("nhom", observed=True)["chech"].mean()
    mau = ["#2E86AB" if v >= 0 else "#C73E1D" for v in gom]
    thanh = ax.bar(gom.index.astype(str), gom.to_numpy(), color=mau)
    ax.bar_label(thanh, fmt="%+.3f", fontsize=9.5, fontweight="bold", padding=2)
    ax.axhline(0, linewidth=1.2, color="#333333")
    ax.set_title("Độ chệch trung bình theo ngũ phân vị giá thật")
    ax.set_xlabel("Ngũ phân vị theo giá thực tế")
    ax.set_ylabel(f"Độ chệch trung bình ({unit})")

    fig.suptitle(title, fontsize=13.5, fontweight="bold")
    cfg.save_fig(fig, name)


def plot_price_area_bedrooms(df: pd.DataFrame, name: str, title: str) -> None:
    """Trực quan hoá khám phá liên hợp Giá – Diện tích – Số phòng ngủ.

    Ba biến này là ba trục mà người mua nhà thật sự cân nhắc. Vẽ riêng lẻ từng
    phân phối thì mỗi hình chỉ trả lời được một câu hỏi biên; vẽ chung sáu bảng
    thì trả lời được câu hỏi *quan hệ*, và trong tập dữ liệu này chính quan hệ
    mới là chỗ có phát hiện bất ngờ.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2))

    # --- Hàng 1: ba phân phối biên ---
    ax = axes[0, 0]
    sns.histplot(df["Price"], bins=32, kde=True, color=cfg.PALETTE[0], ax=ax)
    ax.axvline(df["Price"].median(), linestyle="--", color="#C73E1D", linewidth=1.5)
    ax.set_title(f"Phân phối Giá — độ lệch = {df['Price'].skew():.3f}")
    ax.set_xlabel("Giá (tỷ VNĐ)")
    ax.set_ylabel("Số tin đăng")

    ax = axes[0, 1]
    sns.histplot(df["Area"].clip(upper=300), bins=40, kde=True,
                 color=cfg.PALETTE[1], ax=ax)
    ax.axvline(df["Area"].median(), linestyle="--", color="#C73E1D", linewidth=1.5)
    ax.set_title(f"Phân phối Diện tích — độ lệch = {df['Area'].skew():.3f}")
    ax.set_xlabel("Diện tích (m², cắt hiển thị tại 300)")
    ax.set_ylabel("Số tin đăng")

    ax = axes[0, 2]
    dem = df["Bedrooms"].dropna().astype(int).value_counts().sort_index()
    dem = dem[dem.index <= 8]
    thanh = ax.bar(dem.index.astype(str), dem.to_numpy(), color=cfg.PALETTE[2])
    ax.bar_label(thanh, fmt="%d", fontsize=8.5, padding=1.5)
    ax.set_title("Phân phối Số phòng ngủ")
    ax.set_xlabel("Số phòng ngủ")
    ax.set_ylabel("Số tin đăng")

    # --- Hàng 2: ba quan hệ chéo ---
    ax = axes[1, 0]
    ax.scatter(df["Area"].clip(upper=300), df["Price"], s=8, alpha=0.16,
               color=cfg.PALETTE[0], edgecolors="none")
    r_ap = df[["Area", "Price"]].corr().iloc[0, 1]
    ax.set_title(f"Giá theo Diện tích — Pearson r = {r_ap:.3f}")
    ax.set_xlabel("Diện tích (m², cắt hiển thị tại 300)")
    ax.set_ylabel("Giá (tỷ VNĐ)")

    ax = axes[1, 1]
    con = df.dropna(subset=["Bedrooms"]).copy()
    con["Bedrooms"] = con["Bedrooms"].astype(int)
    con = con[con["Bedrooms"].between(1, 6)]
    sns.boxplot(data=con, x="Bedrooms", y="Price", ax=ax,
                color=cfg.PALETTE[3], width=0.62, fliersize=1.4)
    r_bp = df[["Bedrooms", "Price"]].corr().iloc[0, 1]
    ax.set_title(f"Giá theo Số phòng ngủ — Pearson r = {r_bp:.3f}")
    ax.set_xlabel("Số phòng ngủ")
    ax.set_ylabel("Giá (tỷ VNĐ)")

    ax = axes[1, 2]
    don_gia = (df["Price"] * 1000 / df["Area"].clip(lower=1)).clip(upper=400)
    sns.histplot(don_gia, bins=45, kde=True, color=cfg.PALETTE[4], ax=ax)
    ax.axvline(don_gia.median(), linestyle="--", color="#C73E1D", linewidth=1.5)
    ax.set_title(f"Đơn giá mỗi m² — trung vị {don_gia.median():.1f} triệu đ")
    ax.set_xlabel("Đơn giá (triệu đ/m², cắt hiển thị tại 400)")
    ax.set_ylabel("Số tin đăng")

    fig.suptitle(title, fontsize=13.5, fontweight="bold")
    cfg.save_fig(fig, name)


# ==========================================================================
#  SƠ ĐỒ KIẾN TRÚC ĐỒ THỊ TRI THỨC ĐA TẦNG
# ==========================================================================

def plot_kg_architecture(layers: list[dict], node_colours: dict, name: str,
                         title: str, subtitle: str = "") -> None:
    """Sơ đồ kiến trúc đa tầng của một Đồ thị Tri thức.

    Khác với ảnh chụp đồ thị của một ca cụ thể (``kg_medical.png``,
    ``kg_property.png``) — vốn cho thấy *dữ liệu* — hình này cho thấy *lược đồ*:
    đồ thị có mấy tầng, mỗi tầng chứa loại thực thể nào, và thông tin chảy theo
    hướng nào giữa các tầng.

    Danh sách tầng đọc từ ``ontology.KG_LAYERS`` chứ không khai lại tại đây, nên
    khi bản thể học thêm một tầng thì sơ đồ tự cập nhật theo — không có chuyện
    hình vẽ trong báo cáo mô tả một kiến trúc mà mã nguồn không còn dùng.
    """
    import matplotlib.patches as patches

    n = len(layers)
    fig, ax = plt.subplots(figsize=(12.2, 1.62 * n + 1.5))

    cao = 1.0          # chiều cao mỗi dải tầng
    khoang = 0.42      # khoảng trống giữa hai dải
    rong = 11.4
    VUNG_CHU = 5.9     # bề ngang dành riêng cho tên và mô tả tầng

    for i, layer in enumerate(layers):
        y = (n - 1 - i) * (cao + khoang)

        # Dải nền của tầng.
        ax.add_patch(patches.FancyBboxPatch(
            (0.0, y), rong, cao,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor="#F4F7FA", edgecolor="#B9C6D4", linewidth=1.1, zorder=1))

        # Số thứ tự tầng.
        ax.add_patch(patches.Circle((0.46, y + cao / 2), 0.24,
                                    facecolor="#123A75", edgecolor="none", zorder=3))
        ax.text(0.46, y + cao / 2, str(layer["stt"]), ha="center", va="center",
                color="white", fontsize=12, fontweight="bold", zorder=4)

        # Tên và mô tả tầng.
        ax.text(0.92, y + cao * 0.68, f"Lớp {layer['stt']} — {layer['ten']}",
                ha="left", va="center", fontsize=11.5, fontweight="bold",
                color="#123A75", zorder=4)
        ax.text(0.92, y + cao * 0.33, layer["mo_ta"], ha="left", va="center",
                fontsize=9.2, color="#40515F", zorder=4)

        # Các viên loại thực thể, xếp từ mép phải vào.
        #
        # Bề rộng được co lại khi cần: một tầng có bốn loại thực thể tên dài sẽ
        # tràn sang vùng chữ mô tả bên trái và đè lên nó. Vì matplotlib không tự
        # phát hiện va chạm giữa hai đối tượng vẽ, chữ vẫn hiện ra chồng lên
        # nhau mà không lỗi nào được ném — nên phép co phải làm tường minh ở đây.
        khe = 0.10
        rong_chip = [0.135 * len(t) + 0.30 for t in layer["types"]]
        can = sum(rong_chip) + khe * (len(rong_chip) - 1)
        co_the = rong - 0.28 - VUNG_CHU
        ty_le = min(1.0, co_the / can) if can > 0 else 1.0

        x = rong - 0.28
        for loai, w in zip(reversed(layer["types"]), reversed(rong_chip)):
            do_rong = w * ty_le
            x -= do_rong
            ax.add_patch(patches.FancyBboxPatch(
                (x, y + cao * 0.28), do_rong, cao * 0.44,
                boxstyle="round,pad=0.015,rounding_size=0.07",
                facecolor=node_colours.get(loai, "#9AA5B1"),
                edgecolor="none", alpha=0.92, zorder=3))
            ax.text(x + do_rong / 2, y + cao * 0.50, loai, ha="center", va="center",
                    fontsize=8.6 * min(1.0, ty_le ** 0.5), color="white",
                    fontweight="bold", zorder=4)
            x -= khe * ty_le

        # Mũi tên nối xuống tầng dưới.
        if i < n - 1:
            ax.annotate("", xy=(rong / 2, y - khoang + 0.03),
                        xytext=(rong / 2, y - 0.03),
                        arrowprops=dict(arrowstyle="-|>", linewidth=1.7,
                                        color="#123A75", shrinkA=0, shrinkB=0),
                        zorder=2)

    ax.set_xlim(-0.15, rong + 0.15)
    ax.set_ylim(-0.35, n * (cao + khoang) + 0.30)
    ax.axis("off")
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""),
                 fontsize=13, fontweight="bold", pad=14)
    cfg.save_fig(fig, name)
