"""Đo lường hiệu năng mô hình và chạy các thực nghiệm có kiểm soát.

Toàn bộ mô hình được đánh giá dưới **cùng một giao thức**: cùng tập kiểm tra,
cùng hạt giống ngẫu nhiên, cùng bộ độ đo. Chỉ khi giao thức đồng nhất thì việc
so sánh giữa các mô hình mới có ý nghĩa khoa học (yêu cầu R8, R10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from . import config as cfg


# ==========================================================================
#  BỘ ĐỘ ĐO
# ==========================================================================

def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    """Tính bộ độ đo phân lớp.

    Chỉ báo cáo Accuracy là không đủ với dữ liệu mất cân bằng: một mô hình luôn
    đoán "khoẻ mạnh" vẫn đạt 65% trên tập này trong khi bỏ sót toàn bộ người
    bệnh. Recall mới là độ đo phản ánh đúng rủi ro lâm sàng.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
    return metrics


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Tính bộ độ đo hồi quy.

    MAE và RMSE cùng đơn vị với biến mục tiêu (tỷ VNĐ) nên đọc được trực tiếp;
    RMSE phạt nặng sai số lớn hơn MAE, còn R² cho biết mô hình giải thích được
    bao nhiêu phần phương sai của giá.
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


# ==========================================================================
#  THỰC NGHIỆM 1 — SO SÁNH MÔ HÌNH
# ==========================================================================

def compare_classifiers(fitted: dict, X_test, y_test) -> pd.DataFrame:
    """Chấm điểm mọi mô hình phân lớp trên cùng tập kiểm tra."""
    rows = []
    for name, pipe in fitted.items():
        y_pred = pipe.predict(X_test)
        proba = (
            pipe.predict_proba(X_test)[:, 1]
            if hasattr(pipe.named_steps["model"], "predict_proba")
            else None
        )
        rows.append({"Model": name, **classification_metrics(y_test, y_pred, proba)})
    return pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)


def compare_regressors(fitted: dict, X_test, y_test) -> pd.DataFrame:
    """Chấm điểm mọi mô hình hồi quy trên cùng tập kiểm tra."""
    rows = []
    for name, pipe in fitted.items():
        y_pred = pipe.predict(X_test)
        rows.append({"Model": name, **regression_metrics(y_test, y_pred)})
    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)


# ==========================================================================
#  THỰC NGHIỆM 2 — KHẢO SÁT SIÊU THAM SỐ
# ==========================================================================

def sweep_hyperparameter(
    pipeline_factory,
    values: list,
    X_train,
    y_train,
    X_test,
    y_test,
    metric_fn,
    metric_key: str,
    param_name: str,
) -> pd.DataFrame:
    """Huấn luyện lại mô hình cho từng giá trị siêu tham số và ghi lại kết quả.

    Chỉ **một** siêu tham số thay đổi giữa các lần chạy, mọi yếu tố khác giữ
    nguyên — đó là điều khiến đây là thực nghiệm có kiểm soát chứ không phải
    dò tìm ngẫu nhiên.
    """
    rows = []
    for value in values:
        pipe = pipeline_factory(value)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rows.append({param_name: value, **metric_fn(y_test, y_pred)})
    return pd.DataFrame(rows)


def sweep_decision_threshold(
    pipeline, X_test, y_test, thresholds: list[float]
) -> pd.DataFrame:
    """Khảo sát ảnh hưởng của ngưỡng quyết định tới cân bằng Precision–Recall.

    Ngưỡng mặc định 0.50 tối ưu cho Accuracy, nhưng trong sàng lọc y tế, bỏ sót
    một ca bệnh (FN) tốn kém hơn nhiều so với một cảnh báo nhầm (FP). Hạ ngưỡng
    là cách đánh đổi Precision lấy Recall một cách có chủ đích.
    """
    proba = pipeline.predict_proba(X_test)[:, 1]
    rows = []
    for theta in thresholds:
        y_pred = (proba >= theta).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "Threshold": theta,
                **classification_metrics(y_test, y_pred),
                "TP": int(tp),
                "FN": int(fn),
                "FP": int(fp),
                "TN": int(tn),
            }
        )
    return pd.DataFrame(rows)


# ==========================================================================
#  THỰC NGHIỆM 3 — KHẢO SÁT BIỂU DIỄN DỮ LIỆU
# ==========================================================================

def compare_representations(
    variants: dict[str, tuple], model_factory, metric_fn, sort_key: str
) -> pd.DataFrame:
    """So sánh cùng một thuật toán trên các cách biểu diễn dữ liệu khác nhau.

    Thuật toán và siêu tham số giữ nguyên tuyệt đối; thứ duy nhất thay đổi là
    ma trận đặc trưng X. Mọi chênh lệch kết quả vì thế quy hoàn toàn về chất
    lượng biểu diễn — đúng luận điểm trung tâm của bài tập.

    ``variants`` ánh xạ tên phương án tới bộ ``(X_train, X_test, y_train, y_test)``.
    """
    rows = []
    for name, (X_train, X_test, y_train, y_test) in variants.items():
        pipe = model_factory()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rows.append({"Representation": name, **metric_fn(y_test, y_pred)})
    ascending = sort_key in {"MAE", "MSE", "RMSE"}
    return pd.DataFrame(rows).sort_values(sort_key, ascending=ascending).reset_index(drop=True)


def correlation_feature_selection(
    X: pd.DataFrame, y: pd.Series, threshold: float = 0.10
) -> list[str]:
    """Chọn đặc trưng có tương quan tuyến tính với mục tiêu vượt ngưỡng.

    Đây là phương pháp lọc (filter) đơn giản nhất, chạy trước khi huấn luyện và
    độc lập với thuật toán. Hạn chế đã biết: nó chỉ thấy quan hệ tuyến tính, nên
    có thể loại nhầm đặc trưng có ảnh hưởng phi tuyến mạnh.
    """
    numeric = X.select_dtypes(include="number")
    corr = numeric.corrwith(y).abs()
    return corr[corr >= threshold].index.tolist()


# ==========================================================================
#  XUẤT KẾT QUẢ
# ==========================================================================

def save_table(df: pd.DataFrame, name: str) -> None:
    """Ghi bảng kết quả ra CSV để LaTeX và ứng dụng web đọc lại."""
    df.to_csv(cfg.OUTPUTS_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")


def to_latex_table(df: pd.DataFrame, caption: str, label: str, float_fmt: str = "%.4f") -> str:
    """Kết xuất bảng sang mã LaTeX dạng booktabs dùng ngay trong báo cáo."""
    body = df.to_latex(
        index=False,
        float_format=float_fmt,
        column_format="l" + "r" * (len(df.columns) - 1),
        escape=False,
    )
    lines = body.strip().splitlines()
    inner = "\n".join(lines[1:-1])
    return (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{inner}\n"
        "\\end{table}\n"
    )
