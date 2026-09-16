"""Cấu hình tập trung cho toàn hệ thống.

Mọi đường dẫn, hằng số và tham số ngẫu nhiên đều khai báo tại đây (SSOT), để
notebook, script huấn luyện và ứng dụng web dùng chung một nguồn duy nhất.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Đường dẫn
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
OUTPUTS_DIR = ROOT / "outputs"

DIABETES_CSV = DATA_RAW / "diabetes.csv"
HOUSING_CSV = DATA_RAW / "vietnam_housing_dataset.csv"

for _d in (DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Tham số thực nghiệm — cố định để kết quả tái lập được (yêu cầu R14)
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Tỷ lệ dữ liệu tách riêng làm tập ngoại kiểm (Out-Of-Distribution benchmark).
# Tách trước khi chia train/test để mô hình chưa từng nhìn thấy trong mọi bước.
OOD_SIZE = 0.10

# --------------------------------------------------------------------------
# Định nghĩa đặc trưng
# --------------------------------------------------------------------------

# Các cột sinh học không thể mang giá trị 0 trên người sống.
# Giá trị 0 ở đây là quy ước mã hoá cho "thiếu dữ liệu", không phải số đo thật.
DIABETES_ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

DIABETES_TARGET = "Outcome"
HOUSING_TARGET = "Price"

# Tỷ lệ thiếu vượt ngưỡng này thì cột bị loại: phần quan sát còn lại quá ít để
# việc điền khuyết mang ý nghĩa thống kê.
MISSING_DROP_THRESHOLD = 0.80

# --------------------------------------------------------------------------
# Kiểu trình bày biểu đồ — dùng chung cho mọi hình trong báo cáo
# --------------------------------------------------------------------------
FIG_DPI = 150
PALETTE = ["#2E86AB", "#F18F01", "#C73E1D", "#3B7A57", "#6A4C93"]


def apply_plot_style() -> None:
    """Đặt kiểu vẽ thống nhất cho mọi biểu đồ xuất ra báo cáo."""
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": FIG_DPI,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": False,
        }
    )


def save_fig(fig, name: str) -> Path:
    """Ghi biểu đồ xuống ``figures/`` và trả về đường dẫn đã ghi."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
