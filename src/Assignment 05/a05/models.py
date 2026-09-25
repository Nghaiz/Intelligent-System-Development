"""Mục 4 — một CNN cơ bản và ba mô hình phát triển, mỗi họ có bản 2D (ảnh) và 1D (bảng).

Mỗi bước tiến hoá thêm đúng một cơ chế:
    M0 basic     [Conv -> ReLU -> MaxPool] x 3 -> Flatten -> FC -> FC
    M1 vgg       3 tầng, mỗi tầng 2 khối (conv3 + BN + ReLU) x 2, MaxPool giữa các tầng, GAP -> FC
    M2 resnet    M1 + đường tắt: y = ReLU(F(x) + x), chiếu 1x1 khi đổi số kênh
    M3 seresnet  M2 + SEBlock trên nhánh F trước phép cộng
"""
import torch
from torch import nn

from .blocks import SEBlock

MODEL_NAMES = ["basic", "vgg", "resnet", "seresnet"]
MODEL_LABELS = {"basic": "BasicCNN", "vgg": "VGGNet", "resnet": "ResNet", "seresnet": "SE-ResNet"}


def _ops(dim):
    if dim == 2:
        return nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d
    return nn.Conv1d, nn.BatchNorm1d, nn.MaxPool1d, nn.AdaptiveAvgPool1d


class BasicCNN(nn.Module):
    """M0: mạng gốc kiểu LeNet/AlexNet, không BatchNorm, không đường tắt."""

    def __init__(self, dim, in_shape, n_out, widths, hidden):
        super().__init__()
        Conv, _, Pool, _ = _ops(dim)
        layers, c = [], in_shape[0]
        for w in widths:
            layers += [Conv(c, w, 3, padding=1), nn.ReLU(), Pool(2)]
            c = w
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, *in_shape)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.head(self.features(x))


class Block(nn.Module):
    """Khối hai conv 3x3. residual=False là khối VGG, True là BasicBlock của ResNet,
    se=True thêm chú ý theo kênh vào nhánh F."""

    def __init__(self, dim, c_in, c_out, residual, se):
        super().__init__()
        Conv, BN, _, _ = _ops(dim)
        self.F = nn.Sequential(
            Conv(c_in, c_out, 3, padding=1, bias=False), BN(c_out), nn.ReLU(),
            Conv(c_out, c_out, 3, padding=1, bias=False), BN(c_out))
        self.se = SEBlock(c_out) if se else nn.Identity()
        self.residual = residual
        self.shortcut = nn.Identity()
        if residual and c_in != c_out:
            self.shortcut = nn.Sequential(Conv(c_in, c_out, 1, bias=False), BN(c_out))

    def forward(self, x):
        out = self.se(self.F(x))
        if self.residual:
            out = out + self.shortcut(x)
        return torch.relu(out)


class StagedCNN(nn.Module):
    """M1/M2/M3: cùng khung 3 tầng x 2 khối, chỉ khác hai cờ residual và se."""

    def __init__(self, dim, in_shape, n_out, widths, residual, se):
        super().__init__()
        _, _, Pool, GAP = _ops(dim)
        stages, c = [], in_shape[0]
        for i, w in enumerate(widths):
            stages += [Block(dim, c, w, residual, se), Block(dim, w, w, residual, se)]
            if i < len(widths) - 1:
                stages.append(Pool(2))
            c = w
        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(GAP(1), nn.Flatten(), nn.Linear(c, n_out))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, dim, in_shape, n_out):
    """Dựng mô hình theo tên. dim=2: ảnh (C, H, W); dim=1: bảng (1, số đặc trưng)."""
    basic_w, staged_w = ((32, 64, 128), (64, 128, 256)) if dim == 2 else ((16, 32, 64), (16, 32, 64))
    hidden = 256 if dim == 2 else 64
    if name == "basic":
        m = BasicCNN(dim, in_shape, n_out, basic_w, hidden)
    elif name in ("vgg", "resnet", "seresnet"):
        m = StagedCNN(dim, in_shape, n_out, staged_w, residual=name != "vgg", se=name == "seresnet")
    else:
        raise ValueError(f"Không có mô hình '{name}', chọn một trong {MODEL_NAMES}")
    for mod in m.modules():
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
    return m


def first_conv(model):
    """Tầng tích chập đầu tiên, dùng để xem bộ lọc và feature map ở mục 5."""
    return next(m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d)))
