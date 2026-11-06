"""Chương 3 — bốn kiến trúc CNN bằng PyTorch, chép từ Tiểu luận 01 để nạp lại đúng trọng số đã huấn luyện.

Custom CNN (mô hình cơ sở), VGG-11, MobileNetV2 và ResNet-18 đều dùng stem bước 1 cho ảnh nhỏ. SimpleCNN là
mạng hai tầng dùng riêng cho phép so ba nền tảng. cam_layer(model) trả tầng tích chập cuối cho Grad-CAM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(cin, cout, k=3, s=1, groups=1, act=nn.ReLU):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act is not None:
        layers.append(act(inplace=True))
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ Custom CNN (cơ sở)
class CustomCNN(nn.Module):
    def __init__(self, n_classes=10, widths=(32, 64, 128), drops=(0.2, 0.3, 0.4)):
        super().__init__()
        blocks, cin = [], 3
        for w, p in zip(widths, drops):
            blocks += [conv_bn(cin, w), conv_bn(w, w), nn.MaxPool2d(2), nn.Dropout(p)]
            cin = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(cin, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
                                  nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.features(x)
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ------------------------------------------------------------------ VGG-11
VGG11_CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


class VGG11(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        layers, cin = [], 3
        for v in VGG11_CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(conv_bn(cin, v))
                cin = v
        self.features = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, n_classes))

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1))


# ------------------------------------------------------------------ MobileNetV2
class InvertedResidual(nn.Module):
    """1x1 mở rộng kênh -> 3x3 depthwise (groups = số kênh) -> 1x1 chiếu tuyến tính (không ReLU)."""

    def __init__(self, cin, cout, stride, expand):
        super().__init__()
        hidden = cin * expand
        self.use_res = stride == 1 and cin == cout
        layers = [] if expand == 1 else [conv_bn(cin, hidden, 1, act=nn.ReLU6)]
        layers += [conv_bn(hidden, hidden, 3, stride, groups=hidden, act=nn.ReLU6),
                   conv_bn(hidden, cout, 1, act=None)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetV2(nn.Module):
    # (hệ số mở rộng t, số kênh ra c, số khối n, bước s) - bước của hai nhóm đầu đổi 2 -> 1 cho ảnh nhỏ
    CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, n_classes=10, stem_stride=1):
        """stem_stride=2 cho ảnh 64x64: bản đồ đặc trưng sau thân về 32x32, bằng trường hợp ảnh 32x32 với
        stem_stride=1. Giữ 64x64 qua các khối mở rộng 6 lần cần hơn 8 GB VRAM ở lô 256."""
        super().__init__()
        layers, cin = [conv_bn(3, 32, s=stem_stride, act=nn.ReLU6)], 32
        for t, c, n, s in self.CFG:
            for i in range(n):
                layers.append(InvertedResidual(cin, c, s if i == 0 else 1, t))
                cin = c
        layers.append(conv_bn(cin, 1280, 1, act=nn.ReLU6))
        self.features = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, n_classes))

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1))


# ------------------------------------------------------------------ ResNet-18
class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:                    # chiếu 1x1 khi đổi kích thước hoặc số kênh
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))            # y = F(x) + x: "đường cao tốc" của gradient


class ResNet18(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.stem = conv_bn(3, 64)
        self.cin = 64
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.fc = nn.Linear(512, n_classes)

    def _make_layer(self, cout, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(ResidualBlock(self.cin, cout, s))
            self.cin = cout
        return nn.Sequential(*layers)

    def feature_maps(self, x):
        return self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.feature_maps(x), 1).flatten(1))


class SimpleCNN(nn.Module):
    """Conv32-Pool-Conv64-Pool-FC128-FC(K), đúng kiến trúc bản NumPy và Keras của phép so ba nền tảng."""

    def __init__(self, n_classes, size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * (size // 4) ** 2, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


BUILDERS = {"custom": CustomCNN, "vgg11": VGG11, "resnet18": ResNet18, "mobilenetv2": MobileNetV2}


def build(arch, n_classes, **kwargs):
    if arch not in BUILDERS:
        raise ValueError(f"Kiến trúc không hợp lệ: {arch}")
    return BUILDERS[arch](n_classes=n_classes, **kwargs)


def cam_layer(model):
    """Khối tích chập cuối cùng của mỗi kiến trúc; đầu ra của nó còn giữ cấu trúc không gian."""
    if isinstance(model, ResNet18):
        return model.layer4
    if isinstance(model, MobileNetV2):
        return model.features[-1]
    return [m for m in model.features if isinstance(m, nn.Sequential)][-1]


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))
