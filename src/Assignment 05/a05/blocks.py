"""Mục 2 — các mô hình phát triển của CNN, mỗi cơ chế viết thành một lớp ngắn.

Mỗi lớp là một hàm y = F(x; θ). Notebook 01 chạy một lượt forward trên tensor
giả để kiểm tra shape và đếm tham số; mục này không huấn luyện.
"""
import torch
from torch import nn


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class LeNet5(nn.Module):
    """1998 — Conv -> Pool -> Conv -> Pool -> FC -> FC -> FC."""

    def __init__(self, in_ch=3, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.Tanh(),
            nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class AlexNetMini(nn.Module):
    """2012 — sâu hơn, ReLU thay tanh, Dropout ở phần nối đầy đủ."""

    def __init__(self, in_ch=3, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1), nn.ReLU(),
            nn.Conv2d(192, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(128 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class VGGBlock(nn.Module):
    """2014 — chồng n conv 3×3: hai conv 3×3 nhìn được vùng 5×5 với ít tham số hơn."""

    def __init__(self, c_in, c_out, n_convs=2):
        super().__init__()
        layers = []
        for i in range(n_convs):
            layers += [nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1), nn.ReLU()]
        self.body = nn.Sequential(*layers, nn.MaxPool2d(2))

    def forward(self, x):
        return self.body(x)


class InceptionBlock(nn.Module):
    """2014 — bốn nhánh song song, y = concat(f1(x), f3(x), f5(x), fpool(x))."""

    def __init__(self, c_in, c1, c3, c5, cp):
        super().__init__()
        self.b1 = nn.Conv2d(c_in, c1, 1)
        self.b3 = nn.Sequential(nn.Conv2d(c_in, c3 // 2, 1), nn.ReLU(), nn.Conv2d(c3 // 2, c3, 3, padding=1))
        self.b5 = nn.Sequential(nn.Conv2d(c_in, c5 // 2, 1), nn.ReLU(), nn.Conv2d(c5 // 2, c5, 5, padding=2))
        self.bp = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(c_in, cp, 1))

    def forward(self, x):
        return torch.relu(torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1))


class ResidualBlock(nn.Module):
    """2015 — y = ReLU(F(x) + x): mạng chỉ cần học phần dư F(x) = H(x) - x."""

    def __init__(self, c):
        super().__init__()
        self.F = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return torch.relu(self.F(x) + x)


class DenseBlock(nn.Module):
    """2017 — x_l = H_l([x_0, x_1, ..., x_{l-1}]): mỗi tầng nhận mọi đầu ra trước đó."""

    def __init__(self, c_in, growth, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(nn.BatchNorm2d(c_in + i * growth), nn.ReLU(),
                          nn.Conv2d(c_in + i * growth, growth, 3, padding=1, bias=False))
            for i in range(n_layers))

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            feats.append(layer(torch.cat(feats, dim=1)))
        return torch.cat(feats, dim=1)


class DepthwiseSeparable(nn.Module):
    """2017 — tách conv thường thành depthwise (từng kênh) rồi pointwise 1×1."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.depthwise = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False)
        self.pointwise = nn.Conv2d(c_in, c_out, 1, bias=False)

    def forward(self, x):
        return torch.relu(self.pointwise(torch.relu(self.depthwise(x))))


class SEBlock(nn.Module):
    """2017 — y = x ⊙ σ(W2 ReLU(W1 GAP(x))): mỗi kênh nhận một trọng số trong (0, 1)."""

    def __init__(self, c, r=16):
        super().__init__()
        h = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, h), nn.ReLU(), nn.Linear(h, c), nn.Sigmoid())

    def forward(self, x):  # chạy cho cả (B, C, L) lẫn (B, C, H, W)
        spatial = tuple(range(2, x.ndim))
        w = self.fc(x.mean(dim=spatial))
        return x * w.view(*w.shape, *([1] * len(spatial)))


class CBAM(nn.Module):
    """2018 — chú ý theo kênh (như SE, thêm nhánh max) rồi chú ý theo không gian."""

    def __init__(self, c, r=16):
        super().__init__()
        h = max(c // r, 4)
        self.mlp = nn.Sequential(nn.Linear(c, h), nn.ReLU(), nn.Linear(h, c))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x.mean(dim=(2, 3))) + self.mlp(x.amax(dim=(2, 3))))
        x = x * ca[:, :, None, None]
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa


class PatchEmbed(nn.Module):
    """2020 (ViT) — cắt ảnh thành các ô p×p, chiếu mỗi ô thành một vectơ d chiều."""

    def __init__(self, in_ch=3, patch=4, dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, số ô, d)


class TinyViTBlock(nn.Module):
    """Một khối encoder: z = z + MHA(LN(z)); z = z + MLP(LN(z))."""

    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, z):
        h = self.ln1(z)
        z = z + self.attn(h, h, h, need_weights=False)[0]
        return z + self.mlp(self.ln2(z))
