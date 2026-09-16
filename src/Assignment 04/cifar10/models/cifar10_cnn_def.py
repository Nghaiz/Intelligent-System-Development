"""Định nghĩa CNN 2D cho CIFAR-10 (Assignment 04, miền cifar10).

Mô-đun độc lập để notebook khác (mlp_vs_cnn) nạp lại trọng số đã huấn luyện tại
models/cifar10_cnn_pytorch.pt và trích véc-tơ ẩn 128 chiều bằng extract_features().

Tiền xử lý bắt buộc khi suy luận (xem models/cifar10_preproc.json):
    x = x_uint8 / 255.0                      # ảnh gốc NHWC (N, 32, 32, 3)
    x = (x - mean_c) / std_c                 # mean_c, std_c là ba số cho ba kênh R, G, B
    x = x.transpose(0, 3, 1, 2)              # -> NCHW (N, 3, 32, 32)

Kiến trúc (theo mục 4 hợp đồng tích hợp, biến thể CIFAR-10 ba khối 32 -> 64 -> 64):
    Conv-BN-ReLU-MaxPool-Dropout x3 -> Flatten -> Dense(128) -> ReLU -> Dropout -> Dense(10)

Cách dùng:
    from cifar10_cnn_def import Cifar10CNN
    model = Cifar10CNN()
    model.load_state_dict(torch.load('cifar10_cnn_pytorch.pt', map_location='cpu'))
    model.eval()
    with torch.no_grad():
        z = model.extract_features(x)        # (N, 128)
"""

import torch
import torch.nn as nn


class Cifar10CNN(nn.Module):
    """CNN 2D ba khối phân cấp cho ảnh màu CIFAR-10."""

    FEATURE_DIM = 128          # số chiều véc-tơ ẩn mà extract_features trả về

    def __init__(self, n_classes: int = 10, p_conv: float = 0.25, p_fc: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # 32 x 32 -> 16 x 16
            nn.Dropout(p_conv),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # 16 x 16 -> 8 x 8
            nn.Dropout(p_conv),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # 8 x 8 -> 4 x 4
            nn.Dropout(p_conv),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, self.FEATURE_DIM)
        self.relu_fc = nn.ReLU(inplace=True)
        self.drop_fc = nn.Dropout(p_fc)
        self.fc2 = nn.Linear(self.FEATURE_DIM, n_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về véc-tơ ẩn 128 chiều ở tầng áp chót, dùng cho phân tích PCA.

        Tham số
        -------
        x : Tensor (N, 3, 32, 32) đã chuẩn hóa theo models/cifar10_preproc.json

        Trả về
        ------
        Tensor (N, 128) sau Dense(128) và ReLU, TRƯỚC Dropout và tầng phân loại.
        """
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.flatten(h)
        return self.relu_fc(self.fc1(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về logit chưa qua softmax, (N, 10)."""
        return self.fc2(self.drop_fc(self.extract_features(x)))
