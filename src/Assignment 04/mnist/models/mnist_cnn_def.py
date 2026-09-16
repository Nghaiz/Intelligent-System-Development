"""Định nghĩa CNN 2D cho MNIST (Assignment 04, miền mnist).

Mô-đun độc lập để notebook khác (mlp_vs_cnn) nạp lại trọng số đã huấn luyện
tại models/mnist_cnn_pytorch.pt và trích véc-tơ ẩn 128 chiều bằng extract_features().

Tiền xử lý bắt buộc khi suy luận (xem models/mnist_preproc.json):
    x = x_uint8 / 255.0
    x = (x - mean) / std          # mean, std học từ 48000 ảnh nhánh train
    x -> tensor (N, 1, 28, 28)

Kiến trúc (theo mục 4 hợp đồng tích hợp, biến thể MNIST 2 khối 32 -> 64):
    Conv-BN-ReLU-MaxPool-Dropout x2 -> Flatten -> Dense(128) -> ReLU -> Dropout -> Dense(10)
"""

import torch
import torch.nn as nn


class MnistCNN(nn.Module):
    """CNN 2D hai khối phân cấp cho MNIST."""

    FEATURE_DIM = 128          # số chiều véc-tơ ẩn mà extract_features trả về

    def __init__(self, n_classes: int = 10, p_conv: float = 0.25, p_fc: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(p_conv),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(p_conv),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, self.FEATURE_DIM)
        self.relu_fc = nn.ReLU(inplace=True)
        self.drop_fc = nn.Dropout(p_fc)
        self.fc2 = nn.Linear(self.FEATURE_DIM, n_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về véc-tơ ẩn 128 chiều ở tầng áp chót, dùng cho phân tích PCA.

        Tham số
        -------
        x : Tensor (N, 1, 28, 28) đã chuẩn hóa theo models/mnist_preproc.json

        Trả về
        ------
        Tensor (N, 128) sau Dense(128) và ReLU, TRƯỚC Dropout và tầng phân loại.
        """
        h = self.block1(x)
        h = self.block2(h)
        h = self.flatten(h)
        return self.relu_fc(self.fc1(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về logit chưa qua softmax, (N, 10)."""
        return self.fc2(self.drop_fc(self.extract_features(x)))
