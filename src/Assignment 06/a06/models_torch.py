"""Hạng mục 3 (Ch.3): bốn mô hình hồi quy bằng PyTorch, dùng chung một lớp.

Tầng hồi quy 64 đơn vị -> Linear(·, 32) -> ReLU -> Linear(32, 1). Đầu ra là một số: return đã
chuẩn hoá (AMZN) hoặc logit churn (KKBox).
"""
import torch
from torch import nn

CELLS = ["rnn", "lstm", "gru", "bilstm"]
CELL_LABELS = {"rnn": "Simple RNN", "lstm": "LSTM", "gru": "GRU", "bilstm": "BiLSTM"}
HIDDEN = 64
HEAD = 32


class PyTorchRNN(nn.Module):
    """cell ∈ {rnn, lstm, gru, bilstm}; đầu vào (B, T, D), đầu ra (B,)."""

    def __init__(self, cell, n_features=8, hidden=HIDDEN, head=HEAD):
        super().__init__()
        layer = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU, "bilstm": nn.LSTM}[cell]
        self.bidirectional = cell == "bilstm"
        self.hidden = hidden
        self.rnn = layer(n_features, hidden, batch_first=True, bidirectional=self.bidirectional)
        width = hidden * (2 if self.bidirectional else 1)
        self.head = nn.Sequential(nn.Linear(width, head), nn.ReLU(), nn.Linear(head, 1))

    def forward(self, x):
        out, _ = self.rnn(x)                       # (B, T, H) hoặc (B, T, 2H)
        if self.bidirectional:
            # chiều xuôi đọc xong ở bước cuối, chiều ngược đọc xong ở bước đầu: [h→_T ; h←_1]
            h = torch.cat([out[:, -1, :self.hidden], out[:, 0, self.hidden:]], dim=1)
        else:
            h = out[:, -1]                         # h_T tóm tắt cả chuỗi
        return self.head(h).squeeze(-1)


def build_torch(cell, n_features=8):
    return PyTorchRNN(cell, n_features)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
