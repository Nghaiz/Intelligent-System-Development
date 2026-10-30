# ag2/shop_intent.py
class IntentNet(nn.Module):
    """EmbeddingBag(trung bình) -> Dropout -> Linear -> ReLU -> Linear(6 ý định)."""

    def __init__(self, dim=96, hidden=64, n_classes=len(INTENTS), dropout=0.3):
        super().__init__()
        self.emb = nn.EmbeddingBag(N_BUCKETS, dim, mode="mean")
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, ids, offsets):
        return self.head(self.emb(ids, offsets))
