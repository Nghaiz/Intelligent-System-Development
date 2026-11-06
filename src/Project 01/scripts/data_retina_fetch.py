"""Tải bộ võng mạc 5 cấp từ Hugging Face (chỉ parquet dữ liệu) vào data/raw/retina."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import snapshot_download  # noqa: E402

from dnx.common import DATA  # noqa: E402
from dnx.mem import require_disk  # noqa: E402

require_disk(2, "tải võng mạc")
p = snapshot_download("youssefedweqd/Diabetic_Retinopathy_Detection", repo_type="dataset",
                      allow_patterns=["data/*.parquet", "README.md"], local_dir=DATA / "raw" / "retina")
print("xong", p, flush=True)
