"""Tải các LLM và mô hình embedding vào models/hf/. Chỉ lấy safetensors và cấu hình, không chạy mã từ repo."""
from huggingface_hub import snapshot_download

from tlf.common import HF

REPOS = {
    "gemma3-4b": "unsloth/gemma-3-4b-it-bnb-4bit",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-instruct",
    "qwen2.5-7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "e5-base": "intfloat/multilingual-e5-base",
    "bge-reranker": "BAAI/bge-reranker-v2-m3",
}
KEEP = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "1_Pooling/*", "modules.json"]

if __name__ == "__main__":
    for name, repo in REPOS.items():
        print("->", name, repo, flush=True)
        snapshot_download(repo, local_dir=HF / name, allow_patterns=KEEP, ignore_patterns=["onnx/*", "openvino/*"],
                          max_workers=1)
    print("xong", flush=True)
