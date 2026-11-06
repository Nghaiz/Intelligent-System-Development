# Khởi động Ollama cho Tiểu luận Final: chỉ nghe ở 127.0.0.1, mô hình lưu trong models/ollama/store (ổ D).
$root = Split-Path -Parent $PSScriptRoot
$env:OLLAMA_MODELS = Join-Path $root "models\ollama\store"
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_KEEP_ALIVE = "5m"
& (Join-Path $root "models\ollama\bin\ollama.exe") serve
