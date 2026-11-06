# Ollama của Project 01: dùng lại kho mô hình và tệp chạy của Tiểu luận Final, chỉ nghe ở 127.0.0.1.
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$tlf = Join-Path $root "TieuLuan Final\models\ollama"
$env:OLLAMA_MODELS = Join-Path $tlf "store"
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_KEEP_ALIVE = "5m"
& (Join-Path $tlf "bin\ollama.exe") serve
