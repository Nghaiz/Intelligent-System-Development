# Bật toàn bộ DiaNexus để demo: Neo4j, Ollama, website. Chạy lại nhiều lần vẫn an toàn.
# Bật:  powershell -ExecutionPolicy Bypass -File "src\Project 01\scripts\demo.ps1"
# Tắt web:  thêm tham số -Stop (Neo4j và Ollama giữ nguyên)
param([switch]$Stop)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path (Split-Path -Parent (Split-Path -Parent $project)) ".venv\Scripts\python.exe"
$webPort = 8097

function Test-Port($port) {
    $c = New-Object Net.Sockets.TcpClient
    try { $c.Connect("127.0.0.1", $port); return $true } catch { return $false } finally { $c.Close() }
}

function Wait-Port($port, $name, $seconds) {
    for ($i = 0; $i -lt $seconds; $i++) {
        if (Test-Port $port) { Write-Host "[ok] $name : cong $port"; return }
        Start-Sleep -Seconds 1
    }
    throw "$name khong mo cong $port sau $seconds giay"
}

if ($Stop) {
    $listeners = Get-NetTCPConnection -LocalPort $webPort -State Listen -ErrorAction SilentlyContinue
    foreach ($l in $listeners) { Stop-Process -Id $l.OwningProcess -Force }
    Write-Host "Da tat website."
    return
}

Set-Location $project
if (-not (Test-Port 7687)) {
    docker compose up -d | Out-Host
}
Wait-Port 7687 "Neo4j" 120

if (-not (Test-Port 11434)) {
    $serve = Join-Path $PSScriptRoot "ollama_serve.ps1"
    Start-Process powershell -WindowStyle Hidden -ArgumentList "-ExecutionPolicy Bypass -File `"$serve`""
}
Wait-Port 11434 "Ollama" 60

if (-not (Test-Port $webPort)) {
    Start-Process $python -WorkingDirectory $project -WindowStyle Hidden `
        -ArgumentList "-m uvicorn web.server:app --host 127.0.0.1 --port $webPort" `
        -RedirectStandardOutput "outputs\logs\web_server.log" -RedirectStandardError "outputs\logs\web_server.err.log"
}
Wait-Port $webPort "Website" 60

Write-Host ""
Write-Host "Website      http://127.0.0.1:$webPort"
Write-Host "API docs     http://127.0.0.1:$webPort/api/docs"
Write-Host "Neo4j        http://127.0.0.1:7474"
