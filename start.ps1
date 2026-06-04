# BookKB 一键启动：后端 (FastAPI:8000) + 前端 (Vite:5173)
# 用法：.\start.ps1            正常启动
#       .\start.ps1 -Install   先装/更新依赖再启动

param(
    [switch]$Install
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

# ---------- 前置检查 ----------
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[ERROR] 未找到 .venv，请先创建虚拟环境：python -m venv .venv" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path ".env")) {
    Write-Host "[WARN] 未找到 .env，请确认 DEEPSEEK_API_KEY 已配置" -ForegroundColor Yellow
}

if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "[INFO] 前端依赖未安装，执行 npm install..." -ForegroundColor Cyan
    Push-Location frontend
    npm install
    Pop-Location
}

if ($Install) {
    Write-Host "[INFO] 安装/更新 Python 依赖..." -ForegroundColor Cyan
    & ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    Write-Host "[INFO] 更新前端依赖..." -ForegroundColor Cyan
    Push-Location frontend
    npm install
    Pop-Location
}

# ---------- 启动后端（新窗口） ----------
Write-Host "[INFO] 启动后端 http://localhost:8000 ..." -ForegroundColor Green
$backendCmd = "Set-Location '$ProjectRoot'; & '.venv\Scripts\Activate.ps1'; uvicorn api.main:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

Start-Sleep -Seconds 2

# ---------- 启动前端（新窗口） ----------
Write-Host "[INFO] 启动前端 http://localhost:5173 ..." -ForegroundColor Green
$frontendCmd = "Set-Location '$ProjectRoot\frontend'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Backend : http://localhost:8000"
Write-Host "  Docs    : http://localhost:8000/docs"
Write-Host "  Frontend: http://localhost:5173"
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "关闭对应窗口即可停止服务。"
