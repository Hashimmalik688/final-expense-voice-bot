@echo off
:: =============================================================================
:: test_windows.bat — Windows dev launcher (no GPU / no LiveKit server needed)
::
:: What it does:
::   1. Checks Python is available
::   2. Sets MOCK_MODE=true so token server works without LiveKit / vLLM / TTS
::   3. Starts the token+admin server on port 9000 in a new window
::   4. Waits for it to be ready, then opens the browser
::   5. (Optional) Downloads and starts the LiveKit server binary if found
::
:: Usage:
::   Double-click test_windows.bat   -or-   cd to repo root and run it
::
:: Stop:
::   Close the "Sarah Token Server" window, or press Ctrl+C in it
:: =============================================================================

setlocal enabledelayedexpansion

set "REPO=%~dp0.."
cd /d "%REPO%"

echo.
echo  ============================================================
echo   Sarah Voice Bot -- Windows Dev Mode
echo  ============================================================
echo.

:: ── Check Python ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)

:: ── Install required packages if missing ──────────────────────────────────
echo  [1/4] Checking Python packages...
pip install fastapi uvicorn livekit livekit-api python-multipart pyyaml python-dotenv aiohttp --quiet --disable-pip-version-check
echo        Done.

:: ── Set MOCK_MODE for this session ────────────────────────────────────────
set MOCK_MODE=true
echo  [2/4] Mock mode: ON  (no LiveKit server / vLLM / GPU needed)

:: ── (Optional) Start LiveKit server binary if present ─────────────────────
set "LK_BIN=%REPO%\bin\livekit-server-windows-amd64.exe"
if exist "%LK_BIN%" (
    echo  [3/4] Starting LiveKit server binary...
    start "LiveKit Server" /min "%LK_BIN%" --config "%REPO%\config\livekit.yaml"
    timeout /t 2 /nobreak >nul
    set MOCK_MODE=false
    echo        LiveKit found -- MOCK_MODE disabled, using real LiveKit.
) else (
    echo  [3/4] LiveKit binary not found at bin\livekit-server-windows-amd64.exe
    echo        Running in MOCK_MODE (Sarah uses scripted replies + browser TTS)
    echo        To install: winget install livekit  -or-  download from
    echo        https://github.com/livekit/livekit/releases and place .exe in bin\
)

:: ── Start token server ─────────────────────────────────────────────────────
echo  [4/4] Starting token + admin server on http://localhost:9000 ...
start "Sarah Token Server" cmd /k "cd /d %REPO% && set MOCK_MODE=%MOCK_MODE% && set PYTHONPATH=%REPO% && python -m uvicorn src.token_server:app --host 0.0.0.0 --port 9000"

:: ── Wait for server to start ───────────────────────────────────────────────
echo.
echo  Waiting for server...
:WAIT
timeout /t 1 /nobreak >nul
python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/health')" >nul 2>&1
if errorlevel 1 goto WAIT

:: ── Open browser ──────────────────────────────────────────────────────────
echo  Server is up! Opening browser...
start "" "http://localhost:9000"

echo.
echo  ============================================================
echo   Browser test UI : http://localhost:9000
echo   Admin dashboard : http://localhost:9000/admin
echo   Token endpoint  : http://localhost:9000/token
echo   Health check    : http://localhost:9000/health
echo.
echo   Mock mode = %MOCK_MODE%
echo   Sarah uses scripted replies + your browser's TTS voice.
echo   Speak into mic (Chrome/Edge) or type in the text box.
echo.
echo   Close the "Sarah Token Server" window to stop.
echo  ============================================================
echo.
pause
