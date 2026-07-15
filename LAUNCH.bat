@echo off
setlocal

REM ── Activate conda env and launch the IDS desktop app ───────
call conda activate aegmm_ids 2>nul
if errorlevel 1 (
    echo [ERROR] conda environment 'aegmm_ids' not found.
    echo Please run INSTALL.bat first.
    pause
    exit /b 1
)

cd /d "%~dp0"
echo Starting AE-GMM Real-Time NIDS...
python desktop_app.py
