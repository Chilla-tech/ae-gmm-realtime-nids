@echo off
setlocal

REM ── Legacy Streamlit dashboard (optional alternative to desktop_app.py) ──
call conda activate aegmm_ids 2>nul
if errorlevel 1 (
    echo [ERROR] conda environment 'aegmm_ids' not found.
    echo Please run INSTALL.bat first.
    pause
    exit /b 1
)

cd /d "%~dp0"
echo Starting M3 AE-GMM IDS Dashboard (Streamlit)...
streamlit run app.py
