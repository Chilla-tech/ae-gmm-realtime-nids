@echo off
setlocal

echo ============================================================
echo  AE-GMM Real-Time NIDS  --  Installation Script
echo ============================================================
echo.

REM ── 1. Check for Conda ──────────────────────────────────────
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] conda not found on PATH.
    echo.
    echo Please install Miniconda from:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo Then re-run this script.
    pause
    exit /b 1
)
echo [OK] conda found.

REM ── 2. Create the conda environment ─────────────────────────
echo.
echo [STEP 1/3] Creating conda environment 'aegmm_ids' (Python 3.10)...
call conda create -n aegmm_ids python=3.10 -y
if errorlevel 1 goto :err

REM ── 3. Activate and install pip packages ────────────────────
echo.
echo [STEP 2/3] Installing Python packages (this may take 10-20 min)...
call conda activate aegmm_ids
if errorlevel 1 goto :err

pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :err

REM ── 4. Fix PySide6 plugin path (ensure matched shiboken6) ───
echo.
echo [STEP 3/3] Verifying PySide6 installation...
python -c "from PySide6.QtWidgets import QApplication; print('[OK] PySide6 OK')"
if errorlevel 1 (
    echo [WARN] PySide6 check failed, attempting fix...
    pip install --force-reinstall --no-deps --ignore-installed shiboken6==6.7.3 PySide6-Essentials==6.7.3
)

REM ── 5. Done ─────────────────────────────────────────────────
echo.
echo ============================================================
echo  Installation complete!
echo.
echo  IMPORTANT: You must also install Npcap for live capture:
echo    https://npcap.com/#download
echo    (select "WinPcap API-compatible mode" during install)
echo.
echo  Double-click LAUNCH.bat to start the app.
echo ============================================================
echo.
pause
exit /b 0

:err
echo.
echo [ERROR] Installation failed at the step above.
echo Check the error messages and try again.
pause
exit /b 1
