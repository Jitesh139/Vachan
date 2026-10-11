@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo                     VACHAN - LIVE LAUNCHER
echo =====================================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.11+ and try again.
    pause
    exit /b 1
)

:: 2. Ensure virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [1/3] Setting up virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Could not create virtual environment.
        pause
        exit /b 1
    )
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    echo [1/3] Activating virtual environment...
    call venv\Scripts\activate.bat
)

:: 3. Ensure offline Bible database is initialized
if not exist "data\bibles\vachan_bibles.db" (
    echo [2/3] Building local offline Bible database (KJV, WEB, ASV, Hindi IRV)...
    python scripts\download_bibles.py
) else (
    echo [2/3] Local Bible database verified.
)

:: 4. Launch Application
echo [3/3] Launching Vachan...
echo.
echo =====================================================================
echo - Click 'Start Listening' to start transcription.
echo - Speak into your microphone in English, Hindi, or Hinglish.
echo - When a verse is detected, press Enter to show or Esc to dismiss.
echo - Press F11 on the verse window to toggle full screen on monitor 2.
echo =====================================================================
echo.

python main.py

if %errorlevel% neq 0 (
    echo.
    echo [Notice] The application has closed.
    pause
)
