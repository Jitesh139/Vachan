@echo off
cd /d "%~dp0"

echo =====================================================================
echo                     VACHAN - LIVE LAUNCHER
echo =====================================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
if errorlevel 1 goto NO_PYTHON

:: 2. Check virtual environment
if not exist "venv\Scripts\activate.bat" goto NO_VENV
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat
goto AFTER_ENV

:NO_VENV
echo [1/3] Using system Python environment...

:AFTER_ENV
:: 3. Ensure offline Bible database is initialized
if exist "data\bibles\vachan_bibles.db" goto BIBLE_READY
echo [2/3] Building local offline Bible database (KJV, WEB, ASV, Hindi IRV)...
python scripts\download_bibles.py
goto LAUNCH

:BIBLE_READY
echo [2/3] Local Bible database verified.

:LAUNCH
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

echo.
echo [Notice] The application has closed.
pause
goto :EOF

:NO_PYTHON
echo [ERROR] Python is not installed or not in your PATH.
echo Please install Python 3.11+ and try again.
pause
exit /b 1