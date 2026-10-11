@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo             Vachan Environment Setup
echo ===================================================

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.11+ and try again.
    pause
    exit /b 1
)

echo [1/4] Found Python. Checking version...
python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"
if %errorlevel% neq 0 (
    echo [WARNING] Python 3.11 or newer is strongly recommended.
)

:: Create virtual environment if it does not exist
if not exist "venv" (
    echo [2/4] Creating virtual environment (.venv)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [2/4] Virtual environment (venv) already exists. Skipping creation.
)

:: Activate virtual environment
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip and install packages
echo [4/4] Installing dependencies...
python -m pip install --upgrade pip
if exist "requirements.txt" (
    echo Installing requirements.txt...
    pip install -r requirements.txt
)
if exist "requirements-dev.txt" (
    echo Installing requirements-dev.txt...
    pip install -r requirements-dev.txt
)

:: Create .env from .env.example if missing
if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env from .env.example...
        copy .env.example .env >nul
    )
)

echo.
echo ===================================================
echo Setup complete! To activate your environment run:
echo     venv\Scripts\activate
echo Then launch the application with:
echo     python main.py
echo ===================================================
echo.
pause
