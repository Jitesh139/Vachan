@echo off
setlocal
cd /d "%~dp0"
title Vachan
echo ==============================================================
echo                        VACHAN LAUNCHER
echo ==============================================================

rem ---- 1. Find Python 3.11 - 3.13 (faster-whisper wheels) ----------------
set "PY="
for %%V in (3.11 3.12 3.13) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY goto NO_PYTHON
echo [1/5] Python: %PY%

rem ---- 2. Virtual environment ------------------------------------------------
if not exist "venv\Scripts\python.exe" (
    echo [2/5] Creating virtual environment...
    %PY% -m venv venv || goto FAILED
) else (
    echo [2/5] Virtual environment found.
)
set "VPY=venv\Scripts\python.exe"

rem ---- 3. Dependencies (re-installed only when requirements change) ----------
set "STAMP=venv\installed.txt"
copy /b requirements.txt + requirements-gpu.txt + requirements-openvino.txt "%TEMP%\vachan_req.txt" >nul
fc /b "%TEMP%\vachan_req.txt" "%STAMP%" >nul 2>&1
if errorlevel 1 (
    echo [3/5] Installing dependencies - first run downloads about 300 MB...
    "%VPY%" -m pip install --upgrade pip --quiet
    "%VPY%" -m pip install -r requirements.txt || goto FAILED
    where nvidia-smi >nul 2>&1 && (
        echo       NVIDIA driver found - installing CUDA libraries, about 1 GB...
        "%VPY%" -m pip install -r requirements-gpu.txt || echo [WARN] CUDA libraries failed - CPU will be used.
    )
    where nvidia-smi >nul 2>&1 || (
        echo       No NVIDIA GPU - installing OpenVINO for Intel graphics, about 270 MB...
        "%VPY%" -m pip install -r requirements-openvino.txt || echo [WARN] OpenVINO failed - CPU will be used.
    )
    copy /y "%TEMP%\vachan_req.txt" "%STAMP%" >nul
) else (
    echo [3/5] Dependencies up to date.
)

rem ---- 4. GPU check and Bible texts ------------------------------------------
echo [4/5] Checking GPU...
"%VPY%" scripts\check_gpu.py
"%VPY%" scripts\download_bibles.py
if errorlevel 1 echo [WARN] Bible download failed - the app will retry when it starts.

rem ---- 5. Launch -------------------------------------------------------------
echo [5/5] Starting Vachan... (first start downloads the Whisper model once)
echo.
echo   Enter = show suggested verse   Esc = dismiss   F5 = start/stop listening
echo   F8 = clear screen   F9 = show/hide output window   F11 (output) = fullscreen
echo.
"%VPY%" main.py
if errorlevel 1 goto FAILED
exit /b 0

:NO_PYTHON
echo [ERROR] Python 3.11, 3.12 or 3.13 was not found.
echo         Install Python 3.11 from https://www.python.org/downloads/ ^(tick "Add to PATH"^)
pause
exit /b 1

:FAILED
echo.
echo [ERROR] Something went wrong - see the messages above and logs\vachan.log
pause
exit /b 1
