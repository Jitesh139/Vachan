@echo off
rem Build dist\Vachan\Vachan.exe (one-folder). Run run.bat once first to create venv.
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Run run.bat first to create the virtual environment.
    exit /b 1
)
venv\Scripts\python.exe -m pip install pyinstaller==6.22.3 --quiet || exit /b 1
venv\Scripts\python.exe scripts\download_bibles.py || exit /b 1
venv\Scripts\python.exe -m PyInstaller vachan.spec --noconfirm --clean || exit /b 1
echo.
echo Built dist\Vachan\Vachan.exe - copy the whole dist\Vachan folder to the church PC.
