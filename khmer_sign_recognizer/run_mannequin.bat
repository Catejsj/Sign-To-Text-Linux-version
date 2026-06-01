@echo off
REM Double-click to launch the 3D mannequin.
REM Requires the one-time setup to be done first (see docs/QUICKSTART.md).
cd /d "%~dp0"
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo  venv not found. Run the one-time setup first:
    echo    python -m venv venv
    echo    venv\Scripts\activate
    echo    pip install -r requirements.txt
    echo    pip install torch --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
python scripts\mannequin_local.py
pause
