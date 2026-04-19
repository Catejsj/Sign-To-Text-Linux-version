@echo off
echo ========================================
echo  KHMER SIGN RECOGNIZER - WINDOWS LAYER
echo ========================================
echo.
echo Starting camera capture...
echo Press Q in the window to quit
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat
python run_windows.py

pause