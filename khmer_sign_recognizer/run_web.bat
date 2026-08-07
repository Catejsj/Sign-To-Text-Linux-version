@echo off
REM SignLink Control Center - one-command launch (Windows).
REM Linux/macOS equivalent: ./run_web.sh
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo No venv found. Create one first:
    echo     py -3.11 -m venv venv
    echo     venv\Scripts\activate
    echo     pip install -r requirements.txt
    exit /b 1
)
venv\Scripts\python.exe -m webapp %*
