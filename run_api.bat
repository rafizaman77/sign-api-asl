@echo off
cd /d "%~dp0"
if exist "env\Scripts\activate.bat" (
    call env\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)
echo Starting Sign Language Recognition API...
echo.
python api.py
pause
