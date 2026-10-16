@echo off
REM One-step setup + launch for Windows. Double-click, or run: start.bat --demo
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% --version >nul 2>nul || (echo Python 3 is not installed. Get it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^) and run this again. & pause & exit /b 1)
if not exist .venv (
  echo First run: creating virtual environment and installing dependencies ^(1-2 min^)...
  %PY% -m venv .venv || (pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip -q
  .venv\Scripts\python -m pip install -r requirements.txt -q || (pause & exit /b 1)
)
start "" http://127.0.0.1:5000
.venv\Scripts\python -m optionsapp.app %*
pause
