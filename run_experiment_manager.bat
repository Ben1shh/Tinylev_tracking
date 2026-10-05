@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Create the repository .venv and install requirements.txt first. See README.md.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0run_experiment_manager.py"
if errorlevel 1 pause
