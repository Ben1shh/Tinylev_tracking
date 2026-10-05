@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Install Python 3.12 and follow the README setup steps first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_tracker.py
if errorlevel 1 pause
