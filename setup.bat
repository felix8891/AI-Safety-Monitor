@echo off
setlocal
cd /d "%~dp0"
title AI Safety Monitoring - Setup

echo ============================================
echo   AI Safety Monitoring - One Click Setup
echo ============================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] Python tidak ditemukan.
    echo Install Python 3.10 atau 3.11 dari https://www.python.org/downloads/windows/
    echo Saat instalasi, centang "Add Python to PATH".
    pause
    exit /b 1
  )
  set "PY=python"
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Membuat virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 goto :error
)

echo.
echo [2/3] Memperbarui pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo.
echo [3/3] Menginstal dependency (pertama kali bisa beberapa menit)...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Setup selesai. Menjalankan aplikasi...
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto :error
pause
exit /b 0

:error
echo.
echo [ERROR] Setup atau aplikasi gagal. Periksa pesan error di atas.
echo Pastikan internet aktif dan Python 3.10/3.11 terpasang.
pause
exit /b 1
