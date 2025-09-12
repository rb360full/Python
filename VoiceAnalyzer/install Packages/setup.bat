@echo off
title نصب برنامه Voice Analyzer
color 0a

echo ================================
echo   نصب پیش‌نیازهای Python
echo ================================
echo.

REM نصب کتابخانه‌های پایتون از requirements.txt
pip install -r requirements.txt

echo.
echo ================================
echo   نصب FFmpeg
echo ================================
echo.

REM چک می‌کنیم winget روی سیستم هست یا نه
where winget >nul 2>nul
if %errorlevel%==0 (
    echo در حال نصب ffmpeg با winget...
    winget install -e --id Gyan.FFmpeg
) else (
    echo ❌ winget در سیستم پیدا نشد!
    echo لطفا ffmpeg را دستی نصب کنید:
    echo https://ffmpeg.org/download.html
)

echo.
echo ================================
echo   نصب تمام شد ✅
echo ================================
pause
