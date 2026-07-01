@echo off
cd /d "%~dp0"
echo ============================================================
echo  TRADING SYSTEM — LAUNCHER
echo ============================================================
if "%1"=="" (
    set /p SYMBOL="Ticker (ej: NVDA, AAPL, XAUUSD): "
) else (
    set SYMBOL=%1
)
echo Analizando %SYMBOL%...
python main.py %SYMBOL%
pause
