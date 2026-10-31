@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0installer\build_msi.ps1"
if errorlevel 1 (
    echo [FEIL] MSI-byggingen feilet.
    pause
    exit /b 1
)
pause
