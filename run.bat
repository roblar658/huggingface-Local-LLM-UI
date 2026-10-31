@echo off
chcp 65001 >nul
cd /d "%~dp0"

where cargo >nul 2>nul
if errorlevel 1 (
    set "PATH=%USERPROFILE%\.cargo\bin;C:\Users\rober\scoop\apps\rust-gnu\1.99.1\bin;C:\Users\rober\scoop\apps\rust-gnu\current\bin;C:\Users\rober\scoop\shims;C:\Users\rober\scoop\apps\julia\1.12.6\bin;%PATH%"
)

if not exist "target\release\hf_local_assistant.exe" (
    echo ========================================================
    echo  Bygger Hugging Face Local Assistant (Release)...
    echo ========================================================
    cargo build --release
    if errorlevel 1 (
        echo [FEIL] Byggingen feilet.
        pause
        exit /b 1
    )
)

echo ========================================================
echo  Starter Hugging Face Local Assistant (Release GUI)...
echo ========================================================
start "" "target\release\hf_local_assistant.exe" %*
