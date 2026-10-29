@echo off
chcp 65001 >nul
cd /d "%~dp0"

where cargo >nul 2>nul
if errorlevel 1 set "PATH=C:\Users\rober\scoop\apps\rust-gnu\1.99.1\bin;C:\Users\rober\scoop\apps\rust-gnu\current\bin;C:\Users\rober\scoop\shims;C:\Users\rober\scoop\apps\julia\1.12.6\bin;%PATH%"

echo ========================================================
echo Starter Hugging Face Local Assistant (CLI-modus)...
echo ========================================================
.\target\release\hf_local_assistant.exe --cli
pause

