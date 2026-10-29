# Hugging Face Local Assistant Starter
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:PATH = "$PSScriptRoot;$PSScriptRoot\target\release;C:\Users\rober\scoop\apps\rust-gnu\1.99.1\bin;C:\Users\rober\scoop\apps\rust-gnu\current\bin;C:\Users\rober\scoop\shims;C:\Users\rober\scoop\apps\julia\1.12.6\bin;$env:PATH"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Starter Hugging Face Local Assistant (Release)..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

.\target\release\hf_local_assistant.exe
