# Hugging Face Local Assistant - Startskript for GUI og CLI (PowerShell)
param(
    [switch]$Cli,
    [switch]$Build,
    [switch]$BuildOnly
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Sørg for at cargo/rust er tilgjengelig i PATH
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $env:PATH = "$env:USERPROFILE\.cargo\bin;C:\Users\rober\scoop\apps\rust-gnu\1.99.1\bin;C:\Users\rober\scoop\apps\rust-gnu\current\bin;C:\Users\rober\scoop\shims;C:\Users\rober\scoop\apps\julia\1.12.6\bin;$env:PATH"
}

$exePath = Join-Path $PSScriptRoot "target\release\hf_local_assistant.exe"

if ($Build -or $BuildOnly -or (-not (Test-Path $exePath))) {
    Write-Host "========================================================" -ForegroundColor Cyan
    Write-Host " Bygger release-binær med cargo..." -ForegroundColor Cyan
    Write-Host "========================================================" -ForegroundColor Cyan
    cargo build --release
}

if (-not $BuildOnly) {
    if ($Cli) {
        Write-Host "========================================================" -ForegroundColor Cyan
        Write-Host " Starter Hugging Face Local Assistant (CLI-modus)..." -ForegroundColor Cyan
        Write-Host "========================================================" -ForegroundColor Cyan
        if ($args.Count -gt 0) {
            & $exePath --cli @args
        } else {
            & $exePath --cli
        }
    } else {
        Write-Host "========================================================" -ForegroundColor Cyan
        Write-Host " Starter Hugging Face Local Assistant (Release GUI)..." -ForegroundColor Cyan
        Write-Host "========================================================" -ForegroundColor Cyan
        if ($args.Count -gt 0) {
            Start-Process -FilePath $exePath -ArgumentList $args
        } else {
            Start-Process -FilePath $exePath
        }
    }
}
