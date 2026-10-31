# Hugging Face Local Assistant - CLI-starter (PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

& (Join-Path $PSScriptRoot "run.ps1") -Cli @args
