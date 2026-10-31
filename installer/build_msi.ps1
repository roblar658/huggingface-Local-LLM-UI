# Byggescript for Hugging Face Local Assistant MSI-installasjonspakke
$ErrorActionPreference = "Stop"

$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $repoRoot

# Sørg for at cargo og wix er i PATH
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $env:PATH = "$env:USERPROFILE\.cargo\bin;C:\Users\rober\scoop\apps\rust-gnu\1.99.1\bin;C:\Users\rober\scoop\apps\rust-gnu\current\bin;C:\Users\rober\scoop\shims;C:\Users\rober\scoop\apps\julia\1.12.6\bin;$env:PATH"
}

# Lukk eventuelle kjørende instanser for å unngå fil-lås under kompilering og pakking
Stop-Process -Name "hf_local_assistant" -Force -ErrorAction SilentlyContinue

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " 1. Bygger optimalisert release-binær med cargo..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
cargo build --release

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " 2. Pakker MSI med WiX Toolset v4..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Set-Location $PSScriptRoot
wix build Package.wxs -ext WixToolset.UI.wixext -arch x64 -o HuggingFaceLocalAssistant.msi

$msiFile = Join-Path $PSScriptRoot "HuggingFaceLocalAssistant.msi"
if (Test-Path $msiFile) {
    $item = Get-Item $msiFile
    $sizeMb = [math]::Round(($item.Length / 1MB), 2)
    $hash = (Get-FileHash $msiFile -Algorithm SHA256).Hash
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host " MSI-installasjonsfil er ferdig bygget!" -ForegroundColor Green
    Write-Host " Sti:       $msiFile" -ForegroundColor Yellow
    Write-Host " Størrelse: $sizeMb MB" -ForegroundColor Yellow
    Write-Host " SHA-256:   $hash" -ForegroundColor Yellow
    Write-Host "========================================================" -ForegroundColor Green
}
