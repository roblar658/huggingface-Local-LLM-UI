# Byggescript for Hugging Face Local Assistant MSI-installasjonspakke
$ErrorActionPreference = "Stop"

$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Write-Host "Bygger release-binær med cargo..." -ForegroundColor Cyan
Set-Location $repoRoot
cargo build --release

Write-Host "Pakker MSI med WiX Toolset v4..." -ForegroundColor Cyan
Set-Location $PSScriptRoot
wix build Package.wxs -ext WixToolset.UI.wixext -arch x64 -o HuggingFaceLocalAssistant.msi

$msiFile = Join-Path $PSScriptRoot "HuggingFaceLocalAssistant.msi"
if (Test-Path $msiFile) {
    $sizeMb = [math]::Round(((Get-Item $msiFile).Length / 1MB), 2)
    Write-Host "MSI-installasjonsfil er ferdig bygget!" -ForegroundColor Green
    Write-Host "Sti:  $msiFile" -ForegroundColor Yellow
    Write-Host "Størrelse: $sizeMb MB" -ForegroundColor Yellow
}

