# Run Sign Language Recognition API (Windows PowerShell)
Set-Location $PSScriptRoot
if (Test-Path "env\Scripts\Activate.ps1") {
    & "env\Scripts\Activate.ps1"
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    & ".venv\Scripts\Activate.ps1"
}
Write-Host "Starting API..."
python api.py
