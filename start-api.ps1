# One command to start ASL API + ngrok. Run from this folder: .\start-api.ps1
$apiDir = $PSScriptRoot
if (-not $apiDir) { $apiDir = Get-Location }
Set-Location $apiDir

$hasNgrok = Get-Command ngrok -ErrorAction SilentlyContinue
if ($hasNgrok) {
    Write-Host "[ASL API] Starting ngrok in new window..." -ForegroundColor Cyan
    Start-Process ngrok -ArgumentList "http","5000" -WindowStyle Normal
    Write-Host "[ASL API] Copy the HTTPS URL from ngrok and set DEFAULT_SIGN_API_URL in the widget template.jsx (or paste in widget)." -ForegroundColor Yellow
} else {
    Write-Host "[ASL API] ngrok not installed. For DeepSpace, run 'ngrok http 5000' in another terminal." -ForegroundColor Yellow
}

Write-Host "[ASL API] Starting Flask API on http://localhost:5000 ..." -ForegroundColor Cyan
python api.py
