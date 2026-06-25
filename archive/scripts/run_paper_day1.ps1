# Day 1: test evals + Table 1 + EEG vs EEG+ECG figure
# Usage: .\run_paper_day1.ps1
#        .\run_paper_day1.ps1 -TableOnly   # skip test if already run

param([switch]$TableOnly)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $TableOnly) {
    Write-Host "=== Running TEST evaluations (5 seeds x CNN+RF x EEG+ECG) ===" -ForegroundColor Cyan
    python scripts/paper_day1.py --run-test
}

Write-Host "`n=== Table 1 + figures ===" -ForegroundColor Cyan
python scripts/paper_day1.py --table --fig

Write-Host "`nDone. See results/paper/" -ForegroundColor Green
