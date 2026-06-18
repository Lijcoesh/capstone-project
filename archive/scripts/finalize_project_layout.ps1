# One-time reorganisation: promote sprint artefacts to canonical paths, archive the rest.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$Archive = Join-Path $Root "archive"
$Ad = Join-Path $Archive "data/processed"
$Am = Join-Path $Archive "models"
$Ar = Join-Path $Archive "results"
$As = Join-Path $Archive "scripts"
foreach ($d in $Ad, $Am, $Ar, $As) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

Write-Host "=== Archiving processed data ===" -ForegroundColor Cyan
$proc = Join-Path $Root "data/processed"
Get-ChildItem $proc -Filter "*.npz" -ErrorAction SilentlyContinue | ForEach-Object {
    Move-Item $_.FullName (Join-Path $Ad $_.Name) -Force
}

Write-Host "=== Promoting canonical .npz ===" -ForegroundColor Cyan
Copy-Item (Join-Path $Ad "seizeit2_eeg_bp_w50s5.npz") (Join-Path $proc "eeg_windows.npz")
Copy-Item (Join-Path $Ad "seizeit2_eeg_ecg_bp_w50s5.npz") (Join-Path $proc "eeg_ecg_windows.npz")

Write-Host "=== Archiving + promoting models ===" -ForegroundColor Cyan
$modelRoot = Join-Path $Root "models"
$legacyEeg = Join-Path $Am "seizure_prediction_eeg_legacy"
New-Item -ItemType Directory -Force -Path $legacyEeg | Out-Null
Get-ChildItem (Join-Path $modelRoot "seizure_prediction_eeg") -Filter "*.pt" -ErrorAction SilentlyContinue |
    ForEach-Object { Move-Item $_.FullName (Join-Path $legacyEeg $_.Name) -Force }

$legacyEcge = Join-Path $Am "seizure_prediction_eeg_ecg_legacy"
New-Item -ItemType Directory -Force -Path $legacyEcge | Out-Null
if (Test-Path (Join-Path $modelRoot "seizure_prediction_eeg_ecg")) {
    Get-ChildItem (Join-Path $modelRoot "seizure_prediction_eeg_ecg") -Filter "*.pt" -ErrorAction SilentlyContinue |
        ForEach-Object { Move-Item $_.FullName (Join-Path $legacyEcge $_.Name) -Force }
}

$sprint = Join-Path $modelRoot "sprint"
New-Item -ItemType Directory -Force -Path (Join-Path $modelRoot "seizure_prediction_eeg") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $modelRoot "seizure_prediction_eeg_ecg") | Out-Null

foreach ($s in 42..46) {
    Copy-Item (Join-Path $sprint "cnn_eeg_bp_s$s.pt") `
        (Join-Path $modelRoot "seizure_prediction_eeg/cnn_prediction_eeg_s$s.pt") -Force
    Copy-Item (Join-Path $sprint "cnn_eeg_ecg_bp_s$s.pt") `
        (Join-Path $modelRoot "seizure_prediction_eeg_ecg/cnn_prediction_eeg_ecg_s$s.pt") -Force
}
Copy-Item (Join-Path $sprint "cnn_eeg_bp_s42.pt") `
    (Join-Path $modelRoot "seizure_prediction_eeg/cnn_prediction_eeg.pt") -Force
Copy-Item (Join-Path $sprint "cnn_eeg_ecg_bp_s42.pt") `
    (Join-Path $modelRoot "seizure_prediction_eeg_ecg/cnn_prediction_eeg_ecg.pt") -Force

@("n60", "sprint", "bp_n20", "per_patient", "siena", "window_sweep", "seizure_detection_eeg") | ForEach-Object {
    $src = Join-Path $modelRoot $_
    if (Test-Path $src) { Move-Item $src (Join-Path $Am $_) -Force }
}

Write-Host "=== Archiving + promoting results ===" -ForegroundColor Cyan
$resRoot = Join-Path $Root "results"
if (Test-Path (Join-Path $resRoot "seizure_prediction_eeg")) {
    Move-Item (Join-Path $resRoot "seizure_prediction_eeg") (Join-Path $Ar "seizure_prediction_eeg_legacy") -Force
}
if (Test-Path (Join-Path $resRoot "seizure_prediction_eeg_ecg")) {
    Move-Item (Join-Path $resRoot "seizure_prediction_eeg_ecg") (Join-Path $Ar "seizure_prediction_eeg_ecg_legacy") -Force
}

Move-Item (Join-Path $resRoot "sprint_eeg_bp") (Join-Path $resRoot "seizure_prediction_eeg") -Force
Move-Item (Join-Path $resRoot "sprint_eeg_ecg_bp") (Join-Path $resRoot "seizure_prediction_eeg_ecg") -Force

@(
    "bp_n20_eeg", "comparison", "eda", "n20_s20", "n60_eeg", "n60_eeg_ecg",
    "paper", "per_patient", "per_patient_ecg", "seizure_detection_eeg",
    "siena_eeg", "siena_eeg_ecg", "sprint", "window_sweep", "_smoke"
) | ForEach-Object {
    $src = Join-Path $resRoot $_
    if (Test-Path $src) { Move-Item $src (Join-Path $Ar $_) -Force }
}

Write-Host "=== Archiving old run scripts ===" -ForegroundColor Cyan
@(
    "run_sprint_seizeit2.ps1", "run_rf_test_tuned.ps1", "run_rf_val_tuned.ps1",
    "run_eval_test_tuned.ps1", "run_ensemble_eval.ps1", "run_paper_day1.ps1",
    "run_n60.ps1"
) | ForEach-Object {
    $src = Join-Path $Root $_
    if (Test-Path $src) { Move-Item $src (Join-Path $As $_) -Force }
}

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "Canonical:"
Write-Host "  data/processed/eeg_windows.npz"
Write-Host "  data/processed/eeg_ecg_windows.npz"
Write-Host "  models/seizure_prediction_eeg/cnn_prediction_eeg.pt (+ _s42..46)"
Write-Host "  models/seizure_prediction_eeg_ecg/cnn_prediction_eeg_ecg.pt (+ _s42..46)"
Write-Host "  results/seizure_prediction_eeg/"
Write-Host "  results/seizure_prediction_eeg_ecg/"
Write-Host "  run_final.ps1"
