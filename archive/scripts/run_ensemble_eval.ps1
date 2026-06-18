# Merge 5 seed CNNs -> soft-voting ensemble, evaluate on TEST (and optionally VAL).
# Usage:
#   .\run_ensemble_eval.ps1
#   .\run_ensemble_eval.ps1 -Split val

param(
    [ValidateSet("test", "val")]
    [string]$Split = "test"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$DataEeg  = "data/processed/seizeit2_eeg_bp_w50s5.npz"
$DataEcge = "data/processed/seizeit2_eeg_ecg_bp_w50s5.npz"
$ModelEeg = "models/sprint/cnn_eeg_bp_ensemble5.pt"
$ModelEcge = "models/sprint/cnn_eeg_ecg_bp_ensemble5.pt"

Write-Host "=== Merge 5-seed ensembles ===" -ForegroundColor Cyan
python scripts/merge_ensemble.py
if ($LASTEXITCODE -ne 0) { throw "merge_ensemble failed" }

Write-Host "`n=== Evaluate ensemble ($Split) ===" -ForegroundColor Cyan
python src/seizure_prediction_eeg/evaluate_eeg.py `
    --data $DataEeg --model $ModelEeg `
    --results-dir results/sprint_eeg_bp --feature-set eeg --eval-split $Split
if ($LASTEXITCODE -ne 0) { throw "EEG eval failed" }

python src/seizure_prediction_eeg_ecg/evaluate_eeg_ecg.py `
    --data $DataEcge --model $ModelEcge `
    --results-dir results/sprint_eeg_ecg_bp --feature-set eeg_ecg --eval-split $Split
if ($LASTEXITCODE -ne 0) { throw "EEG+ECG eval failed" }

Write-Host "`nDone. Check mean_subj_auc_smooth in:" -ForegroundColor Green
Write-Host "  results/sprint_eeg_bp/$Split/metrics.csv"
Write-Host "  results/sprint_eeg_ecg_bp/$Split/metrics.csv"
