# Re-run TEST evaluations with val-tuned thresholds (CNN + RF).
# Default thresholds: F1-optimal on VAL, applied to TEST (no manual --pred-threshold).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$DataEeg  = "data/processed/seizeit2_eeg_bp_w50s5.npz"
$DataEcge = "data/processed/seizeit2_eeg_ecg_bp_w50s5.npz"
$Seeds    = 42..46

foreach ($s in $Seeds) {
    Write-Host "`n=== CNN EEG test seed $s ===" -ForegroundColor Cyan
    python src/seizure_prediction_eeg/evaluate_eeg.py `
        --data $DataEeg --model "models/sprint/cnn_eeg_bp_s$s.pt" `
        --results-dir results/sprint_eeg_bp --feature-set eeg --eval-split test --no-gpu

    Write-Host "`n=== CNN EEG+ECG test seed $s ===" -ForegroundColor Cyan
    python src/seizure_prediction_eeg_ecg/evaluate_eeg_ecg.py `
        --data $DataEcge --model "models/sprint/cnn_eeg_ecg_bp_s$s.pt" `
        --results-dir results/sprint_eeg_ecg_bp --feature-set eeg_ecg --eval-split test --no-gpu

    Write-Host "`n=== RF EEG test seed $s ===" -ForegroundColor Cyan
    python src/baseline_rf.py --data $DataEeg --feature-set eeg `
        --results-dir results/sprint_eeg_bp --eval-split test --random-state $s

    Write-Host "`n=== RF EEG+ECG test seed $s ===" -ForegroundColor Cyan
    python src/baseline_rf.py --data $DataEcge --feature-set eeg_ecg `
        --results-dir results/sprint_eeg_ecg_bp --eval-split test --random-state $s
}

Write-Host "`nDone. See results/sprint_eeg_bp/test/ and results/sprint_eeg_ecg_bp/test/" -ForegroundColor Green
