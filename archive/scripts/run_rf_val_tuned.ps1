# Re-run VAL RF baseline with threshold sweep + train-tuned threshold.
# On val: threshold = F1-optimal on TRAIN (same discipline as before threshold-sweep update).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$DataEeg  = "data/processed/seizeit2_eeg_bp_w50s5.npz"
$DataEcge = "data/processed/seizeit2_eeg_ecg_bp_w50s5.npz"
foreach ($s in 42..46) {
    Write-Host "`n=== RF EEG val seed $s ===" -ForegroundColor Cyan
    python src/baseline_rf.py --data $DataEeg --feature-set eeg `
        --results-dir results/sprint_eeg_bp --eval-split val --random-state $s
    Write-Host "`n=== RF EEG+ECG val seed $s ===" -ForegroundColor Cyan
    python src/baseline_rf.py --data $DataEcge --feature-set eeg_ecg `
        --results-dir results/sprint_eeg_ecg_bp --eval-split val --random-state $s
}
Write-Host "`nDone. See results/sprint_eeg_bp/val/ and results/sprint_eeg_ecg_bp/val/" -ForegroundColor Green
