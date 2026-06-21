$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$DataEeg  = "data/processed/seizeit2_eeg_bp_w50s5.npz"
$DataEcge = "data/processed/seizeit2_eeg_ecg_bp_w50s5.npz"
foreach ($s in 42..46) {
    python src/baseline_rf.py --data $DataEeg --feature-set eeg `
        --results-dir results/sprint_eeg_bp --eval-split test --random-state $s
    python src/baseline_rf.py --data $DataEcge --feature-set eeg_ecg `
        --results-dir results/sprint_eeg_ecg_bp --eval-split test --random-state $s
}
