# Final SeizeIT2 seizure-prediction pipeline (bandpower CNN + RF baseline).
# Canonical paths only — no sprint/experiment prefixes.
#
# Usage:
#   .\run_final.ps1                    # full: preprocess + train 5 seeds + eval val/test
#   .\run_final.ps1 -SkipPreprocess    # reuse existing .npz
#   .\run_final.ps1 -SkipTrain         # eval only (models must exist)
#   .\run_final.ps1 -EvalSplit test    # test-only eval pass

param(
    [switch]$SkipPreprocess,
    [switch]$SkipTrain,
    [string]$EvalSplit = "both",   # val | test | both
    [int[]]$Seeds = @(42, 43, 44, 45, 46)
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$DataEeg   = "data/processed/eeg_windows.npz"
$DataEcge  = "data/processed/eeg_ecg_windows.npz"
$ResEeg    = "results/seizure_prediction_eeg"
$ResEcge   = "results/seizure_prediction_eeg_ecg"
$Preictal  = 10
$WindowSec = 50
$StepSec   = 5

function Invoke-Step {
    param([string]$Label, [string[]]$Command)
    Write-Host "`n>> $Label" -ForegroundColor Green
    & python @Command
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $Label (exit $LASTEXITCODE)" }
}

if (-not $SkipPreprocess) {
    Invoke-Step "Preprocess EEG" -Command @(
        "src/seizure_prediction_eeg/preprocess_eeg.py",
        "--window-sec", "$WindowSec", "--step-sec", "$StepSec",
        "--preictal-min", "$Preictal", "--require-ecg",
        "--input-rep", "bandpower_seq", "--out", $DataEeg
    )
    Invoke-Step "Preprocess EEG+ECG" -Command @(
        "src/seizure_prediction_eeg_ecg/preprocess_eeg_ecg.py",
        "--window-sec", "$WindowSec", "--step-sec", "$StepSec",
        "--preictal-min", "$Preictal", "--input-rep", "bandpower_seq", "--out", $DataEcge
    )
}

if (-not $SkipTrain) {
    foreach ($s in $Seeds) {
        $mEeg  = "models/seizure_prediction_eeg/cnn_prediction_eeg_s$s.pt"
        $mEcge = "models/seizure_prediction_eeg_ecg/cnn_prediction_eeg_ecg_s$s.pt"
        Invoke-Step "Train EEG seed $s" -Command @(
            "src/seizure_prediction_eeg/train_model_eeg.py",
            "--data", $DataEeg, "--save-model", $mEeg,
            "--random-state", "$s", "--epochs", "50", "--patience", "8",
            "--batch-size", "512", "--lr", "5e-4"
        )
        Invoke-Step "Train EEG+ECG seed $s" -Command @(
            "src/seizure_prediction_eeg_ecg/train_model_eeg_ecg.py",
            "--data", $DataEcge, "--save-model", $mEcge,
            "--random-state", "$s", "--epochs", "50", "--patience", "8",
            "--batch-size", "512", "--lr", "5e-4"
        )
    }
    Copy-Item "models/seizure_prediction_eeg/cnn_prediction_eeg_s42.pt" `
        "models/seizure_prediction_eeg/cnn_prediction_eeg.pt" -Force
    Copy-Item "models/seizure_prediction_eeg_ecg/cnn_prediction_eeg_ecg_s42.pt" `
        "models/seizure_prediction_eeg_ecg/cnn_prediction_eeg_ecg.pt" -Force
}

$splits = if ($EvalSplit -eq "both") { @("val", "test") } else { @($EvalSplit) }

foreach ($split in $splits) {
    foreach ($s in $Seeds) {
        Invoke-Step "Eval EEG $split s$s" -Command @(
            "src/seizure_prediction_eeg/evaluate_eeg.py",
            "--data", $DataEeg,
            "--model", "models/seizure_prediction_eeg/cnn_prediction_eeg_s$s.pt",
            "--results-dir", $ResEeg, "--feature-set", "eeg",
            "--eval-split", $split, "--no-gpu"
        )
        Invoke-Step "Eval EEG+ECG $split s$s" -Command @(
            "src/seizure_prediction_eeg_ecg/evaluate_eeg_ecg.py",
            "--data", $DataEcge,
            "--model", "models/seizure_prediction_eeg_ecg/cnn_prediction_eeg_ecg_s$s.pt",
            "--results-dir", $ResEcge, "--feature-set", "eeg_ecg",
            "--eval-split", $split, "--no-gpu"
        )
        Invoke-Step "RF EEG $split s$s" -Command @(
            "src/baseline_rf.py", "--data", $DataEeg, "--feature-set", "eeg",
            "--results-dir", $ResEeg, "--eval-split", $split, "--random-state", "$s"
        )
        Invoke-Step "RF EEG+ECG $split s$s" -Command @(
            "src/baseline_rf.py", "--data", $DataEcge, "--feature-set", "eeg_ecg",
            "--results-dir", $ResEcge, "--eval-split", $split, "--random-state", "$s"
        )
    }
}

Write-Host "`nDone. Results: $ResEeg/ and $ResEcge/" -ForegroundColor Green
