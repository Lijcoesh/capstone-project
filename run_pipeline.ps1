# Full seizure-prediction pipeline: preprocess -> train (5-run ensemble) -> eval val/test -> RF baseline.
#
# Usage (from repo root):
#   .\run_pipeline.ps1
#   .\run_pipeline.ps1 -SkipPreprocess
#   .\run_pipeline.ps1 -SkipTrain
#   .\run_pipeline.ps1 -SkipBaseline

param(
    [switch]$SkipPreprocess,
    [switch]$SkipTrain,
    [switch]$SkipBaseline,
    [switch]$NoGpu
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$GpuFlag = if ($NoGpu) { @("--no-gpu") } else { @() }

function Invoke-Step {
    param([string]$Label, [string[]]$Command)
    Write-Host "`n>> $Label" -ForegroundColor Cyan
    Write-Host ("   python " + ($Command -join " "))
    & python @Command
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $Label (exit $LASTEXITCODE)" }
}

Write-Host "=== Seizure prediction pipeline ===" -ForegroundColor Green
Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

if (-not $SkipPreprocess) {
    Invoke-Step "Preprocess EEG" -Command @(
        "src/seizure_prediction_eeg/preprocess_eeg.py",
        "--window-sec", "50", "--step-sec", "5", "--preictal-min", "10",
        "--require-ecg", "--input-rep", "bandpower_seq"
    )
    Invoke-Step "Preprocess EEG+ECG" -Command @(
        "src/seizure_prediction_eeg_ecg/preprocess_eeg_ecg.py",
        "--window-sec", "50", "--step-sec", "5", "--preictal-min", "10",
        "--input-rep", "bandpower_seq"
    )
}

if (-not $SkipTrain) {
    $trainArgs = @("--ensemble-runs", "5", "--random-state", "42",
                   "--epochs", "50", "--patience", "8", "--batch-size", "512", "--lr", "5e-4") + $GpuFlag
    Invoke-Step "Train EEG ensemble (5 runs)" -Command @(
        "src/seizure_prediction_eeg/train_model_eeg.py"
    ) + $trainArgs
    Invoke-Step "Train EEG+ECG ensemble (5 runs)" -Command @(
        "src/seizure_prediction_eeg_ecg/train_model_eeg_ecg.py"
    ) + $trainArgs
}

$evalArgs = $GpuFlag
Invoke-Step "Evaluate EEG (val)" -Command @(
    "src/seizure_prediction_eeg/evaluate_eeg.py", "--eval-split", "val"
) + $evalArgs
Invoke-Step "Evaluate EEG (test)" -Command @(
    "src/seizure_prediction_eeg/evaluate_eeg.py", "--eval-split", "test"
) + $evalArgs
Invoke-Step "Evaluate EEG+ECG (val)" -Command @(
    "src/seizure_prediction_eeg_ecg/evaluate_eeg_ecg.py", "--eval-split", "val"
) + $evalArgs
Invoke-Step "Evaluate EEG+ECG (test)" -Command @(
    "src/seizure_prediction_eeg_ecg/evaluate_eeg_ecg.py", "--eval-split", "test"
) + $evalArgs

if (-not $SkipBaseline) {
    foreach ($split in @("val", "test")) {
        Invoke-Step "RF EEG ($split)" -Command @(
            "src/baseline_rf.py",
            "--data", "data/processed/eeg_windows.npz",
            "--feature-set", "eeg", "--eval-split", $split
        )
        Invoke-Step "RF EEG+ECG ($split)" -Command @(
            "src/baseline_rf.py",
            "--data", "data/processed/eeg_ecg_windows.npz",
            "--feature-set", "eeg_ecg", "--eval-split", $split
        )
    }
}

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Models : models/seizure_prediction_eeg/cnn_prediction_eeg.pt"
Write-Host "         models/seizure_prediction_eeg_ecg/cnn_prediction_eeg_ecg.pt"
Write-Host "Results: results/seizure_prediction_eeg/"
Write-Host "         results/seizure_prediction_eeg_ecg/"
