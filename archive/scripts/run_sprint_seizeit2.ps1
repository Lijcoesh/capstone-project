# SeizeIT2 overnight sprint:
#   Phase 1 - window sweep (3 configs, 1 seed, EEG-only) -> pick best CNN config
#   Phase 2 - full run on winner (5 seeds, EEG + EEG+ECG, RF baseline)
#
# Usage:
#   .\run_sprint_seizeit2.ps1
#
# Log: sprint_overnight.log

param(
    [switch]$SkipSweep,
    [switch]$SweepOnly,
    [switch]$SkipPreprocess,
    [switch]$SkipTrain,
    [int[]]$Seeds = @(42, 43, 44, 45, 46),
    [int]$SweepSeed = 42,
    [int]$WindowSec = 0,
    [int]$StepSec = 0
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$LogFile = Join-Path $PSScriptRoot "sprint_overnight.log"
Start-Transcript -Path $LogFile -Append | Out-Null

$PreictalMin = 10
$WinnerFile = "results/sprint/winning_config.txt"
$SweepConfigs = [ordered]@{
    "w50s5"  = @(50, 5)
    "w60s5"  = @(60, 5)
    "w60s10" = @(60, 10)
}

function Invoke-Step {
    param([string]$Label, [string[]]$Command)
    Write-Host "`n>> $Label" -ForegroundColor Green
    Write-Host ("   " + ($Command -join " "))
    & python @Command
    if ($LASTEXITCODE -ne 0) {
        Stop-Transcript | Out-Null
        throw "FAILED ($Label) - exit code $LASTEXITCODE. See $LogFile"
    }
}

function Test-NpzOk {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path $Path)) { throw "Missing $Label : $Path" }
    $py = "import numpy as np; d=np.load(r'$Path', allow_pickle=True); assert 'X' in d and 'y' in d; print('OK', d['X'].shape, 'windows=', len(d['y']))"
    Invoke-Step "Validate $Label" -Command @("-c", $py)
}

function Get-NpzPaths {
    param([int]$W, [int]$S)
    $tag = "w" + $W + "s" + $S
    @{
        Eeg  = "data/processed/seizeit2_eeg_bp_$tag.npz"
        Ecge = "data/processed/seizeit2_eeg_ecg_bp_$tag.npz"
        Tag  = $tag
    }
}

function Save-WinningConfig {
    param([string]$Name, [int]$W, [int]$S, [double]$CnnAuc, [double]$RfAuc)
    New-Item -ItemType Directory -Force -Path (Split-Path $WinnerFile) | Out-Null
    $lines = @(
        "name=$Name"
        "window_sec=$W"
        "step_sec=$S"
        "cnn_mean_subj_auc_smooth=$CnnAuc"
        "rf_mean_subj_auc_smooth=$RfAuc"
        "preictal_min=$PreictalMin"
        "picked_at=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    )
    $lines | Set-Content $WinnerFile -Encoding utf8
    Write-Host "Winner saved -> $WinnerFile" -ForegroundColor Green
}

function Read-WinningConfig {
    if (-not (Test-Path $WinnerFile)) {
        throw "No $WinnerFile - run sweep first or pass -WindowSec/-StepSec with -SkipSweep"
    }
    $cfg = @{}
    Get-Content $WinnerFile | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { $cfg[$Matches[1]] = $Matches[2] }
    }
    return @{
        Name   = $cfg['name']
        W      = [int]$cfg['window_sec']
        S      = [int]$cfg['step_sec']
        CnnAuc = [double]$cfg['cnn_mean_subj_auc_smooth']
        RfAuc  = [double]$cfg['rf_mean_subj_auc_smooth']
    }
}

function Run-WindowSweep {
    Write-Host "`n========== PHASE 1: WINDOW SWEEP (seed=$SweepSeed) ==========" -ForegroundColor Magenta
    New-Item -ItemType Directory -Force -Path models/window_sweep | Out-Null
    New-Item -ItemType Directory -Force -Path results/window_sweep/val | Out-Null

    foreach ($name in $SweepConfigs.Keys) {
        $w, $s = $SweepConfigs[$name]
        $data = "data/processed/sweep_eeg_bp_$name.npz"
        $model = "models/window_sweep/cnn_eeg_bp_${name}_s$SweepSeed.pt"

        Write-Host "`n--- sweep: $name ($w`s window / $s`s step) ---" -ForegroundColor Yellow
        Invoke-Step "Sweep preprocess $name" -Command @(
            "src/seizure_prediction_eeg/preprocess_eeg.py",
            "--window-sec", "$w", "--step-sec", "$s", "--preictal-min", "$PreictalMin",
            "--require-ecg", "--input-rep", "bandpower_seq", "--out", $data
        )
        Test-NpzOk $data "sweep $name"

        Invoke-Step "Sweep train $name" -Command @(
            "src/seizure_prediction_eeg/train_model_eeg.py",
            "--data", $data, "--save-model", $model,
            "--random-state", "$SweepSeed", "--epochs", "50", "--patience", "8",
            "--batch-size", "512", "--lr", "5e-4"
        )
        Invoke-Step "Sweep eval $name" -Command @(
            "src/seizure_prediction_eeg/evaluate_eeg.py",
            "--data", $data, "--model", $model,
            "--results-dir", "results/window_sweep",
            "--feature-set", "eeg_$name", "--eval-split", "val"
        )
        Invoke-Step "Sweep RF $name" -Command @(
            "src/baseline_rf.py", "--data", $data, "--feature-set", "eeg_$name",
            "--eval-split", "val", "--results-dir", "results/window_sweep",
            "--random-state", "$SweepSeed"
        )
    }

    Write-Host "`n>> Pick sweep winner" -ForegroundColor Green
    $pickOut = python scripts/pick_sweep_winner.py
    if ($LASTEXITCODE -ne 0) { throw "Could not pick sweep winner" }

    Write-Host "`n--- Sweep ranking (CNN mean_subj_auc_smooth) ---" -ForegroundColor Cyan
    $pickOut | ForEach-Object { Write-Host $_ }

    $top = ($pickOut | Select-Object -First 1) -split '\|'
    return @{
        Name   = $top[0]
        W      = [int]$top[1]
        S      = [int]$top[2]
        CnnAuc = [double]$top[3]
        RfAuc  = [double]$top[4]
    }
}

Write-Host "=== SeizeIT2 sprint ===" -ForegroundColor Cyan
Write-Host "Log: $LogFile | Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

Invoke-Step "Preflight" -Command @(
    "-c", "import sys; sys.path.insert(0,'src'); import torch,mne,captum,sklearn; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
)
$freeGb = [math]::Round((Get-PSDrive -Name ((Get-Location).Drive.Name)).Free / 1GB, 1)
Write-Host "Disk free: $freeGb GB"
if ($freeGb -lt 5) { throw "Less than 5 GB free - run .\cleanup_old_npz.ps1 -Delete" }
if (-not (Test-Path "data/raw/seizeit2")) { throw "Missing data/raw/seizeit2" }

New-Item -ItemType Directory -Force -Path models/sprint, data/processed | Out-Null

$winner = $null
if (-not $SkipSweep) {
    $winner = Run-WindowSweep
    Save-WinningConfig $winner.Name $winner.W $winner.S $winner.CnnAuc $winner.RfAuc
} elseif ($WindowSec -gt 0 -and $StepSec -gt 0) {
    $tag = "w" + $WindowSec + "s" + $StepSec
    $winner = @{ Name = $tag; W = $WindowSec; S = $StepSec }
    Write-Host "[SkipSweep] Using manual $tag" -ForegroundColor DarkGray
} else {
    $winner = Read-WinningConfig
    Write-Host "[SkipSweep] Loaded winner: $($winner.Name) (CNN smooth $($winner.CnnAuc))" -ForegroundColor DarkGray
}

$WindowSec = $winner.W
$StepSec = $winner.S
$paths = Get-NpzPaths $WindowSec $StepSec
$DataEeg = $paths.Eeg
$DataEcge = $paths.Ecge

Write-Host "`n>>> Winning config: $($paths.Tag) ($WindowSec`s window, $StepSec`s step, $PreictalMin min horizon)" -ForegroundColor Green

if ($SweepOnly) {
    Write-Host "`n[SweepOnly] Stopping after phase 1." -ForegroundColor Yellow
    Stop-Transcript | Out-Null
    exit 0
}

Write-Host "`n========== PHASE 2: FULL SPRINT (seeds: $($Seeds -join ', ')) ==========" -ForegroundColor Magenta

if (-not $SkipPreprocess) {
    Invoke-Step "Preprocess EEG ($($paths.Tag))" -Command @(
        "src/seizure_prediction_eeg/preprocess_eeg.py",
        "--window-sec", "$WindowSec", "--step-sec", "$StepSec", "--preictal-min", "$PreictalMin",
        "--require-ecg", "--input-rep", "bandpower_seq", "--out", $DataEeg
    )
    Test-NpzOk $DataEeg "EEG"

    Invoke-Step "Preprocess EEG+ECG ($($paths.Tag))" -Command @(
        "src/seizure_prediction_eeg_ecg/preprocess_eeg_ecg.py",
        "--window-sec", "$WindowSec", "--step-sec", "$StepSec", "--preictal-min", "$PreictalMin",
        "--input-rep", "bandpower_seq", "--out", $DataEcge
    )
    Test-NpzOk $DataEcge "EEG+ECG"
} else {
    Write-Host "[SkipPreprocess] Using existing $($paths.Tag) npz files" -ForegroundColor DarkGray
    Test-NpzOk $DataEeg "EEG"
    Test-NpzOk $DataEcge "EEG+ECG"
}

if (-not $SkipTrain) {
    foreach ($s in $Seeds) {
        Write-Host "`n--- EEG seed $s ---" -ForegroundColor Cyan
        Invoke-Step "Train EEG s$s" -Command @(
            "src/seizure_prediction_eeg/train_model_eeg.py",
            "--data", $DataEeg, "--save-model", "models/sprint/cnn_eeg_bp_s$s.pt",
            "--random-state", "$s", "--epochs", "50", "--patience", "8",
            "--batch-size", "512", "--lr", "5e-4"
        )
        Invoke-Step "Eval EEG s$s" -Command @(
            "src/seizure_prediction_eeg/evaluate_eeg.py",
            "--data", $DataEeg, "--model", "models/sprint/cnn_eeg_bp_s$s.pt",
            "--results-dir", "results/sprint_eeg_bp", "--feature-set", "eeg", "--eval-split", "val"
        )
    }

    foreach ($s in $Seeds) {
        Write-Host "`n--- EEG+ECG seed $s ---" -ForegroundColor Cyan
        Invoke-Step "Train EEG+ECG s$s" -Command @(
            "src/seizure_prediction_eeg_ecg/train_model_eeg_ecg.py",
            "--data", $DataEcge, "--save-model", "models/sprint/cnn_eeg_ecg_bp_s$s.pt",
            "--random-state", "$s", "--epochs", "50", "--patience", "8",
            "--batch-size", "512", "--lr", "5e-4"
        )
        Invoke-Step "Eval EEG+ECG s$s" -Command @(
            "src/seizure_prediction_eeg_ecg/evaluate_eeg_ecg.py",
            "--data", $DataEcge, "--model", "models/sprint/cnn_eeg_ecg_bp_s$s.pt",
            "--results-dir", "results/sprint_eeg_ecg_bp", "--feature-set", "eeg_ecg", "--eval-split", "val"
        )
    }

    foreach ($s in $Seeds) {
        Invoke-Step "RF EEG s$s" -Command @(
            "src/baseline_rf.py", "--data", $DataEeg, "--feature-set", "eeg",
            "--eval-split", "val", "--results-dir", "results/sprint_eeg_bp", "--random-state", "$s"
        )
        Invoke-Step "RF EEG+ECG s$s" -Command @(
            "src/baseline_rf.py", "--data", $DataEcge, "--feature-set", "eeg_ecg",
            "--eval-split", "val", "--results-dir", "results/sprint_eeg_ecg_bp", "--random-state", "$s"
        )
    }
}

Write-Host "`n=== ALL DONE ===" -ForegroundColor Green
Write-Host "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""
Write-Host "Winner config: $WinnerFile"
Write-Host "Sweep: results/window_sweep/val/"
Write-Host "Final: results/sprint_eeg_bp/val/ and results/sprint_eeg_ecg_bp/val/"
Write-Host "Resume: .\run_sprint_seizeit2.ps1 -SkipSweep -SkipPreprocess"

Stop-Transcript | Out-Null
