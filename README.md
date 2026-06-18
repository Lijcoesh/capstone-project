# A Comparative Study of EEG-Only vs. EEG-ECG Feature Sets for Seizure Prediction Using a 1D Convolutional Neural Network

Capstone project on **epileptic-seizure prediction** from wearable physiological signals.
A 1-D Convolutional Neural Network is trained on the **SeizeIT2** dataset, and two feature
sets are compared under identical conditions: an **EEG-only** model (2 channels) and an
**EEG + ECG** model (3 channels). Because both models use the same dataset, the only
difference is the ECG channel — a controlled comparison of whether ECG adds predictive value.

**Task.** Seizure *prediction* (not detection): each signal window is labeled **pre-ictal**
(within 10 min before a seizure onset → positive) or **interictal** (far from any seizure →
negative). The seizure itself and a post-ictal guard are excluded from training.

## Project structure

```
capstone-project/                # repo root
├── data/
│   ├── raw/seizeit2/             # SeizeIT2 (BIDS); .edf signals not committed — see "Data"
│   └── processed/                # cached windowed datasets (.npz; regenerated, not committed)
├── src/
│   ├── preprocess_common.py      # shared: BIDS discovery, pre-ictal labeling, windowing, 50 Hz notch, split
│   ├── model_common.py           # shared: 1-D CNN + training
│   ├── evaluate_common.py        # shared: metrics, per-subject AUC, Grad-CAM, report
│   ├── baseline_rf.py            # RandomForest band-power baseline (comparison reference)
│   ├── seizure_prediction_eeg/       # EEG-only pipeline (preprocess → train → evaluate)
│   └── seizure_prediction_eeg_ecg/   # EEG + ECG pipeline (same scripts, +ECG channel)
├── models/          # saved model checkpoints, per pipeline
├── results/         # metrics.csv, per_subject.csv, reports/, plots, per pipeline
├── notebooks/       # results_analysis.ipynb — EEG vs EEG+ECG comparison figure
├── scripts/         # download_seizeit2.py, tune_eeg.py (automated hyperparameter search)
├── docs/            # experiment_log.md — full record of decisions and runs
├── requirements.txt
└── README.md
```

## Installation & Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

This installs the **CPU build** of PyTorch, which works on any machine — no GPU required.
The training scripts automatically fall back to the CPU; training just runs slower.

**Optional — NVIDIA GPU acceleration.** If you have an NVIDIA (CUDA) GPU with drivers
installed, additionally install the CUDA build of PyTorch (pip cannot detect a GPU from a
requirements file, so this is a separate, one-time command):

```bash
pip install --force-reinstall torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
```

Verify which device PyTorch will use (`True` = GPU available, `False` = CPU only):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Data

The dataset is **SeizeIT2** ([OpenNeuro ds005873](https://openneuro.org/datasets/ds005873)),
a BIDS-formatted wearable recording set: behind-the-ear EEG (2 ch), ECG (1 ch), EMG and
movement, at 256 Hz, recorded in European EMUs. The small BIDS metadata and per-recording
annotations (`events.tsv`) are included in this repo; the large `.edf` signals are **not**
committed — download the dataset and place it under `data/raw/seizeit2/` (keeping the BIDS
`sub-XXX/ses-01/{eeg,ecg}/…` layout).

## How the pipelines work

Each pipeline is three stages (one script each); the real logic lives in the shared
`*_common.py` modules so the EEG and EEG+ECG arms differ only by the ECG channel.

### 1. `preprocess_*` — raw EDFs → windowed dataset

Reads the SeizeIT2 recordings and caches a single array (run once; train as often as you
like). Steps:

- **Power-line noise removal** — 50 Hz notch + harmonics (European mains).
- **Sliding-window segmentation** — 2 s windows, 1 s step (512 timepoints at 256 Hz).
- **Per-recording, per-channel z-score normalization** (`--normalize`, default `per_recording`):
  one mean/std per channel over the whole recording, which keeps the cross-window amplitude
  dynamics that carry pre-ictal information (`per_window` is also available).
- **Pre-ictal labeling** — pre-ictal = `[onset − 10 min, onset)` (positive); the seizure +
  a 10 min post-ictal guard are excluded; everything else is interictal.
- **Interictal subsampling** — recordings are ~18 h, so all pre-ictal windows are kept and
  interictal is subsampled at a fixed ratio (`--interictal-ratio`, default 5:1).

Output: `data/processed/eeg_windows.npz` (EEG) or `eeg_ecg_windows.npz` (EEG+ECG).

### 2. `train_model_*` — train the CNN

- **Subject-aware, class-stratified 60/20/20 split:** per subject, pre-ictal and interictal
  windows are split independently and chronologically — the first 60 % → train, the next 20 %
  → validation, the last 20 % → test. Every subject appears in all three sets (within-subject
  prediction), and staying chronological per class avoids temporal leakage. The split is
  deterministic and stored in the checkpoint.
- **Validation set** is used for early stopping (on validation AUC) and any hyperparameter
  tuning; the test set is touched once, at the very end.
- **Class imbalance** is handled with positive class weighting in the loss.
- **Optional ensemble** (`--ensemble-runs N`): N independently initialized models whose class
  probabilities are averaged at evaluation (soft voting).
- The 1-D CNN adapts automatically to the channel count (2 for EEG, 3 for EEG+ECG).

### 3. `evaluate_*` — evaluate on the held-out test set

Reconstructs the same test set and reports a full diagnostic block: precision / recall / F1 /
specificity, confusion matrix, **AUC-ROC / AUC-PR**, an over/underfit check (train vs test F1),
a threshold sweep, and **per-subject AUC** (the headline metric — the pooled AUC is biased by
per-patient probability scales). Outputs to `results/seizure_prediction_*/`:
`metrics.csv` (one row per run), `per_subject.csv`, a timestamped report under `reports/`, an
average pre-ictal window plot, and a **Grad-CAM** figure.

## Usage

```bash
# EEG-only (from src/seizure_prediction_eeg)
python preprocess_eeg.py      # -> data/processed/eeg_windows.npz (run once)
python train_model_eeg.py     # subject-aware 60/20/20 split; default 30 epochs
python evaluate_eeg.py        # -> metrics.csv, per_subject.csv, reports/, plots

# EEG + ECG (from src/seizure_prediction_eeg_ecg)
python preprocess_eeg_ecg.py
python train_model_eeg_ecg.py
python evaluate_eeg_ecg.py

# RandomForest baseline (from repo root; uses the same windows + split as the CNN)
python src/baseline_rf.py --data data/processed/eeg_windows.npz --feature-set eeg
python src/baseline_rf.py --data data/processed/eeg_ecg_windows.npz --feature-set eeg_ecg
```

Every script accepts `--help` for its options. For a robust comparison, repeat
train+evaluate across several `--random-state` seeds and report the mean ± std (single runs
are noisy on these small per-subject test sets).

### Automated hyperparameter tuning (`scripts/tune_eeg.py`)

To avoid manually rerunning the EEG pipeline while changing one variable at a time, use the
tuning script. It runs **preprocess → train → evaluate** in a loop, varying one
hyperparameter at a time (coordinate descent), keeping improvements, and stopping when
validation **AUC-ROC** reaches the target or no further gain is found.

```bash
# Full search (30 epochs per trial — recommended)
python scripts/tune_eeg.py

# Faster screening only (12 epochs; re-confirm top configs with full training)
python scripts/tune_eeg.py --quick

# Resume after an interruption
python scripts/tune_eeg.py --resume

# Sweep a single variable (e.g. dropout, skipping preprocess)
python scripts/tune_eeg.py --param dropout --values 0.1 0.25 0.4 --skip-preprocess
```

**Parameters searched:** step size (`--step-sec`), sampling rate (`--target-sfreq`),
interictal ratio (`--interictal-ratio`), batch size (`--batch-size`), dropout (`--dropout`).
**Fixed:** window size **2.0 s**, notch filter **50 Hz** (European mains; not tuned).

**How many trials?** Each trial is one full preprocess + train + evaluate cycle. The script
tries up to **~121 trials** in the worst case: 1 baseline run + up to **5 rounds**
(`--max-rounds`, default 5) × **24 candidate values** across the five tunable parameters
(6 + 3 + 6 + 4 + 5). It usually runs fewer — already-tried configs are skipped, and the
search stops early if validation AUC-ROC ≥ **0.7** (default `--target`) or a round finds no
improvement. Preprocessing is skipped automatically when only training params change (batch
size, dropout).

**Epochs per trial:** **30** by default (same as `train_model_eeg.py`), with early stopping
after **5** epochs without validation-AUC gain. `--quick` uses **12** epochs and patience
**3** for faster but less reliable screening.

**How long?** Roughly **20–40 minutes per trial** on a mid-range GPU (e.g. GTX 1070): ~3–5
min preprocess, ~15–30 min train, ~1–2 min evaluate. A full search can therefore take from a
few hours (if it stops early) up to **several days** in the worst case. The PC must stay
awake and the terminal open; use `--resume` if interrupted.

**Results** are written under `results/tuning/eeg/`:

| File | Contents |
|------|----------|
| `trials.csv` | Every trial: AUC-ROC, F1, and all parameter values |
| `best_config.json` | Best settings so far + exact commands to rerun on the main pipeline |
| `state.json` | Checkpoint for `--resume` |

The metric used for selection is **validation AUC-ROC** (`--eval-split val`, default in
`evaluate_eeg.py`). Chance level is **0.5** (`--baseline`). When tuning finishes, run the
commands in `best_config.json` on the main paths (`data/processed/eeg_windows.npz`,
`models/seizure_prediction_eeg/`, `results/seizure_prediction_eeg/`) for the final model.
The same preprocessing flags are available on `preprocess_eeg_ecg.py` so the winning EEG
settings can be reused for the EEG+ECG pipeline.

## Baseline

`src/baseline_rf.py` is a **RandomForest baseline** for comparison against the CNN. It uses the
same preprocessed windows and the same within-subject 60/20/20 split, but instead of learning
from the raw waveform it trains on hand-crafted **log band-power features** (delta, theta,
alpha, beta, gamma per channel). It trains on the train block and reports on the held-out test
block, so AUC-ROC / AUC-PR / per-subject AUC / F1 are directly comparable to `evaluate_*.py`.
Results are written to `results/seizure_prediction_<feature-set>/test/baseline_rf_metrics.csv`
and a matching report. A tree-based baseline like this also makes the CNN's added value
explicit and gives an interpretable feature-importance ranking.

## Reporting

Open `notebooks/results_analysis.ipynb` and run all cells. It reads each pipeline's
`per_subject.csv` and `metrics.csv`, compares **per-subject AUC** between EEG and EEG+ECG
(the comparison that answers the research question), and writes its figures + a
`comparison_summary.txt` to a timestamped folder under `results/comparison/<timestamp>/`.
