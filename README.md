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
├── models/          # cnn_prediction_eeg.pt / cnn_prediction_eeg_ecg.pt (5-run ensemble)
├── results/         # seizure_prediction_eeg/, seizure_prediction_eeg_ecg/
├── archive/         # experimental runs + per-seed checkpoints + legacy scripts
├── notebooks/       # results_analysis.ipynb, eda.ipynb
├── notebooks/       # results_analysis.ipynb — EEG vs EEG+ECG comparison figure
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
- **Sliding-window segmentation** — 50 s windows, 5 s step (final config).
- **Band-power sequence** (`--input-rep bandpower_seq`): log band-power per EEG/ECG channel in
  short frames across each window — the CNN input for the final model (not raw waveforms).
- **Per-recording, per-channel z-score normalization** (`--normalize`, default `per_recording`):
  one mean/std per channel over the whole recording, which keeps the cross-window amplitude
  dynamics that carry pre-ictal information (`per_window` is also available).
- **Pre-ictal labeling** — pre-ictal = `[onset − 10 min, onset)` (positive); the seizure +
  a 10 min post-ictal guard are excluded; everything else is interictal.
- **Interictal subsampling** — recordings are ~18 h, so all pre-ictal windows are kept and
  interictal is subsampled at a fixed ratio (`--interictal-ratio`, default 10:1).

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

### 3. `evaluate_*` — LOSO + per-patient calibration

Default evaluation is **leave-one-subject-out (LOSO)** with per-patient threshold
calibration — the clinically honest setup for a *new patient* the model never trained on:

- For each held-out subject `P`: train a fresh CNN on all **other** subjects.
- **Calibration** (first 20% of `P`'s windows, chronological per class): tune a
  patient-specific decision threshold.
- **Test** (remaining 80% of `P`): report AUC and F1 before/after calibration.
- **Before calibration**: population threshold tuned on the other subjects' val blocks.
- **After calibration**: patient threshold tuned on `P`'s calibration block.

Outputs under `results/seizure_prediction_*/loso/`:
`loso_per_subject.csv` (one row per subject), `loso_metrics.csv` (summary),
and a timestamped report under `loso/reports/`.

`train_model_*.py` still trains one **deployment** model on the within-subject split;
`evaluate_*.py` runs the rigorous LOSO protocol (re-trains per fold).

## Usage

```bash
# EEG-only (from src/seizure_prediction_eeg)
python preprocess_eeg.py
python train_model_eeg.py
python evaluate_eeg.py          # LOSO + per-patient calibration (55 folds)

# EEG + ECG (from src/seizure_prediction_eeg_ecg)
python preprocess_eeg_ecg.py
python train_model_eeg_ecg.py
python evaluate_eeg_ecg.py      # LOSO + per-patient calibration

# RandomForest baseline (repo root; train + evaluate in one script)
python src/baseline_rf.py --data data/processed/eeg_windows.npz --feature-set eeg --eval-split test
python src/baseline_rf.py --data data/processed/eeg_ecg_windows.npz --feature-set eeg_ecg --eval-split test
```

Every script accepts `--help` for its options. For a robust comparison, repeat
train+evaluate across several `--random-state` seeds and report the mean ± std (single runs
are noisy on these small per-subject test sets).

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
