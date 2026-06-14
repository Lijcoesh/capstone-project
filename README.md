# A Comparative Study of EEG-Only vs. EEG-ECG Feature Sets for Seizure Prediction Using a 1D Convolutional Neural Network

Capstone project on **epileptic-seizure prediction** from wearable physiological signals.
A 1-D Convolutional Neural Network is trained on the **SeizeIT2** dataset, and two feature
sets are compared under identical conditions: an **EEG-only** model (2 channels) and an
**EEG + ECG** model (3 channels). Because both models use the same dataset, the only
difference is the ECG channel — a controlled comparison of whether ECG adds predictive value.

**Task.** Seizure *prediction* (not detection): each signal window is labeled **pre-ictal**
(within 30 min before a seizure onset → positive) or **interictal** (far from any seizure →
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
│   ├── seizure_prediction_eeg/       # EEG-only pipeline (preprocess → train → evaluate)
│   ├── seizure_prediction_eeg_ecg/   # EEG + ECG pipeline (same scripts, +ECG channel)
│   └── simple_edf_plotter/           # EDF visualization utility
├── models/          # saved model checkpoints, per pipeline
├── results/         # metrics.csv, per_subject.csv, reports/, plots, per pipeline
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
- **Sliding-window segmentation** — 2 s windows, 1 s step (512 timepoints at 256 Hz).
- **Per-window, per-channel z-score normalization.**
- **Pre-ictal labeling** — pre-ictal = `[onset − 30 min, onset)` (positive); the seizure +
  a 10 min post-ictal guard are excluded; everything else is interictal.
- **Interictal subsampling** — recordings are ~18 h, so all pre-ictal windows are kept and
  interictal is subsampled at a fixed ratio (`--interictal-ratio`, default 5:1).

Output: `data/processed/eeg_windows.npz` (EEG) or `eeg_ecg_windows.npz` (EEG+ECG).

### 2. `train_model_*` — train the CNN

- **Subject-aware 80/20 split:** per subject, the first 80 % of their windows (chronological)
  → train, the last 20 % → test, so every subject appears in both (within-subject prediction).
  The split is deterministic and stored in the checkpoint.
- **Class imbalance** is handled with positive class weighting in the loss.
- **Optional ensemble** (`--ensemble-runs N`): N independently initialized models whose class
  probabilities are averaged at evaluation (soft voting).
- The 1-D CNN adapts automatically to the channel count (2 for EEG, 3 for EEG+ECG).

### 3. `evaluate_*` — evaluate on the held-out 20 %

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
python train_model_eeg.py     # subject-aware 80/20 split; default 30 epochs
python evaluate_eeg.py        # -> metrics.csv, per_subject.csv, reports/, plots

# EEG + ECG (from src/seizure_prediction_eeg_ecg)
python preprocess_eeg_ecg.py
python train_model_eeg_ecg.py
python evaluate_eeg_ecg.py
```

Every script accepts `--help` for its options. For a robust comparison, repeat
train+evaluate across several `--random-state` seeds and report the mean ± std (single runs
are noisy on these small per-subject test sets).

## Reporting

Open `notebooks/results_analysis.ipynb` and run all cells. It reads each pipeline's
`per_subject.csv` and `metrics.csv`, compares **per-subject AUC** between EEG and EEG+ECG
(the comparison that answers the research question), and writes
`results/comparison_eeg_vs_eeg_ecg.png`. A narrative record of all decisions and runs is in
[`docs/experiment_log.md`](docs/experiment_log.md).

## Utility

`src/simple_edf_plotter/main.py` plots EEG channels from an EDF over a chosen time window —
handy for quick inspection without running the full pipeline.
