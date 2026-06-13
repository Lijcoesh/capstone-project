# A Comparative Study of EEG-Only vs. EEG-ECG Feature Sets for Seizure Prediction with a 1D Convolutional Neural Network

Capstone project on epileptic-seizure detection from physiological signals. A 1-D
Convolutional Neural Network is trained on the CHB-MIT dataset, and two feature sets are
compared: an **EEG-only** model and an **EEG + ECG** model. The pipeline covers
preprocessing, training, evaluation (precision / recall / F1 / confusion matrix), temporal
post-processing, and Grad-CAM explainability.

## Project structure

```
capstone-project/        # repo root
├── data/
│   ├── raw/         # original CHB-MIT download (immutable, EDF via Git LFS)
│   └── processed/   # preprocessed windowed dataset (.npz; regenerated, not committed)
├── src/
│   ├── seizure_detection_eeg/      # EEG-only pipeline (preprocess → train → evaluate)
│   ├── seizure_detection_eeg_ecg/  # EEG + ECG pipeline (work in progress)
│   └── simple_edf_plotter/         # EEG visualization utility
├── models/          # saved model checkpoints, per pipeline
├── results/         # figures, plots, and metrics.csv logs, per pipeline
├── notebooks/       # results_analysis.ipynb — EEG vs EEG+ECG comparison figure
├── requirements.txt
└── README.md
```

## Installation & Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

This installs the **CPU build** of PyTorch, which works on any machine — no GPU
required. The training scripts automatically fall back to the CPU; training just
runs slower.

**Optional — NVIDIA GPU acceleration.** If you have an NVIDIA (CUDA) GPU with drivers
installed, additionally install the CUDA build of PyTorch. (pip cannot auto-detect a GPU
from a requirements file, so this is a separate, one-time command.)

```bash
pip install --force-reinstall torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
```

Verify which device PyTorch will use (`True` = GPU available, `False` = CPU only):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Data

The small annotation/metadata files (the `chbXX-summary.txt` seizure labels, `RECORDS`,
`SUBJECT-INFO`, etc.) are **included in this repository**. The large raw EEG signals (the
`.edf` files) are **not** committed — download them from
[PhysioNet (CHB-MIT)](https://physionet.org/content/chbmit/1.0.0/) and place them under
`data/raw/physionet.org/files/chbmit/1.0.0/` (the same folder layout PhysioNet provides).

## How the EEG-only pipeline works

The pipeline is split into three stages (one script each) for a clean separation of
concerns. Run them in order from `src/seizure_detection_eeg/`:

### 1. `preprocess_eeg.py` — raw EDFs → windowed dataset

Reads the CHB-MIT EDF files and produces a single cached array, so preprocessing is
decoupled from training (run once; train as often as you like). Steps:

- **Power-line noise removal** — 60 Hz notch filter + harmonics below Nyquist (CHB-MIT was
  recorded in Boston, US mains = 60 Hz).
- **Channel alignment** — keeps only EEG channels present in *all* files (silent intersection).
- **Sliding-window segmentation** — fixed-length windows (default 2 s, 1 s step).
- **Per-window, per-channel z-score normalization.**
- **Labeling** — a window is *ictal* if ≥ 50 % of its duration overlaps a ground-truth
  seizure interval (from the subject summary files), else *interictal*.

Windows from all files are concatenated chronologically and saved to
`data/processed/eeg_windows.npz` (along with metadata: channels, sampling rate, window
size, notch frequency, file boundaries).

### 2. `train_model_eeg.py` — train on the first 80 %

- Loads the `.npz` and takes the first **80 %** of the chronological sequence as the
  training set (the split is chronological to avoid temporal leakage).
- Trains a 1-D CNN. **Class imbalance** (~1.5 % seizure windows) is handled with positive
  class weighting in the loss.
- **Optional ensemble** (`--ensemble-runs N`): trains N independently initialized models;
  their class probabilities are averaged at evaluation time (soft voting).
- Saves the model to `models/seizure_detection_eeg/seizure_cnn.pt`. The split fraction is
  stored in the checkpoint so evaluation reconstructs exactly the same test set.

### 3. `evaluate_eeg.py` — evaluate on the held-out 20 %

- Loads the model + dataset and reconstructs the same **20 %** test set.
- Computes **precision / recall / F1 / confusion matrix**.
- **Temporal post-processing** removes isolated positive runs shorter than a minimum
  duration (default 2 windows) to reduce false positives.
- Appends one row of metrics + config to `results/seizure_detection_eeg/metrics.csv`.
- Writes diagnostic plots (EEG overlay, average seizure morphology, synthetic heart-rate
  overlay\*) and a **Grad-CAM** figure showing which time regions drove each prediction.

\* The heart-rate trace in the EEG-only plots is *synthetic* (a visualization aid), not a
model input. Real ECG features belong to the EEG + ECG pipeline.

## Usage

```bash
cd src/seizure_detection_eeg

# 1. Preprocess raw EDFs -> data/processed/eeg_windows.npz (run once).
python preprocess_eeg.py

# 2. Train on the first 80% (vary --random-state for multiple seeds).
python train_model_eeg.py --epochs 30 --random-state 42

# 3. Evaluate on the held-out 20% (writes metrics.csv + plots + Grad-CAM).
python evaluate_eeg.py
```

Every script accepts `--help` for its full list of options, e.g.
`python train_model_eeg.py --help`. For a robust comparison, repeat steps 2–3 across
several `--random-state` values and report the mean ± std (a single run is noisy because
the test set is essentially one recording).

## Reporting

Open `notebooks/results_analysis.ipynb` and run all cells. It reads the `metrics.csv`
log(s), aggregates **mean ± std** across runs, and produces the EEG-only vs. EEG+ECG
comparison figure and a paste-ready summary table for the paper.

## EEG + ECG pipeline (work in progress)

`src/seizure_detection_eeg_ecg/` will hold the combined model. It currently analyses a
*simulated* heart-rate signal and will be extended with a real EEG+ECG dataset, following
the same `preprocess → train → evaluate` structure as the EEG-only pipeline so the two
feature sets can be compared fairly.

## Utility

`src/simple_edf_plotter/main.py` plots EEG channels from an EDF over a chosen time window —
handy for quick inspection without running the full pipeline.
