# CHB-MIT Multi-File Seizure Detection

A GPU-accelerated deep learning pipeline using a 1-D Convolutional Neural Network (CNN) in PyTorch to detect epileptic seizures from multi-channel CHB-MIT EEG recordings.

---

## Features

* **Multi-File Processing:** Accepts any number of space-separated EDF files via `--edf`.
* **Smart Summary Resolution:** Auto-detects seizure annotations from each subject's text summary (e.g., `chb01/chb01-summary.txt`), with an optional `--summaries` override.
* **Chronological Windowing:** Concatenates signal windows across files chronologically before performing the train/test split to completely prevent data leakage.
* **Channel Alignment:** Performs a silent intersection of EEG channels, keeping only channels present across all loaded files.
* **Hardware Acceleration:** Auto-detects the best available processing unit (CUDA -> MPS -> CPU).
* **Advanced Ensembling:** Supports training an ensemble of N models via `--ensemble-runs`, utilizing majority-vote predictions.
* **Synthetic Heart Rate Overlay:** Generates physiologically plausible heart rate data (resting ~60–75 bpm, shifting via a logarithmic curve up to an ictal peak of ~130–160 bpm) synced with seizure intervals and overlaid onto the evaluation plots.
* **Visualization:** Automatically extracts and exports average seizure morphology metrics (mean ± 1 SD) across channels.

---

## Project Structure & Artifact Outputs

All training outputs are automatically saved to the project's dedicated results architecture:
* **Models:** Saved to `../../models/seizure_detection/seizure_cnn.pt` (configurable via `--save-model` / `--load-model`).
* **Plots & Figures:** PNG artifacts (EEG overlays, Grad-CAM, average morphology) are written directly to `../../results/seizure_detection/`.

---

## Installation & Setup

### 1. Fast Automation (Recommended)
An interactive setup script is provided for in the root directory to initialize the local Python virtual environment and fetch system dependencies:

```bash
chmod +x ./quick_setup.sh
./quick_setup.sh
source .venv/bin/activate
```

If you want to use an NVIDIA GPU for training, install PyTorch with CUDA support. This requires a compatible NVIDIA GPU and NVIDIA drivers installed on the host system.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify that PyTorch detects your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```