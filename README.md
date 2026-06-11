# EEG_Test

CNN-based seizure detector for CHB-MIT EDF files, with GPU support via WSL2.

## What this does

- Loads an EDF file and a seizure summary `.txt` from the CHB-MIT dataset
- Trains a 1-D CNN on windowed EEG segments (or loads a saved model)
- Runs on GPU automatically when CUDA is available (WSL2 / Linux / Windows)
- Saves the trained model to disk for later reuse
- Plots EEG traces with ground-truth seizure intervals (red) and CNN predictions (blue)

## Setup

### CPU only

```powershell
pip install -r requirements.txt
```

### GPU (WSL2 / Linux — CUDA 12.1)

```bash
pip install torch>=2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **WSL2 note:** install the NVIDIA driver on Windows (≥ 512.xx) only — no separate CUDA toolkit inside WSL2 is needed.

## Run

```powershell
python .\train_detect_seizures.py
```

### Common options

| Flag | Default | Description |
|---|---|---|
| `--edf` | `physionet.org/.../chb01_01.edf` | Path to EDF file |
| `--summary` | `physionet.org/.../chb01-summary.txt` | Path to seizure summary |
| `--window-sec` | `2.0` | Sliding window size (s) |
| `--step-sec` | `1.0` | Sliding window step (s) |
| `--train-frac` | `0.7` | Fraction of data used for training |
| `--epochs` | `30` | CNN training epochs |
| `--batch-size` | `64` | Mini-batch size |
| `--lr` | `1e-3` | Adam learning rate |
| `--no-gpu` | off | Force CPU even if GPU is present |
| `--save-model` | `seizure_cnn.pt` | Where to save the trained model |
| `--load-model` | _(none)_ | Load a saved model, skip training |
| `--start` | `0.0` | Plot window start time (s) |
| `--duration` | `60.0` | Plot window duration (s) |
| `--max-channels` | `8` | Max EEG channels to plot |
| `--channels` | _(all EEG)_ | Comma-separated channel names to plot |
| `--save` | `train_detect_chb01_01.png` | Output plot path |
| `--show` | off | Open interactive plot window |

### Examples

Train and save a model:

```powershell
python .\train_detect_chb01_01.py --epochs 50 --save-model seizure_cnn.pt
```

Load a saved model and plot a specific time window:

```powershell
python .\train_detect_chb01_01.py --load-model seizure_cnn.pt --start 2000 --duration 120
```

Select specific channels:

```powershell
python .\train_detect_chb01_01.py --channels "FP1-F7,F7-T7,T7-P7" --show
```

Force CPU:

```powershell
python .\train_detect_chb01_01.py --no-gpu
```