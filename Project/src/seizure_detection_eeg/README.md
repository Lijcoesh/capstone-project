# CHB-MIT Multi-File Seizure Detection

A GPU-accelerated deep learning pipeline using a 1-D Convolutional Neural Network (CNN) in PyTorch to detect epileptic seizures from multichannel CHB-MIT EEG recordings.

---

## Features

* **Multi-File Processing:** Accepts any number of space-separated EDF files via `--edf`.
* **Smart Summary Resolution:** Auto-detects seizure annotations from each subject's text summary (e.g., `chb01/chb01-summary.txt`), with an optional `--summaries` override.
* **Chronological Windowing:** Concatenates signal windows across files chronologically before performing the train/test split to completely prevent data leakage.
* **Channel Alignment:** Performs a silent intersection of EEG channels, keeping only channels present across all loaded files.
* **Hardware Acceleration:** Auto-detects the best available processing unit (CUDA -> MPS -> CPU).
* **Advanced Ensembling:** Supports training an ensemble of N models via `--ensemble-runs`, averaging the predicted class probabilities across runs (soft voting).
* **Synthetic Heart Rate Overlay:** Generates physiologically plausible heart rate data (resting ~60–75 bpm, shifting via a logarithmic curve up to an ictal peak of ~130–160 bpm) synced with seizure intervals and overlaid onto the evaluation plots — for visualization only, not a model input.
* **Visualization:** Automatically extracts and exports average seizure morphology metrics (mean ± 1 SD) across channels.

---

## Project Structure & Artifact Outputs

All training outputs are automatically saved to the project's dedicated results architecture:
* **Models:** Saved to `../../models/seizure_detection_eeg/seizure_cnn.pt` (configurable via `--save-model` / `--load-model`).
* **Plots & Figures:** PNG artifacts (EEG overlays, Grad-CAM, average morphology) are written directly to `../../results/seizure_detection_eeg/`.

## Run

```powershell
python .\train_model_eeg.py
```

### Common options

| Flag                  | Default                                         | Description                                                                       |
|-----------------------|-------------------------------------------------|-----------------------------------------------------------------------------------|
| `--edf`               | *(The 19 files spanning chb01–chb04)*           | Space-separated path(s) to EDF file(s) in chronological order                     |
| `--summaries`         | `None` *(Auto-detected via parent dir)*         | Explicit path(s) to seizure summaries (one per unique subject folder)             |
| `--window-sec`        | `2.0`                                           | Sliding window size in seconds                                                    |
| `--step-sec`          | `1.0`                                           | Sliding window step size in seconds                                               |
| `--train-frac`        | `0.8`                                           | Fraction of concatenated data used for training (rest is held-out test); 80/20    |
| `--epochs`            | `30`                                            | CNN training epochs                                                               |
| `--batch-size`        | `64`                                            | Mini-batch size for training                                                      |
| `--lr`                | `0.001` (`1e-3`)                                | Adam optimizer learning rate                                                      |
| `--pred-threshold`    | `0.5`                                           | Positive-class threshold for seizure prediction (0-1). Lower = more detections    |
| `--pred-min-run`      | `2`                                             | Minimum consecutive positive windows to keep after thresholding                   |
| `--no-gpu`            | *disabled*                                      | Force CPU execution even if CUDA or MPS acceleration is available                 |
| `--random-state`      | `42`                                            | Random seed for reproducibility across PyTorch and NumPy operations               |
| `--save-model`        | `../../models/seizure_detection_eeg/seizure_cnn.pt` | Target file path to export the trained model weights and metadata                 |
| `--load-model`        | `None`                                          | Path to load a pre-trained model checkpoint, skipping the training pipeline       |
| `--ensemble-runs`     | `1`                                             | Train N independent models and aggregate by averaging class probabilities (soft voting) |
| `--plot-edf`          | `None` *(Defaults to last --edf with seizures)* | Explicit path of the specific EDF file to generate visualization plots for        |
| `--start`             | `0.0`                                           | Start time offset for evaluation plotting (seconds)                               |
| `--duration`          | `60.0`                                          | Total duration window length for evaluation plotting (seconds)                    |
| `--max-channels`      | `8`                                             | Maximum number of shared EEG channels to render on the overlay plot               |
| `--channels`          | `""` *(All intersecting EEG channels)*          | Explicit comma-separated channel names to narrow down visualization plotting      |
| `--save`              | `eeg_overlay.png`                               | Output file path for the multi-channel EEG signal overlay plot                    |
| `--show`              | *disabled*                                      | Instantly launch an interactive UI plot window upon execution completion          |
| `--save-seizure-plot` | `train_detect_chb01.png`                        | Output file path for the average seizure morphology ($mean \pm 1\text{ SD}$) plot |
| `--save-hr-plot`      | `heart_rate_seizures.png`                       | Output file path for the standalone synthetic heart rate metric plot              |
| `--save-gradcam-plot` | `gradcam.png`                                   | Output file path for the Grad-CAM model interpretability explanation plot         |
| `--gradcam-n-samples` | `4`                                             | Number of top distinct seizure windows to explain via Grad-CAM processing         |
### Examples

Train and save a model:

```powershell
python .\train_model_eeg.py --epochs 50 --save-model ../../models/seizure_detection_eeg/seizure_cnn.pt
```

Load a saved model and plot a specific time window:

```powershell
python train_model_eeg.py --load-model ../../models/seizure_detection_eeg/seizure_cnn.pt --start 2000 --duration 120
```

Select specific channels:

```powershell
python train_model_eeg.py --channels "FP1-F7,F7-T7,T7-P7" --show
```

Force CPU:

```powershell
python train_model_eeg.py --no-gpu
```
