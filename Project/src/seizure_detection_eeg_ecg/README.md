# EEG + ECG Seizure Detection

The second pipeline in this comparative study: a seizure-detection model that uses **both EEG and ECG-derived features**. Its scores are meant to be compared against the EEG-only model in [`../seizure_detection_eeg/`](../seizure_detection_eeg/).

> **Status: work in progress.** The CHB-MIT dataset is EEG-only, so the ECG side is currently driven by a *simulated* heart-rate signal (`simulate_seizure_heartrate`). This will be replaced by a real EEG+ECG dataset, after which the model is trained on the combined feature set.

---

## Script

**`train_model_eeg_ecg.py`**
- Encodes a heart-rate signal into range bins, runs a 1-D CNN over sliding windows, and reports detected seizure segments with heart-rate statistics (baseline, peak, change from baseline, etc.).
- Includes a synthetic generator (`simulate_seizure_heartrate`) for testing without real ECG data.

## Artifact Outputs

* **Model:** loaded from / saved to `../../models/seizure_detection_eeg_ecg/seizure_cnn.pt`.
* **Plots:** written to `../../results/seizure_detection_eeg_ecg/`.

## Dependencies

Uses the shared `../requirements.txt` (no separate requirements file in this folder).

## Run

```powershell
python .\train_model_eeg_ecg.py
```
