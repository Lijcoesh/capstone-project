# Archive

Experimental artefacts from development (window sweeps, n=20/60 ablations,
per-patient models, Siena external validation, raw-waveform runs, etc.).

**Canonical final paths** (repo root):

| What | Path |
|------|------|
| EEG data | `data/processed/eeg_windows.npz` |
| EEG+ECG data | `data/processed/eeg_ecg_windows.npz` |
| EEG model | `models/seizure_prediction_eeg/cnn_prediction_eeg.pt` |
| EEG+ECG model | `models/seizure_prediction_eeg_ecg/cnn_prediction_eeg_ecg.pt` |
| EEG results | `results/seizure_prediction_eeg/` |
| EEG+ECG results | `results/seizure_prediction_eeg_ecg/` |

Final config: bandpower sequence, 50 s window / 5 s step, 10 min pre-ictal,
60 subjects (`--require-ecg` for paired EEG set). See `run_final.ps1`.

Moved here on 2026-06-18.

**Per-seed checkpoints** (used to build the 5-model ensemble) are in
`archive/models/seed_checkpoints/`. The canonical checkpoints
`cnn_prediction_eeg.pt` / `cnn_prediction_eeg_ecg.pt` each contain all five
`state_dicts` for soft voting at evaluation time.

To retrain the ensemble from scratch (no extra scripts):

```bash
python src/seizure_prediction_eeg/train_model_eeg.py --ensemble-runs 5 --random-state 42
python src/seizure_prediction_eeg_ecg/train_model_eeg_ecg.py --ensemble-runs 5 --random-state 42
```
