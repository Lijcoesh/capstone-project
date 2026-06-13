## Scripts
**seizure_detection_eeg/train_model_eeg.py:**
- Trains a 1-D CNN to detect seizures in the CHB-MIT dataset from **EEG signals only**, using a sliding window approach. Includes data preprocessing (power-line notch filter, per-channel z-score), an 80/20 chronological train/test split, model training, evaluation, and visualization. Saves the trained model to `../../models/seizure_detection_eeg/` and plots to `../../results/seizure_detection_eeg/`.

**seizure_detection_eeg_ecg/train_model_eeg_ecg.py:**
- The **EEG + ECG** pipeline (work in progress). Currently analyses (simulated) heart-rate signals; to be extended with real EEG+ECG data so its scores can be compared against the EEG-only model. Loads/saves its model under `../../models/seizure_detection_eeg_ecg/`.

**simple_edf_plotter/main.py:**
- A utility script to plot EEG signals from an EDF file. It allows users to specify the time window, channels to plot, and whether to display the plot interactively or save it as an image file. This is useful for quick visualization of EEG data without needing to run the full seizure detection script.