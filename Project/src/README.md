## Scripts
**edf_test/debug_summary.py:**
- A diagnostic script that reads the CHBMIT chb01-summary.txt file and uses regular expressions to extract and print the raw, truncated text sections for specific EDF files. It is used to inspect the exact formatting of seizure labels within the dataset.

**heart_rate_seizure/main.py:**
- Defines a 1D Convolutional Neural Network (SeizureCNN) trained to detect seizures from range-encoded heart rate data. It includes a pipeline for binning time-series heart rate values into multi-channel inputs (via occupancy, value, or soft encoding) to format them for model inference and evaluate seizure probabilities over sliding windows.

**seizure_detection/train_detect_seizures.py:**
- A comprehensive training and evaluation pipeline that processes multiple CHB-MIT EDF files concurrently. It extracts matching seizure intervals dynamically from subject summary files, creates a unified multi-channel EEG window dataset, and trains an ensemble or standalone 1D PyTorch CNN. The script automatically handles hardware acceleration, prevents cross-file validation leakage, and outputs extensive post-processing metrics alongside synthetic heart-rate correlation overlays, average morphology plots, and Grad-CAM explanations.

**simple_edf_plotter/main.py:**
- A utility script to plot EEG signals from an EDF file. It allows users to specify the time window, channels to plot, and whether to display the plot interactively or save it as an image file. This is useful for quick visualization of EEG data without needing to run the full seizure detection script.