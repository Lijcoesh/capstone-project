## Scripts
**edf_test/debug_summary.py:**
- A simple script to read and print the seizure summary file for a given patient, useful for debugging and understanding the dataset structure.

**seizure_detection/train_detect_chb01_01.py:**
- Trains a CNN to detect seizures in the CHB-MIT dataset using a sliding window approach. The script includes options for data preprocessing, model training, evaluation, and visualization of results. It can save the trained model and generate plots of EEG signals with seizure predictions.

**simple_edf_plotter/main.py:**
- A utility script to plot EEG signals from an EDF file. It allows users to specify the time window, channels to plot, and whether to display the plot interactively or save it as an image file. This is useful for quick visualization of EEG data without needing to run the full seizure detection script.

**simple_edf_plotter_old/main.py:**
- An older version of the EEG plotting script, this should be removed to avoid confusion.