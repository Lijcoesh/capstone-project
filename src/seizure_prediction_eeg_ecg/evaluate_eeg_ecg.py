# -*- coding: utf-8 -*-
"""
Evaluate the EEG + ECG seizure-prediction model on the held-out 20% test set.

Thin wrapper: the evaluation logic lives in evaluate_common; this just sets the
EEG+ECG dataset/model/results paths and the feature-set label ('eeg_ecg') written
to metrics.csv, so the notebook can compare EEG vs EEG+ECG.

Example:
  python evaluate_eeg_ecg.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_common import add_eval_args, run_evaluation  # noqa: E402
from preprocess_eeg_ecg import DEFAULT_PREPROCESSED_PATH    # noqa: E402
from train_model_eeg_ecg import DEFAULT_MODEL_PATH          # noqa: E402

DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent / "../../results/seizure_prediction_eeg_ecg"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the EEG+ECG seizure-prediction model.")
    add_eval_args(parser, DEFAULT_PREPROCESSED_PATH, DEFAULT_MODEL_PATH, DEFAULT_RESULTS_DIR, "eeg_ecg")
    run_evaluation(parser.parse_args())


if __name__ == "__main__":
    main()
