# -*- coding: utf-8 -*-
"""
Train the EEG + ECG seizure-prediction CNN (SeizeIT2).

Thin wrapper: the training logic lives in model_common; this just sets the
EEG+ECG dataset/model paths. Identical to the EEG-only trainer apart from the
3-channel input (the CNN adapts automatically).

Use:
  python train_model_eeg_ecg.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_common import add_training_args, run_training  # noqa: E402
from preprocess_eeg_ecg import DEFAULT_PREPROCESSED_PATH   # noqa: E402

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "../../models/seizure_prediction_eeg_ecg/cnn_prediction_eeg_ecg.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the EEG+ECG seizure-prediction CNN.")
    add_training_args(parser, DEFAULT_PREPROCESSED_PATH, DEFAULT_MODEL_PATH)
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
