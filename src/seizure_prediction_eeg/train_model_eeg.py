# -*- coding: utf-8 -*-
"""
Train the EEG-only seizure-prediction CNN (SeizeIT2).

Thin wrapper: the training logic lives in model_common; this just sets the
EEG-only dataset/model paths. Uses the within-subject 60/20/20 split (train on
the first 60% chronological, validate on the next 20%) and saves the model.

Use:
  python train_model_eeg.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_common import add_training_args, run_training  # noqa: E402
from preprocess_eeg import DEFAULT_PREPROCESSED_PATH       # noqa: E402

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "../../models/seizure_prediction_eeg/cnn_prediction_eeg.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the EEG-only seizure-prediction CNN.")
    add_training_args(parser, DEFAULT_PREPROCESSED_PATH, DEFAULT_MODEL_PATH)
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
