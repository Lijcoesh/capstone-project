# -*- coding: utf-8 -*-
"""
Train the EEG-only seizure-prediction CNN (SeizeIT2).

Thin wrapper: the training logic lives in model_common; this just sets the
EEG-only dataset/model paths. Trains on the first 80% (chronological) and saves
the model.

Example:
  python train_model_eeg.py --epochs 30 --random-state 42
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_common import add_training_args, run_training  # noqa: E402
from preprocess_eeg import DEFAULT_PREPROCESSED_PATH       # noqa: E402

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "../../models/seizure_prediction_eeg/seizure_cnn.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the EEG-only seizure-prediction CNN.")
    add_training_args(parser, DEFAULT_PREPROCESSED_PATH, DEFAULT_MODEL_PATH)
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
