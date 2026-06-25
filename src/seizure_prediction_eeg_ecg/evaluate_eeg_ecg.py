# -*- coding: utf-8 -*-
"""
LOSO evaluation for the EEG+ECG seizure-prediction pipeline.

Same protocol as evaluate_eeg.py; writes results/seizure_prediction_eeg_ecg/loso/.

Use:
  python evaluate_eeg_ecg.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loso_common import add_loso_eval_args, run_loso_evaluation  # noqa: E402
from preprocess_eeg_ecg import DEFAULT_PREPROCESSED_PATH                    # noqa: E402

DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent / "../../results/seizure_prediction_eeg_ecg"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LOSO evaluation with per-patient calibration (EEG+ECG)."
    )
    add_loso_eval_args(parser, DEFAULT_PREPROCESSED_PATH, DEFAULT_RESULTS_DIR, "eeg_ecg")
    run_loso_evaluation(parser.parse_args())


if __name__ == "__main__":
    main()
