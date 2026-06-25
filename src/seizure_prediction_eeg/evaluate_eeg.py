# -*- coding: utf-8 -*-
"""
LOSO evaluation for the EEG-only seizure-prediction pipeline.

Trains a fresh population model per fold (each held-out subject = new patient),
calibrates a per-patient threshold on that subject's calibration block, and reports
on the held-out test block. Writes results/seizure_prediction_eeg/loso/.

Use:
  python evaluate_eeg.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loso_common import add_loso_eval_args, run_loso_evaluation  # noqa: E402
from preprocess_eeg import DEFAULT_PREPROCESSED_PATH                      # noqa: E402

DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent / "../../results/seizure_prediction_eeg"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LOSO evaluation with per-patient calibration (EEG-only)."
    )
    add_loso_eval_args(parser, DEFAULT_PREPROCESSED_PATH, DEFAULT_RESULTS_DIR, "eeg")
    run_loso_evaluation(parser.parse_args())


if __name__ == "__main__":
    main()
