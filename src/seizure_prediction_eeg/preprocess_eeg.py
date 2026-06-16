# -*- coding: utf-8 -*-
"""
Preprocessing for the EEG-only seizure-prediction pipeline (SeizeIT2).

Thin wrapper around preprocess_common: loads EEG only (2 channels) and writes the
windowed dataset to data/processed/eeg_windows.npz. All logic (BIDS discovery,
pre-ictal labeling, 50 Hz notch, interictal subsampling) lives in preprocess_common
so the EEG-only and EEG+ECG pipelines differ by nothing but the ECG channel.

Use:
  python preprocess_eeg.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess_common import (  # noqa: E402
    DEFAULT_INTERICTAL_RATIO,
    DEFAULT_NORMALIZE,
    DEFAULT_STEP_SEC,
    DEFAULT_WINDOW_SEC,
    NORMALIZE_CHOICES,
    PREICTAL_SEC,
    build_dataset,
    load_preprocessed,
    save_preprocessed,
)

DEFAULT_PREPROCESSED_PATH = (
    Path(__file__).resolve().parent / "../../data/processed/eeg_windows.npz"
)

__all__ = ["DEFAULT_PREPROCESSED_PATH", "load_preprocessed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess SeizeIT2 EEG into a windowed .npz dataset (seizure prediction)."
    )
    parser.add_argument("--interictal-ratio", type=float, default=DEFAULT_INTERICTAL_RATIO,
                        help="Interictal : pre-ictal windows kept (default 5).")
    parser.add_argument("--preictal-min", type=float, default=PREICTAL_SEC / 60.0,
                        help="Pre-ictal horizon in minutes: windows within this many "
                             f"minutes before onset = positive (default {PREICTAL_SEC / 60:.0f}).")
    parser.add_argument("--normalize", choices=NORMALIZE_CHOICES, default=DEFAULT_NORMALIZE,
                        help="Window normalization: 'per_window' (default) z-scores each "
                             "window; 'per_recording' uses one mean/std per channel over "
                             "the whole recording (keeps cross-window amplitude dynamics).")
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--step-sec", type=float, default=DEFAULT_STEP_SEC)
    parser.add_argument("--require-ecg", action="store_true",
                        help="Keep only recordings that also have an ECG file, so this "
                             "EEG-only set is built on the SAME recordings as the EEG+ECG "
                             "set (fair paired comparison).")
    parser.add_argument("--subjects", type=str, nargs="+", default=None,
                        help="Limit to specific subjects, e.g. sub-001 sub-002 (default: all).")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_PREPROCESSED_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset(
        include_ecg=False,
        interictal_ratio=args.interictal_ratio,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        subjects=args.subjects,
        random_state=args.random_state,
        preictal_sec=args.preictal_min * 60.0,
        normalize=args.normalize,
        require_ecg=args.require_ecg,
    )
    save_preprocessed(args.out, result)


if __name__ == "__main__":
    main()
