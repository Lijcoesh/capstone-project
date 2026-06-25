# -*- coding: utf-8 -*-
"""
Preprocessing for the EEG + ECG seizure-prediction pipeline (SeizeIT2).

Thin wrapper around preprocess_common: loads EEG **and** ECG (3 channels) and
writes the windowed dataset to data/processed/eeg_ecg_windows.npz. All the actual
logic (BIDS discovery, pre-ictal labeling, 50 Hz notch, interictal subsampling)
lives in preprocess_common so the EEG-only and EEG+ECG pipelines stay identical
apart from the ECG channel.

Use:
  python preprocess_eeg_ecg.py
"""

import argparse
import sys
from pathlib import Path

# Make the shared module (one level up, in src/) importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess_common import (  # noqa: E402
    DEFAULT_FRAME_SEC,
    DEFAULT_FRAME_STEP_SEC,
    DEFAULT_INPUT_REP,
    DEFAULT_INTERICTAL_RATIO,
    DEFAULT_LABEL_MODE,
    DEFAULT_NORMALIZE,
    DEFAULT_STEP_SEC,
    DEFAULT_WINDOW_SEC,
    INPUT_REP_CHOICES,
    LABEL_MODE_CHOICES,
    NORMALIZE_CHOICES,
    PREICTAL_SEC,
    build_dataset,
    load_preprocessed,
    save_preprocessed,
)

DEFAULT_PREPROCESSED_PATH = (
    Path(__file__).resolve().parent / "../../data/processed/eeg_ecg_windows.npz"
)

__all__ = ["DEFAULT_PREPROCESSED_PATH", "load_preprocessed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess SeizeIT2 EEG+ECG into a windowed .npz dataset (seizure prediction)."
    )
    parser.add_argument("--interictal-ratio", type=float, default=DEFAULT_INTERICTAL_RATIO,
                        help="Interictal : pre-ictal windows kept (default 10).")
    parser.add_argument("--preictal-min", type=float, default=PREICTAL_SEC / 60.0,
                        help="Pre-ictal horizon in minutes: windows within this many "
                             f"minutes before onset = positive (default {PREICTAL_SEC / 60:.0f}).")
    parser.add_argument("--normalize", choices=NORMALIZE_CHOICES, default=DEFAULT_NORMALIZE,
                        help="Window normalization: 'per_window' (default) z-scores each "
                             "window; 'per_recording' uses one mean/std per channel over "
                             "the whole recording (keeps cross-window amplitude dynamics).")
    parser.add_argument("--label-mode", choices=LABEL_MODE_CHOICES, default=DEFAULT_LABEL_MODE,
                        help="'prediction' (default): pre-ictal vs interictal (the real task). "
                             "'detection': ictal vs interictal — an easier sanity check that "
                             "the pipeline can learn anything at all. Use a separate --out.")
    parser.add_argument("--input-rep", choices=INPUT_REP_CHOICES, default=DEFAULT_INPUT_REP,
                        help="'raw' (default): raw waveform for the 1-D CNN. 'bandpower_seq': "
                             "log band-power per band in short frames across the window (the "
                             "RF's features kept as a time sequence) — gives the CNN the spectral "
                             "content WITH its temporal evolution. Tiny vs raw (avoids OOM).")
    parser.add_argument("--frame-sec", type=float, default=DEFAULT_FRAME_SEC,
                        help="bandpower_seq frame length in seconds (default 4).")
    parser.add_argument("--frame-step-sec", type=float, default=DEFAULT_FRAME_STEP_SEC,
                        help="bandpower_seq frame hop in seconds (default 1).")
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--step-sec", type=float, default=DEFAULT_STEP_SEC)
    parser.add_argument("--subjects", type=str, nargs="+", default=None,
                        help="Limit to specific subjects, e.g. sub-001 sub-002 (default: all).")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_PREPROCESSED_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset(
        include_ecg=True,
        interictal_ratio=args.interictal_ratio,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        subjects=args.subjects,
        random_state=args.random_state,
        preictal_sec=args.preictal_min * 60.0,
        normalize=args.normalize,
        label_mode=args.label_mode,
        input_rep=args.input_rep,
        frame_sec=args.frame_sec,
        frame_step_sec=args.frame_step_sec,
    )
    save_preprocessed(args.out, result)


if __name__ == "__main__":
    main()
