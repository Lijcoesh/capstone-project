# -*- coding: utf-8 -*-
"""
RandomForest baseline for seizure prediction (checklist item: "baseline established").

The project proposal specified a RandomForest; the main model is a 1D CNN. This script
provides that RF as a simple, interpretable baseline so the CNN's value can be judged
against a non-deep reference. It uses the SAME preprocessed windows and the SAME
within-subject 60/20/20 split, trains on the train block, and reports on the held-out
TEST block — so the numbers are directly comparable to evaluate_*.py.

Features: log band-power (delta/theta/alpha/beta/gamma) per channel — the standard
hand-crafted EEG feature set a tree model can use (a RF on a raw 512-sample waveform is
meaningless). For EEG+ECG the extra ECG channel contributes the same five band-powers,
so the EEG-vs-EEG+ECG contrast is preserved for the baseline too.

Example:
  python src/baseline_rf.py --data data/processed/eeg_windows.npz --feature-set eeg
  python src/baseline_rf.py --data data/processed/eeg_ecg_windows.npz --feature-set eeg_ecg
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_common import (  # noqa: E402
    load_preprocessed,
    subject_aware_split,
    _subject_from_path,
)

# Bands stay below the 50 Hz mains notch; gamma capped at 45 Hz to avoid the notched bin.
BANDS = [("delta", 0.5, 4), ("theta", 4, 8), ("alpha", 8, 13),
         ("beta", 13, 30), ("gamma", 30, 45)]


def band_power_features(X: np.ndarray, sfreq: float, chunk: int = 20000) -> np.ndarray:
    """(N, ch, T) raw windows -> (N, ch*len(BANDS)) log band-powers (band-major)."""
    n, ch, T = X.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / sfreq)
    masks = [(freqs >= lo) & (freqs < hi) for _, lo, hi in BANDS]
    out = np.empty((n, ch * len(BANDS)), dtype=np.float32)
    for i in range(0, n, chunk):
        power = np.abs(np.fft.rfft(X[i:i + chunk], axis=-1)) ** 2     # (b, ch, F)
        feats = np.concatenate([power[:, :, m].sum(axis=-1) for m in masks], axis=1)
        out[i:i + chunk] = np.log1p(feats)
    return out


def feature_names(ch_names: list[str]) -> list[str]:
    return [f"{c}_{b}" for b, _, _ in BANDS for c in ch_names]       # matches concat order


def per_subject_auc(y, prob, subj) -> tuple[float, float, int]:
    aucs = [roc_auc_score(y[m], prob[m])
            for s in np.unique(subj)
            for m in [subj == s] if len(np.unique(y[m])) > 1]
    if not aucs:
        return float("nan"), float("nan"), 0
    return float(np.mean(aucs)), float(np.std(aucs)), len(aucs)


def main() -> None:
    ap = argparse.ArgumentParser(description="RandomForest band-power baseline (seizure prediction).")
    ap.add_argument("--data", type=Path, required=True, help="Preprocessed .npz (same as the CNN uses).")
    ap.add_argument("--feature-set", default="eeg", help="Label + default results dir (eeg / eeg_ecg).")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="Default: results/seizure_prediction_<feature-set>.")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    data = load_preprocessed(args.data)
    X, y = data["X"], data["y"]
    sfreq, ch_names = data["sfreq"], data["channel_names"]

    # Identical within-subject 60/20/20 split as the CNN; RF needs no val (no early stop),
    # so we train on the train block and report on the held-out test block.
    train_idx, _val_idx, test_idx = subject_aware_split(data, args.train_frac, args.val_frac, None)

    print(f"[Baseline-RF] {args.feature_set}: band-power features for "
          f"{X.shape[0]:,} windows x {X.shape[1]} ch ...")
    feats = band_power_features(X, sfreq)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, class_weight="balanced",
        n_jobs=-1, random_state=args.random_state)
    clf.fit(feats[train_idx], y[train_idx])
    prob = clf.predict_proba(feats[test_idx])[:, 1]

    yt = y[test_idx]
    pred = (prob >= 0.5).astype(int)
    subj_of = np.empty(len(X), dtype=object)
    for (s, e), p in zip(data["file_slices"], data["recording_paths"]):
        subj_of[s:e] = _subject_from_path(p)
    subj_t = subj_of[test_idx]

    auc = roc_auc_score(yt, prob)
    auc_pr = average_precision_score(yt, prob)
    pr, rc, f1, _ = precision_recall_fscore_support(yt, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    msa, msa_sd, nsub = per_subject_auc(yt, prob, subj_t)
    top = sorted(zip(feature_names(ch_names), clf.feature_importances_),
                 key=lambda t: -t[1])[:8]

    bar = "=" * 58
    lines = [bar, f"  RandomForest BASELINE   feature_set={args.feature_set}",
             f"  split=TEST   n_test={len(yt):,}   pre-ictal={int(yt.sum()):,}", bar,
             f"  AUC-ROC                : {auc:.3f}   (0.50 = chance)",
             f"  AUC-PR                 : {auc_pr:.3f}   (baseline {yt.mean():.3f})",
             f"  mean per-subject AUC   : {msa:.3f} +/- {msa_sd:.3f}   (n={nsub})",
             f"  Precision/Recall/F1    : {pr:.3f} / {rc:.3f} / {f1:.3f}",
             f"  Confusion [tn fp fn tp]: [{tn} {fp} {fn} {tp}]", "",
             "  Top band-power features (RF importance):"]
    lines += [f"    {n:<18} {v:.3f}" for n, v in top]
    lines.append(bar)
    report = "\n".join(lines)
    print(report)

    results_dir = args.results_dir or (
        Path(__file__).resolve().parent.parent / f"results/seizure_prediction_{args.feature_set}")
    out = Path(results_dir) / "test"
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_rf_report.txt").write_text(report + "\n", encoding="utf-8")

    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "model": "randomforest_bandpower", "feature_set": args.feature_set,
           "auc_roc": round(auc, 4), "auc_pr": round(auc_pr, 4),
           "mean_subj_auc": round(msa, 4), "precision": round(pr, 4),
           "recall": round(rc, 4), "f1": round(f1, 4),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
           "n_test": int(len(yt)), "n_estimators": args.n_estimators,
           "random_state": args.random_state}
    csv_path = out / "baseline_rf_metrics.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[Baseline-RF] Saved -> {out / 'baseline_rf_report.txt'} and {csv_path}")


if __name__ == "__main__":
    main()
