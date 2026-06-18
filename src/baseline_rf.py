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

Use:
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


def temporal_smooth(prob, centers, subj, k: int):
    """Per-subject centered moving average of prob over k consecutive windows
    (same post-processing the CNN eval applies, so the comparison is fair)."""
    if k <= 1:
        return prob.astype(float)
    out = np.empty(len(prob), dtype=float)
    for s in np.unique(subj):
        idx = np.nonzero(subj == s)[0]
        order = idx[np.argsort(centers[idx])]
        p = prob[order].astype(float)
        kk = min(k, len(p))
        ker = np.ones(kk)
        out[order] = np.convolve(p, ker, "same") / np.convolve(np.ones_like(p), ker, "same")
    return out


def prf_at_threshold(y_true, prob, threshold: float) -> tuple[float, float, float, int, int, int, int]:
    pred = (prob >= threshold).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return float(pr), float(rc), float(f1), int(tn), int(fp), int(fn), int(tp)


def best_f1_threshold(y_true, prob, lo: float = 0.05, hi: float = 0.55, step: float = 0.01
                      ) -> tuple[float, float, float, float]:
    """Return (threshold, precision, recall, f1) maximising F1 on the given set."""
    best_t, best = 0.5, (0.0, 0.0, 0.0)
    for t in np.arange(lo, hi + 1e-9, step):
        pr, rc, f1, *_ = prf_at_threshold(y_true, prob, float(t))
        if f1 > best[2]:
            best_t, best = float(t), (pr, rc, f1)
    return best_t, best[0], best[1], best[2]


def format_threshold_sweep(y_true, prob, thresholds: tuple[float, ...],
                           current: float = 0.5) -> list[str]:
    lines = ["  Threshold sweep (P / R / F1):"]
    for t in thresholds:
        pr, rc, f1, *_ = prf_at_threshold(y_true, prob, t)
        marker = "  <-- current" if abs(t - current) < 1e-9 else ""
        lines.append(f"    thr {t:.2f}:  P {pr:.3f}   R {rc:.3f}   F1 {f1:.3f}{marker}")
    return lines

METRIC_FIELDS = [
    "timestamp", "model", "feature_set", "eval_split", "auc_roc", "auc_pr",
    "mean_subj_auc", "mean_subj_auc_smooth", "pred_threshold",
    "precision", "recall", "f1",
    "precision_thr050", "recall_thr050", "f1_thr050",
    "tn", "fp", "fn", "tp", "n_test", "n_estimators", "random_state",
]


def append_rf_metrics(csv_path: Path, row: dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        header = csv_path.read_text(encoding="utf-8").splitlines()[0].strip()
        if header != ",".join(METRIC_FIELDS):
            bak = csv_path.with_suffix(".csv.bak")
            csv_path.rename(bak)
            print(f"[Baseline-RF] Metrics header changed -> backed up to {bak.name}")
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="RandomForest band-power baseline (seizure prediction).")
    ap.add_argument("--data", type=Path, required=True, help="Preprocessed .npz (same as the CNN uses).")
    ap.add_argument("--feature-set", default="eeg", help="Label + default results dir (eeg / eeg_ecg).")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="Default: results/seizure_prediction_<feature-set>.")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--eval-split", choices=["val", "test"], default="val",
                    help="Which held-out block to report on (same discipline as the CNN): "
                         "'val' (default) while tuning, 'test' for the final one-time run.")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--pred-threshold", type=float, default=None,
                    help="Decision threshold. Default: F1-optimal on VAL (test split) or "
                         "on TRAIN (val split). RF probabilities sit far below 0.5 — do not "
                         "use 0.5 unless comparing explicitly to the CNN default.")
    args = ap.parse_args()

    data = load_preprocessed(args.data)
    X, y = data["X"], data["y"]
    sfreq, ch_names = data["sfreq"], data["channel_names"]
    input_rep = data.get("input_rep", "raw")

    # Identical within-subject 60/20/20 split as the CNN; RF needs no val for early
    # stopping, so we always train on the train block and report on the chosen held-out
    # block (val while tuning, test for the final run) — same discipline as the CNN.
    train_idx, val_idx, test_idx = subject_aware_split(data, args.train_frac, args.val_frac, None)
    eval_idx = val_idx if args.eval_split == "val" else test_idx

    if input_rep == "bandpower_seq":
        # Same band-power features as the CNN, but time-averaged over frames: the RF
        # sees the aggregated spectrum, the CNN sees the sequence -> fair "does temporal
        # modelling help?" contrast. ch_names are already the band-power feature names.
        print(f"[Baseline-RF] {args.feature_set}: mean band-power over frames for "
              f"{X.shape[0]:,} windows x {X.shape[1]} features ...")
        feats = X.mean(axis=2)
        feat_label = list(ch_names)
    else:
        print(f"[Baseline-RF] {args.feature_set}: band-power features for "
              f"{X.shape[0]:,} windows x {X.shape[1]} ch ...")
        feats = band_power_features(X, sfreq)
        feat_label = feature_names(ch_names)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, class_weight="balanced",
        n_jobs=-1, random_state=args.random_state)
    clf.fit(feats[train_idx], y[train_idx])
    prob = clf.predict_proba(feats[eval_idx])[:, 1]
    prob_val = clf.predict_proba(feats[val_idx])[:, 1]
    prob_train = clf.predict_proba(feats[train_idx])[:, 1]

    yt = y[eval_idx]
    y_val, y_train = y[val_idx], y[train_idx]
    subj_of = np.empty(len(X), dtype=object)
    for (s, e), p in zip(data["file_slices"], data["recording_paths"]):
        subj_of[s:e] = _subject_from_path(p)
    subj_t = subj_of[eval_idx]

    val_thr, _, _, val_thr_f1 = best_f1_threshold(y_val, prob_val)
    train_thr, _, _, train_thr_f1 = best_f1_threshold(y_train, prob_train)
    if args.pred_threshold is not None:
        report_thr = float(args.pred_threshold)
        thr_source = "manual"
    elif args.eval_split == "test":
        report_thr, thr_source = val_thr, "val-tuned (best F1 on VAL)"
    else:
        report_thr, thr_source = train_thr, "train-tuned (best F1 on TRAIN)"

    auc = roc_auc_score(yt, prob)
    auc_pr = average_precision_score(yt, prob)
    pr, rc, f1, tn, fp, fn, tp = prf_at_threshold(yt, prob, report_thr)
    pr05, rc05, f105, tn05, fp05, fn05, tp05 = prf_at_threshold(yt, prob, 0.5)
    msa, msa_sd, nsub = per_subject_auc(yt, prob, subj_t)
    # Same temporal smoothing the CNN eval uses, so CNN vs RF is apples-to-apples.
    centers_e = data["centers"][eval_idx]
    step = float(data.get("step_sec", 1.0)) or 1.0
    msa_sm, sm_sec = msa, 0
    for sec in (30, 60, 120):
        k = max(1, int(round(sec / step)))
        m, _, _ = per_subject_auc(yt, temporal_smooth(prob, centers_e, subj_t, k), subj_t)
        if m > msa_sm:
            msa_sm, sm_sec = m, sec
    top = sorted(zip(feat_label, clf.feature_importances_),
                 key=lambda t: -t[1])[:8]

    sweep_thresholds = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60)

    bar = "=" * 58
    lines = [bar, f"  RandomForest BASELINE   feature_set={args.feature_set}",
             f"  split={args.eval_split.upper()}   n_eval={len(yt):,}   pre-ictal={int(yt.sum()):,}", bar,
             f"  AUC-ROC                : {auc:.3f}   (0.50 = chance)",
             f"  AUC-PR                 : {auc_pr:.3f}   (baseline {yt.mean():.3f})",
             f"  mean per-subject AUC   : {msa:.3f} +/- {msa_sd:.3f}   (n={nsub})",
             f"  mean per-subj AUC (sm) : {msa_sm:.3f}   (best smoothing {sm_sec}s)",
             f"  Precision/Recall/F1    : {pr:.3f} / {rc:.3f} / {f1:.3f}   "
             f"(@ thr {report_thr:.2f}, {thr_source})",
             f"  Confusion [tn fp fn tp]: [{tn} {fp} {fn} {tp}]",
             f"  @ thr 0.50 (CNN default): P {pr05:.3f} / R {rc05:.3f} / F1 {f105:.3f}  "
             f"[{tn05} {fp05} {fn05} {tp05}]",
             "",
             f"  Prob pre-ictal  mean/median: {prob[yt == 1].mean():.3f} / {np.median(prob[yt == 1]):.3f}",
             f"  Prob interictal mean/median: {prob[yt == 0].mean():.3f} / {np.median(prob[yt == 0]):.3f}",
             ""]
    lines += format_threshold_sweep(yt, prob, sweep_thresholds, current=report_thr)
    if args.eval_split == "test" and args.pred_threshold is None:
        lines += [
            "",
            f"  Threshold tuning: best F1 on VAL = {val_thr_f1:.3f} at thr {val_thr:.2f}",
        ]
    lines += ["", "  Top band-power features (RF importance):"]
    lines += [f"    {n:<18} {v:.3f}" for n, v in top]
    lines.append(bar)
    report = "\n".join(lines)
    print(report)

    results_dir = args.results_dir or (
        Path(__file__).resolve().parent.parent / f"results/seizure_prediction_{args.feature_set}")
    out = Path(results_dir) / args.eval_split
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_rf_report.txt").write_text(report + "\n", encoding="utf-8")

    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "model": "randomforest_bandpower", "feature_set": args.feature_set,
           "eval_split": args.eval_split,
           "auc_roc": round(auc, 4), "auc_pr": round(auc_pr, 4),
           "mean_subj_auc": round(msa, 4), "mean_subj_auc_smooth": round(msa_sm, 4),
           "pred_threshold": round(report_thr, 4),
           "precision": round(pr, 4),
           "recall": round(rc, 4), "f1": round(f1, 4),
           "precision_thr050": round(pr05, 4),
           "recall_thr050": round(rc05, 4), "f1_thr050": round(f105, 4),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
           "n_test": int(len(yt)), "n_estimators": args.n_estimators,
           "random_state": args.random_state}
    csv_path = out / "baseline_rf_metrics.csv"
    append_rf_metrics(csv_path, row)
    print(f"[Baseline-RF] Saved -> {out / 'baseline_rf_report.txt'} and {csv_path}")


if __name__ == "__main__":
    main()
