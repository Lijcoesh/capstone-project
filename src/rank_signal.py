# -*- coding: utf-8 -*-
"""
Rank subjects by how much *separable signal* their data contains — WITHOUT a CNN.

For each preprocessed npz we compute the four character features (line length,
Hjorth mobility/activity, crest factor) per window and ask how well pre-ictal can
be told from interictal using only those features:

  - best_fauc : best single-feature ROC-AUC (direction-free)
  - cv_auc    : 5-fold cross-validated ROC-AUC of a logistic regression on all
                four features (a cheap multivariate "signal ceiling")

This is a model-free screen: a subject with a high cv_auc has learnable pre-ictal
signal in the raw signal statistics, so a CNN is likely to do well; a subject near
0.5 has little to learn. Use it to pick which subject to actually train.

Note: the CV is window-level and the windows overlap (50%), so cv_auc is mildly
optimistic in absolute terms — but it is fine for *ranking* subjects relative to
one another (every subject is inflated the same way).

Usage (from src/):
  python rank_signal.py                      # all data/processed/eeg_sub-*.npz
  python rank_signal.py --glob "data/processed/eeg_sub-0*.npz"
"""

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess_common import load_preprocessed, _subject_from_path  # noqa: E402
from horizon_features import window_features, FEATURES  # noqa: E402
from explain_horizons import count_seizures  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def signal_score(x, y):
    feats = window_features(x)
    F = np.column_stack([feats[k] for k, _ in FEATURES])
    best_fauc = max(max(a, 1 - a) for a in
                    (roc_auc_score(y, F[:, i]) for i in range(F.shape[1])))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cv_auc = cross_val_score(clf, F, y, cv=cv, scoring="roc_auc").mean()
    return float(best_fauc), float(cv_auc)


def subject_from_npz(path):
    m = re.search(r"(sub-\d+)", Path(path).name)
    return m.group(1) if m else Path(path).stem


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", type=str, default="data/processed/eeg_sub-*.npz",
                    help="Glob (relative to repo root) for the npz files to rank.")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results/explainability/horizons/signal_ranking.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(REPO_ROOT / args.glob)))
    if not paths:
        print(f"No npz matched {args.glob}"); return

    rows = []
    for p in paths:
        data = load_preprocessed(p)
        x, y = data["X"], data["y"]
        # subject id per window (handles both single- and multi-subject npz files)
        subj_of = np.empty(len(x), dtype=object)
        for (s, e), rp in zip(data["file_slices"], data["recording_paths"]):
            subj_of[s:e] = _subject_from_path(rp)
        for subj in sorted(set(subj_of.tolist())):
            m = subj_of == subj
            xs, ys = x[m], y[m]
            if len(np.unique(ys)) < 2:
                print(f"[Skip] {subj}: one class only"); continue
            best_fauc, cv_auc = signal_score(xs, ys)
            rows.append(dict(subject=subj, n_seizures=count_seizures(subj),
                             n_pre=int((ys == 1).sum()), n_int=int((ys == 0).sum()),
                             best_fauc=round(best_fauc, 4), cv_auc=round(cv_auc, 4)))
            print(f"  {subj}: cv_auc {cv_auc:.3f}  best_fauc {best_fauc:.3f}  "
                  f"(pre {int((ys==1).sum())}, int {int((ys==0).sum())})")

    rows.sort(key=lambda r: r["cv_auc"], reverse=True)
    print(f"\n=== Subjects ranked by model-free signal (cv_auc) ===")
    print(f"{'rank':<5}{'subject':<10}{'cv_auc':>8}{'best_fauc':>11}{'seizures':>10}")
    for i, r in enumerate(rows, 1):
        print(f"{i:<5}{r['subject']:<10}{r['cv_auc']:>8.3f}{r['best_fauc']:>11.3f}"
              f"{r['n_seizures']:>10}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWinner: {rows[0]['subject']} (cv_auc {rows[0]['cv_auc']:.3f}). "
          f"Saved ranking to {args.out}")


if __name__ == "__main__":
    main()
