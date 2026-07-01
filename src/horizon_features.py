# -*- coding: utf-8 -*-
"""
Quantify the pre-ictal vs interictal "signal character" per subject, and test
whether its separability explains the model's per-subject AUC.

For each subject we load its own npz + model, reconstruct the held-out test set,
and compute four interpretable per-window features (averaged over the 2 EEG
channels), on ALL test windows (no cherry-picking):

  - line_length    : mean |x[t+1]-x[t]|            -> noisiness / HF content
  - hjorth_mobility: sqrt(var(dx)/var(x))          -> dominant-frequency proxy
  - hjorth_activity: var(x)                         -> power
  - crest_factor   : peak-to-peak / RMS            -> spikiness (flat+spikes vs busy)

Then per feature we measure how well it ALONE separates pre-ictal from interictal
within that subject (univariate ROC-AUC, direction-free), and check whether that
"signal separability" tracks the model's AUC across subjects (Spearman).

Outputs (results/explainability/horizons/ by default):
  - feature_boxplots.png      : pre vs int distribution per feature, per subject
  - feature_separability.png  : best single-feature AUC vs model AUC (scatter)
  - feature_summary.csv       : per subject/feature means, Cohen's d, feature-AUC

Usage (from src/):
  python horizon_features.py
  python horizon_features.py --subjects sub-034,sub-047
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess_common import load_preprocessed, subject_aware_split, _subject_from_path  # noqa: E402
from model_common import pick_device  # noqa: E402
from evaluate_common import load_models, average_positive_prob  # noqa: E402
from explain_horizons import count_seizures  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_TMPL = "data/processed/eeg_{subj}.npz"
DEFAULT_MODEL_TMPL = "models/per_subject/{subj}.pt"
DEFAULT_OUT = REPO_ROOT / "results/explainability/horizons"

PRE_COLOR = "#c0392b"
INT_COLOR = "#2c6fbb"

FEATURES = [
    ("line_length", "Line length  (noisiness / HF content)"),
    ("hjorth_mobility", "Hjorth mobility  (dominant-freq proxy)"),
    ("hjorth_activity", "Hjorth activity  (power, var)"),
    ("crest_factor", "Crest factor  (ptp / RMS, spikiness)"),
]


def window_features(x):
    """x: (n, n_ch, T) -> dict of (n,) feature arrays, averaged over channels."""
    dx = np.diff(x, axis=2)
    var_x = x.var(axis=2)                      # (n, ch)
    var_dx = dx.var(axis=2)
    line_length = np.abs(dx).mean(axis=2)
    mobility = np.sqrt(np.divide(var_dx, var_x, out=np.zeros_like(var_dx),
                                 where=var_x > 0))
    rms = np.sqrt((x ** 2).mean(axis=2))
    ptp = np.ptp(x, axis=2)
    crest = np.divide(ptp, rms, out=np.zeros_like(ptp), where=rms > 0)
    return {
        "line_length": line_length.mean(axis=1),
        "hjorth_mobility": mobility.mean(axis=1),
        "hjorth_activity": var_x.mean(axis=1),
        "crest_factor": crest.mean(axis=1),
    }


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def feature_auc(values, y):
    """Direction-free univariate separability of a feature for labels y (1=pre)."""
    if len(np.unique(y)) < 2:
        return float("nan")
    a = roc_auc_score(y, values)
    return float(max(a, 1 - a))


def process_subject(subj, data_path, model_path, device):
    data = load_preprocessed(str(data_path))
    x, y = data["X"], data["y"]
    models, meta = load_models(model_path, device)
    train_frac = float(meta.get("train_frac", 0.6))
    val_frac = float(meta.get("val_frac", 0.2))
    _, _, test_idx = subject_aware_split(data, train_frac, val_frac, meta.get("train_subjects"))

    subj_of = np.empty(len(x), dtype=object)
    for (s, e), p in zip(data["file_slices"], data["recording_paths"]):
        subj_of[s:e] = _subject_from_path(p)
    mask = np.zeros(len(x), dtype=bool)
    mask[test_idx] = True
    mask &= (subj_of == subj)
    pos = np.where(mask)[0]

    y_test = y[pos]
    probs = average_positive_prob(models, x[pos], device)
    model_auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")

    feats = window_features(x[pos])
    pre, intc = y_test == 1, y_test == 0
    out = dict(subject=subj, model_auc=model_auc, n_seizures=count_seizures(subj),
               n_pre=int(pre.sum()), n_int=int(intc.sum()), feats={})
    for key, _ in FEATURES:
        v = feats[key]
        out["feats"][key] = dict(
            pre=v[pre], int=v[intc],
            d=cohens_d(v[pre], v[intc]),
            fauc=feature_auc(v, y_test),
        )
    return out


# ── figures ───────────────────────────────────────────────────────────────────

def boxplot_figure(results, save_path, show):
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(2.0 * len(results) + 3,
                                                        3.0 * len(FEATURES)))
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, FEATURES):
        data, ticks, ticklabels, colors = [], [], [], []
        p = 1.0
        for r in results:
            f = r["feats"][key]
            for cls, color in [("pre", PRE_COLOR), ("int", INT_COLOR)]:
                vals = f[cls]
                data.append(vals if len(vals) else [np.nan])
                colors.append(color)
                ticks.append(p)
                ticklabels.append(f"{r['subject'].replace('sub-','')}\n{cls}")
                p += 1
            p += 0.7
        bp = ax.boxplot(data, positions=ticks, widths=0.6, showfliers=False,
                        patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=7)
        # annotate per-subject feature-AUC above each subject pair
        p = 1.0
        ymax = ax.get_ylim()[1]
        for r in results:
            ax.text(p + 0.5, ymax, f"fAUC {r['feats'][key]['fauc']:.2f}",
                    ha="center", va="top", fontsize=7, color="#444")
            p += 2.7
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Pre-ictal (red) vs interictal (blue) signal character — subjects "
                 "sorted by model AUC\n(fAUC = how well this single feature alone "
                 "separates pre vs int within the subject)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[Feature] boxplots -> {save_path}")
    plt.show() if show else plt.close(fig)


def separability_figure(results, save_path, show):
    model_aucs = np.array([r["model_auc"] for r in results])
    # best single-feature separability per subject
    best_fauc = np.array([max(r["feats"][k]["fauc"] for k, _ in FEATURES) for r in results])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(best_fauc, model_aucs, s=70, color="#6c3483", zorder=3)
    for r, bf, ma in zip(results, best_fauc, model_aucs):
        ax.annotate(f"{r['subject'].replace('sub-','')}  ({r['n_seizures']} sz)",
                    (bf, ma), textcoords="offset points", xytext=(7, 3), fontsize=9)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Best single-feature separability  (univariate fAUC)", fontsize=10)
    ax.set_ylabel("Model AUC (per-subject test)", fontsize=10)
    rho = spearmanr(best_fauc, model_aucs).correlation if len(results) > 2 else float("nan")
    ax.set_title(f"Signal-character separability vs model AUC\n"
                 f"Spearman ρ = {rho:.2f}  (n={len(results)})", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[Feature] separability -> {save_path}")
    plt.show() if show else plt.close(fig)
    return rho


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", type=str,
                    default="sub-034,sub-035,sub-039,sub-047,sub-060")
    ap.add_argument("--data-tmpl", type=str, default=DEFAULT_DATA_TMPL)
    ap.add_argument("--model-tmpl", type=str, default=DEFAULT_MODEL_TMPL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    device = pick_device(args.no_gpu)
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    for subj in subjects:
        data_path = REPO_ROOT / args.data_tmpl.format(subj=subj)
        model_path = REPO_ROOT / args.model_tmpl.format(subj=subj)
        if not data_path.exists() or not model_path.exists():
            print(f"[Skip] {subj}: missing npz or model")
            continue
        print(f"\n=== {subj} ===")
        results.append(process_subject(subj, data_path, model_path, device))

    if not results:
        print("Nothing to do."); return
    results.sort(key=lambda r: -(r["model_auc"] if not np.isnan(r["model_auc"]) else -1))

    boxplot_figure(results, args.out / "feature_boxplots.png", args.show)
    rho = separability_figure(results, args.out / "feature_separability.png", args.show)

    # summary table + csv
    print(f"\n{'subject':<9}{'mAUC':>6}{'sz':>4}   " +
          "  ".join(f"{k.split('_')[0][:5]:>5}:d/fAUC" for k, _ in FEATURES))
    rows = []
    for r in results:
        cells = []
        row = {"subject": r["subject"], "model_auc": round(r["model_auc"], 4),
               "n_seizures": r["n_seizures"]}
        for key, _ in FEATURES:
            f = r["feats"][key]
            cells.append(f"{f['d']:+5.2f}/{f['fauc']:.2f}")
            row[f"{key}_pre_mean"] = round(float(np.mean(f["pre"])), 5)
            row[f"{key}_int_mean"] = round(float(np.mean(f["int"])), 5)
            row[f"{key}_cohens_d"] = round(f["d"], 4)
            row[f"{key}_feat_auc"] = round(f["fauc"], 4)
        print(f"{r['subject']:<9}{r['model_auc']:>6.3f}{r['n_seizures']:>4}   " +
              "  ".join(f"{c:>11}" for c in cells))
        rows.append(row)
    with open(args.out / "feature_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\nSpearman(best single-feature separability, model AUC) = {rho:.2f}")
    print(f"Done. Figures + feature_summary.csv in {args.out}")


if __name__ == "__main__":
    main()
