# -*- coding: utf-8 -*-
"""Day 1 paper artifacts: Table 1 (val/test, CNN+RF) and EEG vs EEG+ECG figure."""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SEEDS = [42, 43, 44, 45, 46]
DATA_EEG = REPO / "data/processed/seizeit2_eeg_bp_w50s5.npz"
DATA_ECG = REPO / "data/processed/seizeit2_eeg_ecg_bp_w50s5.npz"
OUT = REPO / "results/paper"


def run_test_evals() -> None:
    """Run held-out TEST eval once per seed (CNN + RF, EEG + EEG+ECG)."""
    for s in SEEDS:
        subprocess.run([
            sys.executable, str(REPO / "src/seizure_prediction_eeg/evaluate_eeg.py"),
            "--data", str(DATA_EEG),
            "--model", str(REPO / f"models/sprint/cnn_eeg_bp_s{s}.pt"),
            "--results-dir", str(REPO / "results/sprint_eeg_bp"),
            "--feature-set", "eeg", "--eval-split", "test",
        ], check=True, cwd=REPO)
        subprocess.run([
            sys.executable, str(REPO / "src/seizure_prediction_eeg_ecg/evaluate_eeg_ecg.py"),
            "--data", str(DATA_ECG),
            "--model", str(REPO / f"models/sprint/cnn_eeg_ecg_bp_s{s}.pt"),
            "--results-dir", str(REPO / "results/sprint_eeg_ecg_bp"),
            "--feature-set", "eeg_ecg", "--eval-split", "test",
        ], check=True, cwd=REPO)
        subprocess.run([
            sys.executable, str(REPO / "src/baseline_rf.py"),
            "--data", str(DATA_EEG), "--feature-set", "eeg",
            "--eval-split", "test",
            "--results-dir", str(REPO / "results/sprint_eeg_bp"),
            "--random-state", str(s),
        ], check=True, cwd=REPO)
        subprocess.run([
            sys.executable, str(REPO / "src/baseline_rf.py"),
            "--data", str(DATA_ECG), "--feature-set", "eeg_ecg",
            "--eval-split", "test",
            "--results-dir", str(REPO / "results/sprint_eeg_ecg_bp"),
            "--random-state", str(s),
        ], check=True, cwd=REPO)
        print(f"[test] seed {s} done")
        if s == 42:
            OUT.mkdir(parents=True, exist_ok=True)
            for arm, sub in [("eeg", "sprint_eeg_bp"), ("ecg", "sprint_eeg_ecg_bp")]:
                src = REPO / f"results/{sub}/test/per_subject.csv"
                if src.exists():
                    shutil.copy(src, OUT / f"per_subject_{arm}_test_s42.csv")


def load_smooth_auc(csv_path: Path, is_rf: bool) -> list[float]:
    if not csv_path.exists():
        return []
    key = "mean_subj_auc_smooth"
    vals: list[float] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get(key) or (row.get("mean_subj_auc") if is_rf else "")
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return vals


def mean_std(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    a = np.array(vals, dtype=float)
    return f"{a.mean():.3f} +/- {a.std():.3f}"


def build_table1() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arms = [
        ("EEG", REPO / "results/sprint_eeg_bp"),
        ("EEG+ECG", REPO / "results/sprint_eeg_ecg_bp"),
    ]
    lines = [
        "Table 1. Mean per-subject AUC (temporal smoothing), 5 seeds, SeizeIT2 n=60, w50s5.",
        "",
        "| Feature set | Split | CNN | RF |",
        "|-------------|-------|-----|-----|",
    ]
    for name, base in arms:
        for split in ("val", "test"):
            cnn = load_smooth_auc(base / split / "metrics.csv", is_rf=False)
            rf = load_smooth_auc(base / split / "baseline_rf_metrics.csv", is_rf=True)
            lines.append(f"| {name} | {split} | {mean_std(cnn)} | {mean_std(rf)} |")

    text = "\n".join(lines) + "\n"
    (OUT / "table1.md").write_text(text, encoding="utf-8")
    print(text)

    # CSV for easy import
    with open(OUT / "table1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature_set", "split", "model", "mean", "std", "n_seeds"])
        for name, base in arms:
            for split in ("val", "test"):
                for model, is_rf in (("cnn", False), ("rf", True)):
                    fname = "baseline_rf_metrics.csv" if is_rf else "metrics.csv"
                    vals = load_smooth_auc(base / split / fname, is_rf=is_rf)
                    if vals:
                        a = np.array(vals)
                        w.writerow([name, split, model, round(a.mean(), 4),
                                    round(a.std(), 4), len(vals)])


def plot_eeg_vs_ecg(split: str = "test") -> None:
    """Paired per-subject AUC: EEG vs EEG+ECG (CNN)."""
    if split == "test":
        eeg_path = OUT / "per_subject_eeg_test_s42.csv"
        ecg_path = OUT / "per_subject_ecg_test_s42.csv"
        if not eeg_path.exists():
            eeg_path = REPO / "results/sprint_eeg_bp/test/per_subject.csv"
            ecg_path = REPO / "results/sprint_eeg_ecg_bp/test/per_subject.csv"
    else:
        eeg_path = REPO / f"results/sprint_eeg_bp/{split}/per_subject.csv"
        ecg_path = REPO / f"results/sprint_eeg_ecg_bp/{split}/per_subject.csv"

    if not eeg_path.exists() or not ecg_path.exists():
        print(f"[fig] Missing per_subject for {split}")
        return

    def load_psv(p: Path) -> dict[str, float]:
        out: dict[str, float] = {}
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[row["subject"]] = float(row["auc"])
        return out

    eeg = load_psv(eeg_path)
    ecg = load_psv(ecg_path)
    subjects = sorted(set(eeg) & set(ecg))
    auc_eeg = np.array([eeg[s] for s in subjects])
    auc_ecg = np.array([ecg[s] for s in subjects])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Boxplot
    ax = axes[0]
    ax.boxplot([auc_eeg, auc_ecg], tick_labels=["EEG", "EEG+ECG"])
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Per-subject AUC")
    ax.set_title(f"CNN BandPower — {split} set")
    ax.set_ylim(0, 1)

    # Paired lines per subject
    ax = axes[1]
    x = [0, 1]
    for ae, ac in zip(auc_eeg, auc_ecg):
        ax.plot(x, [ae, ac], "o-", alpha=0.35, color="steelblue", markersize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["EEG", "EEG+ECG"])
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Per-subject AUC")
    ax.set_title("Paired subjects")

    fig.suptitle(
        f"EEG-only vs EEG+ECG (1D CNN)  |  {split.upper()}  |  n={len(subjects)} subjects",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"fig_eeg_vs_ecg_cnn_{split}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] Saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-test", action="store_true",
                    help="Run all TEST evaluations (CNN+RF, 5 seeds)")
    ap.add_argument("--table", action="store_true", help="Build Table 1")
    ap.add_argument("--fig", action="store_true", help="Plot EEG vs EEG+ECG")
    args = ap.parse_args()
    if not any([args.run_test, args.table, args.fig]):
        args.run_test = args.table = args.fig = True
    if args.run_test:
        run_test_evals()
    if args.table:
        build_table1()
    if args.fig:
        plot_eeg_vs_ecg("test")
        plot_eeg_vs_ecg("val")


if __name__ == "__main__":
    main()
