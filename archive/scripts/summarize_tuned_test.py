"""Summarize val-tuned test metrics from reports and RF CSV."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def parse_report(path: Path) -> dict:
    t = path.read_text(encoding="utf-8")

    def g(pat: str) -> float | None:
        m = re.search(pat, t)
        return float(m.group(1)) if m else None

    m120 = re.search(r"120s.*?\s+([0-9.]+)\s+([0-9.]+)", t)
    m_ms = re.search(r"mean per-subject AUC.*?: ([0-9.]+)", t)
    seed_m = re.search(r"seed=(\d+)", t)
    return {
        "thr": g(r"threshold=([0-9.]+)"),
        "P": g(r"Precision\s+: ([0-9.]+)"),
        "R": g(r"Recall \(sens\.\) : ([0-9.]+)"),
        "F1": g(r"F1\s+: ([0-9.]+)"),
        "auc": g(r"AUC-ROC\s+: ([0-9.]+)"),
        "auc_pr": g(r"AUC-PR\s+: ([0-9.]+)"),
        "auc_sm": float(m120.group(1)) if m120 else None,
        "msauc_sm": float(m120.group(2)) if m120 else None,
        "msauc": float(m_ms.group(1)) if m_ms else None,
        "seed": int(seed_m.group(1)) if seed_m else None,
    }


def agg_reports(paths: list[Path], label: str) -> None:
    rows = [parse_report(p) for p in paths]
    print(f"\n{label}")
    print(f"  thresholds: {[round(r['thr'], 2) for r in rows]}")
    for k in ("auc", "auc_sm", "msauc", "msauc_sm", "P", "R", "F1", "auc_pr"):
        v = [r[k] for r in rows if r[k] is not None]
        print(f"  {k}: mean={np.mean(v):.3f}  std={np.std(v, ddof=1):.3f}")


def agg_rf(path: Path, label: str) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    print(f"\n{label}")
    print(f"  thresholds: {[round(float(r['pred_threshold']), 2) for r in rows]}")
    for k in ("auc_roc", "mean_subj_auc", "mean_subj_auc_smooth",
              "precision", "recall", "f1", "auc_pr"):
        v = [float(r[k]) for r in rows]
        print(f"  {k}: mean={np.mean(v):.3f}  std={np.std(v, ddof=1):.3f}")


def main() -> None:
    agg_reports(sorted((ROOT / "results/sprint_eeg_bp/test/reports").glob("eval_eeg_20260618_11*.txt")),
                "CNN EEG (val-tuned thr)")
    agg_reports(sorted((ROOT / "results/sprint_eeg_ecg_bp/test/reports").glob("eval_eeg_ecg_20260618_11*.txt")),
                "CNN EEG+ECG (val-tuned thr)")
    agg_rf(ROOT / "results/sprint_eeg_bp/test/baseline_rf_metrics.csv", "RF EEG (val-tuned thr)")
    agg_rf(ROOT / "results/sprint_eeg_ecg_bp/test/baseline_rf_metrics.csv", "RF EEG+ECG (val-tuned thr)")


if __name__ == "__main__":
    main()
