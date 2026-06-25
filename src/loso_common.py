# -*- coding: utf-8 -*-
"""
Leave-One-Subject-Out (LOSO) evaluation with per-patient calibration.

For each held-out subject P (treated as a *new patient* the global model never saw):
  1. Train a fresh population CNN on all other subjects' train blocks.
  2. Tune a population threshold on the other subjects' val blocks (before calibration).
  3. Tune a patient-specific threshold on P's calibration block (after calibration).
  4. Report ranking (AUC) and classification (F1/P/R) on P's test block only.

This is the clinically honest evaluation: global model + calibration period + held-out
monitoring period on someone who contributed zero windows to training.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from evaluate_common import (
    _block_metrics,
    average_positive_prob,
    best_f1_threshold,
)
from model_common import pick_device, train_model
from preprocess_common import build_subject_index_array, load_preprocessed, loso_fold_indices

# Defaults (no CLI flags required for the standard workflow).
LOSO_CAL_FRAC = 0.2
LOSO_TRAIN_INNER_FRAC = 0.8
LOSO_EPOCHS = 50
LOSO_PATIENCE = 8
LOSO_ENSEMBLE_RUNS = 1          # one model per fold — 55 folds is already heavy
LOSO_BATCH_SIZE = 128
LOSO_LR = 1e-3
LOSO_RANDOM_STATE = 42
LOSO_MIN_RUN = 2
LOSO_MIN_CAL_POSITIVES = 3        # need a few pre-ictal cal windows to tune thr_P

PER_SUBJECT_FIELDS = [
    "feature_set", "subject", "auc",
    "f1_before", "f1_after",
    "precision_before", "precision_after",
    "recall_before", "recall_after",
    "thr_global", "thr_calibrated",
    "n_test", "n_test_preictal", "n_cal", "n_cal_preictal",
    "skipped", "skip_reason",
]


def _train_fold_models(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        device: torch.device,
        *,
        input_rep: str,
        random_state: int,
        no_amp: bool,
) -> list[torch.nn.Module]:
    models: list[torch.nn.Module] = []
    for run in range(max(1, LOSO_ENSEMBLE_RUNS)):
        seed = random_state + run
        models.append(train_model(
            x_train, y_train, device,
            epochs=LOSO_EPOCHS, batch_size=LOSO_BATCH_SIZE,
            lr=LOSO_LR, random_state=seed,
            x_val=x_val if len(x_val) else None,
            y_val=y_val if len(x_val) else None,
            patience=LOSO_PATIENCE, use_amp=not no_amp,
            input_rep=input_rep,
        ))
    return models


def _safe_auc(y: np.ndarray, prob: np.ndarray) -> float:
    return float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")


def _write_per_subject_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_SUBJECT_FIELDS)
        w.writeheader()
        w.writerows(rows)


def add_loso_eval_args(
        parser: argparse.ArgumentParser,
        default_data: Path,
        default_results_dir: Path,
        default_feature_set: str,
) -> None:
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir)
    parser.add_argument("--feature-set", type=str, default=default_feature_set)
    parser.add_argument("--random-state", type=int, default=LOSO_RANDOM_STATE)
    parser.add_argument("--no-gpu", action="store_true")


def run_loso_evaluation(args: argparse.Namespace) -> None:
    device = pick_device(args.no_gpu)
    data = load_preprocessed(args.data)
    x, y = data["X"], data["y"]
    input_rep = data.get("input_rep", "raw")
    subj_of = build_subject_index_array(data)
    subjects = sorted(np.unique(subj_of))

    out_dir = Path(args.results_dir) / "loso"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "loso_per_subject.csv"
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[LOSO] {len(subjects)} folds  |  cal_frac={LOSO_CAL_FRAC}  "
          f"|  input_rep={input_rep}  |  ensemble={LOSO_ENSEMBLE_RUNS}")
    print(f"[LOSO] Results -> {csv_path}\n")

    rows: list[dict] = []
    for fold, held_out in enumerate(subjects, start=1):
        print(f"\n{'=' * 60}")
        print(f"  LOSO fold {fold}/{len(subjects)}  held-out = {held_out}")
        print(f"{'=' * 60}")

        train_idx, val_idx, cal_idx, test_idx = loso_fold_indices(
            data, held_out, subj_of,
            cal_frac=LOSO_CAL_FRAC,
            train_inner_frac=LOSO_TRAIN_INNER_FRAC,
        )

        row: dict = {
            "feature_set": args.feature_set,
            "subject": held_out,
            "auc": "",
            "f1_before": "", "f1_after": "",
            "precision_before": "", "precision_after": "",
            "recall_before": "", "recall_after": "",
            "thr_global": "", "thr_calibrated": "",
            "n_test": len(test_idx),
            "n_test_preictal": int(y[test_idx].sum()) if len(test_idx) else 0,
            "n_cal": len(cal_idx),
            "n_cal_preictal": int(y[cal_idx].sum()) if len(cal_idx) else 0,
            "skipped": 0,
            "skip_reason": "",
        }

        if len(test_idx) == 0:
            row["skipped"] = 1
            row["skip_reason"] = "no_test_windows"
            print("  [skip] no test windows")
            rows.append(row)
            _write_per_subject_csv(csv_path, rows)
            continue
        if row["n_test_preictal"] == 0:
            row["skipped"] = 1
            row["skip_reason"] = "no_test_preictal"
            print("  [skip] no pre-ictal windows in test block")
            rows.append(row)
            _write_per_subject_csv(csv_path, rows)
            continue
        if len(train_idx) == 0:
            row["skipped"] = 1
            row["skip_reason"] = "no_train_windows"
            print("  [skip] empty training pool")
            rows.append(row)
            _write_per_subject_csv(csv_path, rows)
            continue

        print(f"  train={len(train_idx):,}  val={len(val_idx):,}  "
              f"cal={len(cal_idx):,}  test={len(test_idx):,}  "
              f"(test pre-ictal={row['n_test_preictal']})")

        models = _train_fold_models(
            x[train_idx], y[train_idx],
            x[val_idx], y[val_idx],
            device,
            input_rep=input_rep,
            random_state=args.random_state,
            no_amp=False,
        )

        prob_test = average_positive_prob(models, x[test_idx], device)
        y_test = y[test_idx]
        auc = _safe_auc(y_test, prob_test)

        # Population threshold (other subjects' val blocks) — "before calibration".
        if len(val_idx) and len(np.unique(y[val_idx])) > 1:
            prob_val = average_positive_prob(models, x[val_idx], device)
            thr_global, _ = best_f1_threshold(y[val_idx], prob_val, LOSO_MIN_RUN)
        else:
            thr_global = 0.5

        # Patient-specific threshold on P's calibration block — "after calibration".
        n_cal_pos = row["n_cal_preictal"]
        if len(cal_idx) and n_cal_pos >= LOSO_MIN_CAL_POSITIVES:
            prob_cal = average_positive_prob(models, x[cal_idx], device)
            thr_cal, _ = best_f1_threshold(y[cal_idx], prob_cal, LOSO_MIN_RUN)
        else:
            thr_cal = thr_global
            if n_cal_pos < LOSO_MIN_CAL_POSITIVES:
                print(f"  [cal] only {n_cal_pos} pre-ictal cal windows "
                      f"(<{LOSO_MIN_CAL_POSITIVES}) — using population threshold")

        m_before = _block_metrics(prob_test, y_test, thr_global, LOSO_MIN_RUN)
        m_after = _block_metrics(prob_test, y_test, thr_cal, LOSO_MIN_RUN)

        row.update({
            "auc": round(auc, 4),
            "f1_before": round(m_before["f1"], 4),
            "f1_after": round(m_after["f1"], 4),
            "precision_before": round(m_before["precision"], 4),
            "precision_after": round(m_after["precision"], 4),
            "recall_before": round(m_before["recall"], 4),
            "recall_after": round(m_after["recall"], 4),
            "thr_global": round(thr_global, 4),
            "thr_calibrated": round(thr_cal, 4),
        })
        rows.append(row)
        _write_per_subject_csv(csv_path, rows)

        print(f"  AUC={auc:.3f}  F1 {m_before['f1']:.3f} -> {m_after['f1']:.3f}  "
              f"(thr {thr_global:.2f} -> {thr_cal:.2f})")

    # ── aggregate summary ──
    valid = [r for r in rows if not r["skipped"]]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"loso_{ts}.txt"

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in valid if r[key] != ""]
        return float(np.mean(vals)) if vals else float("nan")

    lines = [
        "=" * 60,
        f"  LOSO + per-patient calibration   feature_set={args.feature_set}",
        f"  folds={len(subjects)}  evaluated={len(valid)}  skipped={len(rows) - len(valid)}",
        "=" * 60,
        f"  mean AUC (test)           : {_mean('auc'):.3f}",
        f"  mean F1 before cal        : {_mean('f1_before'):.3f}",
        f"  mean F1 after cal         : {_mean('f1_after'):.3f}",
        f"  mean precision before/after: {_mean('precision_before'):.3f} / {_mean('precision_after'):.3f}",
        f"  mean recall before/after  : {_mean('recall_before'):.3f} / {_mean('recall_after'):.3f}",
        "",
        "  Per-subject rows -> loso_per_subject.csv",
    ]
    report_text = "\n".join(lines)
    report_path.write_text(report_text + "\n", encoding="utf-8")

    summary_csv = out_dir / "loso_metrics.csv"
    summary_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "feature_set": args.feature_set,
        "n_folds": len(subjects),
        "n_evaluated": len(valid),
        "mean_auc": round(_mean("auc"), 4),
        "mean_f1_before": round(_mean("f1_before"), 4),
        "mean_f1_after": round(_mean("f1_after"), 4),
        "mean_precision_before": round(_mean("precision_before"), 4),
        "mean_precision_after": round(_mean("precision_after"), 4),
        "mean_recall_before": round(_mean("recall_before"), 4),
        "mean_recall_after": round(_mean("recall_after"), 4),
        "cal_frac": LOSO_CAL_FRAC,
        "loso_epochs": LOSO_EPOCHS,
        "loso_ensemble_runs": LOSO_ENSEMBLE_RUNS,
    }
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        w.writeheader()
        w.writerow(summary_row)

    print(f"\n{report_text}")
    print(f"\n[LOSO] Summary -> {summary_csv}")
    print(f"[LOSO] Report  -> {report_path}")
