# -*- coding: utf-8 -*-
"""
Automated hyperparameter search for the EEG seizure-prediction pipeline.

Runs preprocess -> train -> evaluate in a loop, varying one parameter at a
time (coordinate descent). Stops when validation AUC-ROC reaches --target or
after --max-rounds with no improvement.

The seven parameters from todo.txt are searched:
  window size, step size, sampling rate, notch filter, batch size, dropout,
  interictal ratio.

Usage (from repo root):
  python scripts/tune_eeg.py
  python scripts/tune_eeg.py --quick
  python scripts/tune_eeg.py --resume
  python scripts/tune_eeg.py --param window_sec --values 1.5 2.0 3.0
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
PREPROCESS = SRC / "seizure_prediction_eeg" / "preprocess_eeg.py"
TRAIN = SRC / "seizure_prediction_eeg" / "train_model_eeg.py"
EVALUATE = SRC / "seizure_prediction_eeg" / "evaluate_eeg.py"

TUNING_DIR = REPO_ROOT / "results" / "tuning" / "eeg"
TRIALS_CSV = TUNING_DIR / "trials.csv"
STATE_JSON = TUNING_DIR / "state.json"
BEST_JSON = TUNING_DIR / "best_config.json"

DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "eeg_windows.npz"
DEFAULT_MODEL = TUNING_DIR / "current_model.pt"

# Search order: preprocessing params first, then training params.
PARAM_ORDER = (
    "window_sec",
    "step_sec",
    "target_sfreq",
    "notch_hz",
    "interictal_ratio",
    "batch_size",
    "dropout",
)

DEFAULT_CONFIG: dict[str, float | int] = {
    "window_sec": 2.0,
    "step_sec": 1.0,
    "target_sfreq": 0.0,      # 0 = native (256 Hz)
    "notch_hz": 50.0,
    "interictal_ratio": 5.0,
    "batch_size": 128,
    "dropout": 0.25,
}

# Full grids used for coordinate descent (sorted by distance to current value).
CANDIDATE_GRIDS: dict[str, list[float | int]] = {
    "window_sec": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "step_sec": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "target_sfreq": [0, 128, 256],          # 0 = keep native sampling rate
    "notch_hz": [0, 50, 60],                # 0 = disable notch
    "interictal_ratio": [2, 3, 5, 7, 10, 15],
    "batch_size": [32, 64, 128, 256],
    "dropout": [0.0, 0.1, 0.25, 0.4, 0.5],
}

PREPROCESS_PARAMS = frozenset({
    "window_sec", "step_sec", "target_sfreq", "notch_hz", "interictal_ratio",
})


@dataclass
class TrialResult:
    trial_id: int
    config: dict[str, float | int]
    auc_roc: float
    f1: float
    mean_subj_auc: float
    duration_sec: float
    status: str
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def _config_key(config: dict[str, float | int]) -> str:
    return json.dumps(config, sort_keys=True)


def _neighbors(param: str, current: float | int, config: dict[str, float | int]) -> list[float | int]:
    """Candidate values for *param*, nearest to *current* first."""
    grid = list(CANDIDATE_GRIDS[param])
    if param == "step_sec":
        window = float(config["window_sec"])
        grid = [v for v in grid if float(v) <= window]
        if not grid:
            grid = [min(float(current), window)]
    ordered = sorted(grid, key=lambda v: (abs(float(v) - float(current)), float(v)))
    # Always try current value first (cheap skip if already best).
    if current in ordered:
        ordered.remove(current)
    return [current] + ordered


def _needs_preprocess(prev: dict[str, float | int] | None, cur: dict[str, float | int]) -> bool:
    if prev is None:
        return True
    return any(prev[k] != cur[k] for k in PREPROCESS_PARAMS)


def _preprocess_cmd(config: dict[str, float | int], data_path: Path) -> list[str]:
    cmd = [
        sys.executable, str(PREPROCESS),
        "--window-sec", str(config["window_sec"]),
        "--step-sec", str(config["step_sec"]),
        "--interictal-ratio", str(config["interictal_ratio"]),
        "--notch-hz", str(config["notch_hz"]),
        "--out", str(data_path),
    ]
    sfreq = float(config["target_sfreq"])
    if sfreq > 0:
        cmd.extend(["--target-sfreq", str(sfreq)])
    return cmd


def _train_cmd(
        config: dict[str, float | int],
        data_path: Path,
        model_path: Path,
        epochs: int,
        patience: int,
) -> list[str]:
    return [
        sys.executable, str(TRAIN),
        "--data", str(data_path),
        "--save-model", str(model_path),
        "--batch-size", str(int(config["batch_size"])),
        "--dropout", str(config["dropout"]),
        "--epochs", str(epochs),
        "--patience", str(patience),
    ]


def _evaluate_cmd(
        data_path: Path,
        model_path: Path,
        results_dir: Path,
) -> list[str]:
    return [
        sys.executable, str(EVALUATE),
        "--data", str(data_path),
        "--model", str(model_path),
        "--results-dir", str(results_dir),
        "--eval-split", "val",
    ]


def _run_cmd(cmd: list[str], label: str) -> None:
    print(f"\n{'─' * 60}\n[{label}]\n  {' '.join(cmd)}\n{'─' * 60}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _read_metrics(results_dir: Path) -> dict[str, float]:
    metrics_path = results_dir / "val" / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {metrics_path}")
    row = rows[-1]
    return {
        "auc_roc": float(row["auc_roc"]),
        "f1": float(row["f1"]),
        "mean_subj_auc": float(row["mean_subj_auc"]) if row.get("mean_subj_auc") else float("nan"),
    }


def _append_trial_csv(result: TrialResult) -> None:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "trial_id": result.trial_id,
        "timestamp": result.timestamp,
        "status": result.status,
        "auc_roc": result.auc_roc,
        "f1": result.f1,
        "mean_subj_auc": result.mean_subj_auc,
        "duration_sec": round(result.duration_sec, 1),
        "error": result.error,
        **{k: result.config[k] for k in PARAM_ORDER},
    }
    write_header = not TRIALS_CSV.exists()
    with open(TRIALS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _save_state(
        trial_id: int,
        best_config: dict[str, float | int],
        best_auc: float,
        completed_keys: set[str],
        round_idx: int,
) -> None:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps({
        "trial_id": trial_id,
        "best_config": best_config,
        "best_auc": best_auc,
        "completed_keys": sorted(completed_keys),
        "round_idx": round_idx,
        "updated": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")


def _load_state() -> dict[str, Any] | None:
    if not STATE_JSON.exists():
        return None
    return json.loads(STATE_JSON.read_text(encoding="utf-8"))


def _save_best(config: dict[str, float | int], auc: float, target: float, baseline: float) -> None:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    BEST_JSON.write_text(json.dumps({
        "config": config,
        "auc_roc": auc,
        "beats_baseline": auc > baseline,
        "reaches_target": auc >= target,
        "commands": {
            "preprocess": " ".join(_preprocess_cmd(config, DEFAULT_DATA)),
            "train": " ".join(_train_cmd(config, DEFAULT_DATA, DEFAULT_MODEL, 30, 5)),
            "evaluate": " ".join(_evaluate_cmd(DEFAULT_DATA, DEFAULT_MODEL, TUNING_DIR / "best_run")),
        },
        "updated": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")


def run_trial(
        trial_id: int,
        config: dict[str, float | int],
        prev_config: dict[str, float | int] | None,
        *,
        epochs: int,
        patience: int,
        data_path: Path,
        model_path: Path,
        results_dir: Path,
        skip_preprocess: bool = False,
) -> TrialResult:
    t0 = time.perf_counter()
    try:
        if not skip_preprocess and _needs_preprocess(prev_config, config):
            _run_cmd(_preprocess_cmd(config, data_path), "preprocess")
        elif skip_preprocess:
            print("\n[skip] preprocess unchanged for this trial")
        else:
            print("\n[skip] preprocess — only training hyperparameters changed")

        _run_cmd(_train_cmd(config, data_path, model_path, epochs, patience), "train")
        _run_cmd(_evaluate_cmd(data_path, model_path, results_dir), "evaluate")
        metrics = _read_metrics(results_dir)
        status = "ok"
        error = ""
        auc = metrics["auc_roc"]
    except (subprocess.CalledProcessError, OSError, ValueError, FileNotFoundError) as exc:
        status = "failed"
        error = str(exc)
        metrics = {"auc_roc": float("nan"), "f1": float("nan"), "mean_subj_auc": float("nan")}
        auc = float("nan")

    result = TrialResult(
        trial_id=trial_id,
        config=dict(config),
        auc_roc=auc,
        f1=metrics["f1"],
        mean_subj_auc=metrics["mean_subj_auc"],
        duration_sec=time.perf_counter() - t0,
        status=status,
        error=error,
    )
    _append_trial_csv(result)
    tag = f"AUC-ROC={auc:.4f}" if status == "ok" else f"FAILED: {error}"
    print(f"\n[Trial {trial_id}] {tag}  ({result.duration_sec / 60:.1f} min)")
    return result


def coordinate_descent(args: argparse.Namespace) -> None:
    state = _load_state() if args.resume else None
    if state and not args.resume:
        state = None

    config = dict(DEFAULT_CONFIG)
    best_auc = float("-inf")
    trial_id = 0
    completed_keys: set[str] = set()
    round_idx = 0
    prev_config: dict[str, float | int] | None = None

    if state:
        config = {k: state["best_config"][k] for k in PARAM_ORDER}
        best_auc = float(state["best_auc"])
        trial_id = int(state["trial_id"])
        completed_keys = set(state.get("completed_keys", []))
        round_idx = int(state.get("round_idx", 0))
        print(f"[resume] trial_id={trial_id}  best_auc={best_auc:.4f}  round={round_idx}")
        print(f"[resume] config={config}")

    if args.param:
        # Single-parameter sweep mode.
        if args.param not in PARAM_ORDER:
            raise ValueError(f"Unknown param {args.param!r}; choose from {PARAM_ORDER}")
        values = args.values or CANDIDATE_GRIDS[args.param]
        for value in values:
            trial_id += 1
            trial_config = dict(config)
            trial_config[args.param] = value
            key = _config_key(trial_config)
            if key in completed_keys:
                print(f"[skip] trial already completed: {trial_config}")
                continue
            result = run_trial(
                trial_id, trial_config, prev_config,
                epochs=args.epochs, patience=args.patience,
                data_path=DEFAULT_DATA, model_path=DEFAULT_MODEL,
                results_dir=TUNING_DIR / f"trial_{trial_id:04d}",
                skip_preprocess=args.skip_preprocess,
            )
            completed_keys.add(key)
            prev_config = trial_config
            if result.status == "ok" and result.auc_roc > best_auc:
                best_auc = result.auc_roc
                config = dict(trial_config)
                _save_best(config, best_auc, args.target, args.baseline)
        _print_summary(config, best_auc, args.baseline, args.target)
        return

    # Baseline run with defaults if we have not started yet.
    if trial_id == 0:
        trial_id = 1
        key = _config_key(config)
        result = run_trial(
            trial_id, config, None,
            epochs=args.epochs, patience=args.patience,
            data_path=DEFAULT_DATA, model_path=DEFAULT_MODEL,
            results_dir=TUNING_DIR / f"trial_{trial_id:04d}",
        )
        completed_keys.add(key)
        prev_config = dict(config)
        if result.status == "ok":
            best_auc = result.auc_roc
            _save_best(config, best_auc, args.target, args.baseline)

    while round_idx < args.max_rounds:
        round_idx += 1
        print(f"\n{'=' * 60}\n  ROUND {round_idx}/{args.max_rounds}  best AUC-ROC={best_auc:.4f}\n{'=' * 60}")
        improved_round = False

        for param in PARAM_ORDER:
            print(f"\n>>> Tuning {param} (current={config[param]})")
            best_for_param = config[param]
            best_param_auc = best_auc

            for value in _neighbors(param, config[param], config):
                trial_config = dict(config)
                trial_config[param] = value
                key = _config_key(trial_config)
                if key in completed_keys:
                    continue

                trial_id += 1
                result = run_trial(
                    trial_id, trial_config, prev_config,
                    epochs=args.epochs, patience=args.patience,
                    data_path=DEFAULT_DATA, model_path=DEFAULT_MODEL,
                    results_dir=TUNING_DIR / f"trial_{trial_id:04d}",
                )
                completed_keys.add(key)
                prev_config = dict(trial_config)
                _save_state(trial_id, config, best_auc, completed_keys, round_idx)

                if result.status != "ok" or result.auc_roc != result.auc_roc:
                    continue
                if result.auc_roc > best_param_auc + args.min_delta:
                    best_param_auc = result.auc_roc
                    best_for_param = value

            if best_param_auc > best_auc + args.min_delta:
                print(f"  ✓ {param}: {config[param]} -> {best_for_param}  "
                      f"(AUC {best_auc:.4f} -> {best_param_auc:.4f})")
                config[param] = best_for_param
                best_auc = best_param_auc
                improved_round = True
                _save_best(config, best_auc, args.target, args.baseline)
                _save_state(trial_id, config, best_auc, completed_keys, round_idx)
                if best_auc >= args.target:
                    print(f"\n[done] Target AUC-ROC {args.target} reached.")
                    _print_summary(config, best_auc, args.baseline, args.target)
                    return

        if not improved_round:
            print(f"\n[stop] No improvement in round {round_idx}.")
            break

    _print_summary(config, best_auc, args.baseline, args.target)


def _print_summary(
        config: dict[str, float | int],
        auc: float,
        baseline: float,
        target: float,
) -> None:
    print(f"\n{'=' * 60}")
    print("  TUNING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Best validation AUC-ROC : {auc:.4f}")
    print(f"  Baseline (chance)       : {baseline:.2f}  "
          f"{'BEAT ✓' if auc > baseline else 'not yet'}")
    print(f"  Target                  : {target:.2f}  "
          f"{'REACHED ✓' if auc >= target else 'not yet'}")
    print("\n  Best config:")
    for k in PARAM_ORDER:
        print(f"    {k:18s} {config[k]}")
    print(f"\n  Saved to {BEST_JSON}")
    print(f"  All trials logged to {TRIALS_CSV}")
    print("\n  To run the best config on the main pipeline paths:")
    print(f"    {' '.join(_preprocess_cmd(config, REPO_ROOT / 'data/processed/eeg_windows.npz'))}")
    main_model = REPO_ROOT / "models/seizure_prediction_eeg/cnn_prediction_eeg.pt"
    print(f"    {' '.join(_train_cmd(config, REPO_ROOT / 'data/processed/eeg_windows.npz', main_model, 30, 5))}")
    print(f"    {' '.join(_evaluate_cmd(REPO_ROOT / 'data/processed/eeg_windows.npz', main_model, REPO_ROOT / 'results/seizure_prediction_eeg'))}")
    print(f"{'=' * 60}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated coordinate-descent tuning for the EEG pipeline.",
    )
    parser.add_argument("--baseline", type=float, default=0.5,
                        help="Chance-level AUC-ROC to beat (default 0.5).")
    parser.add_argument("--target", type=float, default=0.7,
                        help="Stop early when validation AUC-ROC reaches this (default 0.7).")
    parser.add_argument("--max-rounds", type=int, default=5,
                        help="Max coordinate-descent rounds (default 5).")
    parser.add_argument("--min-delta", type=float, default=0.002,
                        help="Minimum AUC improvement to accept a change (default 0.002).")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs per trial (default 30).")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stopping patience (default 5).")
    parser.add_argument("--quick", action="store_true",
                        help="Fast screening: 12 epochs, patience 3.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from results/tuning/eeg/state.json.")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Never rerun preprocessing (train/eval only).")
    parser.add_argument("--param", type=str, default=None,
                        help="Sweep only this parameter (e.g. window_sec).")
    parser.add_argument("--values", type=float, nargs="+", default=None,
                        help="Values to try with --param (default: built-in grid).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = 12
        args.patience = 3
        print("[quick] Using epochs=12, patience=3 for faster screening.")

    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Tuning EEG pipeline  |  baseline={args.baseline}  target={args.target}")
    print(f"Logs: {TRIALS_CSV}")
    coordinate_descent(args)


if __name__ == "__main__":
    main()
