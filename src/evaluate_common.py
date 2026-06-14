# -*- coding: utf-8 -*-
"""
Shared evaluation + explainability core for both seizure-prediction pipelines.

Loads a trained model + the preprocessed dataset, reconstructs the same held-out
20% test set (via the train_frac stored in the checkpoint), and:
  - averages class probabilities across ensemble members (soft voting),
  - applies temporal post-processing (removes short isolated positive runs),
  - computes precision / recall / F1 / confusion matrix,
  - appends one row of metrics + config to the pipeline's metrics.csv,
  - writes an average pre-ictal window plot and a Grad-CAM explainability figure.

The thin evaluate_*.py wrappers only set the dataset/model/results paths and the
feature-set label, then call run_evaluation.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from captum.attr import LayerGradCam

from preprocess_common import load_preprocessed, subject_aware_split, _subject_from_path
from model_common import SeizureCNN, pick_device

REPO_ROOT = Path(__file__).resolve().parent.parent  # src/ -> repo root


def _repo_relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).name


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(path: Path, device: torch.device) -> tuple[list[SeizureCNN], dict]:
    checkpoint = torch.load(path, map_location=device)
    meta = checkpoint["meta"]
    models: list[SeizureCNN] = []
    for sd in checkpoint["state_dicts"]:
        m = SeizureCNN(meta["n_channels"], meta["n_timepoints"]).to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    print(f"[Model] Loaded {len(models)} model(s) from {path}  "
          f"(channels={meta['n_channels']}, T={meta['n_timepoints']}, "
          f"train_frac={meta.get('train_frac')})")
    return models, meta


# ── Inference + post-processing ───────────────────────────────────────────────

@torch.no_grad()
def predict_positive_prob(model: SeizureCNN, x: np.ndarray, device: torch.device,
                          batch_size: int = 256) -> np.ndarray:
    model.eval()
    dl = DataLoader(TensorDataset(torch.from_numpy(x).to(device)), batch_size=batch_size)
    probs = []
    for (xb,) in dl:
        probs.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


def average_positive_prob(models, x, device, batch_size: int = 256) -> np.ndarray:
    probs = np.stack([predict_positive_prob(m, x, device, batch_size) for m in models], axis=0)
    return probs.mean(axis=0)


def smooth_binary_predictions(pred: np.ndarray, min_run: int = 2) -> np.ndarray:
    out = pred.copy()
    n, i = len(out), 0
    while i < n:
        if out[i] == 0:
            i += 1
            continue
        j = i
        while j < n and out[j] == 1:
            j += 1
        if (j - i) < min_run:
            out[i:j] = 0
        i = j
    return out


def compute_prf(y_true, y_pred) -> tuple[float, float, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return float(p), float(r), float(f)


def _block_metrics(prob: np.ndarray, y: np.ndarray, thr: float, min_run: int) -> dict:
    """Precision/recall/F1/specificity/balanced-acc + confusion at one threshold."""
    pred = smooth_binary_predictions((prob >= thr).astype(int), min_run)
    p, r, f = compute_prf(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {"precision": p, "recall": r, "f1": f, "specificity": spec,
            "balanced_acc": 0.5 * (r + spec),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# ── Metrics logging ───────────────────────────────────────────────────────────

def append_metrics_csv(csv_path: Path, row: dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[Metrics] Appended run to {csv_path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_average_preictal(x, y, channel_names, window_sec, save_path, show, max_channels=6):
    """Mean ± 1 SD of every pre-ictal window, one subplot per channel."""
    wins = x[y == 1]
    if len(wins) == 0:
        print("[AvgPreictal] No pre-ictal windows — skipping plot.")
        return
    n_ch = min(len(channel_names), max_channels)
    t_axis = np.linspace(0, window_sec, wins.shape[2])
    mean, std = wins[:, :n_ch, :].mean(axis=0), wins[:, :n_ch, :].std(axis=0)
    cols = min(n_ch, 3)
    rows = (n_ch + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), sharex=True)
    axes = np.array(axes).flatten()
    for i in range(n_ch):
        ax = axes[i]
        ax.plot(t_axis, mean[i], color="#c0392b", linewidth=1.5, label="Mean")
        ax.fill_between(t_axis, mean[i] - std[i], mean[i] + std[i], color="#c0392b",
                        alpha=0.20, label="±1 SD")
        ax.set_title(channel_names[i], fontsize=9)
        ax.set_ylabel("z-score", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    for j in range(n_ch, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Average pre-ictal window  |  n={len(wins)}  |  {window_sec:.1f}s", fontsize=11)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[AvgPreictal] Saved to {save_path}")
    plt.show() if show else plt.close(fig)


def plot_gradcam(model, x_test, y_test, channel_names, window_sec, n_samples,
                 save_path, device, show):
    pos = np.where(y_test == 1)[0]
    if len(pos) == 0:
        print("[GradCAM] No pre-ictal windows in test set — skipping.")
        return
    probs = predict_positive_prob(model, x_test, device)
    best = pos[np.argsort(probs[pos])[::-1][:n_samples]]
    grad_cam = LayerGradCam(model, model.conv_block[10])
    model.eval()
    n_ch = min(len(channel_names), 6)
    t_axis = np.linspace(0, window_sec, x_test.shape[2])
    colors = cm.tab10(np.linspace(0, 1, n_ch))
    fig, axes = plt.subplots(len(best), 1, figsize=(11, 3.5 * len(best)), squeeze=False)
    for row, wi in enumerate(best):
        ax = axes[row, 0]
        inp = torch.from_numpy(x_test[wi:wi + 1]).to(device).requires_grad_(True)
        heat = grad_cam.attribute(inp, target=1).mean(dim=1).squeeze(0).detach().cpu().numpy()
        heat = np.maximum(heat, 0)
        if heat.max() > 0:
            heat = heat / heat.max()
        heat_resized = np.interp(t_axis, np.linspace(0, window_sec, len(heat)), heat)
        for k in range(len(t_axis) - 1):
            ax.axvspan(t_axis[k], t_axis[k + 1],
                       color=(1.0, 0.2, 0.2, float(heat_resized[k]) * 0.5), linewidth=0)
        rng = np.max(np.ptp(x_test[wi, :n_ch], axis=1))
        spacing = rng * 1.3 or 1.0
        offsets = np.arange(n_ch)[::-1] * spacing
        for c in range(n_ch):
            ax.plot(t_axis, x_test[wi, c] + offsets[c], color=colors[c], linewidth=0.9,
                    label=channel_names[c] if c < len(channel_names) else f"ch{c}")
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_yticks(offsets)
        ax.set_yticklabels(channel_names[:n_ch], fontsize=8)
        ax.set_xlabel("Time within window (s)", fontsize=9)
        ax.set_title(f"Grad-CAM | test window #{wi} | p(pre-ictal)={probs[wi]:.2f}", fontsize=10)
        ax.grid(True, axis="x", alpha=0.25)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:, 0], fraction=0.015, pad=0.02, label="Grad-CAM intensity")
    fig.suptitle("Grad-CAM: most confident pre-ictal predictions", fontsize=11)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[GradCAM] Saved to {save_path}")
    plt.show() if show else plt.close(fig)


# ── CLI helper + orchestration ────────────────────────────────────────────────

def add_eval_args(parser, default_data, default_model, default_results_dir, default_feature_set):
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir)
    parser.add_argument("--feature-set", type=str, default=default_feature_set,
                        help="Label written to metrics.csv (e.g. 'eeg' or 'eeg_ecg').")
    parser.add_argument("--train-frac", type=float, default=None,
                        help="Override the split fraction. Default: use the checkpoint value.")
    parser.add_argument("--pred-threshold", type=float, default=0.5)
    parser.add_argument("--pred-min-run", type=int, default=2)
    parser.add_argument("--gradcam-n-samples", type=int, default=4)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--show", action="store_true")


def run_evaluation(args: argparse.Namespace) -> None:
    if not (0.0 < args.pred_threshold < 1.0):
        raise ValueError("--pred-threshold must be in (0, 1)")
    if args.pred_min_run < 1:
        raise ValueError("--pred-min-run must be >= 1")

    device = pick_device(args.no_gpu)

    print(f"\n[Data] Loading preprocessed dataset: {args.data}")
    data = load_preprocessed(args.data)
    x, y = data["X"], data["y"]
    channel_names = data["channel_names"]
    window_sec = data["window_sec"]

    models, meta = load_models(args.model, device)
    train_frac = args.train_frac if args.train_frac is not None else float(meta.get("train_frac", 0.8))
    train_subjects = meta.get("train_subjects")  # list of subject IDs used during training

    train_idx, test_idx = subject_aware_split(data, train_frac, train_subjects)
    x_test, y_test = x[test_idx], y[test_idx]

    if train_subjects:
        from preprocess_common import _subject_from_path
        subj_of_all = np.empty(len(x), dtype=object)
        for (s, e), p in zip(data["file_slices"], data["recording_paths"]):
            subj_of_all[s:e] = _subject_from_path(p)
        test_subjects = sorted(set(subj_of_all[test_idx].tolist()))
        print(f"\nSubject-level split  ->  test subjects: {', '.join(test_subjects)}")
    else:
        print(f"\nWithin-subject stratified split (train_frac={train_frac})")
    print(f"  test windows={len(x_test):,}  (pre-ictal: {int(y_test.sum()):,})\n")

    thr, min_run = args.pred_threshold, args.pred_min_run
    prob_test = average_positive_prob(models, x_test, device)
    m = _block_metrics(prob_test, y_test, thr, min_run)

    # threshold-independent ranking quality
    two_classes = len(np.unique(y_test)) > 1
    auc_roc = roc_auc_score(y_test, prob_test) if two_classes else float("nan")
    auc_pr = average_precision_score(y_test, prob_test) if two_classes else float("nan")

    # over/underfit check: F1 on a subsample of the training data
    rng = np.random.default_rng(0)
    tr_sub = (train_idx if len(train_idx) <= 20000
              else np.sort(rng.choice(train_idx, 20000, replace=False)))
    prob_train = average_positive_prob(models, x[tr_sub], device)
    train_f1 = _block_metrics(prob_train, y[tr_sub], thr, min_run)["f1"]

    # per-subject test breakdown
    subj_of = np.empty(len(x), dtype=object)
    for (s, e), p in zip(data["file_slices"], data["recording_paths"]):
        subj_of[s:e] = _subject_from_path(p)
    test_subj = subj_of[test_idx]

    # ── report (printed AND saved to a timestamped file) ──
    bar = "=" * 60
    lines = [
        bar,
        f"  EVALUATION REPORT   feature_set={args.feature_set}   threshold={thr}",
        f"  model={_repo_relative(args.model)}   epochs={meta.get('epochs')}   "
        f"seed={meta.get('random_state')}",
        bar,
        f"  Precision      : {m['precision']:.3f}",
        f"  Recall (sens.) : {m['recall']:.3f}",
        f"  Specificity    : {m['specificity']:.3f}",
        f"  F1             : {m['f1']:.3f}",
        f"  Balanced acc   : {m['balanced_acc']:.3f}",
        f"  Confusion [tn fp fn tp]: [{m['tn']} {m['fp']} {m['fn']} {m['tp']}]",
        "",
        f"  AUC-ROC        : {auc_roc:.3f}   (0.50 = chance)",
        f"  AUC-PR         : {auc_pr:.3f}   (baseline {y_test.mean():.3f} = positive rate)",
        "",
        f"  Over/underfit  : train F1 {train_f1:.3f}  vs  test F1 {m['f1']:.3f}  "
        f"(gap {train_f1 - m['f1']:+.3f})",
        "",
        "  Threshold sweep (P / R / F1):",
    ]
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        mm = _block_metrics(prob_test, y_test, t, min_run)
        marker = "  <-- current" if abs(t - thr) < 1e-9 else ""
        lines.append(f"    thr {t:.1f}:  P {mm['precision']:.3f}   R {mm['recall']:.3f}   "
                     f"F1 {mm['f1']:.3f}{marker}")
    lines += ["", "  Per-subject test (AUC = within-patient ranking quality):"]
    subj_aucs, per_subject_rows = [], []
    for subj in sorted(set(test_subj.tolist())):
        mask = test_subj == subj
        ys, ps = y_test[mask], prob_test[mask]
        sp, sr, sf = compute_prf(ys, smooth_binary_predictions((ps >= thr).astype(int), min_run))
        sauc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else float("nan")
        if not np.isnan(sauc):
            subj_aucs.append(sauc)
        lines.append(f"    {subj}:  AUC {sauc:.3f}   P {sp:.3f}   R {sr:.3f}   F1 {sf:.3f}   "
                     f"(n_preictal {int(ys.sum())} / {int(mask.sum())})")
        per_subject_rows.append({
            "feature_set": args.feature_set, "subject": subj,
            "auc": ("" if np.isnan(sauc) else round(float(sauc), 4)),
            "precision": round(sp, 4), "recall": round(sr, 4), "f1": round(sf, 4),
            "n_preictal": int(ys.sum()), "n_test": int(mask.sum()),
        })
    if subj_aucs:
        lines.append(f"    mean per-subject AUC (subjects with positives): "
                     f"{np.mean(subj_aucs):.3f} ± {np.std(subj_aucs):.3f}  (n={len(subj_aucs)})")
    lines.append(bar)

    report = "\n".join(lines)
    print(report)
    reports_dir = Path(args.results_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"eval_{args.feature_set}_{stamp}.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"[Report] Saved to {report_path}")

    # per-subject CSV (overwritten each run; the latest breakdown for the notebook)
    ps_path = Path(args.results_dir) / "per_subject.csv"
    with open(ps_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature_set", "subject", "auc",
                                          "precision", "recall", "f1", "n_preictal", "n_test"])
        w.writeheader()
        w.writerows(per_subject_rows)
    print(f"[Per-subject] Saved to {ps_path}")

    append_metrics_csv(Path(args.results_dir) / "metrics.csv", {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "feature_set": args.feature_set,
        "train_frac": train_frac,
        "epochs": meta.get("epochs"),
        "random_state": meta.get("random_state"),
        "n_runs": meta.get("n_runs", len(models)),
        "interictal_ratio": meta.get("interictal_ratio"),
        "pred_threshold": thr,
        "pred_min_run": min_run,
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "auc_roc": round(float(auc_roc), 4),
        "auc_pr": round(float(auc_pr), 4),
        "train_f1": round(train_f1, 4),
        "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
        "n_test": len(y_test), "n_test_preictal": int(y_test.sum()),
        "model_path": _repo_relative(args.model),
    })

    plot_average_preictal(x, y, channel_names, window_sec,
                          Path(args.results_dir) / "average_preictal.png", args.show)
    plot_gradcam(models[0], x_test, y_test, channel_names, window_sec,
                 args.gradcam_n_samples, Path(args.results_dir) / "gradcam.png",
                 device, args.show)
