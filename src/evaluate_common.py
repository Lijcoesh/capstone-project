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

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from captum.attr import LayerGradCam
import torch.nn as nn

from preprocess_common import load_preprocessed, subject_aware_split, _subject_from_path
from model_common import SeizureCNN, build_model, pick_device

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
    models = []
    input_rep = meta.get("input_rep", "raw")
    for sd in checkpoint["state_dicts"]:
        m = build_model(input_rep, meta["n_channels"], meta["n_timepoints"]).to(device)
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
    # Move one batch at a time (keeps VRAM low on small GPUs; x stays in CPU RAM).
    model.eval()
    probs = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).to(device)
        probs.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs).astype(np.float32) if probs else np.empty(0, np.float32)


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


def temporal_smooth_prob(prob: np.ndarray, centers: np.ndarray,
                         subj: np.ndarray, k: int) -> np.ndarray:
    """Centered moving-average of p(pre-ictal) over k consecutive windows, per subject.

    A single 2 s window is a noisy estimate of the pre-ictal state; clinically you
    never alarm on one window, you track the probability over time. Averaging each
    window with its temporal neighbours (ordered by center time, within a subject)
    denoises the ranking — and costs no retraining. Edges are normalised by the
    number of valid taps so the ends aren't biased toward zero.
    """
    if k <= 1:
        return prob.astype(np.float32, copy=True)
    out = np.empty(len(prob), dtype=np.float64)
    for s in np.unique(subj):
        idx = np.nonzero(subj == s)[0]
        order = idx[np.argsort(centers[idx])]          # chronological within subject
        p = prob[order].astype(np.float64)
        # Cap the kernel at the subject's window count: np.convolve(mode="same")
        # returns max(len(p), len(kernel)) samples, so a kernel longer than this
        # subject's test windows would break the assignment back into `out`.
        kk = min(k, len(p))
        kernel = np.ones(kk, dtype=np.float64)
        num = np.convolve(p, kernel, mode="same")
        den = np.convolve(np.ones_like(p), kernel, mode="same")
        out[order] = num / den
    return out.astype(np.float32)


def auc_overall_and_per_subject(prob: np.ndarray, y: np.ndarray,
                                subj: np.ndarray) -> tuple[float, float, int]:
    """(pooled AUC, mean per-subject AUC, n subjects with both classes)."""
    overall = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    saucs = [roc_auc_score(y[m], prob[m])
             for s in np.unique(subj)
             for m in [subj == s] if len(np.unique(y[m])) > 1]
    mean_sauc = float(np.mean(saucs)) if saucs else float("nan")
    return float(overall), mean_sauc, len(saucs)


def _block_metrics(prob: np.ndarray, y: np.ndarray, thr: float, min_run: int) -> dict:
    """Precision/recall/F1/specificity/balanced-acc + confusion at one threshold."""
    pred = smooth_binary_predictions((prob >= thr).astype(int), min_run)
    p, r, f = compute_prf(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    total = tn + fp + fn + tp
    return {"precision": p, "recall": r, "f1": f, "specificity": spec,
            "accuracy": (tp + tn) / total if total else 0.0,
            "balanced_acc": 0.5 * (r + spec),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def best_f1_threshold(y: np.ndarray, prob: np.ndarray, min_run: int,
                      lo: float = 0.05, hi: float = 0.95, step: float = 0.01
                      ) -> tuple[float, float]:
    """Return (threshold, F1) maximising F1 on the given set (with min-run filter)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(lo, hi + 1e-9, step):
        f = _block_metrics(prob, y, float(t), min_run)["f1"]
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1

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

def _last_conv1d_layer(model: nn.Module) -> nn.Conv1d:
    for m in reversed(list(model.conv_block.modules())):
        if isinstance(m, nn.Conv1d):
            return m
    raise ValueError("No Conv1d layer found in model.conv_block")


def _resize_1d(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) == n:
        return arr.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, len(arr))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, arr).astype(np.float32)


def _frame_time_axis(n_frames: int, frame_sec: float, frame_step_sec: float) -> np.ndarray:
    """Center time (seconds) of each band-power frame within the window."""
    return (np.arange(n_frames, dtype=np.float64) * frame_step_sec + frame_sec * 0.5)


def _gradcam_attribution(model, grad_cam: LayerGradCam, inp: torch.Tensor,
                         n_features: int, n_frames: int) -> np.ndarray:
    """(n_features, n_frames) non-negative saliency in [0, 1]."""
    attr = grad_cam.attribute(inp, target=1).detach().cpu().numpy()
    if attr.ndim == 3:
        heat = attr.squeeze(0)
    elif attr.ndim == 2:
        heat = attr
    else:
        raise ValueError(f"Unexpected Grad-CAM shape: {attr.shape}")
    if heat.ndim == 1:
        heat = np.tile(_resize_1d(heat, n_frames)[None, :], (n_features, 1))
    elif heat.shape[0] != n_features or heat.shape[1] != n_frames:
        # Deeper layer: resize time, broadcast across features if needed.
        t = _resize_1d(heat.mean(axis=0), n_frames)
        heat = np.tile(t[None, :], (n_features, 1))
    heat = np.maximum(heat.astype(np.float32), 0.0)
    if heat.max() > 0:
        heat /= heat.max()
    return heat


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
    grad_cam = LayerGradCam(model, _last_conv1d_layer(model))
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


def plot_gradcam_bandpower(model, x_test, y_test, feature_names, window_sec,
                           frame_sec, frame_step_sec, n_samples, save_path,
                           device, show):
    """Grad-CAM for band-power sequences: input heatmap + saliency for the most
    confident true pre-ictal windows (highest p(pre-ictal))."""
    pos = np.where(y_test == 1)[0]
    if len(pos) == 0:
        print("[GradCAM] No pre-ictal windows in test set — skipping.")
        return
    probs = predict_positive_prob(model, x_test, device)
    best = pos[np.argsort(probs[pos])[::-1][:n_samples]]
    grad_cam = LayerGradCam(model, _last_conv1d_layer(model))
    model.eval()
    n_features, n_frames = x_test.shape[1], x_test.shape[2]
    t_axis = _frame_time_axis(n_frames, frame_sec, frame_step_sec)
    fig, axes = plt.subplots(len(best), 2, figsize=(12, 2.8 * len(best)), squeeze=False)
    for row, wi in enumerate(best):
        inp = torch.from_numpy(x_test[wi:wi + 1]).to(device).requires_grad_(True)
        values = x_test[wi]
        heat = _gradcam_attribution(model, grad_cam, inp, n_features, n_frames)

        ax_in, ax_cam = axes[row, 0], axes[row, 1]
        vmax = max(float(np.percentile(np.abs(values), 99)), 0.5)
        im_in = ax_in.imshow(values, aspect="auto", origin="lower", cmap="RdBu_r",
                             vmin=-vmax, vmax=vmax,
                             extent=[t_axis[0], t_axis[-1], -0.5, n_features - 0.5])
        ax_in.set_yticks(range(n_features))
        ax_in.set_yticklabels(feature_names, fontsize=7)
        ax_in.set_xlabel("Time in window (s)", fontsize=8)
        ax_in.set_title(f"Band-power input  |  p={probs[wi]:.2f}", fontsize=9)

        im_cam = ax_cam.imshow(heat, aspect="auto", origin="lower", cmap="hot",
                               vmin=0.0, vmax=1.0,
                               extent=[t_axis[0], t_axis[-1], -0.5, n_features - 0.5])
        ax_cam.set_yticks(range(n_features))
        ax_cam.set_yticklabels(feature_names, fontsize=7)
        ax_cam.set_xlabel("Time in window (s)", fontsize=8)
        ax_cam.set_title("Grad-CAM saliency (why the model cares)", fontsize=9)
        if row == 0:
            fig.colorbar(im_in, ax=axes[0, 0], fraction=0.02, pad=0.02, label="z-scored log power")
            fig.colorbar(im_cam, ax=axes[0, 1], fraction=0.02, pad=0.02, label="attribution")

    fig.suptitle("Most confident true pre-ictal windows  |  band-power + Grad-CAM", fontsize=11)
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
                        help="Override the train split fraction. Default: use the checkpoint value.")
    parser.add_argument("--val-frac", type=float, default=None,
                        help="Override the validation split fraction. Default: checkpoint value.")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val",
                        help="Which held-out set to report on. 'val' (default) for tuning "
                             "hyperparameters/threshold; 'test' for the final, one-time evaluation.")
    parser.add_argument("--pred-threshold", type=float, default=None,
                        help="Decision threshold. Default: F1-optimal on VAL (test split) or "
                             "on TRAIN subsample (val split), with --pred-min-run applied.")
    parser.add_argument("--pred-min-run", type=int, default=2)
    parser.add_argument("--gradcam-n-samples", type=int, default=4)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--show", action="store_true")


def run_evaluation(args: argparse.Namespace) -> None:
    if args.pred_threshold is not None and not (0.0 < args.pred_threshold < 1.0):
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
    train_frac = args.train_frac if args.train_frac is not None else float(meta.get("train_frac", 0.6))
    val_frac = args.val_frac if args.val_frac is not None else float(meta.get("val_frac", 0.2))
    train_subjects = meta.get("train_subjects")  # list of subject IDs used during training

    train_idx, val_idx, test_idx_full = subject_aware_split(
        data, train_frac, val_frac, train_subjects)

    # 'val' for tuning (default), 'test' for the final one-time evaluation.
    eval_split = args.eval_split
    if eval_split == "val" and len(val_idx) == 0:
        print("[Split] No validation set (subject-level split) -> falling back to test.")
        eval_split = "test"
    eval_idx = val_idx if eval_split == "val" else test_idx_full
    # downstream code (per-subject breakdown, plots) reports on the chosen split
    test_idx = eval_idx
    x_test, y_test = x[test_idx], y[test_idx]

    # All outputs go into a per-split subdir so it is unambiguous which split they
    # describe: results/<pipeline>/test/ vs results/<pipeline>/val/. The notebook
    # compares the two pipelines' test/ dirs.
    out_base = Path(args.results_dir) / eval_split
    out_base.mkdir(parents=True, exist_ok=True)

    banner = ("VALIDATION (tuning)" if eval_split == "val"
              else "TEST (final, one-time)")
    if train_subjects:
        subj_of_all = np.empty(len(x), dtype=object)
        for (s, e), p in zip(data["file_slices"], data["recording_paths"]):
            subj_of_all[s:e] = _subject_from_path(p)
        test_subjects = sorted(set(subj_of_all[test_idx].tolist()))
        print(f"\nSubject-level split  ->  test subjects: {', '.join(test_subjects)}")
    else:
        print(f"\nWithin-subject 3-way split (train={train_frac}, val={val_frac}, "
              f"test={1 - train_frac - val_frac:.2f})")
    print(f"  Reporting on: {banner}")
    print(f"  windows={len(x_test):,}  (pre-ictal: {int(y_test.sum()):,})\n")

    min_run = args.pred_min_run
    prob_test = average_positive_prob(models, x_test, device)

    # Pick decision threshold: manual, val-tuned (test), or train-tuned (val).
    rng = np.random.default_rng(0)
    tr_sub = (train_idx if len(train_idx) <= 20000
              else np.sort(rng.choice(train_idx, 20000, replace=False)))
    if args.pred_threshold is not None:
        thr, thr_source = float(args.pred_threshold), "manual"
    elif eval_split == "test":
        prob_val = average_positive_prob(models, x[val_idx], device)
        thr, tune_f1 = best_f1_threshold(y[val_idx], prob_val, min_run)
        thr_source = f"val-tuned (best F1={tune_f1:.3f} on VAL)"
    else:
        prob_tr_tune = average_positive_prob(models, x[tr_sub], device)
        thr, tune_f1 = best_f1_threshold(y[tr_sub], prob_tr_tune, min_run)
        thr_source = f"train-tuned (best F1={tune_f1:.3f} on TRAIN)"

    print(f"[Threshold] {thr:.2f}  ({thr_source})\n")
    m = _block_metrics(prob_test, y_test, thr, min_run)
    m05 = _block_metrics(prob_test, y_test, 0.5, min_run)

    # threshold-independent ranking quality
    two_classes = len(np.unique(y_test)) > 1
    auc_roc = roc_auc_score(y_test, prob_test) if two_classes else float("nan")
    auc_pr = average_precision_score(y_test, prob_test) if two_classes else float("nan")

    # over/underfit check: F1 on a subsample of the training data
    prob_train = average_positive_prob(models, x[tr_sub], device)
    train_f1 = _block_metrics(prob_train, y[tr_sub], thr, min_run)["f1"]

    # per-subject test breakdown
    subj_of = np.empty(len(x), dtype=object)
    for (s, e), p in zip(data["file_slices"], data["recording_paths"]):
        subj_of[s:e] = _subject_from_path(p)
    test_subj = subj_of[test_idx]

    # temporal aggregation: a single 2 s window is noisy, so also score p(pre-ictal)
    # after a per-subject moving average over several seconds (no retraining). We
    # report both the pooled AUC and the mean per-subject AUC (the patient-aware
    # headline for the EEG vs EEG+ECG comparison).
    step_sec = float(data.get("step_sec", 1.0)) or 1.0
    centers_test = data["centers"][test_idx]
    smooth_secs = [0, 30, 60, 120]
    smooth_rows = []  # (sec, k, pooled_auc, mean_subject_auc, n_subj)
    for sec in smooth_secs:
        k = max(1, int(round(sec / step_sec)))
        ps = temporal_smooth_prob(prob_test, centers_test, test_subj, k)
        o, ms, ns = auc_overall_and_per_subject(ps, y_test, test_subj)
        smooth_rows.append((sec, k, o, ms, ns))
    raw_auc, raw_msauc, _ = smooth_rows[0][2], smooth_rows[0][3], smooth_rows[0][4]
    # pick the best smoothing by mean per-subject AUC for the csv summary
    best_row = max(smooth_rows[1:], key=lambda r: (r[3] if not np.isnan(r[3]) else -1),
                   default=smooth_rows[0])

    # ── report (printed AND saved to a timestamped file) ──
    bar = "=" * 60
    lines = [
        bar,
        f"  EVALUATION REPORT   feature_set={args.feature_set}   threshold={thr:.2f}",
        f"  threshold source     : {thr_source}",
        f"  eval split={eval_split.upper()}   model={_repo_relative(args.model)}   "
        f"epochs={meta.get('epochs')}   seed={meta.get('random_state')}",
        bar,
        f"  Precision      : {m['precision']:.3f}",
        f"  Recall (sens.) : {m['recall']:.3f}",
        f"  Specificity    : {m['specificity']:.3f}",
        f"  F1             : {m['f1']:.3f}",
        f"  Accuracy       : {m['accuracy']:.3f}   (misleading under imbalance)",
        f"  Balanced acc   : {m['balanced_acc']:.3f}",
        f"  Confusion [tn fp fn tp]: [{m['tn']} {m['fp']} {m['fn']} {m['tp']}]",
        f"  @ thr 0.50 (legacy)  : P {m05['precision']:.3f} / R {m05['recall']:.3f} / "
        f"F1 {m05['f1']:.3f}  [{m05['tn']} {m05['fp']} {m05['fn']} {m05['tp']}]",
        "",
        f"  AUC-ROC        : {auc_roc:.3f}   (0.50 = chance)",
        f"  AUC-PR         : {auc_pr:.3f}   (baseline {y_test.mean():.3f} = positive rate)",
        "",
        "  Temporal aggregation (moving avg of p over time, per subject):",
        "    smoothing      pooled-AUC   mean-subject-AUC",
    ]
    for sec, k, o, ms, ns in smooth_rows:
        tag = "raw (1 win)" if sec == 0 else f"{sec:>3d}s ({k} win)"
        lines.append(f"    {tag:<14s} {o:>9.3f}   {ms:>14.3f}  (n={ns})")
    lines += [
        "",
        f"  Over/underfit  : train F1 {train_f1:.3f}  vs  test F1 {m['f1']:.3f}  "
        f"(gap {train_f1 - m['f1']:+.3f})",
        "",
        "  Threshold sweep (P / R / F1):",
    ]
    sweep_ts = sorted({0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, round(thr, 2)})
    for t in sweep_ts:
        mm = _block_metrics(prob_test, y_test, t, min_run)
        marker = "  <-- current" if abs(t - thr) < 0.005 else ""
        lines.append(f"    thr {t:.2f}:  P {mm['precision']:.3f}   R {mm['recall']:.3f}   "
                     f"F1 {mm['f1']:.3f}{marker}")
    lines += ["", f"  Per-subject {eval_split} (AUC = within-patient ranking quality):"]
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
    reports_dir = out_base / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"eval_{args.feature_set}_{stamp}.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"[Report] Saved to {report_path}")

    # per-subject CSV (overwritten each run; the latest breakdown for the notebook)
    ps_path = out_base / "per_subject.csv"
    with open(ps_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature_set", "subject", "auc",
                                          "precision", "recall", "f1", "n_preictal", "n_test"])
        w.writeheader()
        w.writerows(per_subject_rows)
    print(f"[Per-subject] Saved to {ps_path}")

    append_metrics_csv(out_base / "metrics.csv", {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "feature_set": args.feature_set,
        "eval_split": eval_split,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "epochs": meta.get("epochs"),
        "random_state": meta.get("random_state"),
        "n_runs": meta.get("n_runs", len(models)),
        "interictal_ratio": meta.get("interictal_ratio"),
        "preictal_sec": meta.get("preictal_sec"),
        "normalize": data.get("normalize", "per_window"),
        "pred_threshold": round(thr, 4),
        "pred_min_run": min_run,
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "precision_thr050": round(m05["precision"], 4),
        "recall_thr050": round(m05["recall"], 4),
        "f1_thr050": round(m05["f1"], 4),
        "accuracy": round(m["accuracy"], 4),
        "balanced_acc": round(m["balanced_acc"], 4),
        "auc_roc": round(float(auc_roc), 4),
        "auc_pr": round(float(auc_pr), 4),
        "mean_subj_auc": ("" if np.isnan(raw_msauc) else round(raw_msauc, 4)),
        "smooth_sec": best_row[0],
        "auc_roc_smooth": ("" if np.isnan(best_row[2]) else round(best_row[2], 4)),
        "mean_subj_auc_smooth": ("" if np.isnan(best_row[3]) else round(best_row[3], 4)),
        "train_f1": round(train_f1, 4),
        "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
        "n_test": len(y_test), "n_test_preictal": int(y_test.sum()),
        "model_path": _repo_relative(args.model),
    })

    plot_average_preictal(x, y, channel_names, window_sec,
                          out_base / "average_preictal.png", args.show)
    input_rep = meta.get("input_rep", "raw")
    if input_rep == "raw":
        plot_gradcam(models[0], x_test, y_test, channel_names, window_sec,
                     args.gradcam_n_samples, out_base / "gradcam.png",
                     device, args.show)
    elif input_rep == "bandpower_seq":
        plot_gradcam_bandpower(
            models[0], x_test, y_test, channel_names, window_sec,
            float(data.get("frame_sec", 4.0)),
            float(data.get("frame_step_sec", 1.0)),
            args.gradcam_n_samples, out_base / "gradcam.png", device, args.show)
    else:
        print(f"[GradCAM] Skipped (input_rep={input_rep}).")
