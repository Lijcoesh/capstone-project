# -*- coding: utf-8 -*-
"""
Visual comparison of pre-ictal vs interictal horizons across subjects.

Goal: explain why per-subject test AUC does NOT track the number of seizures a
patient has. For each subject we put pre-ictal and interictal windows side by
side (Grad-CAM overlaid on the raw waveform) and compare signal *strength*, then
read it against that subject's test AUC and seizure count.

Each subject has its OWN preprocessed npz and its OWN trained model (single-
subject runs). Paths are derived from templates with a {subj} placeholder:
  --data-tmpl   default  data/processed/eeg_{subj}.npz
  --model-tmpl  default  models/per_subject/{subj}.pt

The EEG model is input_rep="raw": 2 behind-the-ear EEG channels x 512 samples
(2 s @ 256 Hz), per-recording z-scored, so the model input *is* the waveform and
Grad-CAM is overlaid directly on it.

Outputs (to results/explainability/horizons/ by default):
  - gradcam_sidebyside_<subject>.png : N pre-ictal (left) vs N interictal (right)
    test windows, Grad-CAM heat overlaid, p(pre-ictal) per window.
  - signal_strength_meanwave.png     : mean +/- SD waveform per subject (rows,
    sorted by AUC) x channel (cols), pre-ictal vs interictal overlaid.
  - signal_strength_amplitude.png    : per-window amplitude (peak-to-peak & RMS)
    distributions, pre-ictal vs interictal, per subject.
  - summary.csv + a printed table (AUC, seizure count, amplitude stats).

Usage (from src/):
  python explain_horizons.py --subjects sub-034,sub-035,sub-039,sub-047,sub-060
  python explain_horizons.py --subjects sub-034,sub-047 --n-windows 10 --show
"""

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import LayerGradCam
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess_common import load_preprocessed, subject_aware_split, _subject_from_path  # noqa: E402
from model_common import pick_device  # noqa: E402
from evaluate_common import load_models, average_positive_prob, _last_conv1d_layer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_TMPL = "data/processed/eeg_{subj}.npz"
DEFAULT_MODEL_TMPL = "models/per_subject/{subj}.pt"
DEFAULT_OUT = REPO_ROOT / "results/explainability/horizons"

PRE_COLOR = "#c0392b"   # pre-ictal red
INT_COLOR = "#2c6fbb"   # interictal blue


# ── seizure count (ties the analysis back to data quantity) ───────────────────

def count_seizures(subj: str) -> int:
    """Seizures for a subject = events.tsv rows whose eventType starts with 'sz'."""
    n = 0
    for f in glob.glob(str(REPO_ROOT / f"data/raw/**/{subj}_*events.tsv"), recursive=True):
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if (row.get("eventType") or "").strip().startswith("sz"):
                    n += 1
    return n


# ── window selection ──────────────────────────────────────────────────────────

def select_windows(probs, label_mask, n, mode, rng):
    """Up to n global positions for one class.

    pre-ictal  -> most confident (highest p) so Grad-CAM is meaningful.
    interictal -> 'confident' (lowest p), 'hard' (highest p = false alarms),
                  or 'random' (representative).
    """
    idx = np.where(label_mask)[0]
    if len(idx) == 0:
        return idx
    p = probs[idx]
    if mode in ("preictal", "hard"):
        order = np.argsort(p)[::-1]
    elif mode == "confident":
        order = np.argsort(p)
    elif mode == "random":
        order = rng.permutation(len(idx))
    else:
        raise ValueError(mode)
    return idx[order[:n]]


# ── Grad-CAM heat for one raw window ──────────────────────────────────────────

def window_heat(grad_cam, x_win, device):
    """Non-negative Grad-CAM saliency over time in [0,1] for a single window."""
    inp = torch.from_numpy(x_win[None]).to(device).requires_grad_(True)
    heat = grad_cam.attribute(inp, target=1).mean(dim=1).squeeze(0).detach().cpu().numpy()
    heat = np.maximum(heat, 0.0)
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat


def draw_window(ax, x_win, prob, heat, t_axis, window_sec, channel_names, colors,
                title_prefix):
    """Plot one window's channels (vertically offset) with Grad-CAM heat behind."""
    n_ch = len(channel_names)
    heat_resized = np.interp(t_axis, np.linspace(0, window_sec, len(heat)), heat)
    for k in range(len(t_axis) - 1):
        ax.axvspan(t_axis[k], t_axis[k + 1],
                   color=(1.0, 0.2, 0.2, float(heat_resized[k]) * 0.5), linewidth=0)
    rng = np.max(np.ptp(x_win[:n_ch], axis=1))
    spacing = (rng * 1.3) or 1.0
    offsets = np.arange(n_ch)[::-1] * spacing
    for c in range(n_ch):
        ax.plot(t_axis, x_win[c] + offsets[c], color=colors[c], linewidth=0.8)
    ax.set_xlim(t_axis[0], t_axis[-1])
    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_names, fontsize=6)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_title(f"{title_prefix} | p={prob:.2f}", fontsize=7)
    ax.grid(True, axis="x", alpha=0.2)


def gradcam_sidebyside(subject, x, probs, pre_pos, int_pos, model, device,
                       channel_names, window_sec, auc, save_path, show):
    grad_cam = LayerGradCam(model, _last_conv1d_layer(model))
    model.eval()
    n_rows = max(len(pre_pos), len(int_pos), 1)
    t_axis = np.linspace(0, window_sec, x.shape[2])
    colors = cm.tab10(np.linspace(0, 1, len(channel_names)))

    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 1.7 * n_rows), squeeze=False)
    for col, (positions, prefix) in enumerate([(pre_pos, "pre"), (int_pos, "int")]):
        for row in range(n_rows):
            ax = axes[row, col]
            if row >= len(positions):
                ax.set_visible(False)
                continue
            gi = positions[row]
            heat = window_heat(grad_cam, x[gi], device)
            draw_window(ax, x[gi], probs[gi], heat, t_axis, window_sec,
                        channel_names, colors, f"{prefix} #{gi}")
    axes[0, 0].annotate("PRE-ICTAL  (before onset)", xy=(0.5, 1.0),
                        xytext=(0.5, 1.5), xycoords="axes fraction",
                        ha="center", fontsize=11, color=PRE_COLOR, weight="bold")
    axes[0, 1].annotate("INTERICTAL  (far from seizure)", xy=(0.5, 1.0),
                        xytext=(0.5, 1.5), xycoords="axes fraction",
                        ha="center", fontsize=11, color=INT_COLOR, weight="bold")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.02, label="Grad-CAM intensity")
    fig.suptitle(f"{subject}  |  test AUC {auc:.3f}  |  pre-ictal vs interictal "
                 f"(Grad-CAM overlay)", fontsize=12, y=0.995)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[GradCAM] {subject} -> {save_path}")
    plt.show() if show else plt.close(fig)


# ── combined cross-subject figures ────────────────────────────────────────────

def combined_meanwave(results, channel_names, window_sec, save_path, show):
    n_ch = len(channel_names)
    t_axis = np.linspace(0, window_sec, results[0]["x_pre"].shape[2])
    fig, axes = plt.subplots(len(results), n_ch,
                             figsize=(5 * n_ch, 2.8 * len(results)),
                             squeeze=False, sharex=True)
    for r, res in enumerate(results):
        for c in range(n_ch):
            ax = axes[r, c]
            for arr, color, lab in [(res["x_pre"], PRE_COLOR, "pre-ictal"),
                                    (res["x_int"], INT_COLOR, "interictal")]:
                if len(arr) == 0:
                    continue
                mean, std = arr[:, c, :].mean(0), arr[:, c, :].std(0)
                ax.plot(t_axis, mean, color=color, linewidth=1.4, label=lab)
                ax.fill_between(t_axis, mean - std, mean + std, color=color, alpha=0.15)
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{res['subject']}  |  AUC {res['auc']:.3f}  |  "
                         f"{res['n_seizures']} sz  |  {channel_names[c]}", fontsize=8)
            if c == 0:
                ax.set_ylabel("z-score", fontsize=8)
            if r == len(results) - 1:
                ax.set_xlabel("Time in window (s)", fontsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Mean ± SD waveform: pre-ictal vs interictal  (subjects sorted by AUC)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[Signal] mean-wave -> {save_path}")
    plt.show() if show else plt.close(fig)


def combined_amplitude(results, save_path, show):
    """Per-window peak-to-peak and RMS (avg over channels), pre vs interictal."""
    fig, axes = plt.subplots(1, 2, figsize=(2.2 * len(results) + 3, 4.5))
    for mi, (key, title) in enumerate([("ptp", "Peak-to-peak amplitude (z)"),
                                       ("rms", "RMS amplitude (z)")]):
        ax = axes[mi]
        data, ticks, ticklabels, box_colors = [], [], [], []
        p = 1.0
        for res in results:
            for cls, color in [("pre", PRE_COLOR), ("int", INT_COLOR)]:
                vals = res[f"{cls}_{key}"]
                data.append(vals if len(vals) else [np.nan])
                box_colors.append(color)
                ticks.append(p)
                ticklabels.append(f"{res['subject'].replace('sub-','')}\n{cls}")
                p += 1
            p += 0.7
        bp = ax.boxplot(data, positions=ticks, widths=0.6, showfliers=False,
                        patch_artist=True)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Per-window signal strength: pre-ictal vs interictal\n"
                 "(per-recording z-scored — compare pre vs int WITHIN a subject, "
                 "not absolute level across subjects)", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[Signal] amplitude -> {save_path}")
    plt.show() if show else plt.close(fig)


# ── per-subject processing ────────────────────────────────────────────────────

def process_subject(subj, data_path, model_path, device, args, rng):
    data = load_preprocessed(str(data_path))
    x, y = data["X"], data["y"]
    channel_names = data["channel_names"]
    window_sec = data["window_sec"]

    models, meta = load_models(model_path, device)
    if meta.get("input_rep", "raw") != "raw":
        print(f"[Warn] {subj}: input_rep={meta.get('input_rep')} — Grad-CAM overlay "
              f"assumes a raw time-domain signal.")
    train_frac = float(meta.get("train_frac", 0.6))
    val_frac = float(meta.get("val_frac", 0.2))
    _, _, test_idx = subject_aware_split(data, train_frac, val_frac, meta.get("train_subjects"))

    subj_of = np.empty(len(x), dtype=object)
    for (s, e), p in zip(data["file_slices"], data["recording_paths"]):
        subj_of[s:e] = _subject_from_path(p)

    eval_mask = np.zeros(len(x), dtype=bool)
    eval_mask[test_idx] = True
    eval_mask &= (subj_of == subj)

    probs = np.full(len(x), np.nan, dtype=np.float32)
    test_pos = np.where(eval_mask)[0]
    probs[test_pos] = average_positive_prob(models, x[test_pos], device)

    y_test = y[test_pos]
    auc = (roc_auc_score(y_test, probs[test_pos])
           if len(np.unique(y_test)) > 1 else float("nan"))

    pre_mask = eval_mask & (y == 1)
    int_mask = eval_mask & (y == 0)
    pre_pos = select_windows(probs, pre_mask, args.n_windows, "preictal", rng)
    int_pos = select_windows(probs, int_mask, args.n_windows, args.interictal_pick, rng)

    x_pre = x[np.where(pre_mask)[0]]
    x_int = x[np.where(int_mask)[0]]
    ptp = lambda a: np.ptp(a, axis=2).mean(axis=1) if len(a) else np.array([])
    rms = lambda a: np.sqrt((a ** 2).mean(axis=2)).mean(axis=1) if len(a) else np.array([])

    gradcam_sidebyside(subj, x, probs, pre_pos, int_pos, models[0], device,
                       channel_names, window_sec, auc,
                       args.out / f"gradcam_sidebyside_{subj}.png", args.show)

    return dict(
        subject=subj, auc=auc, n_seizures=count_seizures(subj),
        n_test=int(eval_mask.sum()), n_pre=len(x_pre), n_int=len(x_int),
        x_pre=x_pre, x_int=x_int,
        pre_ptp=ptp(x_pre), int_ptp=ptp(x_int),
        pre_rms=rms(x_pre), int_rms=rms(x_int),
        channel_names=channel_names, window_sec=window_sec,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", type=str,
                    default="sub-034,sub-035,sub-039,sub-047,sub-060")
    ap.add_argument("--data-tmpl", type=str, default=DEFAULT_DATA_TMPL,
                    help="Path template with {subj}, relative to repo root.")
    ap.add_argument("--model-tmpl", type=str, default=DEFAULT_MODEL_TMPL,
                    help="Path template with {subj}, relative to repo root.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-windows", type=int, default=10,
                    help="Windows per class per subject in the Grad-CAM figure.")
    ap.add_argument("--interictal-pick", choices=["confident", "hard", "random"],
                    default="confident",
                    help="Which interictal windows to display: 'confident' = most "
                         "clearly interictal (low p), 'hard' = false alarms (high p), "
                         "'random' = representative.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.no_gpu)
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    for subj in subjects:
        data_path = REPO_ROOT / args.data_tmpl.format(subj=subj)
        model_path = REPO_ROOT / args.model_tmpl.format(subj=subj)
        if not data_path.exists():
            print(f"[Skip] {subj}: no npz at {data_path}")
            continue
        if not model_path.exists():
            print(f"[Skip] {subj}: no model at {model_path} (train it first)")
            continue
        print(f"\n=== {subj} ===")
        results.append(process_subject(subj, data_path, model_path, device, args, rng))

    if not results:
        print("\nNothing to do — no subject had both an npz and a model.")
        return

    # consistent channel names / window length come from the first subject
    results.sort(key=lambda r: (-(r["auc"] if not np.isnan(r["auc"]) else -1)))
    ch_names = results[0]["channel_names"]
    window_sec = results[0]["window_sec"]

    combined_meanwave(results, ch_names, window_sec,
                      args.out / "signal_strength_meanwave.png", args.show)
    combined_amplitude(results, args.out / "signal_strength_amplitude.png", args.show)

    # summary table + csv (sorted by AUC, descending)
    print(f"\n{'subject':<10}{'AUC':>7}{'seizures':>10}{'n_test':>8}{'n_pre':>7}"
          f"{'pre_ptp':>9}{'int_ptp':>9}{'pre_rms':>9}{'int_rms':>9}")
    rows = []
    for r in results:
        m = lambda a: float(np.nanmean(a)) if len(a) else float("nan")
        print(f"{r['subject']:<10}{r['auc']:>7.3f}{r['n_seizures']:>10}{r['n_test']:>8}"
              f"{r['n_pre']:>7}{m(r['pre_ptp']):>9.3f}{m(r['int_ptp']):>9.3f}"
              f"{m(r['pre_rms']):>9.3f}{m(r['int_rms']):>9.3f}")
        rows.append({
            "subject": r["subject"], "auc": round(r["auc"], 4),
            "n_seizures": r["n_seizures"], "n_test": r["n_test"],
            "n_preictal": r["n_pre"], "n_interictal": r["n_int"],
            "pre_ptp": round(m(r["pre_ptp"]), 4), "int_ptp": round(m(r["int_ptp"]), 4),
            "pre_rms": round(m(r["pre_rms"]), 4), "int_rms": round(m(r["int_rms"]), 4),
        })
    with open(args.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nDone. Figures + summary.csv in {args.out}")


if __name__ == "__main__":
    main()
