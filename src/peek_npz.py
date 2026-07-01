# -*- coding: utf-8 -*-
"""
Inspect the pre-ictal vs interictal windows of a preprocessed npz WITHOUT a model.

Use this right after preprocessing, before training, to (a) eyeball example
pre-ictal and interictal windows and (b) measure whether the two classes are
separable at all — purely from the signal. No trained model is required.

The "feature separability" (fAUC) is a model-free ceiling: if a simple feature
already separates pre from int (fAUC well above 0.5) there is learnable signal;
if every fAUC sits near 0.5 the classes are indistinguishable and a model is
unlikely to do better than chance for this subject.

Outputs (next to the npz, or --out):
  - peek_<name>_windows.png  : N pre-ictal vs N interictal example windows (raw)
  - peek_<name>_features.png : pre vs int distribution per character feature

Usage (from src/):
  python peek_npz.py --data ../data/processed/eeg_sub-055.npz
  python peek_npz.py --data ../data/processed/eeg_sub-055.npz --n-windows 8 --show
"""

import argparse
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess_common import load_preprocessed  # noqa: E402
from horizon_features import window_features, cohens_d, FEATURES  # noqa: E402

PRE_COLOR = "#c0392b"
INT_COLOR = "#2c6fbb"


def draw_window(ax, x_win, t_axis, channel_names, colors, title):
    n_ch = len(channel_names)
    rng = np.max(np.ptp(x_win[:n_ch], axis=1))
    spacing = (rng * 1.3) or 1.0
    offsets = np.arange(n_ch)[::-1] * spacing
    for c in range(n_ch):
        ax.plot(t_axis, x_win[c] + offsets[c], color=colors[c], linewidth=0.8)
    ax.set_xlim(t_axis[0], t_axis[-1])
    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_names, fontsize=6)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_title(title, fontsize=7)
    ax.grid(True, axis="x", alpha=0.2)


def windows_figure(x, pre_pos, int_pos, channel_names, window_sec, name, save_path, show):
    n_rows = max(len(pre_pos), len(int_pos), 1)
    t_axis = np.linspace(0, window_sec, x.shape[2])
    colors = cm.tab10(np.linspace(0, 1, len(channel_names)))
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 1.7 * n_rows), squeeze=False)
    for col, (positions, prefix) in enumerate([(pre_pos, "pre"), (int_pos, "int")]):
        for row in range(n_rows):
            ax = axes[row, col]
            if row >= len(positions):
                ax.set_visible(False); continue
            gi = positions[row]
            draw_window(ax, x[gi], t_axis, channel_names, colors, f"{prefix} #{gi}")
    axes[0, 0].annotate("PRE-ICTAL", xy=(0.5, 1.0), xytext=(0.5, 1.5),
                        xycoords="axes fraction", ha="center", fontsize=11,
                        color=PRE_COLOR, weight="bold")
    axes[0, 1].annotate("INTERICTAL", xy=(0.5, 1.0), xytext=(0.5, 1.5),
                        xycoords="axes fraction", ha="center", fontsize=11,
                        color=INT_COLOR, weight="bold")
    fig.suptitle(f"{name}: random pre-ictal vs interictal windows (from npz, no model)",
                 fontsize=12, y=0.995)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Windows] -> {save_path}")
    plt.show() if show else plt.close(fig)


def features_figure(feats, y, name, save_path, show):
    pre, intc = y == 1, y == 0
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(3.2 * len(FEATURES), 4))
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, FEATURES):
        v = feats[key]
        bp = ax.boxplot([v[pre], v[intc]], positions=[1, 2], widths=0.6,
                        showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], [PRE_COLOR, INT_COLOR]):
            patch.set_facecolor(color); patch.set_alpha(0.45)
        for med in bp["medians"]:
            med.set_color("black")
        fauc = roc_auc_score(y, v)
        fauc = max(fauc, 1 - fauc)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["pre", "int"], fontsize=9)
        ax.set_title(f"{title}\nfAUC {fauc:.2f}  |  d {cohens_d(v[pre], v[intc]):+.2f}",
                     fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"{name}: signal character, pre-ictal vs interictal "
                 f"(fAUC = model-free separability)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[Features] -> {save_path}")
    plt.show() if show else plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="Path to a preprocessed npz.")
    ap.add_argument("--n-windows", type=int, default=10, help="Example windows per class.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: alongside the npz).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    data = load_preprocessed(str(args.data))
    x, y = data["X"], data["y"]
    channel_names = data["channel_names"]
    window_sec = data["window_sec"]
    name = args.data.stem
    out = args.out or args.data.parent
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pre_idx = np.where(y == 1)[0]
    int_idx = np.where(y == 0)[0]
    print(f"\n{name}: {len(x)} windows  |  pre-ictal {len(pre_idx)}  interictal {len(int_idx)}")
    if len(pre_idx) == 0 or len(int_idx) == 0:
        print("[Warn] one class is empty — cannot compare."); return

    pre_pos = rng.choice(pre_idx, min(args.n_windows, len(pre_idx)), replace=False)
    int_pos = rng.choice(int_idx, min(args.n_windows, len(int_idx)), replace=False)
    windows_figure(x, np.sort(pre_pos), np.sort(int_pos), channel_names, window_sec,
                   name, out / f"peek_{name}_windows.png", args.show)

    feats = window_features(x)
    features_figure(feats, y, name, out / f"peek_{name}_features.png", args.show)

    print(f"\n{'feature':<18}{'pre_mean':>10}{'int_mean':>10}{'cohens_d':>10}{'fAUC':>7}")
    for key, _ in FEATURES:
        v = feats[key]
        fauc = roc_auc_score(y, v); fauc = max(fauc, 1 - fauc)
        print(f"{key:<18}{v[y==1].mean():>10.4f}{v[y==0].mean():>10.4f}"
              f"{cohens_d(v[y==1], v[y==0]):>10.2f}{fauc:>7.2f}")
    print(f"\nDone. Figures in {out}")


if __name__ == "__main__":
    main()
