# -*- coding: utf-8 -*-
"""
Evaluation + explainability stage for the EEG-only seizure-detection pipeline.

Loads a trained model (from train_model_eeg.py) and the preprocessed dataset,
reconstructs exactly the same held-out 20% test set (using the train_frac stored
in the checkpoint), and:
  - averages class probabilities across ensemble members (soft voting),
  - applies temporal post-processing (removes short isolated positive runs),
  - computes precision / recall / F1 / confusion matrix,
  - appends one row of metrics + config to results/.../metrics.csv,
  - writes diagnostic plots (average seizure morphology, EEG + HR overlay,
    standalone heart-rate plot) and a Grad-CAM explainability figure.

Example:
  python evaluate_eeg.py
  python evaluate_eeg.py --pred-threshold 0.4 --start 1600 --duration 120
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from captum.attr import LayerGradCam

from preprocess_eeg import (
    DEFAULT_PREPROCESSED_PATH,
    POWERLINE_NOTCH_HZ,
    apply_notch_filter,
    load_preprocessed,
    parse_seizure_intervals,
    resolve_channels,
)
from train_model_eeg import SeizureCNN, DEFAULT_MODEL_PATH, pick_device

RESULTS_DIR = Path(__file__).resolve().parent / "../../results/seizure_detection_eeg"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(path: Path, device: torch.device) -> tuple[list[SeizureCNN], dict]:
    """Load one or more trained models (ensemble) from a checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    meta = checkpoint["meta"]
    state_dicts = checkpoint["state_dicts"]
    models: list[SeizureCNN] = []
    for sd in state_dicts:
        m = SeizureCNN(meta["n_channels"], meta["n_timepoints"]).to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    print(f"[Model] Loaded {len(models)} model(s) from {path}  "
          f"(channels={meta['n_channels']}, T={meta['n_timepoints']}, "
          f"train_frac={meta.get('train_frac')})")
    return models, meta


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_positive_prob(
        model: SeizureCNN, x: np.ndarray, device: torch.device, batch_size: int = 256
) -> np.ndarray:
    """Return P(class=1) for each window."""
    model.eval()
    dl = DataLoader(TensorDataset(torch.from_numpy(x).to(device)), batch_size=batch_size)
    probs = []
    for (xb,) in dl:
        logits = model(xb)
        p1 = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p1.cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


def average_positive_prob(
        models: list[SeizureCNN], x: np.ndarray, device: torch.device, batch_size: int = 256
) -> np.ndarray:
    """Average class-1 probabilities across all models (soft voting)."""
    probs = np.stack(
        [predict_positive_prob(m, x, device, batch_size) for m in models], axis=0
    )
    return probs.mean(axis=0)


# ── Post-processing ───────────────────────────────────────────────────────────

def smooth_binary_predictions(pred: np.ndarray, min_run: int = 2) -> np.ndarray:
    """Remove isolated positive runs shorter than *min_run* windows."""
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


def windows_to_intervals(
        centers: np.ndarray, pred: np.ndarray, window_sec: float
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    i, n = 0, len(pred)
    while i < n:
        if pred[i] == 0:
            i += 1
            continue
        j = i
        while j < n and pred[j] == 1:
            j += 1
        start = centers[i] - 0.5 * window_sec
        end = centers[j - 1] + 0.5 * window_sec
        intervals.append((max(0.0, start), max(start, end)))
        i = j
    return intervals


def compute_prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(p), float(r), float(f)


def has_overlap_with_window(
        intervals: list[tuple[float, float]], start: float, end: float
) -> bool:
    """True when any interval overlaps the [start, end] plot window."""
    return any((se > start) and (ss < end) for ss, se in intervals)


# ── Metrics logging ───────────────────────────────────────────────────────────

def append_metrics_csv(csv_path: Path, row: dict) -> None:
    """Append one metrics row to a CSV, writing the header if the file is new."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[Metrics] Appended run to {csv_path}")


# ── Heart rate generation (synthetic, visualization only) ─────────────────────

# Physiological reference values for a 25-35 year-old adult.
_HR_BASELINE_BPM   = 68.0
_HR_PEAK_BPM       = 145.0
_HR_BASELINE_NOISE = 2.5
_HR_ICTAL_NOISE    = 6.0
_HR_RISE_TAU       = 15.0
_HR_FALL_TAU       = 25.0


def _log_ramp(t_elapsed: float, tau: float) -> float:
    if tau <= 0:
        return 1.0
    return float(np.clip(np.log1p(t_elapsed / tau) / np.log(2.0), 0.0, 1.0))


def generate_heart_rate(
        total_duration: float,
        seizure_intervals: list[tuple[float, float]],
        fs: float = 1.0,
        rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic heart-rate time-series (visualization aid; not a model input)."""
    if rng is None:
        rng = np.random.default_rng(42)

    times = np.arange(0.0, total_duration, 1.0 / fs)
    n = len(times)
    hr = np.full(n, _HR_BASELINE_BPM, dtype=np.float64)
    delta = _HR_PEAK_BPM - _HR_BASELINE_BPM

    for t_idx, t in enumerate(times):
        ictal = False
        postictal_frac = 0.0
        for ss, se in seizure_intervals:
            if ss <= t <= se:
                frac = _log_ramp(t - ss, _HR_RISE_TAU)
                hr[t_idx] = _HR_BASELINE_BPM + delta * frac
                ictal = True
                break
            elif t > se:
                later = any(ss2 <= t for ss2, _ in seizure_intervals if ss2 > se)
                if not later:
                    frac = _log_ramp(t - se, _HR_FALL_TAU)
                    postictal_level = _HR_PEAK_BPM - delta * frac
                    postictal_frac = max(postictal_frac, postictal_level)
        if not ictal and postictal_frac > 0.0:
            hr[t_idx] = max(_HR_BASELINE_BPM, postictal_frac)

    noise_amp = np.where(hr > _HR_BASELINE_BPM + 10, _HR_ICTAL_NOISE, _HR_BASELINE_NOISE)
    hr += rng.normal(0.0, noise_amp)
    hr = np.clip(hr, 40.0, 200.0)
    return times, hr


def plot_heart_rate_with_seizures(
        times: np.ndarray,
        hr: np.ndarray,
        seizure_intervals: list[tuple[float, float]],
        predicted_intervals: list[tuple[float, float]],
        plot_start: float,
        plot_duration: float,
        save_path: Path,
        show: bool,
        title_suffix: str = "",
) -> None:
    plot_end = plot_start + plot_duration
    mask = (times >= plot_start) & (times <= plot_end)
    t_win = times[mask]
    hr_win = hr[mask]

    fig, ax = plt.subplots(figsize=(13, 3.5))
    for ss, se in seizure_intervals:
        x0, x1 = max(ss, plot_start), min(se, plot_end)
        if x1 > x0:
            ax.axvspan(x0, x1, color="#e74c3c", alpha=0.18, label="_gt")
    for ps, pe in predicted_intervals:
        x0, x1 = max(ps, plot_start), min(pe, plot_end)
        if x1 > x0:
            ax.axvspan(x0, x1, color="#2980b9", alpha=0.15, label="_pred")

    from matplotlib.collections import LineCollection
    points = np.array([t_win, hr_win]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(40, 170)
    lc = LineCollection(segments, cmap="RdYlGn_r", norm=norm, linewidth=2.0, zorder=3)
    lc.set_array(hr_win)
    ax.add_collection(lc)

    ax.axhline(_HR_BASELINE_BPM, color="#27ae60", linewidth=0.8,
               linestyle="--", alpha=0.7, label=f"Resting baseline ({_HR_BASELINE_BPM:.0f} bpm)")
    ax.axhline(_HR_PEAK_BPM, color="#e74c3c", linewidth=0.8,
               linestyle="--", alpha=0.7, label=f"Ictal peak ({_HR_PEAK_BPM:.0f} bpm)")

    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label("HR (bpm)", fontsize=8)

    legend_handles = [
        mpatches.Patch(color="#e74c3c", alpha=0.5, label="Ground truth seizure"),
        mpatches.Patch(color="#2980b9", alpha=0.5, label="CNN predicted seizure"),
        plt.Line2D([0], [0], color="#27ae60", linewidth=1.2, linestyle="--",
                   label=f"Resting baseline ({_HR_BASELINE_BPM:.0f} bpm)"),
        plt.Line2D([0], [0], color="#e74c3c", linewidth=1.2, linestyle="--",
                   label=f"Ictal peak ({_HR_PEAK_BPM:.0f} bpm)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.85, edgecolor="none")

    ax.set_xlim(plot_start, plot_end)
    ax.set_ylim(40, 180)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Heart rate (bpm)", fontsize=9)
    title = "Synthetic heart rate  |  age 25–35 reference"
    if title_suffix:
        title += f"  |  {title_suffix}"
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    save_path = RESULTS_DIR / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[HeartRate] Saved standalone HR plot to {save_path}")
    plt.show() if show else plt.close(fig)


# ── Average seizure morphology plot ──────────────────────────────────────────

def plot_average_seizure(
        x: np.ndarray,
        y: np.ndarray,
        channel_names: list[str],
        window_sec: float,
        save_path: Path,
        show: bool,
        max_channels: int = 6,
) -> None:
    """Mean ± 1 SD of every labelled seizure window, one subplot per channel."""
    seiz_wins = x[y == 1]
    if len(seiz_wins) == 0:
        print("[AvgSeizure] No seizure windows found — skipping plot.")
        return

    n_ch = min(len(channel_names), max_channels)
    t_axis = np.linspace(0, window_sec, seiz_wins.shape[2])
    mean = seiz_wins[:, :n_ch, :].mean(axis=0)
    std = seiz_wins[:, :n_ch, :].std(axis=0)

    cols = min(n_ch, 3)
    rows = (n_ch + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), sharex=True)
    axes = np.array(axes).flatten()

    for i in range(n_ch):
        ax = axes[i]
        ax.plot(t_axis, mean[i], color="#c0392b", linewidth=1.5, label="Mean")
        ax.fill_between(t_axis, mean[i] - std[i], mean[i] + std[i],
                        color="#c0392b", alpha=0.20, label="±1 SD")
        ax.set_title(channel_names[i], fontsize=9)
        ax.set_ylabel("z-score", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    for j in range(n_ch, len(axes)):
        axes[j].set_visible(False)
    for ax in axes[(rows - 1) * cols: (rows - 1) * cols + cols]:
        ax.set_xlabel("Time within window (s)", fontsize=8)

    fig.suptitle(
        f"Average seizure morphology  |  n={len(seiz_wins)} windows  |  {window_sec:.1f}s window",
        fontsize=11,
    )
    fig.tight_layout()
    save_path = RESULTS_DIR / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[AvgSeizure] Saved to {save_path}")
    plt.show() if show else plt.close(fig)


# ── EEG overlay plot (with HR subplot) ───────────────────────────────────────

def plot_results(
        raw: mne.io.BaseRaw,
        seizure_intervals: list[tuple[float, float]],
        predicted_intervals: list[tuple[float, float]],
        start: float,
        duration: float,
        channels_arg: str,
        max_channels: int,
        save_path: Path,
        show: bool,
        title_suffix: str = "",
        heart_rate_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    sfreq = float(raw.info["sfreq"])
    total_duration = raw.n_times / sfreq
    if start < 0 or start >= total_duration:
        raise ValueError("Invalid --start for plot.")
    if duration <= 0:
        raise ValueError("--duration must be > 0")

    end = min(start + duration, total_duration)
    ch = resolve_channels(raw, channels_arg, max_channels)

    cropped = raw.copy().pick(ch).crop(tmin=start, tmax=end, include_tmax=False)
    data, times = cropped.get_data(return_times=True)
    data_uv = data * 1e6

    has_hr = heart_rate_data is not None
    if has_hr:
        fig, (ax_eeg, ax_hr) = plt.subplots(
            2, 1, figsize=(13, 9),
            gridspec_kw={"height_ratios": [3, 1]}, sharex=False,
        )
    else:
        fig, ax_eeg = plt.subplots(figsize=(13, 7))

    spacing = np.max(np.ptp(data_uv, axis=1)) * 1.2 or 1.0
    offsets = np.arange(len(ch))[::-1] * spacing
    channel_colors = cm.tab10(np.linspace(0, 1, len(ch)))

    for i in range(len(ch)):
        ax_eeg.plot(times + start, data_uv[i] + offsets[i],
                    color=channel_colors[i], linewidth=0.8)
    for ss, se in seizure_intervals:
        x0, x1 = max(ss, start), min(se, end)
        if x1 > x0:
            ax_eeg.axvspan(x0, x1, color="red", alpha=0.18)
    for ps, pe in predicted_intervals:
        x0, x1 = max(ps, start), min(pe, end)
        if x1 > x0:
            ax_eeg.axvspan(x0, x1, color="blue", alpha=0.15)

    legend_handles = [
        mpatches.Patch(color="red",  alpha=0.5, label="Ground truth seizure"),
        mpatches.Patch(color="blue", alpha=0.5, label="CNN predicted seizure"),
    ]
    for i, name in enumerate(ch):
        legend_handles.append(
            plt.Line2D([0], [0], color=channel_colors[i], linewidth=1.5, label=name)
        )
    ax_eeg.legend(handles=legend_handles, loc="upper right", fontsize=8,
                  framealpha=0.85, edgecolor="none", ncols=2)
    ax_eeg.set_yticks(offsets)
    ax_eeg.set_yticklabels(ch)
    ax_eeg.set_xlabel("Time (s)")
    ax_eeg.set_ylabel("Channel")
    title = f"CNN seizure detector  |  {start:.1f}s – {end:.1f}s"
    if title_suffix:
        title += f"  |  {title_suffix}"
    ax_eeg.set_title(title)
    ax_eeg.grid(True, axis="x", alpha=0.3)

    if has_hr:
        hr_times, hr_vals = heart_rate_data
        mask = (hr_times >= start) & (hr_times <= end)
        t_win = hr_times[mask]
        hr_win = hr_vals[mask]
        for ss, se in seizure_intervals:
            x0, x1 = max(ss, start), min(se, end)
            if x1 > x0:
                ax_hr.axvspan(x0, x1, color="#e74c3c", alpha=0.18)
        for ps, pe in predicted_intervals:
            x0, x1 = max(ps, start), min(pe, end)
            if x1 > x0:
                ax_hr.axvspan(x0, x1, color="#2980b9", alpha=0.15)

        from matplotlib.collections import LineCollection
        points = np.array([t_win, hr_win]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(40, 170)
        lc = LineCollection(segments, cmap="RdYlGn_r", norm=norm, linewidth=2.2, zorder=3)
        lc.set_array(hr_win)
        ax_hr.add_collection(lc)

        ax_hr.axhline(_HR_BASELINE_BPM, color="#27ae60", linewidth=0.9,
                      linestyle="--", alpha=0.8, label=f"Resting {_HR_BASELINE_BPM:.0f} bpm")
        ax_hr.axhline(_HR_PEAK_BPM, color="#e74c3c", linewidth=0.9,
                      linestyle="--", alpha=0.8, label=f"Ictal peak {_HR_PEAK_BPM:.0f} bpm")
        ax_hr.set_xlim(start, end)
        ax_hr.set_ylim(40, 185)
        ax_hr.set_xlabel("Time (s)", fontsize=9)
        ax_hr.set_ylabel("Heart rate (bpm)", fontsize=9)
        ax_hr.set_title("Synthetic heart rate  |  age 25–35 reference", fontsize=9)
        ax_hr.legend(fontsize=8, loc="upper right", framealpha=0.85, edgecolor="none")
        ax_hr.grid(True, axis="x", alpha=0.3)
        ax_hr.grid(True, axis="y", alpha=0.2)
        sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_hr, pad=0.01, fraction=0.02)
        cbar.set_label("bpm", fontsize=8)

    fig.tight_layout()
    save_path = RESULTS_DIR / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[Plot] Saved to {save_path}")
    plt.show() if show else plt.close(fig)


# ── Grad-CAM ─────────────────────────────────────────────────────────────────

def plot_gradcam(
        model: SeizureCNN,
        x_test: np.ndarray,
        y_test: np.ndarray,
        channel_names: list[str],
        window_sec: float,
        n_samples: int,
        save_path: Path,
        device: torch.device,
        show: bool,
) -> None:
    seizure_windows = np.where(y_test == 1)[0]
    if len(seizure_windows) == 0:
        print("[GradCAM] No seizure windows in test set — skipping.")
        return

    probabilities = predict_positive_prob(model, x_test, device)
    seizure_probabilities = probabilities[seizure_windows]
    best_seizure_windows = seizure_windows[
        np.argsort(seizure_probabilities)[::-1][:n_samples]
    ]

    grad_cam = LayerGradCam(model, model.conv_block[10])
    model.eval()
    channels_to_plot = min(len(channel_names), 6)
    time_axis = np.linspace(0, window_sec, x_test.shape[2])
    colors = cm.tab10(np.linspace(0, 1, channels_to_plot))

    fig, axes = plt.subplots(
        len(best_seizure_windows), 1,
        figsize=(11, 3.5 * len(best_seizure_windows)), squeeze=False,
    )

    for row, window_index in enumerate(best_seizure_windows):
        ax = axes[row, 0]
        input_window = torch.from_numpy(
            x_test[window_index: window_index + 1]
        ).to(device).requires_grad_(True)
        attribution = grad_cam.attribute(input_window, target=1)
        heatmap = attribution.mean(dim=1).squeeze(0).detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        heatmap_time_axis = np.linspace(0, window_sec, len(heatmap))
        heatmap_resized = np.interp(time_axis, heatmap_time_axis, heatmap)

        for k in range(len(time_axis) - 1):
            ax.axvspan(time_axis[k], time_axis[k + 1],
                       color=(1.0, 0.2, 0.2, float(heatmap_resized[k]) * 0.5), linewidth=0)

        signal_range = np.max(np.ptp(x_test[window_index, :channels_to_plot], axis=1))
        spacing = signal_range * 1.3 or 1.0
        offsets = np.arange(channels_to_plot)[::-1] * spacing
        for channel_index in range(channels_to_plot):
            ax.plot(time_axis, x_test[window_index, channel_index] + offsets[channel_index],
                    color=colors[channel_index], linewidth=0.9,
                    label=channel_names[channel_index] if channel_index < len(channel_names)
                    else f"ch{channel_index}")

        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_yticks(offsets)
        ax.set_yticklabels(channel_names[:channels_to_plot], fontsize=8)
        ax.set_xlabel("Time within window (s)", fontsize=9)
        ax.set_title(f"Grad-CAM | test window #{window_index} | "
                     f"p(seizure)={probabilities[window_index]:.2f}", fontsize=10)
        ax.grid(True, axis="x", alpha=0.25)

    colorbar_source = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(0, 1))
    colorbar_source.set_array([])
    fig.colorbar(colorbar_source, ax=axes[:, 0], fraction=0.015, pad=0.02,
                 label="Grad-CAM intensity")
    fig.suptitle("Grad-CAM: most confident seizure predictions", fontsize=11)
    fig.tight_layout()
    save_path = RESULTS_DIR / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[GradCAM] Saved to {save_path}")
    plt.show() if show else plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained EEG seizure-detection model on the held-out test set."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_PREPROCESSED_PATH,
                        help="Path to the preprocessed dataset (.npz).")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--train-frac", type=float, default=None,
                        help="Override the split fraction. Default: use the value stored "
                             "in the checkpoint (so the test set matches training).")
    parser.add_argument("--feature-set", type=str, default="eeg",
                        help="Label written to metrics.csv (e.g. 'eeg' or 'eeg_ecg').")
    parser.add_argument("--pred-threshold", type=float, default=0.5,
                        help="Positive-class threshold (0-1). Lower = more detections.")
    parser.add_argument("--pred-min-run", type=int, default=2,
                        help="Minimum consecutive positive windows to keep after thresholding.")
    parser.add_argument("--no-gpu", action="store_true")
    # plotting
    parser.add_argument("--plot-edf", type=Path, default=None,
                        help="Which EDF to plot. Defaults to the last file with seizures.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--max-channels", type=int, default=8)
    parser.add_argument("--channels", type=str, default="")
    parser.add_argument("--save", type=Path, default=Path("eeg_overlay.png"))
    parser.add_argument("--save-seizure-plot", type=Path, default=Path("train_detect_chb01.png"))
    parser.add_argument("--save-hr-plot", type=Path, default=Path("heart_rate_seizures.png"))
    parser.add_argument("--save-gradcam-plot", type=Path, default=Path("gradcam.png"))
    parser.add_argument("--gradcam-n-samples", type=int, default=4)
    parser.add_argument("--metrics-csv", type=Path, default=RESULTS_DIR / "metrics.csv")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    if not (0.0 < args.pred_threshold < 1.0):
        raise ValueError("--pred-threshold must be in (0, 1)")
    if args.pred_min_run < 1:
        raise ValueError("--pred-min-run must be >= 1")

    device = pick_device(args.no_gpu)

    # ── load data + model ──
    print(f"\n[Data] Loading preprocessed dataset: {args.data}")
    data = load_preprocessed(args.data)
    x, y, centers = data["X"], data["y"], data["centers"]
    channel_names = data["channel_names"]
    window_sec = data["window_sec"]
    edf_paths = data["edf_paths"]
    summary_map = dict(zip(edf_paths, data["summary_paths"]))
    file_slices = data["file_slices"]

    models, meta = load_models(args.model, device)
    train_frac = args.train_frac if args.train_frac is not None else float(meta.get("train_frac", 0.8))

    # ── reconstruct the SAME chronological split as training ──
    split = int(len(x) * train_frac)
    split = max(10, min(split, len(x) - 10))
    x_test, y_test = x[split:], y[split:]
    centers_test = centers[split:]
    print(f"\nSplit train_frac={train_frac}  ->  test windows={len(x_test)}  "
          f"(seizure: {int(y_test.sum())})")
    test_file_names = [edf_paths[i].name for i, (fs, fe) in enumerate(file_slices) if fe > split]
    print(f"Test set spans: {test_file_names}\n")

    # ── average probabilities (soft voting), threshold, post-process ──
    prob_test = average_positive_prob(models, x_test, device)
    pred_test = (prob_test >= args.pred_threshold).astype(int)
    pred_test = smooth_binary_predictions(pred_test, min_run=args.pred_min_run)

    precision, recall, f1 = compute_prf(y_test, pred_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test, labels=[0, 1]).ravel()

    label = f"Ensemble ({len(models)} models)" if len(models) > 1 else "SeizureCNN (1-D Conv)"
    print(f"Model: {label}")
    print(f"Test Precision : {precision:.3f}")
    print(f"Test Recall    : {recall:.3f}")
    print(f"Test F1        : {f1:.3f}")
    print(f"Confusion [tn fp fn tp]: [{tn} {fp} {fn} {tp}]")

    # ── log metrics row ──
    append_metrics_csv(args.metrics_csv, {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "feature_set": args.feature_set,
        "train_frac": train_frac,
        "epochs": meta.get("epochs"),
        "random_state": meta.get("random_state"),
        "n_runs": meta.get("n_runs", len(models)),
        "pred_threshold": args.pred_threshold,
        "pred_min_run": args.pred_min_run,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_test": len(y_test), "n_test_seizure": int(y_test.sum()),
        "model_path": str(args.model),
    })

    # ── average seizure morphology (over all labelled seizure windows) ──
    plot_average_seizure(x, y, channel_names, window_sec=window_sec,
                         save_path=args.save_seizure_plot, show=args.show,
                         max_channels=args.max_channels)

    # ── pick the EDF to plot (last file with seizures by default) ──
    if args.plot_edf:
        plot_edf_path = args.plot_edf
    else:
        seizure_files = [p for p in edf_paths if parse_seizure_intervals(summary_map[p], p.name)]
        plot_edf_path = seizure_files[-1] if seizure_files else edf_paths[-1]
        if seizure_files:
            print(f"[Plot] Auto-selected file with seizures: {plot_edf_path.name}")
    plot_file_idx = next((i for i, p in enumerate(edf_paths) if p == plot_edf_path),
                         len(edf_paths) - 1)

    fs, fe = file_slices[plot_file_idx]
    t0 = max(fs, split) - split
    t1 = max(fe, split) - split
    if t0 < t1 and t1 <= len(pred_test):
        plot_pred = pred_test[t0:t1]
        plot_centers = centers_test[t0:t1]
    else:
        plot_pred = np.zeros(0, dtype=int)
        plot_centers = np.zeros(0, dtype=float)
        print(f"[Plot] {plot_edf_path.name} is entirely in the training set; "
              "no test predictions to overlay.")

    predicted_intervals = windows_to_intervals(plot_centers, plot_pred, window_sec)
    plot_seizures = parse_seizure_intervals(summary_map[plot_edf_path], plot_edf_path.name)
    raw_plot = mne.io.read_raw_edf(str(plot_edf_path), preload=True, verbose="ERROR")
    apply_notch_filter(raw_plot, POWERLINE_NOTCH_HZ)
    print(f"[Plot] {plot_edf_path.name}: GT intervals={len(plot_seizures)}  "
          f"Pred intervals={len(predicted_intervals)}")

    plot_start = args.start
    plot_end = min(plot_start + args.duration, raw_plot.n_times / float(raw_plot.info["sfreq"]))
    has_visible_event = (has_overlap_with_window(plot_seizures, plot_start, plot_end)
                         or has_overlap_with_window(predicted_intervals, plot_start, plot_end))
    if (not has_visible_event) and args.start == 0.0:
        if plot_seizures:
            plot_start = max(0.0, min(s for s, _ in plot_seizures) - 10.0)
            print(f"[Plot] No events in 0-{plot_end:.1f}s; auto-shifting to start={plot_start:.1f}s (GT).")
        elif predicted_intervals:
            plot_start = max(0.0, min(s for s, _ in predicted_intervals) - 10.0)
            print(f"[Plot] No events in 0-{plot_end:.1f}s; auto-shifting to start={plot_start:.1f}s (pred).")

    total_dur = raw_plot.n_times / float(raw_plot.info["sfreq"])
    print(f"[HeartRate] Generating synthetic HR signal for {plot_edf_path.name} ({total_dur:.0f}s) …")
    hr_times, hr_vals = generate_heart_rate(
        total_duration=total_dur, seizure_intervals=plot_seizures,
        fs=4.0, rng=np.random.default_rng(int(meta.get("random_state", 42))),
    )

    plot_results(raw=raw_plot, seizure_intervals=plot_seizures,
                 predicted_intervals=predicted_intervals, start=plot_start,
                 duration=args.duration, channels_arg=args.channels,
                 max_channels=args.max_channels, save_path=args.save, show=args.show,
                 title_suffix=plot_edf_path.name, heart_rate_data=(hr_times, hr_vals))

    plot_heart_rate_with_seizures(times=hr_times, hr=hr_vals,
                                  seizure_intervals=plot_seizures,
                                  predicted_intervals=predicted_intervals,
                                  plot_start=plot_start, plot_duration=args.duration,
                                  save_path=args.save_hr_plot, show=args.show,
                                  title_suffix=plot_edf_path.name)

    plot_gradcam(model=models[0], x_test=x_test, y_test=y_test,
                 channel_names=channel_names, window_sec=window_sec,
                 n_samples=args.gradcam_n_samples, save_path=args.save_gradcam_plot,
                 device=device, show=args.show)


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
