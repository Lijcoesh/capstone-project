# -*- coding: utf-8 -*-
"""
Seizure detection on CHB-MIT EEG data — multi-file edition.

  - Accepts any number of EDF files via --edf (space-separated).
  - Each EDF's seizure annotations are read from its own subject summary file,
    auto-detected as <edf_parent>/<subject>-summary.txt  (e.g. chb01/chb01-summary.txt).
    Override with --summaries to supply explicit paths (one per unique subject folder).
  - Windows from all files are concatenated in chronological order.
  - Train/test split is made on that concatenated sequence (no leakage).
  - Only channels present in ALL files are used (silent intersection).
  - Model: 1-D CNN in PyTorch; GPU auto-detected (CUDA → MPS → CPU).
  - Trained model saved with torch.save (--save-model / --load-model).
  - Ensemble training via --ensemble-runs N (majority-vote predictions).
  - Average seizure morphology plot saved to --save-seizure-plot.
  - Synthetic heart rate (resting ~60-75 bpm, seizure ~130-160 bpm) is generated
    using log-curve ramps around each seizure interval and overlaid on the EEG plot.

WSL2 GPU setup (one-time):
  1. Install the NVIDIA driver on Windows (>=512.xx for CUDA 12).
  2. Inside WSL2:
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  3. Verify:  python -c "import torch; print(torch.cuda.is_available())"

Command Examples:
  - python train_detect_seizures.py --epochs=5
  - python train_detect_seizures.py --edf chb01/chb01_03.edf chb02/chb02_16.edf --epochs=10
  - python train_detect_seizures.py --summaries chb01/chb01-summary.txt chb02/chb02-summary.txt
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mne
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from captum.attr import LayerGradCam

from typing import List, Tuple, Dict, Optional

# ── Defaults ──────────────────────────────────────────────────────────────────

_BASE = Path("physionet.org/files/chbmit/1.0.0/")

DEFAULT_EDFS = [
    _BASE / "chb01/chb01_03.edf",
    _BASE / "chb01/chb01_04.edf",
    _BASE / "chb01/chb01_15.edf",
    _BASE / "chb01/chb01_16.edf",
    _BASE / "chb01/chb01_18.edf",
    _BASE / "chb01/chb01_21.edf",
    _BASE / "chb01/chb01_26.edf",
    _BASE / "chb02/chb02_16.edf",
    _BASE / "chb02/chb02_19.edf",
    _BASE / "chb03/chb03_01.edf",
    _BASE / "chb03/chb03_02.edf",
    _BASE / "chb03/chb03_03.edf",
    _BASE / "chb03/chb03_04.edf",
    _BASE / "chb03/chb03_34.edf",
    _BASE / "chb03/chb03_35.edf",
    _BASE / "chb03/chb03_36.edf",
    _BASE / "chb04/chb04_05.edf",
    _BASE / "chb04/chb04_08.edf",
    _BASE / "chb04/chb04_28.edf",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    model_name: str = ""


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GPU-accelerated CNN seizure detector on multiple EDF files."
    )
    # ── data ──
    parser.add_argument(
        "--edf", type=Path, nargs="+", default=DEFAULT_EDFS,
        metavar="EDF",
        help="One or more EDF files (chronological order). "
             "Default: the 19 files spanning chb01–chb04.",
    )
    parser.add_argument(
        "--summaries", type=Path, nargs="+", default=None,
        metavar="SUMMARY",
        help=(
            "Explicit summary file(s) to use. Supply one path per unique subject "
            "folder (e.g. chb01/chb01-summary.txt chb02/chb02-summary.txt). "
            "When omitted, the summary is auto-detected from each EDF's parent "
            "directory as <parent>/<subject>-summary.txt."
        ),
    )
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    # ── plot (applied to the last file by default) ──
    parser.add_argument("--plot-edf", type=Path, default=None,
                        help="Which EDF to plot. Defaults to the last --edf file with seizures.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--max-channels", type=int, default=8)
    parser.add_argument("--channels", type=str, default="")
    parser.add_argument("--save", type=Path, default=Path("eeg_overlay.png"))
    parser.add_argument("--show", action="store_true")
    # ── training ──
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pred-threshold", type=float, default=0.5,
                        help="Positive-class threshold for seizure prediction (0-1). Lower = more detections.")
    parser.add_argument("--pred-min-run", type=int, default=2,
                        help="Minimum consecutive positive windows to keep after thresholding.")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    # ── model I/O ──
    parser.add_argument("--save-model", type=Path, default=Path("seizure_cnn.pt"))
    parser.add_argument("--load-model", type=Path, default=None)
    # ── ensemble / average seizure plot ──
    parser.add_argument("--ensemble-runs", type=int, default=1,
                        help="Train N models and majority-vote predictions (default: 1 = no ensemble).")
    parser.add_argument("--save-seizure-plot", type=Path, default=Path("train_detect_chb01.png"),
                        help="Output path for the average seizure morphology plot (all files).")
    # ── heart rate plot ──
    parser.add_argument("--save-hr-plot", type=Path, default=Path("heart_rate_seizures.png"),
                        help="Output path for the standalone heart rate + seizure plot.")
    # ── grad-cam ──
    parser.add_argument("--save-gradcam-plot", type=Path, default=Path("gradcam.png"),
                        help="Output path for the Grad-CAM explanation plot.")
    parser.add_argument("--gradcam-n-samples", type=int, default=4,
                        help="Number of top seizure windows to explain with Grad-CAM.")
    return parser.parse_args()


# ── Summary resolution ────────────────────────────────────────────────────────

def _auto_summary_for(edf_path: Path) -> Path:
    """
    Derive the canonical summary path from an EDF path.

    Convention used by CHB-MIT:
        <base>/<subject>/<subject>-summary.txt
    e.g. physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf
      →  physionet.org/files/chbmit/1.0.0/chb01/chb01-summary.txt
    """
    subject_dir = edf_path.parent          # …/chb01
    subject_id  = subject_dir.name         # chb01
    return subject_dir / f"{subject_id}-summary.txt"


def build_summary_map(
    edf_paths: list[Path],
    explicit_summaries: list[Path] | None,
) -> dict[Path, Path]:
    """
    Return a mapping  edf_path → summary_path  for every EDF in *edf_paths*.
    """
    explicit_map: dict[Path, Path] = {}
    if explicit_summaries:
        for sp in explicit_summaries:
            sp_resolved = sp.resolve()
            if not sp_resolved.exists():
                raise FileNotFoundError(f"Summary file not found: {sp}")
            explicit_map[sp_resolved.parent] = sp_resolved

    result: dict[Path, Path] = {}
    for edf in edf_paths:
        edf_parent = edf.resolve().parent
        if edf_parent in explicit_map:
            result[edf] = explicit_map[edf_parent]
        else:
            auto = _auto_summary_for(edf)
            if not auto.exists():
                raise FileNotFoundError(
                    f"Could not find summary for {edf.name}. "
                    f"Expected: {auto}  "
                    f"Use --summaries to supply the path explicitly."
                )
            result[edf] = auto

    used = sorted({str(v) for v in result.values()})
    print(f"[Summaries] Using {len(used)} summary file(s):")
    for s in used:
        print(f"  {s}")

    return result


# ── EDF / summary parsing ─────────────────────────────────────────────────────

def _extract_file_section(summary_text: str, edf_name: str) -> str:
    escaped = re.escape(edf_name)
    match = re.search(
        rf"File Name:\s*{escaped}\s*(.*?)(?=\nFile Name:|\Z)", summary_text, re.S
    )
    return match.group(1) if match else ""


def parse_seizure_intervals(summary_path: Path, edf_name: str) -> list[tuple[float, float]]:
    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    section = _extract_file_section(text, edf_name)
    if not section:
        return []
    starts = [float(x) for x in re.findall(
        r"Seizure(?:\s*\d+)?\s*Start Time:\s*(\d+)\s*seconds", section)]
    ends = [float(x) for x in re.findall(
        r"Seizure(?:\s*\d+)?\s*End Time:\s*(\d+)\s*seconds", section)]
    n = min(len(starts), len(ends))
    return [(starts[i], ends[i]) for i in range(n) if ends[i] > starts[i]]


# ── Channel helpers ───────────────────────────────────────────────────────────

def common_eeg_channels(raws: list[mne.io.BaseRaw]) -> list[str]:
    """Return EEG channels present in every raw object, preserving order of first file."""
    sets = [
        {ch for ch, t in zip(r.ch_names, r.get_channel_types()) if t == "eeg"}
        for r in raws
    ]
    shared = sets[0].intersection(*sets[1:])
    first_order = [
        ch for ch, t in zip(raws[0].ch_names, raws[0].get_channel_types())
        if t == "eeg" and ch in shared
    ]
    dropped = sets[0] - shared
    if dropped:
        print(f"[Channels] Dropped {len(dropped)} channel(s) not shared across all files: "
              f"{sorted(dropped)}")
    return first_order


def resolve_channels(raw: mne.io.BaseRaw, channels_arg: str, max_channels: int) -> list[str]:
    if channels_arg.strip():
        selected = [c.strip() for c in channels_arg.split(",") if c.strip()]
        missing = [c for c in selected if c not in raw.ch_names]
        if missing:
            raise ValueError(f"Channel(s) not found: {missing}")
        return selected
    eeg_channels = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
    selected = eeg_channels[: max(1, max_channels)]
    if not selected:
        raise ValueError("No EEG channels available.")
    return selected


# ── Windowing ─────────────────────────────────────────────────────────────────

def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def build_window_dataset_single(
        raw: mne.io.BaseRaw,
        channel_names: list[str],
        seizure_intervals: list[tuple[float, float]],
        window_sec: float,
        step_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sfreq = float(raw.info["sfreq"])
    total_duration = raw.n_times / sfreq
    data = raw.copy().pick(channel_names).get_data()

    starts = np.arange(0.0, max(0.0, total_duration - window_sec), step_sec, dtype=float)
    if starts.size == 0:
        empty_seg = np.empty((0, len(channel_names), 0), dtype=np.float32)
        return empty_seg, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    win_n = int(round(window_sec * sfreq))
    segs, labels, centers = [], [], []

    for ws in starts:
        i0 = int(round(ws * sfreq))
        i1 = i0 + win_n
        if i1 > data.shape[1]:
            break

        seg = data[:, i0:i1].astype(np.float32)
        mu = seg.mean(axis=1, keepdims=True)
        std = seg.std(axis=1, keepdims=True) + 1e-8
        seg = (seg - mu) / std

        segs.append(seg)
        overlap = sum(
            interval_overlap(ws, ws + window_sec, ss, se)
            for ss, se in seizure_intervals
        )
        labels.append(1 if overlap >= 0.5 * window_sec else 0)
        centers.append(ws + 0.5 * window_sec)

    return np.stack(segs), np.array(labels, dtype=np.int64), np.array(centers)


def build_multi_file_dataset(
        edf_paths: list[Path],
        summary_map: dict[Path, Path],
        channel_names: list[str],
        window_sec: float,
        step_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    all_x, all_y, all_centers = [], [], []
    slices: list[tuple[int, int]] = []

    for edf_path in tqdm(edf_paths, desc="Loading files", unit="file"):
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
        summary_path = summary_map[edf_path]
        seizures = parse_seizure_intervals(summary_path, edf_path.name)
        x, y, centers = build_window_dataset_single(
            raw, channel_names, seizures, window_sec, step_sec
        )
        start_idx = sum(len(a) for a in all_x)
        all_x.append(x)
        all_y.append(y)
        all_centers.append(centers)
        slices.append((start_idx, start_idx + len(x)))
        subject = edf_path.parent.name
        print(f"  [{subject}] {edf_path.name}: {len(x):>5} windows, "
              f"{int(y.sum())} seizure window(s)  (summary: {summary_path.name})")

    return (
        np.concatenate(all_x, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_centers, axis=0),
        slices,
    )


# ── 1-D CNN model ─────────────────────────────────────────────────────────────

class SeizureCNN(nn.Module):
    def __init__(self, n_channels: int, n_timepoints: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.25),

            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.25),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block(x))


# ── Device ────────────────────────────────────────────────────────────────────

def pick_device(no_gpu: bool) -> torch.device:
    if not no_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU] Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif not no_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[GPU] Using Apple MPS device.")
    else:
        device = torch.device("cpu")
        print("[CPU] No GPU found (or --no-gpu set); training on CPU.")
    return device


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
        x_train: np.ndarray,
        y_train: np.ndarray,
        device: torch.device,
        epochs: int,
        batch_size: int,
        lr: float,
        random_state: int,
) -> SeizureCNN:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    n_channels, n_timepoints = x_train.shape[1], x_train.shape[2]
    model = SeizureCNN(n_channels, n_timepoints).to(device)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_w = (n_neg / n_pos) if n_pos > 0 else 1.0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.from_numpy(x_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)
    dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model.train()
    epoch_bar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        total_loss = 0.0
        batch_bar = tqdm(dl, desc=f"  Epoch {epoch:3d}", unit="batch", leave=False)
        for xb, yb in batch_bar:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        epoch_bar.set_postfix(loss=f"{total_loss / len(x_train):.4f}")

    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(
        model: SeizureCNN, x: np.ndarray, device: torch.device, batch_size: int = 256
) -> np.ndarray:
    model.eval()
    dl = DataLoader(TensorDataset(torch.from_numpy(x).to(device)), batch_size=batch_size)
    preds = [model(xb).argmax(dim=1).cpu().numpy() for (xb,) in dl]
    return np.concatenate(preds).astype(int)


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


# ── Model I/O ─────────────────────────────────────────────────────────────────

def save_model(model: SeizureCNN, path: Path, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    print(f"[Model] Saved to {path}")


def load_model(path: Path, device: torch.device) -> tuple[SeizureCNN, dict]:
    checkpoint = torch.load(path, map_location=device)
    meta = checkpoint["meta"]
    model = SeizureCNN(meta["n_channels"], meta["n_timepoints"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"[Model] Loaded from {path}  "
          f"(channels={meta['n_channels']}, T={meta['n_timepoints']})")
    return model, meta


# ── Post-processing ───────────────────────────────────────────────────────────

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


# ── Heart rate generation ─────────────────────────────────────────────────────

# Physiological reference values for a 25-35 year-old adult:
#   Resting HR  : 60–75 bpm  (we use 68 bpm as the baseline)
#   Peak seizure HR : 130–160 bpm  (we use ~145 bpm as the ictal peak)
#   Rise time   : HR climbs with a logarithmic curve (fast early, slower near peak)
#   Recovery    : mirror log curve back to baseline (fast drop, slow tail)

_HR_BASELINE_BPM   = 68.0   # mean resting HR
_HR_PEAK_BPM       = 145.0  # mean ictal peak HR
_HR_BASELINE_NOISE = 2.5    # ± bpm random walk noise at rest
_HR_ICTAL_NOISE    = 6.0    # ± bpm noise during seizure
_HR_RISE_TAU       = 15.0   # seconds from seizure onset to ~63 % of peak (log scale)
_HR_FALL_TAU       = 25.0   # seconds from seizure end to ~63 % recovery


def _log_ramp(t_elapsed: float, tau: float) -> float:
    """
    Normalised [0, 1] log-shaped ramp.

    Uses  f(t) = log(1 + t/tau) / log(2)  so that f(tau) ≈ 1.
    Clamped to [0, 1].
    """
    if tau <= 0:
        return 1.0
    return float(np.clip(np.log1p(t_elapsed / tau) / np.log(2.0), 0.0, 1.0))


def generate_heart_rate(
        total_duration: float,
        seizure_intervals: list[tuple[float, float]],
        fs: float = 1.0,
        rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a synthetic heart-rate time-series for a recording of *total_duration* seconds.

    Parameters
    ----------
    total_duration : float
        Length of the recording in seconds.
    seizure_intervals : list of (start, end) tuples
        Ground-truth seizure intervals in seconds.
    fs : float
        Sampling rate of the output signal in Hz (default 1 Hz = one sample per second).
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    times : np.ndarray  shape (N,)
    hr    : np.ndarray  shape (N,)   — instantaneous HR in bpm
    """
    if rng is None:
        rng = np.random.default_rng(42)

    times = np.arange(0.0, total_duration, 1.0 / fs)
    n = len(times)
    hr = np.full(n, _HR_BASELINE_BPM, dtype=np.float64)

    delta = _HR_PEAK_BPM - _HR_BASELINE_BPM  # ~77 bpm swing

    for t_idx, t in enumerate(times):
        # Find which seizure phase this sample belongs to
        ictal = False
        postictal_frac = 0.0   # 0 = fully recovered, 1 = just ended

        for ss, se in seizure_intervals:
            if ss <= t <= se:
                # ── ictal: log rise from seizure onset ──
                t_elapsed = t - ss
                frac = _log_ramp(t_elapsed, _HR_RISE_TAU)
                hr[t_idx] = _HR_BASELINE_BPM + delta * frac
                ictal = True
                break
            elif t > se:
                # ── post-ictal: log recovery after seizure end ──
                t_since_end = t - se
                # Check that no later seizure has started
                later = any(ss2 <= t for ss2, _ in seizure_intervals if ss2 > se)
                if not later:
                    frac = _log_ramp(t_since_end, _HR_FALL_TAU)
                    postictal_level = _HR_PEAK_BPM - delta * frac
                    postictal_frac = max(postictal_frac, postictal_level)

        if not ictal and postictal_frac > 0.0:
            hr[t_idx] = max(_HR_BASELINE_BPM, postictal_frac)

    # Add physiologically plausible noise
    noise_amp = np.where(hr > _HR_BASELINE_BPM + 10, _HR_ICTAL_NOISE, _HR_BASELINE_NOISE)
    hr += rng.normal(0.0, noise_amp)

    # Enforce realistic floor / ceiling
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
    """
    Standalone heart-rate plot with ground-truth and predicted seizure shading.
    The plot window is clipped to [plot_start, plot_start + plot_duration].
    """
    plot_end = plot_start + plot_duration
    mask = (times >= plot_start) & (times <= plot_end)
    t_win = times[mask]
    hr_win = hr[mask]

    fig, ax = plt.subplots(figsize=(13, 3.5))

    # Shade seizure regions first (behind the curve)
    for ss, se in seizure_intervals:
        x0, x1 = max(ss, plot_start), min(se, plot_end)
        if x1 > x0:
            ax.axvspan(x0, x1, color="#e74c3c", alpha=0.18, label="_gt")

    for ps, pe in predicted_intervals:
        x0, x1 = max(ps, plot_start), min(pe, plot_end)
        if x1 > x0:
            ax.axvspan(x0, x1, color="#2980b9", alpha=0.15, label="_pred")

    # Heart rate curve with gradient-like coloring driven by HR value
    from matplotlib.collections import LineCollection
    points = np.array([t_win, hr_win]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(40, 170)
    lc = LineCollection(segments, cmap="RdYlGn_r", norm=norm, linewidth=2.0, zorder=3)
    lc.set_array(hr_win)
    ax.add_collection(lc)

    # Reference lines
    ax.axhline(_HR_BASELINE_BPM, color="#27ae60", linewidth=0.8,
               linestyle="--", alpha=0.7, label=f"Resting baseline ({_HR_BASELINE_BPM:.0f} bpm)")
    ax.axhline(_HR_PEAK_BPM, color="#e74c3c", linewidth=0.8,
               linestyle="--", alpha=0.7, label=f"Ictal peak ({_HR_PEAK_BPM:.0f} bpm)")

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label("HR (bpm)", fontsize=8)

    # Legend patches
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
    fig.savefig(save_path, dpi=150)
    print(f"[HeartRate] Saved standalone HR plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ── Average seizure morphology plot ──────────────────────────────────────────

def plot_average_seizure(
        x: np.ndarray,
        y: np.ndarray,
        channel_names: list[str],
        sfreq: float,
        window_sec: float,
        save_path: Path,
        show: bool,
        max_channels: int = 6,
) -> None:
    """
    Plot mean ± 1 SD of every labelled seizure window, one subplot per channel.
    """
    seiz_wins = x[y == 1]
    if len(seiz_wins) == 0:
        print("[AvgSeizure] No seizure windows found — skipping plot.")
        return

    n_ch = min(len(channel_names), max_channels)
    t_axis = np.linspace(0, window_sec, seiz_wins.shape[2])

    mean = seiz_wins[:, :n_ch, :].mean(axis=0)
    std  = seiz_wins[:, :n_ch, :].std(axis=0)

    cols = min(n_ch, 3)
    rows = (n_ch + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), sharex=True)
    axes = np.array(axes).flatten()

    for i in range(n_ch):
        ax = axes[i]
        ax.plot(t_axis, mean[i], color="#c0392b", linewidth=1.5, label="Mean")
        ax.fill_between(
            t_axis,
            mean[i] - std[i],
            mean[i] + std[i],
            color="#c0392b", alpha=0.20, label="±1 SD",
        )
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
        f"Average seizure morphology  |  n={len(seiz_wins)} windows  |  "
        f"{window_sec:.1f}s window",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[AvgSeizure] Saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ── Ensemble helpers ──────────────────────────────────────────────────────────

def ensemble_predict(
        models: list[SeizureCNN],
        x: np.ndarray,
        device: torch.device,
        threshold: float,
        batch_size: int = 256,
) -> np.ndarray:
    """Average class-1 probabilities across models, then threshold."""
    probs = np.stack(
        [predict_positive_prob(m, x, device, batch_size) for m in models], axis=0
    )
    return (probs.mean(axis=0) >= threshold).astype(int)


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
    """
    Two-panel figure:
      Top   — multi-channel EEG traces with GT/pred seizure shading.
      Bottom — synthetic heart-rate curve with the same shading.

    Pass *heart_rate_data* = (times, hr) to enable the HR panel.
    If None, falls back to the original single-panel layout.
    """
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
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=False,          # EEG times are relative; HR times are absolute
        )
    else:
        fig, ax_eeg = plt.subplots(figsize=(13, 7))

    # ── EEG panel ──
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

    # ── Heart-rate panel ──
    if has_hr:
        hr_times, hr_vals = heart_rate_data
        mask = (hr_times >= start) & (hr_times <= end)
        t_win  = hr_times[mask]
        hr_win = hr_vals[mask]

        # GT shading
        for ss, se in seizure_intervals:
            x0, x1 = max(ss, start), min(se, end)
            if x1 > x0:
                ax_hr.axvspan(x0, x1, color="#e74c3c", alpha=0.18)
        # Pred shading
        for ps, pe in predicted_intervals:
            x0, x1 = max(ps, start), min(pe, end)
            if x1 > x0:
                ax_hr.axvspan(x0, x1, color="#2980b9", alpha=0.15)

        # Colour-mapped HR curve (green → yellow → red with rising HR)
        from matplotlib.collections import LineCollection
        points   = np.array([t_win, hr_win]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(40, 170)
        lc   = LineCollection(segments, cmap="RdYlGn_r", norm=norm,
                               linewidth=2.2, zorder=3)
        lc.set_array(hr_win)
        ax_hr.add_collection(lc)

        # Reference dashed lines
        ax_hr.axhline(_HR_BASELINE_BPM, color="#27ae60", linewidth=0.9,
                      linestyle="--", alpha=0.8,
                      label=f"Resting {_HR_BASELINE_BPM:.0f} bpm")
        ax_hr.axhline(_HR_PEAK_BPM, color="#e74c3c", linewidth=0.9,
                      linestyle="--", alpha=0.8,
                      label=f"Ictal peak {_HR_PEAK_BPM:.0f} bpm")

        ax_hr.set_xlim(start, end)
        ax_hr.set_ylim(40, 185)
        ax_hr.set_xlabel("Time (s)", fontsize=9)
        ax_hr.set_ylabel("Heart rate (bpm)", fontsize=9)
        ax_hr.set_title("Synthetic heart rate  |  age 25–35 reference", fontsize=9)
        ax_hr.legend(fontsize=8, loc="upper right", framealpha=0.85, edgecolor="none")
        ax_hr.grid(True, axis="x", alpha=0.3)
        ax_hr.grid(True, axis="y", alpha=0.2)

        # Colourbar on the right of the HR panel
        sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_hr, pad=0.01, fraction=0.02)
        cbar.set_label("bpm", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[Plot] Saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


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
        print("[GradCAM] No seizure windows in test set")
        print("[GradCAM] Skipping...")
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
        len(best_seizure_windows),
        1,
        figsize=(11, 3.5 * len(best_seizure_windows)),
        squeeze=False
    )

    for row, window_index in enumerate(best_seizure_windows):
        ax = axes[row, 0]

        input_window = torch.from_numpy(
            x_test[window_index: window_index + 1]
        ).to(device).requires_grad_(True)

        attribution = grad_cam.attribute(input_window, target=1)

        heatmap = attribution.mean(dim=1)
        heatmap = heatmap.squeeze(0).detach().cpu().numpy()

        heatmap = np.maximum(heatmap, 0)

        # make heatmap values between 0 and 1
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Stretch it back to the original time length.
        heatmap_time_axis = np.linspace(0, window_sec, len(heatmap))
        heatmap_resized = np.interp(time_axis, heatmap_time_axis, heatmap)

        # Draw red background.
        # Darker red means that time was more important.
        for k in range(len(time_axis) - 1):
            ax.axvspan(
                time_axis[k],
                time_axis[k + 1],
                color=(1.0, 0.2, 0.2, float(heatmap_resized[k]) * 0.5),
                linewidth=0
            )

        # Space the EEG channels vertically so they do not overlap
        signal_range = np.max(np.ptp(x_test[window_index, :channels_to_plot], axis=1))
        spacing = signal_range * 1.3 or 1.0
        offsets = np.arange(channels_to_plot)[::-1] * spacing

        # Draw the EEG signals
        for channel_index in range(channels_to_plot):
            ax.plot(
                time_axis,
                x_test[window_index, channel_index] + offsets[channel_index],
                color=colors[channel_index],
                linewidth=0.9,
                label=channel_names[channel_index]
                if channel_index < len(channel_names)
                else f"ch{channel_index}"
            )

        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_yticks(offsets)
        ax.set_yticklabels(channel_names[:channels_to_plot], fontsize=8)
        ax.set_xlabel("Time within window (s)", fontsize=9)

        ax.set_title(
            f"Grad-CAM | test window #{window_index} | "
            f"p(seizure)={probabilities[window_index]:.2f}",
            fontsize=10
        )

        ax.grid(True, axis="x", alpha=0.25)

    # colorbar explaining heatmap intensity
    colorbar_source = plt.cm.ScalarMappable(
        cmap=plt.cm.Reds,
        norm=plt.Normalize(0, 1)
    )
    colorbar_source.set_array([])

    fig.colorbar(
        colorbar_source,
        ax=axes[:, 0],
        fraction=0.015,
        pad=0.02,
        label="Grad-CAM intensity"
    )

    fig.suptitle(
        "Grad-CAM: most confident seizure predictions",
        fontsize=11
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)

    print(f"[GradCAM] Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def train_and_detect(args: argparse.Namespace) -> Metrics:
    # ── validate ──
    for p in args.edf:
        if not p.exists():
            raise FileNotFoundError(f"EDF file not found: {p}")
    if args.window_sec < 0.1:
        raise ValueError("--window-sec must be >= 0.1")
    if not (0.1 <= args.step_sec <= args.window_sec):
        raise ValueError("--step-sec must be in [0.1, window-sec]")
    if not (0.1 <= args.train_frac < 1.0):
        raise ValueError("--train-frac must be in [0.1, 1.0)")
    if not (0.0 < args.pred_threshold < 1.0):
        raise ValueError("--pred-threshold must be in (0, 1)")
    if args.pred_min_run < 1:
        raise ValueError("--pred-min-run must be >= 1")

    device = pick_device(args.no_gpu)

    # ── resolve summary files ──
    print()
    summary_map = build_summary_map(args.edf, args.summaries)

    # ── find shared channels across all files ──
    print(f"\nScanning channels across {len(args.edf)} file(s)…")
    raws_probe = [
        mne.io.read_raw_edf(str(p), preload=False, verbose="ERROR") for p in args.edf
    ]
    channel_names = common_eeg_channels(raws_probe)
    if not channel_names:
        raise ValueError("No EEG channels shared across all files.")
    print(f"[Channels] Using {len(channel_names)} shared EEG channel(s).\n")
    del raws_probe

    # ── build dataset from all files ──
    x, y, centers, file_slices = build_multi_file_dataset(
        args.edf, summary_map, channel_names, args.window_sec, args.step_sec
    )
    n_channels, n_timepoints = x.shape[1], x.shape[2]

    # ── chronological train/test split ──
    split = int(len(x) * args.train_frac)
    split = max(10, min(split, len(x) - 10))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test   = x[split:], y[split:]
    centers_test      = centers[split:]

    print(f"\nDataset  total={len(x)}  train={split}  test={len(x_test)}")
    print(f"Seizure windows — train: {int(y_train.sum())}  test: {int(y_test.sum())}")
    test_file_names = [args.edf[i].name for i, (fs, fe) in enumerate(file_slices) if fe > split]
    print(f"Test set spans: {test_file_names}\n")

    # ── average seizure morphology ──
    raw0 = mne.io.read_raw_edf(str(args.edf[0]), preload=False, verbose="ERROR")
    plot_average_seizure(
        x, y, channel_names,
        sfreq=float(raw0.info["sfreq"]),
        window_sec=args.window_sec,
        save_path=args.save_seizure_plot,
        show=args.show,
        max_channels=args.max_channels,
    )

    # ── train ensemble (or load a single saved model) ──
    if args.load_model and args.load_model.exists():
        model, _ = load_model(args.load_model, device)
        models = [model]
    else:
        n_runs = max(1, args.ensemble_runs)
        models: list[SeizureCNN] = []
        best_f1, best_idx = -1.0, 0

        for run in range(n_runs):
            run_seed = args.random_state + run
            print(f"\nTraining CNN (run {run + 1}/{n_runs}, seed={run_seed}) …")
            m = train_model(
                x_train, y_train, device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                random_state=run_seed,
            )
            run_pred = smooth_binary_predictions(predict(m, x_test, device), min_run=2)
            _, _, run_f1 = compute_prf(y_test, run_pred)
            print(f"  Run {run + 1} F1: {run_f1:.3f}")
            models.append(m)
            if run_f1 > best_f1:
                best_f1, best_idx = run_f1, run

        save_model(
            models[best_idx], args.save_model,
            meta={"n_channels": n_channels, "n_timepoints": n_timepoints},
        )
        if n_runs > 1:
            print(f"\nBest single-run F1: {best_f1:.3f} (run {best_idx + 1})")

    # ── evaluate ──
    if len(models) > 1:
        print(f"\nEnsemble voting across {len(models)} models …")
        pred_test = ensemble_predict(models, x_test, device, threshold=args.pred_threshold)
    else:
        prob_test = predict_positive_prob(models[0], x_test, device)
        pred_test = (prob_test >= args.pred_threshold).astype(int)
    pred_test = smooth_binary_predictions(pred_test, min_run=args.pred_min_run)

    precision, recall, f1 = compute_prf(y_test, pred_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test, labels=[0, 1]).ravel()

    label = f"Ensemble ({len(models)} models)" if len(models) > 1 else "SeizureCNN (1-D Conv)"
    print(f"\nModel: {label}")
    print(f"Test Precision : {precision:.3f}")
    print(f"Test Recall    : {recall:.3f}")
    print(f"Test F1        : {f1:.3f}")
    print(f"Confusion [tn fp fn tp]: [{tn} {fp} {fn} {tp}]")

    # ── EEG overlay plot: default to last file with seizures ──
    if args.plot_edf:
        plot_edf_path = args.plot_edf
    else:
        seizure_files = [
            p for p in args.edf
            if parse_seizure_intervals(summary_map[p], p.name)
        ]
        plot_edf_path = seizure_files[-1] if seizure_files else args.edf[-1]
        if seizure_files:
            print(f"[Plot] Auto-selected file with seizures: {plot_edf_path.name}")
    plot_file_idx = next(
        (i for i, p in enumerate(args.edf) if p == plot_edf_path), len(args.edf) - 1
    )

    fs, fe = file_slices[plot_file_idx]
    t0 = max(fs, split) - split
    t1 = max(fe, split) - split
    if t0 < t1 and t1 <= len(pred_test):
        plot_pred    = pred_test[t0:t1]
        plot_centers = centers_test[t0:t1]
    else:
        plot_pred    = np.zeros(0, dtype=int)
        plot_centers = np.zeros(0, dtype=float)
        print(f"[Plot] {plot_edf_path.name} is entirely in the training set; "
              "no test predictions to overlay.")

    predicted_intervals = windows_to_intervals(plot_centers, plot_pred, args.window_sec)
    plot_seizures       = parse_seizure_intervals(summary_map[plot_edf_path], plot_edf_path.name)
    raw_plot            = mne.io.read_raw_edf(str(plot_edf_path), preload=True, verbose="ERROR")
    print(f"[Plot] {plot_edf_path.name}: GT intervals={len(plot_seizures)}  "
          f"Pred intervals={len(predicted_intervals)}")

    plot_start = args.start
    plot_end   = min(plot_start + args.duration,
                     raw_plot.n_times / float(raw_plot.info["sfreq"]))
    has_visible_event = (
        has_overlap_with_window(plot_seizures, plot_start, plot_end)
        or has_overlap_with_window(predicted_intervals, plot_start, plot_end)
    )

    if (not has_visible_event) and args.start == 0.0:
        if plot_seizures:
            plot_start = max(0.0, min(s for s, _ in plot_seizures) - 10.0)
            print(
                f"[Plot] No events in {args.start:.1f}-{plot_end:.1f}s; "
                f"auto-shifting to start={plot_start:.1f}s (ground truth)."
            )
        elif predicted_intervals:
            plot_start = max(0.0, min(s for s, _ in predicted_intervals) - 10.0)
            print(
                f"[Plot] No events in {args.start:.1f}-{plot_end:.1f}s; "
                f"auto-shifting to start={plot_start:.1f}s (prediction)."
            )

    # ── Generate synthetic heart rate for the plot EDF ──
    total_dur = raw_plot.n_times / float(raw_plot.info["sfreq"])
    print(f"[HeartRate] Generating synthetic HR signal for {plot_edf_path.name} "
          f"({total_dur:.0f}s total) …")
    hr_times, hr_vals = generate_heart_rate(
        total_duration=total_dur,
        seizure_intervals=plot_seizures,
        fs=4.0,               # 4 samples/s — smooth enough, lightweight
        rng=np.random.default_rng(args.random_state),
    )

    # ── Combined EEG + HR overlay plot ──
    plot_results(
        raw=raw_plot,
        seizure_intervals=plot_seizures,
        predicted_intervals=predicted_intervals,
        start=plot_start,
        duration=args.duration,
        channels_arg=args.channels,
        max_channels=args.max_channels,
        save_path=args.save,
        show=args.show,
        title_suffix=plot_edf_path.name,
        heart_rate_data=(hr_times, hr_vals),
    )

    # ── Standalone heart-rate plot ──
    plot_heart_rate_with_seizures(
        times=hr_times,
        hr=hr_vals,
        seizure_intervals=plot_seizures,
        predicted_intervals=predicted_intervals,
        plot_start=plot_start,
        plot_duration=args.duration,
        save_path=args.save_hr_plot,
        show=args.show,
        title_suffix=plot_edf_path.name,
    )

    # ── Grad-CAM explanation ──
    plot_gradcam(
        model=models[0],
        x_test=x_test,
        y_test=y_test,
        channel_names=channel_names,
        window_sec=args.window_sec,
        n_samples=args.gradcam_n_samples,
        save_path=args.save_gradcam_plot,
        device=device,
        show=args.show,
    )

    return Metrics(precision=precision, recall=recall, f1=f1, model_name="cnn")


def main() -> None:
    args = parse_args()
    _ = train_and_detect(args)


if __name__ == "__main__":
    main()