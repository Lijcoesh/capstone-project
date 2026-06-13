# -*- coding: utf-8 -*-
"""
Preprocessing stage for the EEG-only seizure-detection pipeline.

Reads raw CHB-MIT EDF files, applies the standard preprocessing steps, and
saves the result as a single compressed NumPy archive (.npz) that the training
script (train_model_eeg.py) consumes. Run this ONCE; then train as often as you
like without re-reading the EDFs.

Preprocessing steps (in order):
  1. Read each EDF (multi-channel raw EEG).
  2. Power-line notch filter at 60 Hz + harmonics (US mains; CHB-MIT = Boston).
  3. Keep only EEG channels present in ALL files (silent intersection).
  4. Sliding-window segmentation (fixed-length windows, configurable step).
  5. Per-window, per-channel z-score normalization.
  6. Label each window ictal (1) if >=50% of its duration overlaps a seizure
     interval from the subject summary file, else interictal (0).

Windows from all files are concatenated in chronological order. The train/test
split is NOT done here — it is left to the training script so the split ratio
can be varied without re-preprocessing.

Example:
  python preprocess_eeg.py
  python preprocess_eeg.py --edf chb01/chb01_03.edf chb02/chb02_16.edf --out my_data.npz
"""

import argparse
import re
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

# ── Defaults ──────────────────────────────────────────────────────────────────

_BASE = Path(__file__).resolve().parent / "../../data/raw/physionet.org/files/chbmit/1.0.0/"

# Power-line interference is ALWAYS removed as a standard preprocessing step.
# CHB-MIT was recorded at Children's Hospital Boston (USA), where the mains
# frequency is 60 Hz. Harmonics below the Nyquist frequency are removed as well.
POWERLINE_NOTCH_HZ = 60.0

# Default location of the preprocessed dataset produced by this script.
DEFAULT_PREPROCESSED_PATH = (
    Path(__file__).resolve().parent / "../../data/processed/eeg_windows.npz"
)

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


# ── Summary resolution ────────────────────────────────────────────────────────

def _auto_summary_for(edf_path: Path) -> Path:
    """
    Derive the canonical summary path from an EDF path.

    Convention used by CHB-MIT:
        <base>/<subject>/<subject>-summary.txt
    e.g. raw/physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf
      →  raw/physionet.org/files/chbmit/1.0.0/chb01/chb01-summary.txt
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


# ── Preprocessing ─────────────────────────────────────────────────────────────

def apply_notch_filter(raw: mne.io.BaseRaw, notch_freq: float) -> None:
    """
    Remove power-line noise at *notch_freq* and its harmonics, in place.

    No-op when notch_freq <= 0 (baseline / filtering disabled). Only harmonics
    below the Nyquist frequency are filtered (CHB-MIT is sampled at 256 Hz →
    Nyquist = 128 Hz, so for 60 Hz mains we notch 60 and 120 Hz).
    """
    if notch_freq <= 0:
        return
    nyquist = float(raw.info["sfreq"]) / 2.0
    freqs = list(np.arange(notch_freq, nyquist, notch_freq))
    if not freqs:
        return
    raw.notch_filter(freqs=freqs, verbose="ERROR")


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
        apply_notch_filter(raw, POWERLINE_NOTCH_HZ)
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


# ── Save / load the preprocessed dataset ──────────────────────────────────────

def save_preprocessed(
        out_path: Path,
        x: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        channel_names: list[str],
        sfreq: float,
        window_sec: float,
        step_sec: float,
        edf_paths: list[Path],
        summary_map: dict[Path, Path],
        file_slices: list[tuple[int, int]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=x,
        y=y,
        centers=centers,
        channel_names=np.array(channel_names),
        sfreq=np.array(sfreq, dtype=np.float64),
        window_sec=np.array(window_sec, dtype=np.float64),
        step_sec=np.array(step_sec, dtype=np.float64),
        notch_freq=np.array(POWERLINE_NOTCH_HZ, dtype=np.float64),
        edf_paths=np.array([str(p) for p in edf_paths]),
        summary_paths=np.array([str(summary_map[p]) for p in edf_paths]),
        file_slices=np.array(file_slices, dtype=np.int64).reshape(-1, 2),
    )
    print(f"[Preprocess] Saved preprocessed dataset to {out_path}")


def load_preprocessed(path: Path) -> dict:
    """Load a .npz produced by this script into a plain dict of arrays/metadata."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found: {path}\n"
            f"Run  python preprocess_eeg.py  first to generate it."
        )
    npz = np.load(path, allow_pickle=False)
    return {
        "X": npz["X"],
        "y": npz["y"],
        "centers": npz["centers"],
        "channel_names": [str(c) for c in npz["channel_names"]],
        "sfreq": float(npz["sfreq"]),
        "window_sec": float(npz["window_sec"]),
        "step_sec": float(npz["step_sec"]),
        "notch_freq": float(npz["notch_freq"]),
        "edf_paths": [Path(p) for p in npz["edf_paths"]],
        "summary_paths": [Path(p) for p in npz["summary_paths"]],
        "file_slices": [tuple(int(v) for v in row) for row in npz["file_slices"]],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess CHB-MIT EDF files into a windowed .npz dataset for training."
    )
    parser.add_argument(
        "--edf", type=Path, nargs="+", default=DEFAULT_EDFS, metavar="EDF",
        help="One or more EDF files (chronological order). "
             "Default: the 19 files spanning chb01–chb04.",
    )
    parser.add_argument(
        "--summaries", type=Path, nargs="+", default=None, metavar="SUMMARY",
        help="Explicit summary file(s) (one per unique subject folder). "
             "When omitted, auto-detected from each EDF's parent directory.",
    )
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_PREPROCESSED_PATH,
        help="Output .npz path for the preprocessed dataset.",
    )
    return parser.parse_args()


def preprocess(
        edf_paths: list[Path],
        summaries: list[Path] | None,
        window_sec: float,
        step_sec: float,
) -> dict:
    """Run the full preprocessing pipeline and return arrays + metadata."""
    for p in edf_paths:
        if not p.exists():
            raise FileNotFoundError(f"EDF file not found: {p}")
    if window_sec < 0.1:
        raise ValueError("--window-sec must be >= 0.1")
    if not (0.1 <= step_sec <= window_sec):
        raise ValueError("--step-sec must be in [0.1, window-sec]")

    print()
    summary_map = build_summary_map(edf_paths, summaries)

    print(f"\nScanning channels across {len(edf_paths)} file(s)…")
    raws_probe = [
        mne.io.read_raw_edf(str(p), preload=False, verbose="ERROR") for p in edf_paths
    ]
    channel_names = common_eeg_channels(raws_probe)
    if not channel_names:
        raise ValueError("No EEG channels shared across all files.")
    sfreq = float(raws_probe[0].info["sfreq"])
    print(f"[Channels] Using {len(channel_names)} shared EEG channel(s).\n")
    del raws_probe

    x, y, centers, file_slices = build_multi_file_dataset(
        edf_paths, summary_map, channel_names, window_sec, step_sec,
    )

    print(f"\nDataset  total={len(x)}  seizure windows={int(y.sum())}  "
          f"channels={x.shape[1]}  timepoints={x.shape[2]}")

    return {
        "x": x, "y": y, "centers": centers,
        "channel_names": channel_names, "sfreq": sfreq,
        "window_sec": window_sec, "step_sec": step_sec,
        "edf_paths": edf_paths, "summary_map": summary_map,
        "file_slices": file_slices,
    }


def main() -> None:
    args = parse_args()
    result = preprocess(args.edf, args.summaries, args.window_sec, args.step_sec)
    save_preprocessed(
        args.out,
        result["x"], result["y"], result["centers"],
        result["channel_names"], result["sfreq"],
        result["window_sec"], result["step_sec"],
        result["edf_paths"], result["summary_map"], result["file_slices"],
    )


if __name__ == "__main__":
    main()
