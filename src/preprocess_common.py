# -*- coding: utf-8 -*-
"""
Shared SeizeIT2 preprocessing — single source of truth for both pipelines.

Both the EEG-only and the EEG+ECG pipeline call into this module; the ONLY
difference between them is whether the ECG channel is included (`include_ecg`).
Keeping the logic here guarantees the two feature sets differ by nothing other
than that channel, so the comparison stays fair.

Task: seizure PREDICTION (pre-ictal vs. interictal), not detection.

Per seizure onset `o` (from the BIDS events.tsv):
  - pre-ictal  : window in [o - PREICTAL_SEC, o)              -> label 1 (positive)
  - ictal+post : window in [o, o + duration + POSTICTAL_GUARD] -> excluded (dropped)
  - interictal : everything else                              -> label 0 (negative)
Priority on overlap: exclude > pre-ictal > interictal.

Because recordings are ~18 h long, keeping every interictal window is infeasible
(millions of windows) and absurdly imbalanced. We therefore keep ALL pre-ictal
windows and subsample interictal at a fixed ratio (interictal : pre-ictal).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

# ── Defaults / constants ──────────────────────────────────────────────────────

SEIZEIT2_BASE = Path(__file__).resolve().parent / "../data/raw/seizeit2"

# SeizeIT2 was recorded in European EMUs (Belgium, Germany, Sweden, Portugal):
# mains frequency is 50 Hz (NOT 60 Hz like the US-recorded CHB-MIT).
SEIZEIT2_NOTCH_HZ = 50.0

PREICTAL_SEC = 30 * 60          # 30 min before onset = pre-ictal (positive)
POSTICTAL_GUARD_SEC = 10 * 60   # exclude the seizure + 10 min after it

DEFAULT_INTERICTAL_RATIO = 5.0  # interictal : pre-ictal windows kept
DEFAULT_WINDOW_SEC = 2.0
DEFAULT_STEP_SEC = 1.0


@dataclass
class Recording:
    subject: str
    run: str
    eeg_path: Path
    ecg_path: Path | None
    events_path: Path


# ── BIDS discovery ────────────────────────────────────────────────────────────

def discover_recordings(base: Path = SEIZEIT2_BASE) -> list[Recording]:
    """Find every EEG recording in the BIDS tree, paired with its ECG + events."""
    base = Path(base)
    recordings: list[Recording] = []
    for eeg_edf in sorted(base.glob("sub-*/ses-*/eeg/*_eeg.edf")):
        stem = eeg_edf.name.replace("_eeg.edf", "")
        events = eeg_edf.with_name(f"{stem}_events.tsv")
        ecg = eeg_edf.parent.parent / "ecg" / f"{stem}_ecg.edf"
        subject = eeg_edf.parents[2].name      # sub-XXX
        recordings.append(
            Recording(
                subject=subject,
                run=stem,
                eeg_path=eeg_edf,
                ecg_path=ecg if ecg.exists() else None,
                events_path=events if events.exists() else None,
            )
        )
    return recordings


def parse_seizure_events(events_path: Path | None) -> list[tuple[float, float]]:
    """Return seizure intervals (onset, duration) from a BIDS events.tsv.

    Seizures are rows whose eventType starts with 'sz' (sz, sz_foc_*, sz_gen_* …).
    Background ('bckg') and impedance ('impd') rows are ignored.
    """
    if events_path is None or not Path(events_path).exists():
        return []
    seizures: list[tuple[float, float]] = []
    with open(events_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            etype = (row.get("eventType") or "").strip()
            if etype.startswith("sz"):
                try:
                    onset = float(row["onset"])
                    dur = float(row["duration"])
                except (KeyError, ValueError):
                    continue
                if dur > 0:
                    seizures.append((onset, dur))
    return sorted(seizures)


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def apply_notch_filter(raw: mne.io.BaseRaw, notch_freq: float = SEIZEIT2_NOTCH_HZ) -> None:
    """Remove power-line noise at *notch_freq* and harmonics below Nyquist, in place."""
    if notch_freq <= 0:
        return
    nyquist = float(raw.info["sfreq"]) / 2.0
    freqs = list(np.arange(notch_freq, nyquist, notch_freq))
    if freqs:
        raw.notch_filter(freqs=freqs, verbose="ERROR")


def _classify_window(
        ws: float, we: float, seizures: list[tuple[float, float]]
) -> str:
    """Classify a window [ws, we) as 'exclude', 'preictal', or 'interictal'."""
    # exclude takes priority: ictal + post-ictal guard
    for onset, dur in seizures:
        ictal_start = onset
        post_end = onset + dur + POSTICTAL_GUARD_SEC
        if we > ictal_start and ws < post_end:
            return "exclude"
    # pre-ictal: within [onset - PREICTAL_SEC, onset)
    for onset, _dur in seizures:
        pre_start = onset - PREICTAL_SEC
        if we > pre_start and ws < onset:
            return "preictal"
    return "interictal"


def _load_recording_data(
        rec: Recording, include_ecg: bool
) -> tuple[np.ndarray, float, list[str]]:
    """Load (and notch-filter) the EEG (+ECG) signal as a (channels, samples) array."""
    eeg = mne.io.read_raw_edf(str(rec.eeg_path), preload=True, verbose="ERROR")
    apply_notch_filter(eeg)
    sfreq = float(eeg.info["sfreq"])
    data = eeg.get_data()                       # (2, N)
    channel_names = list(eeg.ch_names)

    if include_ecg:
        if rec.ecg_path is None:
            raise FileNotFoundError(f"ECG missing for {rec.run}")
        ecg = mne.io.read_raw_edf(str(rec.ecg_path), preload=True, verbose="ERROR")
        apply_notch_filter(ecg)
        if ecg.n_times != eeg.n_times:
            # align to the shorter length (defensive; they normally match exactly)
            n = min(ecg.n_times, eeg.n_times)
            data = data[:, :n]
            ecg_data = ecg.get_data()[:, :n]
        else:
            ecg_data = ecg.get_data()
        data = np.concatenate([data, ecg_data], axis=0)   # (3, N)
        channel_names = channel_names + list(ecg.ch_names)

    return data, sfreq, channel_names


def build_recording_windows(
        rec: Recording,
        include_ecg: bool,
        window_sec: float,
        step_sec: float,
        interictal_ratio: float,
        rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    """
    Build the kept windows for one recording: ALL pre-ictal + a subsampled set of
    interictal (ratio : 1). Returns (X, y, centers, channel_names, sfreq).
    Recordings with no seizures contribute nothing (no pre-ictal → returns empty).
    """
    seizures = parse_seizure_events(rec.events_path)
    if not seizures:
        return (np.empty((0, 0, 0), np.float32), np.empty(0, np.int64),
                np.empty(0, np.float64), [], 0.0)

    data, sfreq, channel_names = _load_recording_data(rec, include_ecg)
    n_samples = data.shape[1]
    total_dur = n_samples / sfreq
    win_n = int(round(window_sec * sfreq))

    starts = np.arange(0.0, max(0.0, total_dur - window_sec), step_sec, dtype=float)
    preictal_starts: list[float] = []
    interictal_starts: list[float] = []
    for ws in starts:
        label = _classify_window(ws, ws + window_sec, seizures)
        if label == "preictal":
            preictal_starts.append(ws)
        elif label == "interictal":
            interictal_starts.append(ws)
        # 'exclude' -> dropped

    if not preictal_starts:
        return (np.empty((0, 0, 0), np.float32), np.empty(0, np.int64),
                np.empty(0, np.float64), channel_names, sfreq)

    # subsample interictal to ratio : 1
    target = int(round(interictal_ratio * len(preictal_starts)))
    if len(interictal_starts) > target:
        idx = rng.choice(len(interictal_starts), size=target, replace=False)
        interictal_starts = [interictal_starts[i] for i in sorted(idx)]

    kept = ([(ws, 1) for ws in preictal_starts]
            + [(ws, 0) for ws in interictal_starts])
    kept.sort(key=lambda t: t[0])               # chronological order

    segs, labels, centers = [], [], []
    for ws, lab in kept:
        i0 = int(round(ws * sfreq))
        i1 = i0 + win_n
        if i1 > n_samples:
            continue
        seg = data[:, i0:i1].astype(np.float32)
        mu = seg.mean(axis=1, keepdims=True)
        std = seg.std(axis=1, keepdims=True) + 1e-8
        segs.append((seg - mu) / std)
        labels.append(lab)
        centers.append(ws + 0.5 * window_sec)

    return (np.stack(segs), np.array(labels, dtype=np.int64),
            np.array(centers, dtype=np.float64), channel_names, sfreq)


# ── Dataset assembly ──────────────────────────────────────────────────────────

def build_dataset(
        include_ecg: bool,
        interictal_ratio: float = DEFAULT_INTERICTAL_RATIO,
        window_sec: float = DEFAULT_WINDOW_SEC,
        step_sec: float = DEFAULT_STEP_SEC,
        base: Path = SEIZEIT2_BASE,
        subjects: list[str] | None = None,
        random_state: int = 42,
) -> dict:
    """Build the full windowed dataset across all (or selected) SeizeIT2 recordings."""
    rng = np.random.default_rng(random_state)
    recordings = discover_recordings(base)
    if subjects:
        wanted = set(subjects)
        recordings = [r for r in recordings if r.subject in wanted]
    if include_ecg:
        missing_ecg = [r.run for r in recordings if r.ecg_path is None]
        if missing_ecg:
            print(f"  [WARN] Skipping {len(missing_ecg)} recording(s) with no ECG file: "
                  f"{', '.join(missing_ecg)}")
        recordings = [r for r in recordings if r.ecg_path is not None]
    if not recordings:
        raise ValueError(f"No recordings found under {base}")

    all_x, all_y, all_centers = [], [], []
    rec_paths, event_paths, file_slices = [], [], []
    channel_names: list[str] = []
    sfreq = 0.0
    start_idx = 0

    for rec in tqdm(recordings, desc="Recordings", unit="rec"):
        x, y, centers, ch, sf = build_recording_windows(
            rec, include_ecg, window_sec, step_sec, interictal_ratio, rng
        )
        if len(x) == 0:
            continue
        if not channel_names:
            channel_names, sfreq = ch, sf
        all_x.append(x)
        all_y.append(y)
        all_centers.append(centers)
        rec_paths.append(str(rec.eeg_path))
        event_paths.append(str(rec.events_path))
        file_slices.append((start_idx, start_idx + len(x)))
        start_idx += len(x)
        print(f"  [{rec.subject}] {rec.run}: {len(x):>5} windows "
              f"({int(y.sum())} pre-ictal, {int((y == 0).sum())} interictal)")

    if not all_x:
        raise ValueError("No pre-ictal windows found — no usable recordings.")

    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    centers = np.concatenate(all_centers, axis=0)
    print(f"\nDataset  total={len(x)}  pre-ictal={int(y.sum())}  "
          f"interictal={int((y == 0).sum())}  channels={x.shape[1]}  "
          f"timepoints={x.shape[2]}")

    return {
        "x": x, "y": y, "centers": centers,
        "channel_names": channel_names, "sfreq": sfreq,
        "window_sec": window_sec, "step_sec": step_sec,
        "interictal_ratio": interictal_ratio,
        "recording_paths": rec_paths, "event_paths": event_paths,
        "file_slices": file_slices,
    }


# ── Save / load the preprocessed dataset ──────────────────────────────────────

def save_preprocessed(out_path: Path, result: dict) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=result["x"], y=result["y"], centers=result["centers"],
        channel_names=np.array(result["channel_names"]),
        sfreq=np.array(result["sfreq"], dtype=np.float64),
        window_sec=np.array(result["window_sec"], dtype=np.float64),
        step_sec=np.array(result["step_sec"], dtype=np.float64),
        notch_freq=np.array(SEIZEIT2_NOTCH_HZ, dtype=np.float64),
        interictal_ratio=np.array(result["interictal_ratio"], dtype=np.float64),
        preictal_sec=np.array(PREICTAL_SEC, dtype=np.float64),
        recording_paths=np.array(result["recording_paths"]),
        event_paths=np.array(result["event_paths"]),
        file_slices=np.array(result["file_slices"], dtype=np.int64).reshape(-1, 2),
    )
    print(f"[Preprocess] Saved preprocessed dataset to {out_path}")


def load_preprocessed(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found: {path}\n"
            f"Run the matching preprocess script first to generate it."
        )
    npz = np.load(path, allow_pickle=False)
    return {
        "X": npz["X"], "y": npz["y"], "centers": npz["centers"],
        "channel_names": [str(c) for c in npz["channel_names"]],
        "sfreq": float(npz["sfreq"]),
        "window_sec": float(npz["window_sec"]),
        "step_sec": float(npz["step_sec"]),
        "notch_freq": float(npz["notch_freq"]),
        "interictal_ratio": float(npz["interictal_ratio"]),
        "preictal_sec": float(npz["preictal_sec"]),
        "recording_paths": [Path(p) for p in npz["recording_paths"]],
        "event_paths": [Path(p) for p in npz["event_paths"]],
        "file_slices": [tuple(int(v) for v in row) for row in npz["file_slices"]],
    }


# ── Subject-aware split ───────────────────────────────────────────────────────

def _subject_from_path(path) -> str:
    """Extract the BIDS subject id from a recording path (e.g. 'sub-004')."""
    return Path(path).name.split("_")[0]


def subject_aware_split(
        data: dict,
        train_frac: float = 0.8,
        train_subjects: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subject-level split: every window of a subject goes *entirely* to train or
    *entirely* to test — no subject appears in both sets.

    If `train_subjects` is provided, those subjects go to train and the rest to
    test.  Otherwise the subjects are sorted by ID and the first `train_frac`
    fraction is used for training (e.g. sub-001..sub-016 for 20 subjects at 0.8).

    This avoids the chronological-window-split pitfall where pre-ictal windows
    (which precede seizure onset) end up in train while the post-seizure
    interictal tail fills the test set, leaving several subjects with zero
    pre-ictal test windows.

    Returns sorted (train_idx, test_idx) into the concatenated window array.
    """
    n = len(data["X"])
    subj_of = np.empty(n, dtype=object)
    all_subjects: list[str] = []
    for (s, e), p in zip(data["file_slices"], data["recording_paths"]):
        subj = _subject_from_path(p)
        subj_of[s:e] = subj
        if subj not in all_subjects:
            all_subjects.append(subj)

    if train_subjects is None:
        sorted_subjects = sorted(all_subjects)
        k = min(max(int(round(len(sorted_subjects) * train_frac)), 1),
                len(sorted_subjects) - 1)
        train_set = set(sorted_subjects[:k])
    else:
        train_set = set(train_subjects)

    train_mask = np.array([s in train_set for s in subj_of], dtype=bool)
    return np.where(train_mask)[0], np.where(~train_mask)[0]
