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

PREICTAL_SEC = 30 * 60          # 10 min before onset = pre-ictal (positive)
POSTICTAL_GUARD_SEC = 10 * 60   # exclude the seizure + 10 min after it

# Labeling task:
#   "prediction" — pre-ictal (before onset) vs interictal. The real research task.
#   "detection"  — ictal (during the seizure) vs interictal. Much easier; used as a
#                  sanity check ("can this pipeline learn anything at all?"). If
#                  detection AUC is high while prediction sits at chance, the pipeline
#                  is sound and prediction is simply the hard part.
LABEL_MODE_CHOICES = ("prediction", "detection")
DEFAULT_LABEL_MODE = "prediction"
# In detection mode, drop windows within this guard around each seizure so the
# interictal class is clearly away from any seizure (no pre/post-ictal contamination).
DETECTION_GUARD_SEC = 10 * 60

DEFAULT_INTERICTAL_RATIO = 5.0  # interictal : positive (pre-ictal/ictal) windows kept
DEFAULT_WINDOW_SEC = 2.0
DEFAULT_STEP_SEC = 1.0

# Window normalization scheme:
#   "per_window"    — z-score each 2 s window independently (removes within-window
#                     amplitude).
#   "per_recording" — z-score with one mean/std per channel over the whole recording
#                     (keeps amplitude/power dynamics across windows; only removes the
#                     per-patient/per-electrode scale). This is the default — it scored
#                     higher on validation AUC than per_window.
NORMALIZE_CHOICES = ("per_window", "per_recording")
DEFAULT_NORMALIZE = "per_recording"

# Input representation:
#   "raw"           — the raw (notched, z-scored) waveform, shape (channels, time).
#   "bandpower_seq" — log band-power computed in short frames stepped across the
#                     window, shape (channels*bands, n_frames). This is the RF's
#                     band-power feature set, but kept as a TIME SEQUENCE so the CNN
#                     can see how the spectral content evolves toward onset — the
#                     temporal dynamics the RF averages away. Tiny vs raw (~270x
#                     smaller), so it also sidesteps the raw-waveform RAM blow-up.
INPUT_REP_CHOICES = ("raw", "bandpower_seq")
DEFAULT_INPUT_REP = "raw"
# Frames for bandpower_seq: 4 s frame, 1 s hop (e.g. a 60 s window -> 57 frames).
DEFAULT_FRAME_SEC = 4.0
DEFAULT_FRAME_STEP_SEC = 1.0
# Band-power bands (stay below the 50 Hz mains notch; gamma capped at 45 Hz).
BANDPOWER_BANDS = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
                   ("beta", 13.0, 30.0), ("gamma", 30.0, 45.0)]


def bandpower_sequence(seg: np.ndarray, sfreq: float,
                       frame_sec: float = DEFAULT_FRAME_SEC,
                       frame_step_sec: float = DEFAULT_FRAME_STEP_SEC) -> np.ndarray:
    """Raw window (channels, time) -> log band-power sequence (channels*bands, n_frames).

    Each frame's per-channel band-powers are flattened channel-major
    (ch0_delta, ch0_theta, ..., ch1_delta, ...), one column per frame.
    """
    n_ch, T = seg.shape
    frame_n = max(1, int(round(frame_sec * sfreq)))
    step_n = max(1, int(round(frame_step_sec * sfreq)))
    if T < frame_n:
        frame_n = T
    freqs = np.fft.rfftfreq(frame_n, d=1.0 / sfreq)
    masks = [(freqs >= lo) & (freqs < hi) for _, lo, hi in BANDPOWER_BANDS]
    cols = []
    for s0 in range(0, T - frame_n + 1, step_n):
        frame = seg[:, s0:s0 + frame_n]
        power = np.abs(np.fft.rfft(frame, axis=-1)) ** 2          # (n_ch, F)
        bp = np.stack([power[:, m].sum(axis=-1) for m in masks], axis=1)  # (n_ch, bands)
        cols.append(np.log1p(bp).reshape(-1))                     # (n_ch*bands,)
    return np.stack(cols, axis=1).astype(np.float32)              # (n_ch*bands, n_frames)


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
        ws: float, we: float, seizures: list[tuple[float, float]],
        preictal_sec: float = PREICTAL_SEC,
        label_mode: str = DEFAULT_LABEL_MODE,
) -> str:
    """Classify a window [ws, we) as 'exclude', 'positive', or 'interictal'.

    prediction mode (default): positive = pre-ictal [onset - preictal_sec, onset);
        exclude = the seizure itself + a post-ictal guard.
    detection mode: positive = ictal (overlaps [onset, onset + dur]); exclude = a
        guard band around each seizure so interictal stays clear of pre/post-ictal.
    """
    if label_mode == "detection":
        # positive: window overlaps the seizure itself
        for onset, dur in seizures:
            if we > onset and ws < onset + dur:
                return "positive"
        # exclude a guard band around each seizure (keeps interictal clean)
        for onset, dur in seizures:
            if we > onset - DETECTION_GUARD_SEC and ws < onset + dur + DETECTION_GUARD_SEC:
                return "exclude"
        return "interictal"

    # prediction (default)
    # exclude takes priority: ictal + post-ictal guard
    for onset, dur in seizures:
        ictal_start = onset
        post_end = onset + dur + POSTICTAL_GUARD_SEC
        if we > ictal_start and ws < post_end:
            return "exclude"
    # pre-ictal: within [onset - preictal_sec, onset)
    for onset, _dur in seizures:
        pre_start = onset - preictal_sec
        if we > pre_start and ws < onset:
            return "positive"
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
        preictal_sec: float = PREICTAL_SEC,
        normalize: str = DEFAULT_NORMALIZE,
        label_mode: str = DEFAULT_LABEL_MODE,
        input_rep: str = DEFAULT_INPUT_REP,
        frame_sec: float = DEFAULT_FRAME_SEC,
        frame_step_sec: float = DEFAULT_FRAME_STEP_SEC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    """
    Build the kept windows for one recording: ALL positive (pre-ictal in prediction
    mode, ictal in detection mode) + a subsampled set of interictal (ratio : 1).
    Returns (X, y, centers, channel_names, sfreq). Recordings with no seizures
    contribute nothing (no positives → returns empty).
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
    positive_starts: list[float] = []
    interictal_starts: list[float] = []
    for ws in starts:
        label = _classify_window(ws, ws + window_sec, seizures, preictal_sec, label_mode)
        if label == "positive":
            positive_starts.append(ws)
        elif label == "interictal":
            interictal_starts.append(ws)
        # 'exclude' -> dropped

    if not positive_starts:
        return (np.empty((0, 0, 0), np.float32), np.empty(0, np.int64),
                np.empty(0, np.float64), channel_names, sfreq)

    # subsample interictal to ratio : 1
    target = int(round(interictal_ratio * len(positive_starts)))
    if len(interictal_starts) > target:
        idx = rng.choice(len(interictal_starts), size=target, replace=False)
        interictal_starts = [interictal_starts[i] for i in sorted(idx)]

    kept = ([(ws, 1) for ws in positive_starts]
            + [(ws, 0) for ws in interictal_starts])
    kept.sort(key=lambda t: t[0])               # chronological order

    # Extract raw windows first (un-normalized); representation + normalization below.
    raw_segs, labels, centers = [], [], []
    for ws, lab in kept:
        i0 = int(round(ws * sfreq))
        i1 = i0 + win_n
        if i1 > n_samples:
            continue
        raw_segs.append(data[:, i0:i1].astype(np.float32))
        labels.append(lab)
        centers.append(ws + 0.5 * window_sec)

    if input_rep == "bandpower_seq":
        # Transform each raw window to its log band-power sequence, then z-score the
        # band-power features. per_recording z-scores each feature over ALL frames of
        # the recording (removes the per-subject spectral baseline -> attacks the
        # between-subject shortcut), keeping the temporal dynamics; per_window
        # z-scores within each window.
        X = np.stack([bandpower_sequence(s, sfreq, frame_sec, frame_step_sec)
                      for s in raw_segs])                 # (N, ch*bands, n_frames)
        if normalize == "per_recording":
            mu = X.mean(axis=(0, 2), keepdims=True)
            std = X.std(axis=(0, 2), keepdims=True) + 1e-8
        else:  # per_window
            mu = X.mean(axis=2, keepdims=True)
            std = X.std(axis=2, keepdims=True) + 1e-8
        X = (X - mu) / std
        feat_names = [f"{c}_{b}" for c in channel_names for b, _, _ in BANDPOWER_BANDS]
        return (X.astype(np.float32), np.array(labels, dtype=np.int64),
                np.array(centers, dtype=np.float64), feat_names, sfreq)

    # raw representation (default)
    # per-recording stats: one mean/std per channel over the whole recording, used
    # only for normalize == "per_recording".
    if normalize == "per_recording":
        rec_mu = data.mean(axis=1, keepdims=True).astype(np.float32)
        rec_std = data.std(axis=1, keepdims=True).astype(np.float32) + 1e-8

    segs = []
    for seg in raw_segs:
        if normalize == "per_recording":
            seg = (seg - rec_mu) / rec_std
        else:  # per_window (default)
            mu = seg.mean(axis=1, keepdims=True)
            std = seg.std(axis=1, keepdims=True) + 1e-8
            seg = (seg - mu) / std
        segs.append(seg)

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
        preictal_sec: float = PREICTAL_SEC,
        normalize: str = DEFAULT_NORMALIZE,
        require_ecg: bool = False,
        label_mode: str = DEFAULT_LABEL_MODE,
        input_rep: str = DEFAULT_INPUT_REP,
        frame_sec: float = DEFAULT_FRAME_SEC,
        frame_step_sec: float = DEFAULT_FRAME_STEP_SEC,
) -> dict:
    """Build the full windowed dataset across all (or selected) SeizeIT2 recordings.

    Set `require_ecg=True` on the EEG-only pipeline to restrict it to exactly the
    recordings that also have an ECG file, so the EEG-only and EEG+ECG datasets are
    built on the *same* recordings — a fair paired comparison (the two feature sets
    then differ only by the presence of the ECG channel, nothing else).
    """
    rng = np.random.default_rng(random_state)
    recordings = discover_recordings(base)
    if subjects:
        wanted = set(subjects)
        recordings = [r for r in recordings if r.subject in wanted]
    # include_ecg needs the ECG channel; require_ecg keeps the EEG-only set on the
    # identical recordings so the comparison is paired. Both => filter to ECG-bearing.
    if include_ecg or require_ecg:
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
            rec, include_ecg, window_sec, step_sec, interictal_ratio, rng,
            preictal_sec=preictal_sec, normalize=normalize, label_mode=label_mode,
            input_rep=input_rep, frame_sec=frame_sec, frame_step_sec=frame_step_sec,
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
        pos_label = "ictal" if label_mode == "detection" else "pre-ictal"
        print(f"  [{rec.subject}] {rec.run}: {len(x):>5} windows "
              f"({int(y.sum())} {pos_label}, {int((y == 0).sum())} interictal)")

    if not all_x:
        raise ValueError("No positive windows found — no usable recordings.")

    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    centers = np.concatenate(all_centers, axis=0)
    pos_label = "ictal" if label_mode == "detection" else "pre-ictal"
    dim2 = "n_frames" if input_rep == "bandpower_seq" else "timepoints"
    print(f"\nDataset  total={len(x)}  {pos_label}={int(y.sum())}  "
          f"interictal={int((y == 0).sum())}  features={x.shape[1]}  "
          f"{dim2}={x.shape[2]}  preictal_sec={int(preictal_sec)}  "
          f"normalize={normalize}  label_mode={label_mode}  input_rep={input_rep}")

    return {
        "x": x, "y": y, "centers": centers,
        "channel_names": channel_names, "sfreq": sfreq,
        "window_sec": window_sec, "step_sec": step_sec,
        "interictal_ratio": interictal_ratio,
        "preictal_sec": preictal_sec,
        "normalize": normalize,
        "label_mode": label_mode,
        "input_rep": input_rep,
        "frame_sec": frame_sec,
        "frame_step_sec": frame_step_sec,
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
        preictal_sec=np.array(result.get("preictal_sec", PREICTAL_SEC), dtype=np.float64),
        normalize=np.array(result.get("normalize", DEFAULT_NORMALIZE)),
        label_mode=np.array(result.get("label_mode", DEFAULT_LABEL_MODE)),
        input_rep=np.array(result.get("input_rep", DEFAULT_INPUT_REP)),
        frame_sec=np.array(result.get("frame_sec", DEFAULT_FRAME_SEC), dtype=np.float64),
        frame_step_sec=np.array(result.get("frame_step_sec", DEFAULT_FRAME_STEP_SEC), dtype=np.float64),
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
        "normalize": str(npz["normalize"]) if "normalize" in npz else DEFAULT_NORMALIZE,
        "label_mode": str(npz["label_mode"]) if "label_mode" in npz else DEFAULT_LABEL_MODE,
        "input_rep": str(npz["input_rep"]) if "input_rep" in npz else DEFAULT_INPUT_REP,
        "frame_sec": float(npz["frame_sec"]) if "frame_sec" in npz else DEFAULT_FRAME_SEC,
        "frame_step_sec": float(npz["frame_step_sec"]) if "frame_step_sec" in npz else DEFAULT_FRAME_STEP_SEC,
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
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        train_subjects: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Within-subject class-stratified chronological 3-way split (train/val/test).

    For each subject, pre-ictal and interictal windows are split independently and
    chronologically: the first `train_frac` of each class → train, the next
    `val_frac` → validation, the remainder → test. Staying chronological within
    each class avoids look-ahead / temporal leakage, and train < val < test in time
    (train is oldest, test is newest). The validation set is what you tune
    hyperparameters on (pre-ictal horizon, decision threshold); the test set is
    touched once, at the very end.

    Contrast with a plain chronological window split: pre-ictal windows precede
    seizure onset, so the post-seizure interictal tail fills the last 20% and
    subjects whose seizures fall early end up with zero pre-ictal test windows.

    If `train_subjects` is provided (list of subject IDs), a subject-level split is
    used instead: those subjects → train, all others → test, with NO validation set
    (val is returned empty). Subject-level is a secondary cross-patient analysis.

    Returns sorted (train_idx, val_idx, test_idx) into the concatenated window array.
    """
    n = len(data["X"])
    y = data["y"]
    empty = np.array([], dtype=int)
    subj_of = np.empty(n, dtype=object)
    order: list[str] = []
    for (s, e), p in zip(data["file_slices"], data["recording_paths"]):
        subj = _subject_from_path(p)
        subj_of[s:e] = subj
        if subj not in order:
            order.append(subj)

    # ── subject-level split (optional override): no validation set ─────────────
    if train_subjects is not None:
        train_set = set(train_subjects)
        train_mask = np.array([s in train_set for s in subj_of], dtype=bool)
        return np.where(train_mask)[0], empty, np.where(~train_mask)[0]

    # ── within-subject class-stratified chronological 3-way split (default) ────
    train_parts, val_parts, test_parts = [], [], []
    for subj in order:
        idx = np.nonzero(subj_of == subj)[0]   # already ascending = chronological
        for cls in (1, 0):                       # 1 = pre-ictal, 0 = interictal
            cls_idx = idx[y[idx] == cls]
            n_cls = len(cls_idx)
            if n_cls == 0:
                continue
            # chronological cut points; train gets >=1, val/test may be empty for
            # very small per-subject classes (rare once the cohort is large).
            k_tr = min(max(int(round(n_cls * train_frac)), 1), n_cls)
            k_va = min(max(int(round(n_cls * (train_frac + val_frac))), k_tr), n_cls)
            train_parts.append(cls_idx[:k_tr])
            if k_va > k_tr:
                val_parts.append(cls_idx[k_tr:k_va])
            if n_cls > k_va:
                test_parts.append(cls_idx[k_va:])

    def _cat(parts):
        return np.sort(np.concatenate(parts)) if parts else empty

    return _cat(train_parts), _cat(val_parts), _cat(test_parts)
