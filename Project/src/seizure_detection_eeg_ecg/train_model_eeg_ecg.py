from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

N_CHANNELS = 23
N_TIMEPOINTS = 512
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "../../models/seizure_detection_eeg_ecg/seizure_cnn.pt"
)

# These bounds are based on typical human heart rates and the training data distribution.
HR_MIN_BPM = 40.0
HR_MAX_BPM = 162.0


class SeizureCNN(nn.Module):
    def __init__(self, n_channels: int = N_CHANNELS, n_classes: int = 2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=16),
            nn.Identity(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block(x))


@dataclass
class HeartRateRange:
    index: int
    low_bpm: float
    high_bpm: float
    label: str


@dataclass
class SeizureHeartRateResult:
    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float
    seizure_probability: float
    heartrate_bpm: np.ndarray
    timestamps_s: np.ndarray
    baseline_bpm: float
    peak_bpm: float
    min_bpm: float
    mean_bpm: float
    change_from_baseline_bpm: float
    max_instantaneous_change_bpm: float
    mean_change_bpm_per_sample: float


def heartrate_range_bins(
    hr_min: float = HR_MIN_BPM,
    hr_max: float = HR_MAX_BPM,
    n_channels: int = N_CHANNELS,
) -> list[HeartRateRange]:
    edges = np.linspace(hr_min, hr_max, n_channels + 1)
    bins: list[HeartRateRange] = []
    for i in range(n_channels):
        low, high = float(edges[i]), float(edges[i + 1])
        bins.append(
            HeartRateRange(
                index=i,
                low_bpm=low,
                high_bpm=high,
                label=f"{low:.0f}-{high:.0f} bpm",
            )
        )
    return bins


def load_seizure_model(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    device: str | torch.device = "cpu",
) -> tuple[SeizureCNN, dict]:
    path = Path(model_path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    meta = checkpoint.get("meta", {})
    model = SeizureCNN(
        n_channels=int(meta.get("n_channels", N_CHANNELS)),
        n_classes=2,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, meta


def _resample_1d(values: np.ndarray, target_len: int) -> np.ndarray:
    if len(values) == target_len:
        return values.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, values).astype(np.float32)


def encode_heartrate_by_ranges(
    heartrate_bpm: Iterable[float],
    bins: list[HeartRateRange] | None = None,
    n_timepoints: int = N_TIMEPOINTS,
    mode: str = "occupancy",
) -> np.ndarray:
    bins = bins or heartrate_range_bins()
    hr = np.asarray(list(heartrate_bpm), dtype=np.float32)
    hr = _resample_1d(hr, n_timepoints)
    encoded = np.zeros((len(bins), n_timepoints), dtype=np.float32)
    for b in bins:
        in_bin = (hr >= b.low_bpm) & (hr < b.high_bpm)
        if b.index == len(bins) - 1:
            in_bin = (hr >= b.low_bpm) & (hr <= b.high_bpm)
        if mode == "occupancy":
            encoded[b.index] = in_bin.astype(np.float32)
        elif mode == "value":
            encoded[b.index] = np.where(in_bin, hr, 0.0).astype(np.float32)
        elif mode == "soft":
            center = (b.low_bpm + b.high_bpm) / 2.0
            sigma = max((b.high_bpm - b.low_bpm) / 2.0, 1.0)
            encoded[b.index] = np.exp(-0.5 * ((hr - center) / sigma) ** 2).astype(
                np.float32
            )
        else:
            raise ValueError(f"Unknown encoding mode: {mode}")
    return encoded


def predict_window_probs(
    model: SeizureCNN,
    encoded_window: np.ndarray,
    device: str | torch.device = "cpu",
    seizure_class: int = 0,
) -> tuple[float, float, float]:
    x = torch.from_numpy(encoded_window).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    p_seizure = float(probs[seizure_class].cpu())
    p_other = float(probs[1 - seizure_class].cpu())
    return p_seizure, p_other, p_seizure - p_other


def predict_seizure_score(
    model: SeizureCNN,
    encoded_window: np.ndarray,
    device: str | torch.device = "cpu",
    seizure_class: int = 0,
) -> float:
    return predict_window_probs(model, encoded_window, device, seizure_class)[0]


def _find_elevated_hr_segments(
    heartrate_bpm: np.ndarray,
    times: np.ndarray,
    resting_baseline: float,
    rise_bpm: float,
) -> list[tuple[int, int]]:
    elevated = heartrate_bpm >= resting_baseline + rise_bpm
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(elevated):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(heartrate_bpm)))
    return segments


def _baseline_before_segment(
    heartrate_bpm: np.ndarray, start: int, baseline_samples: int = 32
) -> float:
    lo = max(0, start - baseline_samples)
    if lo >= start:
        return float(np.mean(heartrate_bpm[: max(1, len(heartrate_bpm) // 10)]))
    return float(np.mean(heartrate_bpm[lo:start]))


def analyze_seizure_heartrate(
    heartrate_bpm: Iterable[float],
    timestamps_s: Iterable[float] | None = None,
    sample_rate_hz: float = 4.0,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    seizure_threshold: float | None = None,
    score_percentile: float = 92.0,
    min_mean_hr_rise_bpm: float = 8.0,
    use_hr_elevation_gate: bool = True,
    encoding_mode: str = "occupancy",
    seizure_class: int = 0,
    stride: int = 64,
    device: str | torch.device = "cpu",
    plot: bool = False,
    plot_path: Path | str | None = None,
) -> list[SeizureHeartRateResult]:
    hr = np.asarray(list(heartrate_bpm), dtype=np.float32)
    n = len(hr)
    if n < N_TIMEPOINTS:
        raise ValueError(
            f"Need at least {N_TIMEPOINTS} heart-rate samples, got {n}."
        )

    if timestamps_s is None:
        times = np.arange(n, dtype=np.float32) / sample_rate_hz
    else:
        times = np.asarray(list(timestamps_s), dtype=np.float32)
        if len(times) != n:
            raise ValueError("timestamps_s must match heartrate_bpm length.")

    model, _ = load_seizure_model(model_path, device=device)
    bins = heartrate_range_bins()

    window_probs: list[tuple[int, int, float, float]] = []
    scores: list[float] = []
    for start in range(0, n - N_TIMEPOINTS + 1, stride):
        end = start + N_TIMEPOINTS
        window_hr = hr[start:end]
        encoded = encode_heartrate_by_ranges(
            window_hr, bins=bins, mode=encoding_mode
        )
        prob, _, margin = predict_window_probs(
            model, encoded, device=device, seizure_class=seizure_class
        )
        window_probs.append((start, end, prob, margin))
        scores.append(prob)

    resting_baseline = float(np.median(hr[: max(1, n // 10)]))

    if seizure_threshold is None:
        score_arr = np.asarray(scores, dtype=np.float32)
        seizure_threshold = float(np.percentile(score_arr, score_percentile))
        seizure_threshold = max(seizure_threshold, 1e-6)

    elevated_spans = _find_elevated_hr_segments(
        hr, times, resting_baseline, min_mean_hr_rise_bpm
    )

    def _overlaps_elevated(win_start: int, win_end: int) -> bool:
        if not use_hr_elevation_gate:
            return True
        return any(
            not (win_end <= e0 or win_start >= e1) for e0, e1 in elevated_spans
        )

    def _best_cnn_score(span_start: int, span_end: int) -> float:
        best = 0.0
        for win_start, win_end, prob, _margin in window_probs:
            if not (win_end <= span_start or win_start >= span_end):
                best = max(best, prob)
        return best

    segments: list[tuple[int, int, float]] = []

    if use_hr_elevation_gate and elevated_spans:
        # Anchor segments on elevated heart rate (CNN scores are often miscalibrated
        # for range-encoded inputs that differ from training data).
        for e0, e1 in elevated_spans:
            segments.append((e0, e1, _best_cnn_score(e0, e1)))
    else:
        active_start: int | None = None
        active_end = 0
        active_prob = 0.0
        for start, end, prob, _margin in window_probs:
            if prob >= seizure_threshold and _overlaps_elevated(start, end):
                if active_start is None:
                    active_start, active_end, active_prob = start, end, prob
                else:
                    active_end = end
                    active_prob = max(active_prob, prob)
            elif active_start is not None:
                segments.append((active_start, active_end, active_prob))
                active_start = None
        if active_start is not None:
            segments.append((active_start, active_end, active_prob))

    results: list[SeizureHeartRateResult] = []
    for start, end, prob in segments:
        seg_hr = hr[start:end]
        seg_t = times[start:end]
        delta = np.diff(seg_hr, prepend=seg_hr[0])
        baseline = _baseline_before_segment(hr, start)

        results.append(
            SeizureHeartRateResult(
                start_index=start,
                end_index=end,
                start_time_s=float(seg_t[0]),
                end_time_s=float(seg_t[-1]),
                seizure_probability=float(prob),
                heartrate_bpm=seg_hr.copy(),
                timestamps_s=seg_t.copy(),
                baseline_bpm=baseline,
                peak_bpm=float(np.max(seg_hr)),
                min_bpm=float(np.min(seg_hr)),
                mean_bpm=float(np.mean(seg_hr)),
                change_from_baseline_bpm=float(np.mean(seg_hr) - baseline),
                max_instantaneous_change_bpm=float(np.max(np.abs(delta))),
                mean_change_bpm_per_sample=float(np.mean(delta)),
            )
        )

    if plot or plot_path is not None:
        _plot_seizure_heartrate(
            times, hr, results, window_probs, plot_path, seizure_threshold
        )
        if plot:
            plt.show()
        else:
            plt.close()

    return results


def _plot_seizure_heartrate(
    times: np.ndarray,
    heartrate_bpm: np.ndarray,
    seizures: list[SeizureHeartRateResult],
    window_probs: list[tuple[int, int, float, float]],
    plot_path: Path | str | None,
    seizure_threshold: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(times, heartrate_bpm, color="#2563eb", linewidth=1.2, label="Heart rate")
    for seg in seizures:
        axes[0].axvspan(
            seg.start_time_s,
            seg.end_time_s,
            color="#ef4444",
            alpha=0.2,
            label="Seizure segment" if seg is seizures[0] else None,
        )
    axes[0].set_ylabel("BPM")
    axes[0].set_title("Heart rate with detected seizure segments")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    mid_t = [(times[s] + times[e - 1]) / 2 for s, e, _, _ in window_probs]
    mid_p = [p for _, _, p, _ in window_probs]
    mid_margin = [m for _, _, _, m in window_probs]

    # Relative scores: raw CNN output is ~0.98–1.0, so plot rise above the minimum.
    p_min = min(mid_p) if mid_p else 0.0
    m_min = min(mid_margin) if mid_margin else 0.0
    delta_p = [p - p_min for p in mid_p]
    delta_m = [m - m_min for m in mid_margin]
    th_delta = seizure_threshold - p_min

    ax = axes[1]
    for seg in seizures:
        ax.axvspan(
            seg.start_time_s,
            seg.end_time_s,
            color="#ef4444",
            alpha=0.12,
        )
    ax.plot(
        mid_t,
        delta_p,
        color="#7c3aed",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="P(seizure) rise",
    )
    ax.plot(
        mid_t,
        delta_m,
        color="#059669",
        linewidth=1.2,
        linestyle="--",
        label="class margin rise",
    )
    ax.axhline(
        th_delta,
        color="#6b7280",
        linestyle=":",
        linewidth=1.0,
        label=f"threshold ({seizure_threshold:.3f} raw)",
    )
    y_hi = max(delta_p + delta_m + [th_delta, 0.001]) * 1.25
    ax.set_ylim(0.0, max(y_hi, 0.008))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score rise above minimum window")
    ax.set_title(
        "Sliding-window CNN score (relative; each point = 128 s window)"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if plot_path is not None:
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")


def simulate_seizure_heartrate(
    duration_s: float = 320.0,
    sample_rate_hz: float = 4.0,
    seizure_center_s: float = 160.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate_hz)
    t = np.arange(n, dtype=np.float32) / sample_rate_hz
    hr = np.full(n, 72.0, dtype=np.float32)
    hr += rng.normal(0, 1.5, size=n)

    center = int(seizure_center_s * sample_rate_hz)
    width = int(25 * sample_rate_hz)
    lo, hi = max(0, center - width), min(n, center + width)
    idx = np.arange(lo, hi)
    phase = (idx - lo) / max(1, hi - lo - 1)
    hr[idx] = 72 + 35 * np.sin(np.pi * phase) ** 0.7 + 20 * phase
    hr[idx] += rng.normal(0, 2.0, size=len(idx))

    return t, hr


def seizure_heartrate_changes(
    results: list[SeizureHeartRateResult],
) -> list[dict]:
    output: list[dict] = []
    for r in results:
        change = np.diff(r.heartrate_bpm, prepend=r.heartrate_bpm[0])
        output.append(
            {
                "start_time_s": r.start_time_s,
                "end_time_s": r.end_time_s,
                "heartrate_bpm": r.heartrate_bpm,
                "timestamps_s": r.timestamps_s,
                "change_bpm": change,
                "baseline_bpm": r.baseline_bpm,
                "mean_bpm": r.mean_bpm,
                "change_from_baseline_bpm": r.change_from_baseline_bpm,
                "peak_bpm": r.peak_bpm,
                "min_bpm": r.min_bpm,
                "max_instantaneous_change_bpm": r.max_instantaneous_change_bpm,
            }
        )
    return output


def format_seizure_summary(results: list[SeizureHeartRateResult]) -> str:
    if not results:
        return "No seizure segments detected."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"Seizure {i}: t={r.start_time_s:.1f}-{r.end_time_s:.1f}s, "
            f"P={r.seizure_probability:.2f}, "
            f"HR mean={r.mean_bpm:.1f} (baseline {r.baseline_bpm:.1f}, "
            f"delta={r.change_from_baseline_bpm:+.1f}), "
            f"peak={r.peak_bpm:.1f}, min={r.min_bpm:.1f}, "
            f"max |dHR|={r.max_instantaneous_change_bpm:.1f}/sample"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    times, hr = simulate_seizure_heartrate()
    seizures = analyze_seizure_heartrate(
        hr,
        timestamps_s=times,
        plot=False,
        plot_path=Path(__file__).resolve().parent
        / "../../results/seizure_detection_eeg_ecg/seizure_heartrate_plot.png",
    )
    print(format_seizure_summary(seizures))
    for s in seizures:
        print(
            f"\nReturned arrays: heartrate_bpm shape={s.heartrate_bpm.shape}, "
            f"changes (diff) sample={np.diff(s.heartrate_bpm)[:5]} ..."
        )
