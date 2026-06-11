import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np


DEFAULT_EDF = Path("physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf")
DEFAULT_SUMMARY = Path("physionet.org/files/chbmit/1.0.0/chb01/chb01-summary.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CHB-MIT EEG and highlight seizure intervals from summary text."
    )
    parser.add_argument("--edf", type=Path, default=DEFAULT_EDF, help="Path to EDF file")
    parser.add_argument(
        "--summary", type=Path, default=DEFAULT_SUMMARY, help="Path to chb01-summary.txt"
    )
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=10.0, help="Window duration in seconds")
    parser.add_argument(
        "--max-channels",
        type=int,
        default=8,
        help="Maximum number of EEG channels to display",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="",
        help="Comma-separated channel names (optional)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("plot_chb01_01_overlay.png"),
        help="Output image path. If omitted, default file is used.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also show interactive window in addition to saving the file.",
    )
    return parser.parse_args()


def _extract_file_section(summary_text: str, edf_name: str) -> str:
    escaped = re.escape(edf_name)
    match = re.search(rf"File Name:\s*{escaped}\s*(.*?)(?=\nFile Name:|\Z)", summary_text, re.S)
    if not match:
        return ""
    return match.group(1)


def parse_seizure_intervals(summary_path: Path, edf_name: str) -> list[tuple[float, float]]:
    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    section = _extract_file_section(text, edf_name)
    if not section:
        return []

    starts = [float(v) for v in re.findall(r"Seizure\s*\d+\s*Start Time:\s*(\d+)\s*seconds", section)]
    ends = [float(v) for v in re.findall(r"Seizure\s*\d+\s*End Time:\s*(\d+)\s*seconds", section)]

    count = min(len(starts), len(ends))
    intervals: list[tuple[float, float]] = []
    for i in range(count):
        start, end = starts[i], ends[i]
        if end > start:
            intervals.append((start, end))
    return intervals


def resolve_channels(raw: mne.io.BaseRaw, channels_arg: str, max_channels: int) -> list[str]:
    if channels_arg.strip():
        selected = [c.strip() for c in channels_arg.split(",") if c.strip()]
        missing = [c for c in selected if c not in raw.ch_names]
        if missing:
            raise ValueError(f"Channel(s) not found: {missing}")
        return selected

    eeg_channels = [
        ch for ch, typ in zip(raw.ch_names, raw.get_channel_types()) if typ == "eeg"
    ]
    selected = eeg_channels[: max(1, max_channels)]
    if not selected:
        raise ValueError("No EEG channels available for plotting.")
    return selected


def plot_with_seizure_overlays(
    edf_path: Path,
    summary_path: Path,
    start: float,
    duration: float,
    max_channels: int,
    channels_arg: str,
    save_path: Path,
    show: bool,
) -> None:
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    if start < 0:
        raise ValueError("--start must be >= 0")
    if duration <= 0:
        raise ValueError("--duration must be > 0")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    total_duration = raw.n_times / sfreq

    if start >= total_duration:
        raise ValueError(
            f"--start ({start:.2f}s) exceeds recording length ({total_duration:.2f}s)."
        )

    end = min(start + duration, total_duration)
    picked_channels = resolve_channels(raw, channels_arg, max_channels)
    seizure_intervals = parse_seizure_intervals(summary_path, edf_path.name)

    cropped = raw.copy().pick(picked_channels).crop(tmin=start, tmax=end, include_tmax=False)
    data, times = cropped.get_data(return_times=True)
    data_uV = data * 1e6

    fig, ax = plt.subplots(figsize=(12, 6))
    spacing = np.max(np.ptp(data_uV, axis=1)) * 1.2
    if spacing == 0:
        spacing = 1.0

    offsets = np.arange(len(picked_channels))[::-1] * spacing
    for i, name in enumerate(picked_channels):
        ax.plot(times + start, data_uV[i] + offsets[i], linewidth=0.9, color="black")

    has_overlap = False
    for seizure_start, seizure_end in seizure_intervals:
        overlap_start = max(seizure_start, start)
        overlap_end = min(seizure_end, end)
        if overlap_end > overlap_start:
            ax.axvspan(overlap_start, overlap_end, color="red", alpha=0.2)
            has_overlap = True

    if has_overlap:
        ax.text(
            0.99,
            0.99,
            "Red region = seizure interval",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    ax.set_yticks(offsets)
    ax.set_yticklabels(picked_channels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(f"{edf_path.name} with seizure overlays ({start:.2f}-{end:.2f}s)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    fig.savefig(save_path, dpi=150)
    print(f"Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_with_seizure_overlays(
        edf_path=args.edf,
        summary_path=args.summary,
        start=args.start,
        duration=args.duration,
        max_channels=args.max_channels,
        channels_arg=args.channels,
        save_path=args.save,
        show=args.show,
    )


if __name__ == "__main__":
    main()

