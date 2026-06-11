import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize EEG channels from an EDF file.")
    parser.add_argument(
        "--edf",
        type=Path,
        default=Path("physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf"),
        help="Path to EDF file.",
    )
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--duration", type=float, default=10.0, help="Window length in seconds.")
    parser.add_argument(
        "--max-channels",
        type=int,
        default=8,
        help="Maximum number of channels to plot when --channels is not used.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="",
        help="Comma-separated channel names to plot (e.g. 'FP1-F7,F7-T7').",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output image path (e.g. plot.png).",
    )
    return parser.parse_args()


def _resolve_channels(raw: mne.io.BaseRaw, channels_arg: str, max_channels: int) -> list[str]:
    if channels_arg.strip():
        requested = [name.strip() for name in channels_arg.split(",") if name.strip()]
        missing = [name for name in requested if name not in raw.ch_names]
        if missing:
            raise ValueError(f"Channel(s) not found in EDF: {missing}")
        return requested

    eeg_channels = [
        ch for ch, ch_type in zip(raw.ch_names, raw.get_channel_types()) if ch_type == "eeg"
    ]
    selected = eeg_channels[: max(1, max_channels)]
    if not selected:
        raise ValueError("No EEG channels available to plot.")
    return selected


def plot_eeg(
    edf_path: Path,
    start: float,
    duration: float,
    max_channels: int,
    channels_arg: str,
    save_path: Path | None,
) -> None:
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    if duration <= 0:
        raise ValueError("--duration must be > 0")
    if start < 0:
        raise ValueError("--start must be >= 0")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    total_duration = raw.n_times / sfreq

    if start >= total_duration:
        raise ValueError(
            f"--start ({start:.2f}s) exceeds recording length ({total_duration:.2f}s)."
        )

    end = min(start + duration, total_duration)
    picked_channels = _resolve_channels(raw, channels_arg, max_channels)

    cropped = raw.copy().pick(picked_channels).crop(tmin=start, tmax=end, include_tmax=False)
    data, times = cropped.get_data(return_times=True)

    # Convert to microvolts to make y-axis labels intuitive for EEG users.
    data_uV = data * 1e6

    fig, ax = plt.subplots(figsize=(12, 6))
    spacing = np.max(np.ptp(data_uV, axis=1)) * 1.2
    if spacing == 0:
        spacing = 1.0

    offsets = np.arange(len(picked_channels))[::-1] * spacing
    for i, ch_name in enumerate(picked_channels):
        ax.plot(times + start, data_uV[i] + offsets[i], linewidth=0.9)

    ax.set_yticks(offsets)
    ax.set_yticklabels(picked_channels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(f"EEG segment: {edf_path.name} | {start:.2f}-{end:.2f}s")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    plot_eeg(
        edf_path=args.edf,
        start=args.start,
        duration=args.duration,
        max_channels=args.max_channels,
        channels_arg=args.channels,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()

