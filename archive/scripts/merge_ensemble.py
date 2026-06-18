"""One-off: merge per-seed checkpoints into canonical ensemble .pt files."""
from __future__ import annotations

import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [42, 43, 44, 45, 46]


def merge(pattern: str, out_rel: str, seeds: list[int]) -> None:
    paths = [ROOT / pattern.format(s=s) for s in seeds]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    meta = dict(first["meta"])
    meta["n_runs"] = len(seeds)
    meta["ensemble_seeds"] = seeds
    meta["ensemble_merged_from"] = [
        p.resolve().relative_to(ROOT).as_posix() for p in paths
    ]
    state_dicts = []
    for p in paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        state_dicts.append(ckpt["state_dicts"][0])
    out = ROOT / out_rel
    torch.save({"state_dicts": state_dicts, "meta": meta}, out)
    print(f"Merged {len(seeds)} models -> {out}  (n_runs={meta['n_runs']})")


def main() -> None:
    merge(
        "models/seizure_prediction_eeg/cnn_prediction_eeg_s{s}.pt",
        "models/seizure_prediction_eeg/cnn_prediction_eeg.pt",
        SEEDS,
    )
    merge(
        "models/seizure_prediction_eeg_ecg/cnn_prediction_eeg_ecg_s{s}.pt",
        "models/seizure_prediction_eeg_ecg/cnn_prediction_eeg_ecg.pt",
        SEEDS,
    )

    archive = ROOT / "archive/models/seed_checkpoints"
    for sub in ("seizure_prediction_eeg", "seizure_prediction_eeg_ecg"):
        dst = archive / sub
        dst.mkdir(parents=True, exist_ok=True)
        for s in SEEDS:
            for p in (ROOT / "models" / sub).glob(f"*_s{s}.pt"):
                shutil.move(str(p), str(dst / p.name))
                print(f"Archived {p.name} -> {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
