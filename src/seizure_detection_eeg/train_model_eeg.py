# -*- coding: utf-8 -*-
"""
Training stage for the EEG-only seizure-detection pipeline.

Loads the preprocessed dataset (.npz from preprocess_eeg.py), takes the first
80% of the chronological sequence as the training set, trains a 1-D CNN, and
saves the model. Nothing else: evaluation, metrics, and plots live in
evaluate_eeg.py — this keeps a clean separation of concerns.

The train/test split fraction is stored in the checkpoint so evaluate_eeg.py
reconstructs exactly the same held-out 20% test set.

Example:
  python train_model_eeg.py
  python train_model_eeg.py --epochs 30 --ensemble-runs 5
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from preprocess_eeg import DEFAULT_PREPROCESSED_PATH, load_preprocessed

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "../../models/seizure_detection_eeg/seizure_cnn.pt"
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

    # Class imbalance: weight the positive (seizure) class by how much rarer it is.
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


# ── Model I/O ─────────────────────────────────────────────────────────────────

def save_models(models: list[SeizureCNN], path: Path, meta: dict) -> None:
    """Save one or more trained models (ensemble) in a single checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dicts": [m.state_dict() for m in models], "meta": meta},
        path,
    )
    print(f"[Model] Saved {len(models)} model(s) to {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN seizure detector on the preprocessed EEG dataset."
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_PREPROCESSED_PATH,
        help="Path to the preprocessed dataset (.npz) from preprocess_eeg.py.",
    )
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="Fraction used for training (rest is held-out test). "
                             "Default 0.8 = 80%% train / 20%% test.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ensemble-runs", type=int, default=1,
                        help="Train N independent models; evaluate_eeg.py averages "
                             "their class probabilities (soft voting). Default 1 = no ensemble.")
    parser.add_argument("--save-model", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    if not (0.1 <= args.train_frac < 1.0):
        raise ValueError("--train-frac must be in [0.1, 1.0)")

    device = pick_device(args.no_gpu)

    # ── load the preprocessed dataset ──
    print(f"\n[Data] Loading preprocessed dataset: {args.data}")
    data = load_preprocessed(args.data)
    x, y = data["X"], data["y"]
    print(f"[Data] {len(x)} windows, {x.shape[1]} channels, {x.shape[2]} timepoints/window, "
          f"notch={data['notch_freq']:.0f} Hz, window={data['window_sec']:.1f}s")

    # ── chronological split: train on the first train_frac only ──
    split = int(len(x) * args.train_frac)
    split = max(10, min(split, len(x) - 10))
    x_train, y_train = x[:split], y[:split]
    n_channels, n_timepoints = x.shape[1], x.shape[2]
    print(f"\nTraining on first {split} of {len(x)} windows "
          f"({args.train_frac:.0%}); {int(y_train.sum())} seizure window(s).")

    # ── train one or more models ──
    n_runs = max(1, args.ensemble_runs)
    models: list[SeizureCNN] = []
    for run in range(n_runs):
        run_seed = args.random_state + run
        print(f"\nTraining CNN (run {run + 1}/{n_runs}, seed={run_seed}) …")
        m = train_model(
            x_train, y_train, device,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, random_state=run_seed,
        )
        models.append(m)

    # ── save (split fraction stored so evaluate reconstructs the same test set) ──
    meta = {
        "n_channels": n_channels,
        "n_timepoints": n_timepoints,
        "train_frac": args.train_frac,
        "window_sec": data["window_sec"],
        "notch_freq": data["notch_freq"],
        "n_runs": n_runs,
        "epochs": args.epochs,
        "random_state": args.random_state,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "data_path": str(args.data),
    }
    save_models(models, args.save_model, meta)
    print("\nDone. Evaluate with:  python evaluate_eeg.py")


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
