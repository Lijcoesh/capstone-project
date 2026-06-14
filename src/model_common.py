# -*- coding: utf-8 -*-
"""
Shared training core for both seizure-prediction pipelines (EEG and EEG+ECG).

The 1-D CNN adapts to the number of input channels (2 for EEG, 3 for EEG+ECG),
so the architecture and training are identical between pipelines. The thin
train_model_*.py wrappers only set the dataset/model paths and call run_training.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from preprocess_common import load_preprocessed, subject_aware_split


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

    # Class imbalance: weight the positive (pre-ictal) class by how much rarer it is.
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


def save_models(models: list[SeizureCNN], path: Path, meta: dict) -> None:
    """Save one or more trained models (ensemble) in a single checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dicts": [m.state_dict() for m in models], "meta": meta}, path)
    print(f"[Model] Saved {len(models)} model(s) to {path}")


# ── CLI helper + orchestration ────────────────────────────────────────────────

def add_training_args(parser: argparse.ArgumentParser, default_data: Path, default_model: Path) -> None:
    parser.add_argument("--data", type=Path, default=default_data,
                        help="Path to the preprocessed dataset (.npz).")
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="Fraction of subjects used for training (default 0.8 = first 80%% "
                             "by subject ID). Ignored when --train-subjects is set.")
    parser.add_argument("--train-subjects", type=str, default=None,
                        help="Comma-separated subject IDs to use for training, e.g. "
                             "'sub-001,sub-002,...,sub-016'. Overrides --train-frac.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ensemble-runs", type=int, default=1,
                        help="Train N independent models; evaluation averages their "
                             "class probabilities (soft voting). Default 1 = no ensemble.")
    parser.add_argument("--save-model", type=Path, default=default_model)


def run_training(args: argparse.Namespace) -> None:
    if not (0.1 <= args.train_frac < 1.0):
        raise ValueError("--train-frac must be in [0.1, 1.0)")

    device = pick_device(args.no_gpu)

    print(f"\n[Data] Loading preprocessed dataset: {args.data}")
    data = load_preprocessed(args.data)
    x, y = data["X"], data["y"]
    print(f"[Data] {len(x)} windows, {x.shape[1]} channels, {x.shape[2]} timepoints/window, "
          f"notch={data['notch_freq']:.0f} Hz, window={data['window_sec']:.1f}s, "
          f"interictal_ratio={data['interictal_ratio']:.0f}")

    # within-subject class-stratified chronological split (default)
    # or subject-level split when --train-subjects is given
    train_subjects = (
        [s.strip() for s in args.train_subjects.split(",")]
        if args.train_subjects else None
    )
    train_idx, _ = subject_aware_split(data, args.train_frac, train_subjects)
    x_train, y_train = x[train_idx], y[train_idx]
    n_channels, n_timepoints = x.shape[1], x.shape[2]

    if train_subjects:
        print(f"\nSubject-level split: train subjects = {', '.join(sorted(train_subjects))}")
    else:
        print(f"\nWithin-subject stratified split (train_frac={args.train_frac}): "
              f"train={len(x_train):,} windows ({int(y_train.sum()):,} pre-ictal)")

    n_runs = max(1, args.ensemble_runs)
    models: list[SeizureCNN] = []
    for run in range(n_runs):
        run_seed = args.random_state + run
        print(f"\nTraining CNN (run {run + 1}/{n_runs}, seed={run_seed}) ...")
        models.append(train_model(
            x_train, y_train, device,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, random_state=run_seed,
        ))

    meta = {
        "task": "prediction",
        "split": "subject_level" if train_subjects else "within_subject_stratified",
        "n_channels": n_channels,
        "n_timepoints": n_timepoints,
        "train_frac": args.train_frac,
        "train_subjects": train_subjects,
        "window_sec": data["window_sec"],
        "notch_freq": data["notch_freq"],
        "interictal_ratio": data["interictal_ratio"],
        "preictal_sec": data["preictal_sec"],
        "n_runs": n_runs,
        "epochs": args.epochs,
        "random_state": args.random_state,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "data_path": str(args.data),
    }
    save_models(models, args.save_model, meta)
    print("\nDone. Evaluate next.")
