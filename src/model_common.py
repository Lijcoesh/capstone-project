# -*- coding: utf-8 -*-
"""
Shared training core for both seizure-prediction pipelines (EEG and EEG+ECG).

The 1-D CNN adapts to the number of input channels (2 for EEG, 3 for EEG+ECG),
so the architecture and training are identical between pipelines. The thin
train_model_*.py wrappers only set the dataset/model paths and call run_training.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from preprocess_common import load_preprocessed, subject_aware_split


# ── 1-D CNN model ─────────────────────────────────────────────────────────────

class SeizureCNN(nn.Module):
    def __init__(self, n_channels: int, n_timepoints: int) -> None:
        super().__init__()
        # Guard: after two MaxPool1d(4) the sequence is n_timepoints//16.
        # AdaptiveAvgPool1d(8) requires at least 1 element going in.
        min_len = n_timepoints // 16
        if min_len < 1:
            raise ValueError(
                f"n_timepoints={n_timepoints} is too short for two MaxPool1d(4) layers "
                f"(minimum required: 16)."
            )

        self.conv_block = nn.Sequential(
            # bias=False: BatchNorm subtracts the channel mean, making conv bias a no-op.
            nn.Conv1d(n_channels, 32, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout1d(0.25),   # zeros full channels, correct for 1-D conv

            nn.Conv1d(32, 64, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout1d(0.25),

            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Dropout1d(0.25),
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


class BandPowerSeqCNN(nn.Module):
    """1-D CNN over a band-power sequence: input (channels*bands, n_frames).

    Lighter than SeizureCNN (small kernels, no aggressive pooling) because the input
    is a short feature sequence (~tens of frames), not a long raw waveform. It convolves
    over time so it can pick up how the spectral content evolves toward onset — the
    temporal dynamics a per-window RandomForest averages away.
    """

    def __init__(self, n_features: int, n_frames: int) -> None:
        super().__init__()
        if n_frames < 1:
            raise ValueError(f"n_frames={n_frames} must be >= 1.")

        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout1d(0.3),

            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Dropout1d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block(x))


def build_model(input_rep: str, n_channels: int, n_timepoints: int) -> nn.Module:
    """Pick the architecture for the input representation stored in the dataset."""
    if input_rep == "bandpower_seq":
        return BandPowerSeqCNN(n_channels, n_timepoints)
    return SeizureCNN(n_channels, n_timepoints)


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

@torch.no_grad()
def _val_positive_prob(model: SeizureCNN, x: np.ndarray, device: torch.device,
                       batch_size: int, amp_on: bool) -> np.ndarray:
    """p(pre-ictal) over a held-out array, batched (keeps VRAM low; x stays on CPU)."""
    device_type = device.type if device.type in ("cuda", "cpu") else "cpu"
    model.eval()
    probs = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).to(device)
        with torch.amp.autocast(device_type, enabled=amp_on):
            out = model(xb)
        probs.append(torch.softmax(out.float(), dim=1)[:, 1].cpu().numpy())
    model.train()
    return np.concatenate(probs) if probs else np.empty(0, np.float32)


def train_model(
        x_train: np.ndarray,
        y_train: np.ndarray,
        device: torch.device,
        epochs: int,
        batch_size: int,
        lr: float,
        random_state: int,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        patience: int = 5,
        use_amp: bool = True,
        input_rep: str = "raw",
) -> nn.Module:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    n_channels, n_timepoints = x_train.shape[1], x_train.shape[2]
    model = build_model(input_rep, n_channels, n_timepoints).to(device)

    # Class imbalance: weight the positive (pre-ictal) class by how much rarer it is.
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_w = (n_neg / n_pos) if n_pos > 0 else 1.0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision ~2x on CUDA; silently a no-op on CPU/MPS.
    device_type = device.type if device.type in ("cuda", "cpu") else "cpu"
    amp_on = bool(use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device_type, enabled=amp_on)

    # Keep the full training set in CPU RAM and move only each batch to the GPU.
    # Moving the whole set with .to(device) OOMs on small cards (e.g. 4 GB laptop
    # GPUs) once the cohort is large; per-batch transfer is a tiny PCIe copy.
    pin = device.type == "cuda"
    dl = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                    batch_size=batch_size, shuffle=True, pin_memory=pin)

    # Early stopping on validation AUC (ranking quality, the metric we report). The
    # model underfits more than it overfits here, so this mostly saves time: it stops
    # once val-AUC plateaus instead of always running all `epochs`.
    can_early_stop = (x_val is not None and y_val is not None
                      and len(x_val) > 0 and len(np.unique(y_val)) > 1)
    best_auc, best_state, no_improve, best_epoch = -1.0, None, 0, 0

    model.train()
    epoch_bar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        total_loss = 0.0
        batch_bar = tqdm(dl, desc=f"  Epoch {epoch:3d}", unit="batch", leave=False)
        for xb, yb in batch_bar:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type, enabled=amp_on):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(xb)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        if can_early_stop:
            val_auc = roc_auc_score(
                y_val, _val_positive_prob(model, x_val, device, batch_size * 4, amp_on))
            epoch_bar.set_postfix(loss=f"{total_loss / len(x_train):.4f}",
                                  val_auc=f"{val_auc:.4f}", best=f"{best_auc:.4f}")
            if val_auc > best_auc + 1e-4:
                best_auc, best_epoch, no_improve = val_auc, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n[EarlyStop] No val-AUC gain for {patience} epochs; "
                          f"stopping at epoch {epoch} (best {best_auc:.4f} @ epoch {best_epoch}).")
                    break
        else:
            epoch_bar.set_postfix(loss=f"{total_loss / len(x_train):.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[BestModel] Restored val-AUC {best_auc:.4f} from epoch {best_epoch}.")

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
    parser.add_argument("--train-frac", type=float, default=0.6,
                        help="Fraction of each subject's windows (chronological) used for "
                             "training (default 0.6). Ignored when --train-subjects is set.")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Fraction used for validation, taken right after the train block "
                             "(default 0.2). The remaining 1-train-val is the held-out test set.")
    parser.add_argument("--train-subjects", type=str, default=None,
                        help="Comma-separated subject IDs to use for training, e.g. "
                             "'sub-001,sub-002,...,sub-016'. Overrides --train-frac.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max epochs; early stopping usually halts well before this.")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early-stop after this many epochs without val-AUC improvement "
                             "(default 8). Ignored when there is no validation set.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable CUDA mixed-precision (AMP). AMP is ~2x faster and on "
                             "by default; use this only to rule it out as a cause of issues.")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ensemble-runs", type=int, default=5,
                        help="Train N independent models; evaluation averages their "
                             "class probabilities (soft voting). Default 5.")
    parser.add_argument("--save-model", type=Path, default=default_model)


def run_training(args: argparse.Namespace) -> None:
    if not (0.1 <= args.train_frac < 1.0):
        raise ValueError("--train-frac must be in [0.1, 1.0)")
    if not (0.0 <= args.val_frac < 1.0):
        raise ValueError("--val-frac must be in [0.0, 1.0)")
    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError("--train-frac + --val-frac must be < 1.0 (leave room for test)")

    device = pick_device(args.no_gpu)

    print(f"\n[Data] Loading preprocessed dataset: {args.data}")
    data = load_preprocessed(args.data)
    x, y = data["X"], data["y"]
    input_rep = data.get("input_rep", "raw")
    print(f"[Data] {len(x)} windows, {x.shape[1]} features, {x.shape[2]} steps/window, "
          f"input_rep={input_rep}, notch={data['notch_freq']:.0f} Hz, "
          f"window={data['window_sec']:.1f}s, interictal_ratio={data['interictal_ratio']:.0f}")

    # within-subject class-stratified chronological split (default)
    # or subject-level split when --train-subjects is given
    train_subjects = (
        [s.strip() for s in args.train_subjects.split(",")]
        if args.train_subjects else None
    )
    train_idx, val_idx, _ = subject_aware_split(
        data, args.train_frac, args.val_frac, train_subjects)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    n_channels, n_timepoints = x.shape[1], x.shape[2]

    if train_subjects:
        print(f"\nSubject-level split: train subjects = {', '.join(sorted(train_subjects))}")
    else:
        print(f"\nWithin-subject 3-way split (train={args.train_frac}, val={args.val_frac}, "
              f"test={1 - args.train_frac - args.val_frac:.2f}): "
              f"train={len(x_train):,} windows ({int(y_train.sum()):,} pre-ictal), "
              f"val={len(val_idx):,}")

    n_runs = max(1, args.ensemble_runs)
    models: list[nn.Module] = []
    for run in range(n_runs):
        run_seed = args.random_state + run
        print(f"\nTraining CNN (run {run + 1}/{n_runs}, seed={run_seed}) ...")
        models.append(train_model(
            x_train, y_train, device,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, random_state=run_seed,
            x_val=x_val, y_val=y_val,
            patience=args.patience, use_amp=not args.no_amp,
            input_rep=input_rep,
        ))

    meta = {
        "task": "prediction",
        "split": "subject_level" if train_subjects else "within_subject_stratified",
        "input_rep": input_rep,
        "n_channels": n_channels,
        "n_timepoints": n_timepoints,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "train_subjects": train_subjects,
        "window_sec": data["window_sec"],
        "notch_freq": data["notch_freq"],
        "interictal_ratio": data["interictal_ratio"],
        "preictal_sec": data["preictal_sec"],
        "n_runs": n_runs,
        "epochs": args.epochs,
        "patience": args.patience,
        "amp": not args.no_amp,
        "random_state": args.random_state,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "data_path": str(args.data),
    }
    save_models(models, args.save_model, meta)
    print("\nDone. Evaluate next.")
