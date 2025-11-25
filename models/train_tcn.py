# models/train_tcn.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from .config import (
    CHECKPOINT_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    INPUT_FEATURES,
    TCN_NUM_CHANNELS,
    TCN_KERNEL_SIZE,
    TCN_DROPOUT,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
)
from .datasets import DischargeSequenceDataset, make_train_val_test_loaders
from .tcn import TCN
from . import plots


def train_tcn_main() -> None:
    torch.manual_seed(RANDOM_SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = DischargeSequenceDataset()
    train_loader, val_loader, test_loader = make_train_val_test_loaders(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TCN(
        input_dim=len(INPUT_FEATURES),
        num_channels=TCN_NUM_CHANNELS,
        kernel_size=TCN_KERNEL_SIZE,
        dropout=TCN_DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for X, y, _meta in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            X = X.to(device)  # [B, T, F]
            y = y.to(device)  # [B, T]

            optimizer.zero_grad()
            y_hat, _ = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            batch_size = X.size(0)
            train_loss += loss.item() * batch_size
            n_train += batch_size

        train_loss /= max(n_train, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for X, y, _meta in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                X = X.to(device)
                y = y.to(device)

                y_hat, _ = model(X)
                loss = criterion(y_hat, y)

                batch_size = X.size(0)
                val_loss += loss.item() * batch_size
                n_val += batch_size

        val_loss /= max(n_val, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # checkpoint on best val
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = CHECKPOINT_DIR / "tcn_best.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "input_dim": len(INPUT_FEATURES),
                        "num_channels": TCN_NUM_CHANNELS,
                        "kernel_size": TCN_KERNEL_SIZE,
                        "dropout": TCN_DROPOUT,
                    },
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path}")

    # Plot training curves
    plots.plot_loss_curves(history, PLOTS_DIR / "tcn_loss_curves.png")

    # Simple test-set evaluation (MSE)
    test_loss = evaluate_on_loader(model, dataset, test_loader, device)
    print(f"Test MSE (SOH): {test_loss:.6f}")


def evaluate_on_loader(
    model: TCN,
    dataset: DischargeSequenceDataset,
    loader,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for X, y, _meta in loader:
            X = X.to(device)
            y = y.to(device)

            y_hat, _ = model(X)
            loss = criterion(y_hat, y)

            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            n += batch_size

    return total_loss / max(n, 1)


if __name__ == "__main__":
    train_tcn_main()
