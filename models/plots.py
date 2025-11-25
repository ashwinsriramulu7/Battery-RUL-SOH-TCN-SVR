# models/plots.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import MASTER_DATA_PATH, INPUT_FEATURES, TARGET_SOH_COL, TARGET_RUL_COL


def plot_loss_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (SOH)")
    plt.legend()
    plt.title("TCN Training and Validation Loss")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_example_cycle(battery_id: str, cycle_index: int, save_path: Path) -> None:
    df = pd.read_csv(MASTER_DATA_PATH)
    df = df[(df["battery_id"] == battery_id) & (df["cycle_index"] == cycle_index)].copy()
    if df.empty:
        raise ValueError(f"No data for battery {battery_id}, cycle {cycle_index}")

    t = df["time_s"].to_numpy()

    plt.figure(figsize=(10, 8))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, df["voltage"], label="Voltage (V)")
    ax1.set_ylabel("Voltage (V)")
    ax1.legend(loc="best")

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t, df["current"], label="Current (A)")
    ax2.set_ylabel("Current (A)")
    ax2.legend(loc="best")

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(t, df[TARGET_SOH_COL], label="SOH")
    ax3.set_ylabel("SOH")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc="best")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_rul_predictions(true_rul: np.ndarray, pred_rul: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(true_rul, pred_rul, alpha=0.5)
    lims = [0, max(true_rul.max(), pred_rul.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("True RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title("SVR RUL Predictions")
    plt.grid(True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
