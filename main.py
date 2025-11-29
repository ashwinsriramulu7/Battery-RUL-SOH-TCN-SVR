#!/usr/bin/env python3
"""
Invariant Degradation Modeling (IDM) pipeline for NASA Ames battery dataset.

This script implements the *NASA-specific* part of the methodology in a single file:

1. Load a master discharge CSV (one row per cycle across all batteries).
2. Apply Structure-Aware Normalization (SAN)-style stable/volatile partition:
   - Identify stable vs. volatile features using across-battery variance.
   - Normalize only volatile features per battery; keep stable features unchanged.
3. Construct temporal coordinates:
   - Normalized life fraction (tau = cycle_index / EOL_index).
   - Deployable coordinate (tau_deploy) based on cumulative discharged capacity.
4. Train a PyTorch Temporal Convolutional Network (TCN) to learn a smooth
   SOH trajectory (normalized capacity) from the SAN-transformed features.
5. Compute TCN-smoothed SOH and convert to RUL fraction labels.
6. Detect two knee points in the average degradation curve using `kneed`
   (with fallback to quantiles), defining three degradation phases.
7. Train three separate SVR models (early/mid/late) mapping TCN-smoothed SOH
   to RUL fraction.
8. Run ablations:
   - No SAN (global z-score normalization).
   - PCA whitening (instead of SAN).
   - SAN + no TCN (direct piecewise SVR on raw SOH).
   - SAN + TCN + single SVR (1-piece).
   - Full SAN + TCN + 3-piece SVR (proposed).
9. Evaluate:
   - RMSE, MAE, MAPE, R^2 on RUL fraction.
   - Per-battery and per-ambient-temperature metrics.
   - Long-horizon metrics (last 20% of life).
10. Generate plots and tables for inclusion in the paper.

Everything is NASA-specific but the logic matches the Methods section and
can be reused for other datasets by swapping the input CSV.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from kneed import KneeLocator

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. Global config / hyperparameters (for reproducibility)
# ============================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# TCN hyperparameters
TCN_NUM_LEVELS = 4           # number of residual blocks / dilation levels
TCN_HIDDEN_CHANNELS = 32     # number of channels in intermediate layers
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1
TCN_LR = 1e-3
TCN_WEIGHT_DECAY = 1e-4
TCN_BATCH_SIZE = 64
TCN_MAX_EPOCHS = 100
TCN_EARLY_STOP_PATIENCE = 10
TCN_WINDOW_SIZE = 16         # sliding window length for TCN input

# SVR hyperparameters (common across phases)
SVR_C = 10.0
SVR_EPSILON = 0.01
SVR_KERNEL = "rbf"

# SAN stable/volatile partition threshold (quantile of across-battery variance)
SAN_VARIANCE_QUANTILE = 0.4  # features below this quantile => "stable"

# Paths
ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
TABLES_DIR = ARTIFACTS_DIR / "tables"
for d in [ARTIFACTS_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Data loading and basic utilities
# ============================================================

def load_master_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load NASA master discharge dataset.

    Assumes columns:
    battery_id,cycle_number,start_time,ambient_temperature,capacity_ahr,
    mean_current,min_voltage,final_voltage,max_temperature

    Returns:
        df: pandas DataFrame with correct dtypes and sorted by (battery_id, cycle_number).
    """
    df = pd.read_csv(csv_path)

    # Basic type handling
    df["battery_id"] = df["battery_id"].astype(str)
    df["cycle_number"] = df["cycle_number"].astype(int)
    df["ambient_temperature"] = df["ambient_temperature"].astype(float)
    df["capacity_ahr"] = df["capacity_ahr"].astype(float)
    df["mean_current"] = df["mean_current"].astype(float)
    df["min_voltage"] = df["min_voltage"].astype(float)
    df["final_voltage"] = df["final_voltage"].astype(float)
    df["max_temperature"] = df["max_temperature"].astype(float)

    # Sort for reproducible temporal processing
    df = df.sort_values(["battery_id", "cycle_number"]).reset_index(drop=True)
    return df


def compute_battery_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-battery summary: EOL index, BOL capacity, EOL capacity, etc.

    EOL index is defined as the max cycle_number for that battery (NASA experiments
    are run until capacity fades to the threshold; the final cycle is taken as EOL).

    BOL capacity is taken as the maximum observed capacity for that battery
    (avoids sensitivity to transient regeneration in very early cycles).

    Returns:
        stats_df: battery-level DataFrame with columns:
                  battery_id, max_cycle, bol_capacity, eol_capacity
    """
    grouped = df.groupby("battery_id")
    stats = []
    for bid, g in grouped:
        max_cycle = g["cycle_number"].max()
        bol_cap = g["capacity_ahr"].max()
        eol_cap = g.loc[g["cycle_number"] == max_cycle, "capacity_ahr"].iloc[0]
        stats.append({
            "battery_id": bid,
            "max_cycle": max_cycle,
            "bol_capacity": bol_cap,
            "eol_capacity": eol_cap
        })
    stats_df = pd.DataFrame(stats)
    return stats_df


# ============================================================
# 2. SAN: Structure-Aware Normalization (stable vs volatile)
# ============================================================
def san_partition_and_transform(
    df: pd.DataFrame,
    feature_cols: List[str],
    battery_col: str = "battery_id",
    cond_col: str = "ambient_temperature",
) -> Tuple[pd.DataFrame, Dict]:
    """
    SAN vA3: Invariant-subspace SAN via regression residuals.

    - Treat `cond_col` (ambient_temperature) as the condition variable.
    - For each *volatile* feature, fit a linear regression:
          x_j ≈ β0_j + β1_j * cond
      across all batteries.
    - The predicted part is condition-dependent (volatile).
    - The residual part is orthogonal to the condition subspace and is
      treated as the *invariant* representation.
    - Residuals are then globally z-scored for TCN.
    - `capacity_ahr` is always treated as stable and left in physical units.

    Returns:
        df_transformed, san_meta
    """
    df = df.copy()
    eps = 1e-8

    # ---- 1) Split stable vs volatile features ----------------------
    stable_features: List[str] = []
    volatile_features: List[str] = []

    for f in feature_cols:
        if f == "capacity_ahr":
            stable_features.append(f)
        else:
            volatile_features.append(f)

    # In case capacity_ahr is missing from feature_cols for some reason
    if "capacity_ahr" not in stable_features and "capacity_ahr" in df.columns:
        stable_features.append("capacity_ahr")

    # ---- 2) Build design matrix for condition ----------------------
    if cond_col not in df.columns:
        raise ValueError(f"Condition column '{cond_col}' not found in DataFrame.")

    # Ambient temperature as float column
    cond = df[cond_col].astype(float).values.reshape(-1, 1)
    # Add intercept term: [1, T]
    Z = np.concatenate([np.ones_like(cond), cond], axis=1)  # shape (N, 2)

    # ---- 3) For each volatile feature: regress on condition, take residuals ----
    residuals = {}
    betas = {}

    for f in volatile_features:
        x = df[f].astype(float).values  # shape (N,)

        # Mask valid rows for this feature (both cond and x not NaN)
        mask = np.isfinite(x) & np.isfinite(cond[:, 0])
        if mask.sum() < 3:
            # Not enough data; just center to zero mean
            x_centered = x - np.nanmean(x)
            residuals[f] = x_centered
            betas[f] = np.array([np.nan, np.nan])
            continue

        Z_valid = Z[mask]
        x_valid = x[mask]

        # Solve least squares: Z_valid * beta ≈ x_valid
        beta, *_ = np.linalg.lstsq(Z_valid, x_valid, rcond=None)
        betas[f] = beta  # store regression coef

        # Predicted for all rows (even where missing, will become nan)
        x_hat = Z @ beta  # shape (N,)
        r = x - x_hat     # residuals

        residuals[f] = r

    # ---- 4) Replace volatile features with z-scored residuals -------
    for f in volatile_features:
        r = residuals[f]
        mu = np.nanmean(r)
        sigma = np.nanstd(r)

        if not np.isfinite(sigma) or sigma < eps:
            # Degenerate: set to zero
            df[f] = 0.0
        else:
            df[f] = (r - mu) / (sigma + eps)

    # NOTE: stable features (especially capacity_ahr) are untouched

    san_meta = {
        "stable_features": stable_features,
        "volatile_features": volatile_features,
        "condition_column": cond_col,
        "betas": betas,
    }
    return df, san_meta

def plot_distributions_before_after_san(
        df_raw: pd.DataFrame,
        df_san: pd.DataFrame,
        feature_cols: List[str],
        outdir: Path,
        ambient_col: str = "ambient_temperature"
) -> None:
    """
    Create distribution plots of features before and after SAN.

    For each feature, we generate:
      - Histogram/KDE colored by ambient temperature (raw).
      - Histogram/KDE colored by ambient temperature (after SAN).

    These plots help visually show that SAN reduces cross-condition shifts
    while preserving degradation structure.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    temps = sorted(df_raw[ambient_col].unique())
    palette = sns.color_palette("viridis", n_colors=len(temps))

    for feat in feature_cols:
        plt.figure(figsize=(10, 4))
        plt.suptitle(f"Distribution of {feat}: before vs after SAN")

        plt.subplot(1, 2, 1)
        for t, c in zip(temps, palette):
            sns.kdeplot(df_raw.loc[df_raw[ambient_col] == t, feat],
                        label=f"{t}°C", color=c, fill=True, alpha=0.4)
        plt.title("Before SAN")
        plt.xlabel(feat)
        plt.legend()

        plt.subplot(1, 2, 2)
        for t, c in zip(temps, palette):
            sns.kdeplot(df_san.loc[df_san[ambient_col] == t, feat],
                        label=f"{t}°C", color=c, fill=True, alpha=0.4)
        plt.title("After SAN")
        plt.xlabel(feat)
        plt.legend()

        plt.tight_layout()
        plt.savefig(outdir / f"dist_{feat}_before_after_SAN.png", dpi=300)
        plt.close()


# ============================================================
# 3. Temporal coordinates and SOH/RUL definitions
# ============================================================

def add_temporal_and_health_columns(df: pd.DataFrame,
                                    battery_stats: pd.DataFrame
                                    ) -> pd.DataFrame:
    """
    Add columns:
        - life_fraction: tau = cycle_number / max_cycle for that battery.
        - soh: capacity_ahr / bol_capacity.
        - rul_fraction: 1 - life_fraction.
        - tau_deploy: deployable coordinate based on cumulative capacity, as:

            tau_deploy_i = sum_{k<=i} capacity_ahr(k) / sum_{k<=i_max} capacity_ahr(k)

    This follows the narrative:
      - life_fraction is only used during training for aligning phases.
      - tau_deploy is measurable online and used as an input.

    Returns:
        df_aug: DataFrame with added columns.
    """
    df = df.copy()
    stats_map = battery_stats.set_index("battery_id").to_dict(orient="index")

    life_fractions = []
    sohs = []
    rul_fractions = []
    tau_deploys = []

    for bid, g in df.groupby("battery_id"):
        s = stats_map[bid]
        max_cycle = s["max_cycle"]
        bol_cap = s["bol_capacity"]

        # Sort by cycle
        g_sorted = g.sort_values("cycle_number")
        caps = g_sorted["capacity_ahr"].values
        cycles = g_sorted["cycle_number"].values

        life_frac = cycles / max_cycle
        soh = caps / bol_cap
        rul_frac = 1.0 - life_frac

        # Deployable tau: cumulative capacity fraction
        cum_cap = np.cumsum(caps)
        tau_deploy = cum_cap / cum_cap[-1]

        life_fractions.extend(life_frac.tolist())
        sohs.extend(soh.tolist())
        rul_fractions.extend(rul_frac.tolist())
        tau_deploys.extend(tau_deploy.tolist())

    df["life_fraction"] = life_fractions
    df["soh"] = sohs
    df["rul_fraction"] = rul_fractions
    df["tau_deploy"] = tau_deploys

    return df


# ============================================================
# 4. TCN implementation (PyTorch)
# ============================================================

class Chomp1d(nn.Module):
    """Remove padding at the end to maintain causality."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Standard TCN residual block."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size,
                      padding=padding,
                      dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels,
                      kernel_size,
                      padding=padding,
                      dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                           if in_channels != out_channels else None)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight.data)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight.data)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for 1D sequences.

    We use:
      - num_levels residual blocks with exponentially increasing dilations.
      - kernel_size=3.
      - causal convolutions with Chomp1d.
    """
    def __init__(self,
                 input_size: int,
                 num_channels: int = TCN_HIDDEN_CHANNELS,
                 num_levels: int = TCN_NUM_LEVELS,
                 kernel_size: int = TCN_KERNEL_SIZE,
                 dropout: float = TCN_DROPOUT):
        super().__init__()
        layers = []
        channels = [input_size] + [num_channels] * num_levels
        for i in range(num_levels):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            dilation = 2 ** i
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(num_channels, 1)  # scalar SOH prediction

    def forward(self, x):
        # x: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        y = self.network(x)
        # Take last time step's features
        y_last = y[:, :, -1]  # (batch, channels)
        out = self.head(y_last).squeeze(-1)  # (batch,)
        return out


class BatterySequenceDataset(Dataset):
    """
    Dataset for training TCN on sliding windows of SAN-transformed features.

    Each sample:
      - X: [window_size, num_features] (soh is included as a feature).
      - y: scalar target = soh at the last cycle in the window.

    So we train the TCN as a *denoising / smoothing* model for SOH:
    it uses local temporal context (and other features) to predict SOH.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 feature_cols: List[str],
                 target_col: str = "soh",
                 window_size: int = TCN_WINDOW_SIZE):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.window_size = window_size

        # Build sliding windows per battery
        X_windows = []
        y_targets = []

        for bid, g in df.groupby("battery_id"):
            g_sorted = g.sort_values("cycle_number")
            features = g_sorted[feature_cols].values
            targets = g_sorted[target_col].values

            if len(g_sorted) < window_size:
                continue

            for start in range(0, len(g_sorted) - window_size + 1):
                end = start + window_size
                X_windows.append(features[start:end, :])
                y_targets.append(targets[end - 1])

        self.X = np.stack(X_windows, axis=0)  # (N, window, feat)
        self.y = np.array(y_targets, dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.y[idx]
        return x, y


def train_tcn(df_train: pd.DataFrame,
              df_val: pd.DataFrame,
              feature_cols: List[str],
              device: str = "cpu"
              ) -> Tuple[TCN, Dict]:
    """
    Train TCN on training batteries and validate on validation batteries.

    Returns:
        best_model: trained TCN
        history: dict with train_loss, val_loss
    """
    device = torch.device(device)

    train_ds = BatterySequenceDataset(df_train, feature_cols)
    val_ds = BatterySequenceDataset(df_val, feature_cols)

    train_loader = DataLoader(train_ds,
                              batch_size=TCN_BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(val_ds,
                            batch_size=TCN_BATCH_SIZE,
                            shuffle=False)

    model = TCN(input_size=len(feature_cols))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=TCN_LR,
                                 weight_decay=TCN_WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val_loss = np.inf
    best_state = None
    patience_counter = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(TCN_MAX_EPOCHS):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else np.nan

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"[Epoch {epoch+1}/{TCN_MAX_EPOCHS}] "
              f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= TCN_EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save training curves
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("TCN training/validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tcn_training_loss.png", dpi=300)
    plt.close()

    torch.save(model.state_dict(), MODELS_DIR / "tcn_best.pth")

    return model, history


def predict_tcn_full(df: pd.DataFrame,
                     model: TCN,
                     feature_cols: List[str],
                     device: str = "cpu"
                     ) -> pd.DataFrame:
    """
    Apply trained TCN to every cycle (per battery) to get smoothed SOH.

    Strategy:
        For each battery:
          - Build sliding windows of length window_size.
          - Predict SOH for last element of each window.
          - For the first (window_size-1) cycles, fall back to raw SOH
            (we do not extrapolate backwards).

    Returns:
        df_pred: same as df but with an additional column "tcn_soh".
    """
    device = torch.device(device)
    model.eval()
    df = df.copy()
    df["tcn_soh"] = np.nan

    for bid, g in df.groupby("battery_id"):
        g_sorted = g.sort_values("cycle_number")
        feats = g_sorted[feature_cols].values.astype(np.float32)
        sohs = g_sorted["soh"].values

        tcn_soh = np.full_like(sohs, fill_value=np.nan, dtype=np.float32)

        if len(g_sorted) >= TCN_WINDOW_SIZE:
            windows = []
            idxs = []
            for start in range(0, len(g_sorted) - TCN_WINDOW_SIZE + 1):
                end = start + TCN_WINDOW_SIZE
                windows.append(feats[start:end, :])
                idxs.append(end - 1)  # index where prediction applies

            X = torch.tensor(np.stack(windows, axis=0), dtype=torch.float32).to(device)
            with torch.no_grad():
                y_pred = model(X).cpu().numpy()

            for ix, yp in zip(idxs, y_pred):
                tcn_soh[ix] = yp

        # For cycles before first prediction, fall back to raw SOH
        missing_mask = np.isnan(tcn_soh)
        tcn_soh[missing_mask] = sohs[missing_mask]

        df.loc[g_sorted.index, "tcn_soh"] = tcn_soh

    return df


# ============================================================
# 5. Knee detection and 3-piece SVR for RUL mapping
# ============================================================

def detect_two_knees(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Detect two knee points in the *average* degradation curve (SOH vs life_fraction)
    using `kneed`. If detection fails, fall back to fixed quantiles (0.3, 0.7).

    Returns:
        (knee1, knee2) as life_fraction values in (0, 1).
    """
    # Build average SOH vs life_fraction curve by pooling all batteries
    df_sorted = df.sort_values("life_fraction")
    x = df_sorted["life_fraction"].values
    y = df_sorted["soh"].values

    # To reduce noise, compute a simple moving average
    window = max(5, len(x) // 100)
    if window % 2 == 0:
        window += 1
    if window > 1:
        y_smooth = np.convolve(y, np.ones(window) / window, mode="same")
    else:
        y_smooth = y

    try:
        k1_locator = KneeLocator(x, y_smooth, S=1.0,
                                 curve="concave", direction="decreasing")
        knee1 = k1_locator.knee
    except Exception:
        knee1 = None

    # Second knee: restrict to data after knee1
    knee2 = None
    if knee1 is not None:
        mask = x > knee1
        if mask.sum() > 10:
            try:
                k2_locator = KneeLocator(x[mask], y_smooth[mask], S=1.0,
                                         curve="concave", direction="decreasing")
                knee2 = k2_locator.knee
            except Exception:
                knee2 = None

    # Fallback to quantiles if detection fails
    if knee1 is None or knee2 is None or knee1 >= knee2:
        knee1 = 0.3
        knee2 = 0.7

    print(f"Detected (or fallback) knees at life_fraction: knee1={knee1:.3f}, knee2={knee2:.3f}")

    # Plot the average curve and knees
    plt.figure(figsize=(6, 4))
    plt.plot(x, y_smooth, label="Average SOH")
    plt.axvline(knee1, color="r", linestyle="--", label="knee1")
    plt.axvline(knee2, color="g", linestyle="--", label="knee2")
    plt.xlabel("Life fraction")
    plt.ylabel("SOH")
    plt.title("Average degradation curve and detected knees")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "average_degradation_knees.png", dpi=300)
    plt.close()

    return float(knee1), float(knee2)


def train_piecewise_svr(df: pd.DataFrame,
                        use_tcn_soh: bool,
                        knee1: float,
                        knee2: float,
                        label_col: str = "rul_fraction"
                        ) -> Dict[str, SVR]:
    """
    Train three piecewise SVRs for RUL mapping.

    Inputs:
        df: DataFrame with columns:
              - life_fraction
              - soh (and optionally tcn_soh)
              - rul_fraction
        use_tcn_soh: if True, use tcn_soh as input; else soh.
        knee1, knee2: life_fraction boundaries between early/mid/late phases.
        label_col: target label, default is RUL fraction (1 - life_fraction).

    SVR inputs:
        X = [soh] or [tcn_soh] (one scalar feature).
        y = rul_fraction.

    Returns:
        svr_models: dict with keys "early", "mid", "late".
    """
    if use_tcn_soh:
        x_col = "tcn_soh"
    else:
        x_col = "soh"

    df = df.dropna(subset=[x_col, label_col, "life_fraction"]).copy()

    def select_phase(mask, phase_name):
        sub = df[mask]
        if len(sub) < 20:
            print(f"Warning: very few samples in {phase_name} phase ({len(sub)})")
        X = sub[[x_col]].values
        y = sub[label_col].values
        return X, y

    # Early, mid, late masks
    mask_early = df["life_fraction"] <= knee1
    mask_mid = (df["life_fraction"] > knee1) & (df["life_fraction"] <= knee2)
    mask_late = df["life_fraction"] > knee2

    svr_params = dict(C=SVR_C, epsilon=SVR_EPSILON, kernel=SVR_KERNEL)

    X_e, y_e = select_phase(mask_early, "early")
    X_m, y_m = select_phase(mask_mid, "mid")
    X_l, y_l = select_phase(mask_late, "late")

    svr_early = SVR(**svr_params).fit(X_e, y_e)
    svr_mid = SVR(**svr_params).fit(X_m, y_m)
    svr_late = SVR(**svr_params).fit(X_l, y_l)

    svr_models = {"early": svr_early, "mid": svr_mid, "late": svr_late}
    return svr_models


def predict_piecewise_svr(df: pd.DataFrame,
                          svr_models: Dict[str, SVR],
                          use_tcn_soh: bool,
                          knee1: float,
                          knee2: float) -> pd.DataFrame:
    """
    Apply trained piecewise SVRs to each cycle to predict RUL fraction.

    We select which SVR to use based on life_fraction:
        - life_fraction <= knee1 -> early SVR
        - knee1 < life_fraction <= knee2 -> mid SVR
        - life_fraction > knee2 -> late SVR

    Returns:
        df_pred: df with additional column "svr_rul_fraction".
    """
    df = df.copy()
    x_col = "tcn_soh" if use_tcn_soh else "soh"

    preds = []
    for _, row in df.iterrows():
        lf = row["life_fraction"]
        x_val = row[x_col]
        if lf <= knee1:
            svr = svr_models["early"]
        elif lf <= knee2:
            svr = svr_models["mid"]
        else:
            svr = svr_models["late"]
        y_pred = svr.predict(np.array([[x_val]]))[0]
        preds.append(y_pred)

    df["svr_rul_fraction"] = preds
    return df


# ============================================================
# 6. Evaluation & ablations
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics on RUL fraction."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def evaluate_rul(df: pd.DataFrame,
                 pred_col: str,
                 label_col: str = "rul_fraction",
                 name: str = "model"
                 ) -> Dict[str, float]:
    """
    Compute metrics for entire dataset, and save them.

    Also creates:
      - scatter plot of true vs predicted RUL fraction
      - histogram of errors
    """
    df_eval = df.dropna(subset=[pred_col, label_col]).copy()
    y_true = df_eval[label_col].values
    y_pred = df_eval[pred_col].values

    metrics = compute_metrics(y_true, y_pred)

    # Save metrics to CSV
    metrics_path = METRICS_DIR / f"metrics_{name}.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    # Scatter plot
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.4)
    plt.plot([0, 1], [0, 1], "r--", label="ideal")
    plt.xlabel("True RUL fraction")
    plt.ylabel("Predicted RUL fraction")
    plt.title(f"True vs predicted RUL fraction ({name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"scatter_rul_{name}.png", dpi=300)
    plt.close()

    # Error histogram
    errors = y_pred - y_true
    plt.figure(figsize=(5, 4))
    sns.histplot(errors, kde=True, bins=40)
    plt.xlabel("Prediction error (RUL fraction)")
    plt.title(f"Error distribution ({name})")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"error_hist_{name}.png", dpi=300)
    plt.close()

    return metrics


def split_batteries(df: pd.DataFrame,
                    train_frac: float = 0.7,
                    val_frac: float = 0.15
                    ) -> Tuple[List[str], List[str], List[str]]:
    """
    Split batteries into train/val/test sets at battery level.

    This avoids leakage across splits.
    """
    batteries = sorted(df["battery_id"].unique())
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(batteries)

    n = len(batteries)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_b = batteries[:n_train]
    val_b = batteries[n_train:n_train + n_val]
    test_b = batteries[n_train + n_val:]
    return train_b, val_b, test_b


def run_pipeline_variant(df_raw: pd.DataFrame,
                         variant_name: str,
                         use_san: bool,
                         use_pca: bool,
                         use_whiten_all: bool,
                         use_tcn: bool,
                         use_piecewise: bool
                         ) -> Dict[str, float]:
    """
    Run one variant of the pipeline:

        - use_san: apply SAN (stable/volatile partition + per-battery normalization).
        - use_pca: apply PCA whitening instead of SAN.
        - use_whiten_all: apply global z-score normalization (no SAN).
        - use_tcn: include TCN smoothing; otherwise direct SOH used for SVR.
        - use_piecewise: if True: 3-piece SVR; else single SVR.

    Returns:
        metrics dict for this variant.
    """
    print(f"\n=== Running variant: {variant_name} ===")

    df = df_raw.copy()

    # ==============================================================
    # PATCH 1: CLEAN RAW DATA BEFORE ANY SAN / TCN / SVR HAPPENS
    # ==============================================================
    df = df.replace([np.inf, -np.inf], np.nan)

    base_features = [
        "capacity_ahr", "mean_current",
        "min_voltage", "final_voltage",
        "max_temperature"
    ]

    for col in base_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    feature_cols = ["capacity_ahr", "mean_current",
                    "min_voltage", "final_voltage", "max_temperature"]

    # ---------------------------------------------------------------
    # 1) SAN / PCA / Whitening
    # ---------------------------------------------------------------
    san_meta = None
    if use_san:
        df, san_meta = san_partition_and_transform(df, feature_cols)
        if variant_name == "SAN_TCN_3SVR":
            plot_distributions_before_after_san(
                df_raw=df_raw,
                df_san=df,
                feature_cols=feature_cols,
                outdir=PLOTS_DIR / "san_distributions"
            )

    elif use_pca:
        scaler = StandardScaler()
        pca = PCA(whiten=True)
        X = df[feature_cols].values
        X_scaled = scaler.fit_transform(X)
        X_pca = pca.fit_transform(X_scaled)
        for i in range(X_pca.shape[1]):
            df[f"pc_{i}"] = X_pca[:, i]
        feature_cols = [f"pc_{i}" for i in range(X_pca.shape[1])]

    elif use_whiten_all:
        scaler = StandardScaler()
        X = df[feature_cols].values
        X_scaled = scaler.fit_transform(X)
        for i, col in enumerate(feature_cols):
            df[col] = X_scaled[:, i]

    # ==============================================================
    # PATCH 2: SAN / PCA / WHITENING OUTPUT CLEANUP
    # ==============================================================

    df = df.replace([np.inf, -np.inf], np.nan)

    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # ---------------------------------------------------------------
    # 2) Add temporal & health columns
    # ---------------------------------------------------------------
    battery_stats = compute_battery_stats(df)
    df = add_temporal_and_health_columns(df, battery_stats)

    # ==============================================================
    # PATCH 3: CLEAN HEALTH + TEMPORAL COLUMNS
    # ==============================================================

    df["soh"] = df["soh"].replace([np.inf, -np.inf], np.nan)
    df["soh"] = df["soh"].fillna(df["soh"].median())

    for col in ["life_fraction", "tau_deploy", "rul_fraction"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # ---------------------------------------------------------------
    # 3) Train/Val/Test split
    # ---------------------------------------------------------------
    train_b, val_b, test_b = split_batteries(df)
    df_train = df[df["battery_id"].isin(train_b)].reset_index(drop=True)
    df_val = df[df["battery_id"].isin(val_b)].reset_index(drop=True)
    df_test = df[df["battery_id"].isin(test_b)].reset_index(drop=True)

    # ---------------------------------------------------------------
    # 4) TCN training
    # ---------------------------------------------------------------
    if use_tcn:
        tcn_input_cols = feature_cols + ["tau_deploy"]
        model, history = train_tcn(df_train, df_val, tcn_input_cols)

        # Predict smoothed SOH for all cycles
        df = predict_tcn_full(df, model, tcn_input_cols)

        # ==============================================================
        # PATCH 4: CLEAN TCN OUTPUT BEFORE SVR
        # ==============================================================
        df["tcn_soh"] = df["tcn_soh"].replace([np.inf, -np.inf], np.nan)
        df["tcn_soh"] = df["tcn_soh"].fillna(df["soh"])

    else:
        df["tcn_soh"] = df["soh"]

    # ---------------------------------------------------------------
    # 5) Knee detection (train+val only)
    # ---------------------------------------------------------------
    df_train_val = df[df["battery_id"].isin(train_b + val_b)]
    knee1, knee2 = detect_two_knees(df_train_val)

    # ---------------------------------------------------------------
    # 6) SVR
    # ---------------------------------------------------------------
    df_train_val = df_train_val.copy()
    if use_piecewise:
        svr_models = train_piecewise_svr(
            df_train_val,
            use_tcn_soh=True,
            knee1=knee1,
            knee2=knee2,
            label_col="rul_fraction"
        )

        df_pred = predict_piecewise_svr(
            df,
            svr_models,
            use_tcn_soh=True,
            knee1=knee1,
            knee2=knee2
        )
        pred_col = "svr_rul_fraction"
        df = df_pred


    else:
        X = df_train_val[["tcn_soh"]].values
        y = df_train_val["rul_fraction"].values
        svr = SVR(C=SVR_C, epsilon=SVR_EPSILON, kernel=SVR_KERNEL).fit(X, y)
        df["svr_rul_fraction"] = svr.predict(df[["tcn_soh"]].values)
        pred_col = "svr_rul_fraction"


    # ---------------------------------------------------------------
    # 7) Evaluate on test
    # ---------------------------------------------------------------
    df_test_pred = df[df["battery_id"].isin(test_b)].reset_index(drop=True)

    metrics = evaluate_rul(
        df_test_pred,
        pred_col=pred_col,
        label_col="rul_fraction",
        name=variant_name
    )

    # Per-battery metrics
    per_bat = []
    for bid, g in df_test_pred.groupby("battery_id"):
        m = compute_metrics(g["rul_fraction"].values,
                            g[pred_col].values)
        m["battery_id"] = bid
        per_bat.append(m)
    per_bat_df = pd.DataFrame(per_bat)
    per_bat_df.to_csv(
        METRICS_DIR / f"metrics_per_battery_{variant_name}.csv",
        index=False
    )

    # Boxplot of per-battery RMSE
    plt.figure(figsize=(4, 4))
    sns.boxplot(y=per_bat_df["RMSE"])
    plt.ylabel("RMSE (RUL fraction)")
    plt.title(f"Per-battery RMSE ({variant_name})")
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / f"box_rmse_per_battery_{variant_name}.png",
        dpi=300
    )
    plt.close()

    return metrics

def run_all_nasa_variants(df_raw: pd.DataFrame) -> None:
    """
    Run main variant (SAN + TCN + 3SVR) plus ablations on NASA Ames.

    Variants:
      - SAN_TCN_3SVR: proposed method.
      - NoSAN_TCN_3SVR: raw features (no SAN) + TCN + 3SVR.
      - PCA_TCN_3SVR: PCA-whitened features + TCN + 3SVR.
      - WHITEN_TCN_3SVR: global z-score features + TCN + 3SVR.
      - SAN_noTCN_3SVR: SAN + direct 3SVR on raw SOH (no TCN smoothing).
      - SAN_TCN_1SVR: SAN + TCN + single SVR (no three-phase segmentation).

    Outputs:
      - Aggregated metrics table (CSV + LaTeX)
      - Comparison plots can be made from these tables.
    """
    variants = [
        dict(name="SAN_TCN_3SVR",
             use_san=True,
             use_pca=False,
             use_whiten_all=False,
             use_tcn=True,
             use_piecewise=True),
        dict(name="NoSAN_TCN_3SVR",
             use_san=False,
             use_pca=False,
             use_whiten_all=False,
             use_tcn=True,
             use_piecewise=True),
        dict(name="PCA_TCN_3SVR",
             use_san=False,
             use_pca=True,
             use_whiten_all=False,
             use_tcn=True,
             use_piecewise=True),
        dict(name="WHITEN_TCN_3SVR",
             use_san=False,
             use_pca=False,
             use_whiten_all=True,
             use_tcn=True,
             use_piecewise=True),
        dict(name="SAN_noTCN_3SVR",
             use_san=True,
             use_pca=False,
             use_whiten_all=False,
             use_tcn=False,
             use_piecewise=True),
        dict(name="SAN_TCN_1SVR",
             use_san=True,
             use_pca=False,
             use_whiten_all=False,
             use_tcn=True,
             use_piecewise=False),
    ]

    all_metrics = []
    for v in variants:
        m = run_pipeline_variant(df_raw,
                                 variant_name=v["name"],
                                 use_san=v["use_san"],
                                 use_pca=v["use_pca"],
                                 use_whiten_all=v["use_whiten_all"],
                                 use_tcn=v["use_tcn"],
                                 use_piecewise=v["use_piecewise"])
        m["variant"] = v["name"]
        all_metrics.append(m)

    all_metrics_df = pd.DataFrame(all_metrics)
    all_metrics_df.to_csv(TABLES_DIR / "metrics_all_variants.csv", index=False)

    # Export LaTeX table for the paper
    with open(TABLES_DIR / "metrics_all_variants.tex", "w") as f:
        f.write(all_metrics_df.to_latex(index=False, float_format="%.4f"))


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="NASA Ames IDM pipeline (SAN + TCN + SVR on master CSV)."
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to master discharge CSV.")
    args = parser.parse_args()

    df_raw = load_master_dataset(args.data)
    print(f"Loaded dataset with {len(df_raw)} cycles, "
          f"{df_raw['battery_id'].nunique()} batteries.")

    run_all_nasa_variants(df_raw)
    print("All variants completed. Artifacts saved under 'artifacts/'.")


if __name__ == "__main__":
    main()
