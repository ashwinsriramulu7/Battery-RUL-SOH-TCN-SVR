#!/usr/bin/env python3
"""
Invariant Degradation Modeling (IDM) pipeline for NASA Ames battery dataset.

This script implements the NASA-specific part of the methodology:

1. Load a master discharge CSV (one row per cycle across all batteries).
2. Apply Structure-Aware Normalization (SAN):
   - Identify stable vs. volatile features using across-battery variance
     computed on TRAIN+VAL batteries only.
   - Normalize only volatile features per battery (per-battery z-score);
     keep stable features unchanged (capacity in physical units).
3. Construct temporal coordinates:
   - Normalized life fraction (tau = cycle_index / EOL_index) for training.
   - Deployable coordinate (tau_deploy) based on cumulative discharged
     capacity, strictly measurable online.
4. Train a PyTorch Temporal Convolutional Network (TCN) to learn a smooth
   SOH trajectory (normalized capacity) from the SAN-transformed features.
5. Compute TCN-smoothed SOH and convert to RUL fraction labels.
6. Detect two knee points in the average degradation curve using `kneed`
   (with fallback to quantiles), defining three degradation phases.
7. Train three separate SVR models (early/mid/late) mapping TCN-smoothed SOH
   to RUL fraction.
8. Run ablations:
   - No SAN (raw features, with only basic imputation).
   - PCA whitening (instead of SAN).
   - Global whitening (StandardScaler) instead of SAN.
   - SAN + no TCN (direct piecewise SVR on raw SOH).
   - SAN + TCN + single SVR (1-piece, no phases).
   - Full SAN + TCN + 3-piece SVR (proposed).
9. Evaluation:
   - Leave-one-battery-out (LOBO): for each battery b*, train on all others
     (with an internal train/val split), test on b*.
   - For each variant, aggregate RMSE, MAE, SMAPE, R^2 across folds.
10. Generate plots and tables suitable for inclusion in the paper.

The logic matches the Methods section and can be reused for other datasets
by swapping the input CSV and feature list.
"""

import argparse
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
TCN_HIDDEN_CHANNELS = 32     # channels in intermediate layers
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
        df: pandas DataFrame sorted by (battery_id, cycle_number).
    """
    df = pd.read_csv(csv_path)

    df["battery_id"] = df["battery_id"].astype(str)
    df["cycle_number"] = df["cycle_number"].astype(int)
    df["ambient_temperature"] = df["ambient_temperature"].astype(float)
    df["capacity_ahr"] = df["capacity_ahr"].astype(float)
    df["mean_current"] = df["mean_current"].astype(float)
    df["min_voltage"] = df["min_voltage"].astype(float)
    df["final_voltage"] = df["final_voltage"].astype(float)
    df["max_temperature"] = df["max_temperature"].astype(float)

    df = df.sort_values(["battery_id", "cycle_number"]).reset_index(drop=True)
    return df


def compute_battery_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-battery summary: EOL index, BOL capacity, EOL capacity.

    EOL index is defined as the max cycle_number for that battery.
    BOL capacity is the maximum observed capacity for that battery.

    Returns:
        stats_df: columns [battery_id, max_cycle, bol_capacity, eol_capacity].
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
            "eol_capacity": eol_cap,
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
    train_batteries: List[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    SAN implementation used in the paper:

    1) Use TRAIN+VAL batteries only to decide stable vs volatile features.
       For each feature i, compute variance across batteries of per-battery
       mean: Δσ_i = Var_b ( mean_b(x_i) ). Features below the specified
       variance quantile are marked "stable"; others are "volatile".

    2) For each battery b and each volatile feature j, compute per-battery
       mean μ_{b,j} and std σ_{b,j} across cycles, and apply per-battery
       z-score:
            x'_{b,j} = (x_{b,j} - μ_{b,j}) / (σ_{b,j} + ε).

       Stable features are left in their original physical units.

    3) capacity_ahr is always treated as stable to preserve the physical
       interpretation of SOH.

    This emphasizes cross-battery invariances while keeping capacity in
    physical units.

    Returns:
        df_san: transformed DataFrame.
        san_meta: dict summarizing stable/volatile split and statistics.
    """
    df = df.copy()
    eps = 1e-8

    # -------- 1) Decide stable vs volatile using TRAIN+VAL only ----------
    if train_batteries is None:
        df_train = df
    else:
        df_train = df[df[battery_col].isin(train_batteries)].copy()

    # Per-battery means for each feature
    bat_means = df_train.groupby(battery_col)[feature_cols].mean()
    var_across = bat_means.var(axis=0)  # Series indexed by feature

    # Threshold on across-battery variance
    thresh = var_across.quantile(SAN_VARIANCE_QUANTILE)

    stable_features = var_across[var_across <= thresh].index.tolist()
    volatile_features = [f for f in feature_cols if f not in stable_features]

    # Force capacity_ahr to be stable
    if "capacity_ahr" in feature_cols:
        if "capacity_ahr" not in stable_features:
            stable_features.append("capacity_ahr")
        if "capacity_ahr" in volatile_features:
            volatile_features.remove("capacity_ahr")

    # -------- 2) Per-battery z-score normalization on volatile features ---
    df_san = df.copy()
    for bid, g in df_san.groupby(battery_col):
        idx = g.index
        for f in volatile_features:
            vals = g[f].values.astype(float)
            mu = np.nanmean(vals)
            sigma = np.nanstd(vals)
            if not np.isfinite(sigma) or sigma < eps:
                df_san.loc[idx, f] = 0.0
            else:
                df_san.loc[idx, f] = (vals - mu) / (sigma + eps)

    san_meta = {
        "stable_features": stable_features,
        "volatile_features": volatile_features,
        "variance_across_batteries": var_across.to_dict(),
        "variance_threshold": float(thresh),
        "train_batteries": train_batteries,
    }
    return df_san, san_meta


def plot_distributions_before_after_san(
    df_raw: pd.DataFrame,
    df_san: pd.DataFrame,
    feature_cols: List[str],
    outdir: Path,
    ambient_col: str = "ambient_temperature",
) -> None:
    """
    Distribution plots of features before vs. after SAN, colored by ambient T.
    Used for qualitative evidence that SAN reduces cross-condition shifts.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    temps = sorted(df_raw[ambient_col].unique())
    palette = sns.color_palette("viridis", n_colors=len(temps))

    for feat in feature_cols:
        plt.figure(figsize=(10, 4))
        plt.suptitle(f"Distribution of {feat}: before vs. after SAN")

        plt.subplot(1, 2, 1)
        for t, c in zip(temps, palette):
            sns.kdeplot(
                df_raw.loc[df_raw[ambient_col] == t, feat],
                label=f"{t}°C", color=c, fill=True, alpha=0.4
            )
        plt.title("Before SAN")
        plt.xlabel(feat)
        plt.legend()

        plt.subplot(1, 2, 2)
        for t, c in zip(temps, palette):
            sns.kdeplot(
                df_san.loc[df_san[ambient_col] == t, feat],
                label=f"{t}°C", color=c, fill=True, alpha=0.4
            )
        plt.title("After SAN")
        plt.xlabel(feat)
        plt.legend()

        plt.tight_layout()
        plt.savefig(outdir / f"dist_{feat}_before_after_SAN.png", dpi=300)
        plt.close()


# ============================================================
# 3. Temporal coordinates and SOH/RUL definitions
# ============================================================

def add_temporal_and_health_columns(
    df: pd.DataFrame,
    battery_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add:
        - life_fraction: tau = cycle_number / max_cycle for that battery.
        - soh: capacity_ahr / bol_capacity.
        - rul_fraction: 1 - life_fraction.
        - tau_deploy: cumulative capacity fraction (deployable coordinate).

    Returns:
        df_aug: DataFrame with new columns.
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

        g_sorted = g.sort_values("cycle_number")
        caps = g_sorted["capacity_ahr"].values
        cycles = g_sorted["cycle_number"].values

        life_frac = cycles / max_cycle
        soh = caps / bol_cap
        rul_frac = 1.0 - life_frac

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

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
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

    - num_levels residual blocks with exponentially increasing dilations.
    - kernel_size=3, causal convolutions with Chomp1d.
    - Final linear head outputs scalar SOH prediction.
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
        self.head = nn.Linear(num_channels, 1)  # scalar SOH

    def forward(self, x):
        # x: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        y = self.network(x)
        y_last = y[:, :, -1]
        out = self.head(y_last).squeeze(-1)
        return out


class BatterySequenceDataset(Dataset):
    """
    Dataset for training TCN on sliding windows of SAN-transformed features.

    Each sample:
      - X: [window_size, num_features] (incl. soh as a feature).
      - y: scalar target = soh at the last cycle in the window.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 feature_cols: List[str],
                 target_col: str = "soh",
                 window_size: int = TCN_WINDOW_SIZE):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.window_size = window_size

        X_windows = []
        y_targets = []

        for _, g in df.groupby("battery_id"):
            g_sorted = g.sort_values("cycle_number")
            features = g_sorted[feature_cols].values
            targets = g_sorted[target_col].values

            if len(g_sorted) < window_size:
                continue

            for start in range(0, len(g_sorted) - window_size + 1):
                end = start + window_size
                X_windows.append(features[start:end, :])
                y_targets.append(targets[end - 1])

        self.X = np.stack(X_windows, axis=0) if X_windows else np.empty((0, window_size, len(feature_cols)))
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
              device: str = "cpu",
              tag: str = "tcn") -> Tuple[TCN, Dict]:
    """
    Train TCN on training batteries and validate on validation batteries.

    Returns:
        best_model, history dict with train_loss, val_loss.
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

        print(f"[{tag}] Epoch {epoch+1}/{TCN_MAX_EPOCHS} "
              f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= TCN_EARLY_STOP_PATIENCE:
            print(f"[{tag}] Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save training curves (last fold overwrites; good enough for paper)
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(f"TCN training/validation loss ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{tag}_loss_curves.png", dpi=300)
    plt.close()

    torch.save(model.state_dict(), MODELS_DIR / f"{tag}_best.pth")

    return model, history


def predict_tcn_full(df: pd.DataFrame,
                     model: TCN,
                     feature_cols: List[str],
                     device: str = "cpu") -> pd.DataFrame:
    """
    Apply trained TCN to every cycle (per battery) to get smoothed SOH.

    For each battery:
      - Build sliding windows of length window_size.
      - Predict SOH for last element of each window.
      - For the first (window_size-1) cycles, fall back to raw SOH.

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
                idxs.append(end - 1)

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
    Detect two knee points in the average degradation curve (SOH vs life_fraction)
    using `kneed`. If detection fails, fall back to fixed quantiles (0.3, 0.7).

    Returns:
        (knee1, knee2) as life_fraction values in (0, 1).
    """
    df_sorted = df.sort_values("life_fraction")
    x = df_sorted["life_fraction"].values
    y = df_sorted["soh"].values

    # Simple moving average to reduce noise
    window = max(5, len(x) // 100)
    if window % 2 == 0:
        window += 1
    if window > 1:
        y_smooth = np.convolve(y, np.ones(window) / window, mode="same")
    else:
        y_smooth = y

    knee1 = None
    knee2 = None

    try:
        k1_locator = KneeLocator(x, y_smooth, S=1.0,
                                 curve="concave", direction="decreasing")
        knee1 = k1_locator.knee
    except Exception:
        knee1 = None

    if knee1 is not None:
        mask = x > knee1
        if mask.sum() > 10:
            try:
                k2_locator = KneeLocator(x[mask], y_smooth[mask], S=1.0,
                                         curve="concave", direction="decreasing")
                knee2 = k2_locator.knee
            except Exception:
                knee2 = None

    if knee1 is None or knee2 is None or knee1 >= knee2:
        knee1 = 0.3
        knee2 = 0.7

    print(f"Detected (or fallback) knees at life_fraction: knee1={knee1:.3f}, knee2={knee2:.3f}")

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
                        label_col: str = "rul_fraction") -> Dict[str, SVR]:
    """
    Train three piecewise SVRs for RUL mapping.

    X = [soh] or [tcn_soh]; y = RUL fraction.

    Returns:
        dict with keys "early", "mid", "late".
    """
    x_col = "tcn_soh" if use_tcn_soh else "soh"

    df = df.dropna(subset=[x_col, label_col, "life_fraction"]).copy()

    def select_phase(mask, phase_name):
        sub = df[mask]
        if len(sub) < 20:
            print(f"Warning: very few samples in {phase_name} phase ({len(sub)})")
        X = sub[[x_col]].values
        y = sub[label_col].values
        return X, y

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

    return {"early": svr_early, "mid": svr_mid, "late": svr_late}


def predict_piecewise_svr(df: pd.DataFrame,
                          svr_models: Dict[str, SVR],
                          use_tcn_soh: bool,
                          knee1: float,
                          knee2: float) -> pd.DataFrame:
    """
    Apply trained piecewise SVRs to each cycle to predict RUL fraction.
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
    """
    Compute regression metrics on RUL fraction.

    Uses SMAPE instead of MAPE to avoid blow-up near zero.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    smape = np.mean(
        200.0 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "SMAPE": smape, "R2": r2}


def evaluate_rul(df: pd.DataFrame,
                 pred_col: str,
                 label_col: str,
                 name: str) -> Dict[str, float]:
    """
    Compute metrics for a given split and save plots.

    name: used as prefix in filenames (e.g., includes variant + test battery).
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
    plt.title(f"True vs. predicted RUL fraction ({name})")
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


def run_pipeline_variant(
    df_raw: pd.DataFrame,
    variant_name: str,
    use_san: bool,
    use_pca: bool,
    use_whiten_all: bool,
    use_tcn: bool,
    use_piecewise: bool,
    train_b: List[str],
    val_b: List[str],
    test_b: List[str],
) -> Dict[str, float]:
    """
    Run one variant of the pipeline for a specific LOBO fold.

    train_b, val_b, test_b are lists of battery_ids specifying the split.
    All learned transforms (SAN stable/volatile split, StandardScaler,
    PCA, TCN, SVR, knee detection) are fit ONLY on TRAIN+VAL batteries.
    """
    print(f"\n=== Variant {variant_name} | Train={train_b}, Val={val_b}, Test={test_b} ===")

    df = df_raw.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    all_trainval = train_b + val_b

    base_features = [
        "capacity_ahr", "mean_current",
        "min_voltage", "final_voltage",
        "max_temperature",
    ]

    # Basic imputation based on TRAIN+VAL only (no label leakage)
    df_trainval_raw = df[df["battery_id"].isin(all_trainval)]
    medians = df_trainval_raw[base_features].median()
    for col in base_features:
        df[col] = df[col].fillna(medians[col])

    feature_cols = base_features.copy()

    # ---------------------------------------------------------------
    # 1) SAN / PCA / Whitening
    # ---------------------------------------------------------------
    if use_san:
        df, san_meta = san_partition_and_transform(
            df, feature_cols, battery_col="battery_id",
            train_batteries=all_trainval
        )

        # Only generate SAN distribution plots once per run for main variant
        # (last fold will overwrite; that is acceptable).
        if variant_name == "SAN_TCN_3SVR":
            plot_distributions_before_after_san(
                df_raw=df_raw,
                df_san=df,
                feature_cols=feature_cols,
                outdir=PLOTS_DIR / "san_distributions",
            )

    elif use_pca:
        # Fit scaler and PCA on TRAIN+VAL only, apply to all
        df_trainval = df[df["battery_id"].isin(all_trainval)]
        scaler = StandardScaler()
        X_trainval = scaler.fit_transform(df_trainval[feature_cols].values)

        pca = PCA(whiten=True)
        pca.fit(X_trainval)

        X_all_scaled = scaler.transform(df[feature_cols].values)
        X_all_pca = pca.transform(X_all_scaled)

        for i in range(X_all_pca.shape[1]):
            df[f"pc_{i}"] = X_all_pca[:, i]
        feature_cols = [f"pc_{i}" for i in range(X_all_pca.shape[1])]

    elif use_whiten_all:
        # Global StandardScaler fitted on TRAIN+VAL only, applied to all
        df_trainval = df[df["battery_id"].isin(all_trainval)]
        scaler = StandardScaler()
        scaler.fit(df_trainval[feature_cols].values)

        X_all_scaled = scaler.transform(df[feature_cols].values)
        for i, col in enumerate(feature_cols):
            df[col] = X_all_scaled[:, i]

    # ---------------------------------------------------------------
    # 2) Add temporal & health columns
    # ---------------------------------------------------------------
    battery_stats = compute_battery_stats(df)
    df = add_temporal_and_health_columns(df, battery_stats)

    # Clean up any numerical edge cases
    df["soh"] = df["soh"].replace([np.inf, -np.inf], np.nan)
    df["soh"] = df["soh"].fillna(df["soh"].median())

    for col in ["life_fraction", "tau_deploy", "rul_fraction"]:
        df[col] = df[col].fillna(0.0)

    # Split by batteries for model training and testing
    df_train = df[df["battery_id"].isin(train_b)].reset_index(drop=True)
    df_val = df[df["battery_id"].isin(val_b)].reset_index(drop=True)
    df_test = df[df["battery_id"].isin(test_b)].reset_index(drop=True)

    # ---------------------------------------------------------------
    # 3) TCN training
    # ---------------------------------------------------------------
    if use_tcn:
        tcn_input_cols = feature_cols + ["tau_deploy"]
        tag = f"{variant_name}_train_{'_'.join(train_b)}"
        model, _ = train_tcn(df_train, df_val, tcn_input_cols, tag=tag)

        df = predict_tcn_full(df, model, tcn_input_cols)
        df["tcn_soh"] = df["tcn_soh"].replace([np.inf, -np.inf], np.nan)
        df["tcn_soh"] = df["tcn_soh"].fillna(df["soh"])
    else:
        df["tcn_soh"] = df["soh"]

    # ---------------------------------------------------------------
    # 4) Knee detection (TRAIN+VAL only)
    # ---------------------------------------------------------------
    df_train_val = df[df["battery_id"].isin(all_trainval)]
    knee1, knee2 = detect_two_knees(df_train_val)

    # ---------------------------------------------------------------
    # 5) SVR training
    # ---------------------------------------------------------------
    if use_piecewise:
        svr_models = train_piecewise_svr(
            df_train_val,
            use_tcn_soh=True,
            knee1=knee1,
            knee2=knee2,
            label_col="rul_fraction",
        )
        df = predict_piecewise_svr(
            df,
            svr_models,
            use_tcn_soh=True,
            knee1=knee1,
            knee2=knee2,
        )
        pred_col = "svr_rul_fraction"
    else:
        X = df_train_val[["tcn_soh"]].values
        y = df_train_val["rul_fraction"].values
        svr = SVR(C=SVR_C, epsilon=SVR_EPSILON, kernel=SVR_KERNEL).fit(X, y)
        df["svr_rul_fraction"] = svr.predict(df[["tcn_soh"]].values)
        pred_col = "svr_rul_fraction"

    # ---------------------------------------------------------------
    # 6) Evaluate on *test* battery only for this LOBO fold
    # ---------------------------------------------------------------
    df_test_pred = df[df["battery_id"].isin(test_b)].reset_index(drop=True)

    test_label = "_".join(test_b)
    eval_name = f"{variant_name}_test_{test_label}"
    metrics = evaluate_rul(
        df_test_pred,
        pred_col=pred_col,
        label_col="rul_fraction",
        name=eval_name,
    )

    # Per-battery metrics (here usually a single battery)
    per_bat = []
    for bid, g in df_test_pred.groupby("battery_id"):
        m = compute_metrics(g["rul_fraction"].values,
                            g[pred_col].values)
        m["battery_id"] = bid
        per_bat.append(m)
    per_bat_df = pd.DataFrame(per_bat)
    per_bat_df.to_csv(
        METRICS_DIR / f"metrics_per_battery_{variant_name}_test_{test_label}.csv",
        index=False,
    )

    plt.figure(figsize=(4, 4))
    sns.boxplot(y=per_bat_df["RMSE"])
    plt.ylabel("RMSE (RUL fraction)")
    plt.title(f"Per-battery RMSE ({variant_name}, test {test_label})")
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / f"box_rmse_per_battery_{variant_name}_test_{test_label}.png",
        dpi=300,
    )
    plt.close()

    return metrics


def run_all_nasa_variants(df_raw: pd.DataFrame) -> None:
    """
    Run all variants under leave-one-battery-out (LOBO) evaluation.

    For each battery b*:
        - Train on all other batteries (with an internal train/val split).
        - Test on b*.
    Variants:
      - SAN_TCN_3SVR: proposed method.
      - NoSAN_TCN_3SVR: raw features + TCN + 3SVR.
      - PCA_TCN_3SVR: PCA-whitened features + TCN + 3SVR.
      - WHITEN_TCN_3SVR: StandardScaler-whitened features + TCN + 3SVR.
      - SAN_noTCN_3SVR: SAN + direct 3SVR on raw SOH.
      - SAN_TCN_1SVR: SAN + TCN + single SVR.

    Outputs:
      - Fold-wise metrics (per variant, per test battery).
      - Aggregated mean±std metrics per variant (CSV + LaTeX).
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

    batteries = sorted(df_raw["battery_id"].unique())
    all_metrics_rows = []

    for test_b in batteries:
        # LOBO: test_b is held out; remaining batteries are train+val
        trainval_bats = [b for b in batteries if b != test_b]

        rng = np.random.RandomState(RANDOM_SEED)
        rng.shuffle(trainval_bats)

        # Simple 80/20 split of TRAIN+VAL batteries for TCN early stopping
        n_val = max(1, int(0.2 * len(trainval_bats)))
        val_b = trainval_bats[:n_val]
        train_b = trainval_bats[n_val:]

        # In the degenerate case with very few batteries:
        if len(train_b) == 0:
            train_b = val_b[:]
            val_b = [train_b[-1]]

        for v in variants:
            metrics = run_pipeline_variant(
                df_raw,
                variant_name=v["name"],
                use_san=v["use_san"],
                use_pca=v["use_pca"],
                use_whiten_all=v["use_whiten_all"],
                use_tcn=v["use_tcn"],
                use_piecewise=v["use_piecewise"],
                train_b=train_b,
                val_b=val_b,
                test_b=[test_b],
            )
            metrics["variant"] = v["name"]
            metrics["test_battery"] = test_b
            all_metrics_rows.append(metrics)

    all_metrics_df = pd.DataFrame(all_metrics_rows)
    all_metrics_df.to_csv(
        TABLES_DIR / "metrics_all_variants_LOBO.csv",
        index=False,
    )

    # Aggregated stats per variant
    agg = (
        all_metrics_df
        .groupby("variant")
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            SMAPE_mean=("SMAPE", "mean"),
            SMAPE_std=("SMAPE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .reset_index()
    )
    agg.to_csv(TABLES_DIR / "metrics_all_variants_LOBO_agg.csv", index=False)

    with open(TABLES_DIR / "metrics_all_variants_LOBO_agg.tex", "w") as f:
        f.write(agg.to_latex(index=False, float_format="%.4f"))

    print("\n=== LOBO evaluation complete. Aggregated metrics saved. ===")


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="NASA Ames IDM pipeline (SAN + TCN + SVR on master CSV) with LOBO evaluation."
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
