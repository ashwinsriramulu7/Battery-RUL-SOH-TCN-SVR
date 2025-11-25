# models/eval_extended.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import json
import math

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from .config import (
    MASTER_DATA_PATH,
    INPUT_FEATURES,
    PLOTS_DIR,
    RESULTS_DIR,
    EOL_CAPACITY_AH,
)
from .train_svr import _load_best_tcn  # reuse the TCN loader


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _ensure_dirs() -> Tuple[Path, Path]:
    """Create aggregated/per-battery plot dirs if they do not exist."""
    agg_dir = PLOTS_DIR / "plots_aggregated"
    per_batt_dir = PLOTS_DIR / "plots_per_battery"

    agg_dir.mkdir(parents=True, exist_ok=True)
    per_batt_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    return agg_dir, per_batt_dir


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------
# 1. Build cycle-level predictions from TCN
# -------------------------------------------------------------------------

def build_cycle_level_predictions() -> pd.DataFrame:
    """
    For each (battery_id, cycle_index), run the trained TCN on the full
    discharge time series and aggregate to:
        - soh_true, soh_pred
        - capacity_true, capacity_pred
        - initial_capacity

    Returns a DataFrame with one row per (battery, cycle).
    """
    if not MASTER_DATA_PATH.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_DATA_PATH}")

    df = pd.read_csv(MASTER_DATA_PATH)
    required = {"battery_id", "cycle_index", "capacity_ahr", "initial_capacity", "soh"} | set(INPUT_FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Master dataset missing required columns: {missing}")

    # Sort for reproducibility
    df = df.sort_values(["battery_id", "cycle_index", "time_s"]).reset_index(drop=True)

    # Load TCN
    device = _get_device()
    model = _load_best_tcn().to(device)
    model.eval()

    rows = []
    grouped = df.groupby(["battery_id", "cycle_index"], sort=True)

    for (bid, cid), g in grouped:
        g = g.reset_index(drop=True)

        X_np = g[INPUT_FEATURES].to_numpy(dtype=np.float32)  # [L, F]
        X_t = torch.from_numpy(X_np).unsqueeze(0).to(device)  # [1, L, F]

        with torch.no_grad():
            soh_pred_seq, _ = model(X_t)  # [1, L]
        soh_pred_seq = soh_pred_seq.squeeze(0).cpu().numpy()

        # Aggregate prediction over the cycle.
        # Since target SOH is constant within a cycle, mean is robust.
        soh_pred = float(np.mean(soh_pred_seq))

        soh_true = float(g["soh"].iloc[0])
        init_cap = float(g["initial_capacity"].iloc[0])
        cap_true = float(g["capacity_ahr"].max())   # constant over the cycle
        cap_pred = float(soh_pred * init_cap)

        rows.append(
            {
                "battery_id": str(bid),
                "cycle_index": int(cid),
                "soh_true": soh_true,
                "soh_pred": soh_pred,
                "capacity_true": cap_true,
                "capacity_pred": cap_pred,
                "initial_capacity": init_cap,
            }
        )

    cycle_df = pd.DataFrame(rows).sort_values(["battery_id", "cycle_index"]).reset_index(drop=True)

    out_csv = RESULTS_DIR / "cycle_level_predictions.csv"
    cycle_df.to_csv(out_csv, index=False)
    print(f"[eval_extended] Wrote cycle-level predictions to: {out_csv}")

    return cycle_df


# -------------------------------------------------------------------------
# 2. Smoothing + simple RUL construction
# -------------------------------------------------------------------------

def _savgol_params(n: int) -> Tuple[int, int]:
    """
    Choose a reasonable (window_length, polyorder) for Savitzky-Golay
    given a sequence length n.
    """
    if n < 5:
        return 1, 1  # no real smoothing possible
    # window: odd, <= n, up to 11
    w = min(11, n if n % 2 == 1 else n - 1)
    if w < 3:
        w = 3
    p = 2 if n >= 7 else 1
    if p >= w:
        p = w - 1
    return w, p


def add_smoothing_and_rul(cycle_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each battery:
        - smooth capacity_true, capacity_pred over cycle_index
        - compute a simple RUL estimate:
            RUL_true = EOL_cycle_true - cycle_index
            RUL_pred = EOL_cycle_pred - cycle_index

    Returns the augmented DataFrame.
    """
    cycle_df = cycle_df.copy()
    cycle_df["capacity_true_smooth"] = np.nan
    cycle_df["capacity_pred_smooth"] = np.nan
    cycle_df["rul_true"] = np.nan
    cycle_df["rul_pred"] = np.nan

    for bid, g in cycle_df.groupby("battery_id", sort=True):
        idx = g.index
        g = g.sort_values("cycle_index")

        c_true = g["capacity_true"].to_numpy()
        c_pred = g["capacity_pred"].to_numpy()
        cycles = g["cycle_index"].to_numpy()

        w, p = _savgol_params(len(cycles))
        if w > 1:
            c_true_s = savgol_filter(c_true, window_length=w, polyorder=p)
            c_pred_s = savgol_filter(c_pred, window_length=w, polyorder=p)
        else:
            c_true_s = c_true
            c_pred_s = c_pred

        # Simple "EOL cycle" (first cycle where smoothed capacity <= EOL)
        def _eol_cycle(cap_s, cyc):
            mask = cap_s <= EOL_CAPACITY_AH
            if mask.any():
                return float(cyc[mask][0])
            # If not reached, pretend EOL at last observed cycle
            return float(cyc[-1])

        eol_true = _eol_cycle(c_true_s, cycles)
        eol_pred = _eol_cycle(c_pred_s, cycles)

        rul_true = eol_true - cycles
        rul_pred = eol_pred - cycles

        # Assign back
        cycle_df.loc[idx, "capacity_true_smooth"] = c_true_s
        cycle_df.loc[idx, "capacity_pred_smooth"] = c_pred_s
        cycle_df.loc[idx, "rul_true"] = rul_true
        cycle_df.loc[idx, "rul_pred"] = rul_pred

    return cycle_df


# -------------------------------------------------------------------------
# 3. Global metrics
# -------------------------------------------------------------------------

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_global_metrics(cycle_df: pd.DataFrame) -> Dict[str, float]:
    """Compute aggregate MAE/RMSE for SOH, capacity, and RUL."""
    metrics: Dict[str, float] = {}

    for name, true_col, pred_col in [
        ("soh", "soh_true", "soh_pred"),
        ("capacity_smooth", "capacity_true_smooth", "capacity_pred_smooth"),
        ("rul", "rul_true", "rul_pred"),
    ]:
        mask = cycle_df[true_col].notna() & cycle_df[pred_col].notna()
        if not mask.any():
            continue
        t = cycle_df.loc[mask, true_col].to_numpy()
        p = cycle_df.loc[mask, pred_col].to_numpy()
        mae = float(np.mean(np.abs(p - t)))
        rmse = _rmse(p, t)
        metrics[f"{name}_mae"] = mae
        metrics[f"{name}_rmse"] = rmse

    out_path = RESULTS_DIR / "extended_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval_extended] Wrote global metrics to: {out_path}")
    print(json.dumps(metrics, indent=2))

    return metrics


# -------------------------------------------------------------------------
# 4. Aggregated plots
# -------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_aggregated_plots(cycle_df: pd.DataFrame, agg_dir: Path) -> None:
    """Create cross-battery, aggregated plots."""
    # 1. Capacity: true vs predicted (smoothed)
    mask = cycle_df["capacity_true_smooth"].notna() & cycle_df["capacity_pred_smooth"].notna()
    ct = cycle_df.loc[mask, "capacity_true_smooth"].to_numpy()
    cp = cycle_df.loc[mask, "capacity_pred_smooth"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ct, cp, s=18, alpha=0.6)
    lims = [min(ct.min(), cp.min()), max(ct.max(), cp.max())]
    ax.plot(lims, lims, "k--", alpha=0.7)
    ax.set_xlabel("True capacity (Ah, smoothed)")
    ax.set_ylabel("Predicted capacity (Ah, smoothed)")
    ax.set_title("Aggregated: True vs Predicted Capacity")
    _savefig(fig, agg_dir / "agg_capacity_true_vs_pred.png")

    # 2. Capacity residual histogram
    resid = cp - ct
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(resid, bins=30, alpha=0.7)
    ax.set_xlabel("Residual (pred - true) [Ah]")
    ax.set_ylabel("Count")
    ax.set_title("Aggregated: Capacity Residual Distribution")
    _savefig(fig, agg_dir / "agg_capacity_residual_hist.png")

    # 3. RUL: true vs predicted
    mask = cycle_df["rul_true"].notna() & cycle_df["rul_pred"].notna()
    rt = cycle_df.loc[mask, "rul_true"].to_numpy()
    rp = cycle_df.loc[mask, "rul_pred"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(rt, rp, s=18, alpha=0.6)
    lims = [min(rt.min(), rp.min()), max(rt.max(), rp.max())]
    ax.plot(lims, lims, "k--", alpha=0.7)
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title("Aggregated: True vs Predicted RUL")
    _savefig(fig, agg_dir / "agg_rul_true_vs_pred.png")

    # 4. Per-battery RMSE (capacity)
    cap_rmse_per_batt = []
    for bid, g in cycle_df.groupby("battery_id", sort=True):
        mask_b = g["capacity_true_smooth"].notna() & g["capacity_pred_smooth"].notna()
        if not mask_b.any():
            continue
        t = g.loc[mask_b, "capacity_true_smooth"].to_numpy()
        p = g.loc[mask_b, "capacity_pred_smooth"].to_numpy()
        cap_rmse_per_batt.append((bid, _rmse(t, p)))

    if cap_rmse_per_batt:
        bids, rmses = zip(*cap_rmse_per_batt)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(bids)), rmses)
        ax.set_xticks(range(len(bids)))
        ax.set_xticklabels([str(b) for b in bids], rotation=90)
        ax.set_ylabel("RMSE (Ah)")
        ax.set_title("Per-battery RMSE (Capacity, smoothed)")
        _savefig(fig, agg_dir / "agg_per_battery_capacity_rmse.png")

    # 5. Overlay of capacity curves (smoothed)
    fig, ax = plt.subplots(figsize=(8, 5))
    for bid, g in cycle_df.groupby("battery_id", sort=True):
        g = g.sort_values("cycle_index")
        ax.plot(
            g["cycle_index"],
            g["capacity_true_smooth"],
            alpha=0.5,
            label=f"B{str(bid).zfill(4)}" if len(g) > 0 else str(bid),
        )
    ax.axhline(EOL_CAPACITY_AH, color="k", linestyle="--", alpha=0.7, label="EOL")
    ax.set_xlabel("Cycle index")
    ax.set_ylabel("Capacity (Ah, smoothed)")
    ax.set_title("True Capacity Curves (All Batteries)")
    # avoid huge legends; only show if few batteries
    if cycle_df["battery_id"].nunique() <= 12:
        ax.legend()
    _savefig(fig, agg_dir / "agg_capacity_curves_true.png")


# -------------------------------------------------------------------------
# 5. Per-battery plots
# -------------------------------------------------------------------------

def make_per_battery_plots(cycle_df: pd.DataFrame, per_batt_dir: Path) -> None:
    """
    For each battery, write several diagnostic plots to:

        artifacts/plots/plots_per_battery/BXXXX/*.png
    """
    for bid, g in cycle_df.groupby("battery_id", sort=True):
        bid_str = str(bid).zfill(4)
        out_dir = per_batt_dir / f"B{bid_str}"
        out_dir.mkdir(parents=True, exist_ok=True)

        g = g.sort_values("cycle_index")
        cycles = g["cycle_index"].to_numpy()

        # Capacity vs cycle
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(cycles, g["capacity_true_smooth"], label="True (smoothed)")
        ax.plot(cycles, g["capacity_pred_smooth"], label="Pred (smoothed)")
        ax.axhline(EOL_CAPACITY_AH, color="k", linestyle="--", alpha=0.7, label="EOL")
        ax.set_xlabel("Cycle index")
        ax.set_ylabel("Capacity (Ah)")
        ax.set_title(f"Battery B{bid_str}: Capacity vs Cycle")
        ax.legend()
        _savefig(fig, out_dir / "capacity_vs_cycle.png")

        # Residual vs cycle
        resid = g["capacity_pred_smooth"] - g["capacity_true_smooth"]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(cycles, resid, marker="o", linestyle="-", alpha=0.7)
        ax.axhline(0.0, color="k", linestyle="--", alpha=0.7)
        ax.set_xlabel("Cycle index")
        ax.set_ylabel("Residual (Ah)")
        ax.set_title(f"Battery B{bid_str}: Capacity Residuals")
        _savefig(fig, out_dir / "capacity_residuals.png")

        # RUL curves
        if g["rul_true"].notna().any() and g["rul_pred"].notna().any():
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(cycles, g["rul_true"], label="True RUL", alpha=0.8)
            ax.plot(cycles, g["rul_pred"], label="Pred RUL", alpha=0.8)
            ax.set_xlabel("Cycle index")
            ax.set_ylabel("RUL (cycles)")
            ax.set_title(f"Battery B{bid_str}: RUL vs Cycle")
            ax.legend()
            _savefig(fig, out_dir / "rul_vs_cycle.png")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

def run_extended_evaluation() -> None:
    agg_dir, per_batt_dir = _ensure_dirs()

    print("[eval_extended] Building cycle-level predictions from TCN...")
    cycle_df = build_cycle_level_predictions()

    print("[eval_extended] Adding smoothing + simple RUL estimates...")
    cycle_df = add_smoothing_and_rul(cycle_df)

    # Save augmented cycle-level CSV
    out_csv = RESULTS_DIR / "cycle_level_predictions_with_rul.csv"
    cycle_df.to_csv(out_csv, index=False)
    print(f"[eval_extended] Wrote augmented cycle-level CSV to: {out_csv}")

    print("[eval_extended] Computing global metrics...")
    compute_global_metrics(cycle_df)

    print("[eval_extended] Creating aggregated plots...")
    make_aggregated_plots(cycle_df, agg_dir)

    print("[eval_extended] Creating per-battery plots...")
    make_per_battery_plots(cycle_df, per_batt_dir)

    print("[eval_extended] Done.")


if __name__ == "__main__":
    run_extended_evaluation()
