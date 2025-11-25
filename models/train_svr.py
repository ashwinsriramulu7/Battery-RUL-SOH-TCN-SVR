# models/train_svr.py

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVR

from .config import (
    MASTER_DATA_PATH,
    INPUT_FEATURES,
    TARGET_SOH_COL,
    TARGET_RUL_COL,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    SOH_SEGMENT_THRESHOLDS,
    SVR_KERNEL,
    SVR_C,
    SVR_GAMMA,
    SVR_EPSILON,
)
from .tcn import TCN


def _load_best_tcn() -> TCN:
    ckpt_path = CHECKPOINT_DIR / "tcn_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"TCN checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    model = TCN(
        input_dim=cfg["input_dim"],
        num_channels=cfg["num_channels"],
        kernel_size=cfg["kernel_size"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _build_cycle_representations(model: TCN):
    """
    For each (battery_id, cycle_index), compute:
      - TCN representation (last hidden state)
      - SOH (scalar per cycle)
      - RUL (scalar per cycle)
    Returns:
      features: [N, repr_dim]
      soh:      [N]
      rul:      [N]
    """
    df = pd.read_csv(MASTER_DATA_PATH)
    required_cols = {"battery_id", "cycle_index", TARGET_SOH_COL, TARGET_RUL_COL, "time_s"} | set(INPUT_FEATURES)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in master dataset: {missing}")

    df = df.sort_values(["battery_id", "cycle_index", "time_s"]).reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    feature_list = []
    soh_list = []
    rul_list = []

    grouped = df.groupby(["battery_id", "cycle_index"], sort=True)

    with torch.no_grad():
        for (bid, cid), g in grouped:
            X = g[INPUT_FEATURES].to_numpy(dtype=np.float32)  # [T, F]
            soh_seq = g[TARGET_SOH_COL].to_numpy(dtype=np.float32)
            rul_seq = g[TARGET_RUL_COL].to_numpy(dtype=np.float32)

            # Use complete cycle as a sequence
            X_t = torch.from_numpy(X).unsqueeze(0).to(device)  # [1, T, F]
            _, repr_vec = model(X_t)                           # [1, repr_dim]

            repr_np = repr_vec.squeeze(0).cpu().numpy()
            soh_mean = float(soh_seq.mean())
            rul_last = float(rul_seq[-1])   # RUL at end-of-discharge for this cycle

            feature_list.append(repr_np)
            soh_list.append(soh_mean)
            rul_list.append(rul_last)

    features = np.stack(feature_list, axis=0)
    soh = np.array(soh_list, dtype=np.float32)
    rul = np.array(rul_list, dtype=np.float32)

    return features, soh, rul


def train_piecewise_svr():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = _load_best_tcn()
    X, soh, rul = _build_cycle_representations(model)

    # Drop any NaNs
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(soh) & ~np.isnan(rul)
    X = X[mask]
    soh = soh[mask]
    rul = rul[mask]

    print(f"Training SVR on {X.shape[0]} cycles with repr_dim={X.shape[1]}")

    # Segment by SOH into early/mid/late
    from .datasets import RULSVRDataset

    full_ds = RULSVRDataset(features=X, targets=rul, soh=soh)
    segments = full_ds.segment_by_soh(SOH_SEGMENT_THRESHOLDS)

    models = {}
    segment_stats = {}

    for name, seg_ds in segments.items():
        if seg_ds.X.shape[0] == 0:
            print(f"Segment '{name}' has no samples; skipping.")
            continue

        svr = SVR(
            kernel=SVR_KERNEL,
            C=SVR_C,
            gamma=SVR_GAMMA,
            epsilon=SVR_EPSILON,
        )
        svr.fit(seg_ds.X, seg_ds.y)

        pred = svr.predict(seg_ds.X)
        rmse = float(np.sqrt(np.mean((pred - seg_ds.y) ** 2)))
        print(f"Segment '{name}': N={seg_ds.X.shape[0]}, RMSE={rmse:.3f} cycles")

        models[name] = svr
        segment_stats[name] = {"N": int(seg_ds.X.shape[0]), "RMSE_cycles": rmse}

    # Persist models
    for name, svr in models.items():
        path = RESULTS_DIR / f"svr_{name}.joblib"
        joblib.dump(svr, path)
        print(f"Saved SVR for segment '{name}' to {path}")

    # Save segment stats
    stats_path = RESULTS_DIR / "svr_segment_stats.json"
    with stats_path.open("w") as f:
        json.dump(segment_stats, f, indent=2)
    print(f"Saved segment stats to {stats_path}")


if __name__ == "__main__":
    train_piecewise_svr()
