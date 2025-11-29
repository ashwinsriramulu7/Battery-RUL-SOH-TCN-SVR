#!/usr/bin/env python3
"""
build_master_dataset.py

- Reads all discharge CSVs from processed-data/discharge_csv/
- Builds:
    artifacts/master_datasets/cycle_summary.csv   (one row per battery-cycle)
    artifacts/master_datasets/discharge_sequences_raw.npz
      X_raw: (N_cycles, L, 3)  -> [voltage, current, temperature]
      cycle_indices: (N_cycles,) -> int index into cycle_summary.csv
"""

import os
import glob
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Tuple

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent
DISCHARGE_DIR = PROJECT_ROOT / "processed-data" / "discharge_csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MASTER_DIR = ARTIFACTS_DIR / "master_datasets"

# Fixed sequence length after resampling
SEQ_LEN = 100


def ensure_dirs():
    MASTER_DIR.mkdir(parents=True, exist_ok=True)


def list_discharge_files() -> List[Path]:
    return sorted(DISCHARGE_DIR.glob("B*_discharge.csv"))


def extract_battery_id(path: Path) -> str:
    """
    From 'B0005_discharge.csv' -> 'B0005'
    """
    stem = path.stem  # B0005_discharge
    return stem.split("_")[0]


def resample_cycle(
    df_cycle: pd.DataFrame, seq_len: int = SEQ_LEN
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a single cycle (time_s, voltage, current, temperature)
    onto a fixed-length grid using linear interpolation.

    Returns:
        times_norm: (seq_len,) normalized time in [0, 1]
        seq: (seq_len, 3) float32 -> [voltage, current, temperature]
    """
    t = df_cycle["time_s"].values.astype(float)
    v = df_cycle["voltage"].values.astype(float)
    i = df_cycle["current"].values.astype(float)
    temp = df_cycle["temperature"].values.astype(float)

    # Handle degenerate cycles
    if len(t) < 2 or np.allclose(t[0], t[-1]):
        # Just tile the first sample
        times_norm = np.linspace(0.0, 1.0, seq_len)
        v_res = np.full(seq_len, v[0], dtype=float)
        i_res = np.full(seq_len, i[0], dtype=float)
        temp_res = np.full(seq_len, temp[0], dtype=float)
    else:
        t0 = t[0]
        t1 = t[-1]
        t_norm = (t - t0) / (t1 - t0)
        grid = np.linspace(0.0, 1.0, seq_len)

        v_res = np.interp(grid, t_norm, v)
        i_res = np.interp(grid, t_norm, i)
        temp_res = np.interp(grid, t_norm, temp)
        times_norm = grid

    seq = np.stack([v_res, i_res, temp_res], axis=-1).astype(np.float32)
    return times_norm.astype(np.float32), seq


def process_discharge_file(path: Path, battery_id: str) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]:
    """
    Process a single discharge CSV:
    - Group by cycle_number
    - Build cycle-level summary rows
    - Build sequence-level resampled arrays

    Returns:
        cycle_rows: list of dicts to be appended to summary DataFrame
        seq_list: list of (L, 3) arrays
        tnorm_list: list of (L,) normalized time arrays
    """
    df = pd.read_csv(path)
    required_cols = {
        "time_s",
        "voltage",
        "current",
        "temperature",
        "capacity_ahr",
        "cycle_number",
        "start_time",
        "ambient_temperature",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    cycle_rows = []
    seq_list = []
    tnorm_list = []

    grouped = df.groupby("cycle_number")

    for cycle_number, df_cycle in grouped:
        df_cycle = df_cycle.sort_values("time_s")
        ambient_temp = df_cycle["ambient_temperature"].iloc[0]
        start_time = df_cycle["start_time"].iloc[0]
        capacity_vals = df_cycle["capacity_ahr"].unique()
        # Sanity: capacity is constant per cycle
        capacity = float(capacity_vals[0])

        mean_current = float(df_cycle["current"].mean())
        min_voltage = float(df_cycle["voltage"].min())
        final_voltage = float(df_cycle["voltage"].iloc[-1])
        max_temp = float(df_cycle["temperature"].max())

        times_norm, seq = resample_cycle(df_cycle)

        cycle_rows.append(
            dict(
                battery_id=battery_id,
                cycle_number=int(cycle_number),
                start_time=start_time,
                ambient_temperature=float(ambient_temp),
                capacity_ahr=capacity,
                mean_current=mean_current,
                min_voltage=min_voltage,
                final_voltage=final_voltage,
                max_temperature=max_temp,
            )
        )
        tnorm_list.append(times_norm)
        seq_list.append(seq)

    return pd.DataFrame(cycle_rows), tnorm_list, seq_list


def main():
    ensure_dirs()

    discharge_files = list_discharge_files()
    if not discharge_files:
        raise RuntimeError(f"No discharge CSVs found in {DISCHARGE_DIR}")

    all_cycle_rows = []
    all_seq = []
    all_tnorm = []
    cycle_index_map = []  # (battery_id, cycle_number) for each sequence index

    for path in discharge_files:
        battery_id = extract_battery_id(path)
        print(f"Processing {path.name} (battery {battery_id})")
        cycle_df, tnorm_list, seq_list = process_discharge_file(path, battery_id)

        # Append rows and sequences
        all_cycle_rows.append(cycle_df)
        all_tnorm.extend(tnorm_list)
        all_seq.extend(seq_list)
        # Pair mapping for each cycle
        for _, row in cycle_df.iterrows():
            cycle_index_map.append((row["battery_id"], int(row["cycle_number"])))

    # Build master cycle summary
    summary_df = pd.concat(all_cycle_rows, ignore_index=True)
    # Sort by battery_id then cycle_number
    summary_df = summary_df.sort_values(["battery_id", "cycle_number"]).reset_index(drop=True)
    summary_path = MASTER_DIR / "cycle_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved cycle summary to {summary_path} with {len(summary_df)} cycles.")

    # Build mapping from (battery_id, cycle_number) -> row index in summary_df
    key_to_idx = {
        (row["battery_id"], int(row["cycle_number"])): idx
        for idx, row in summary_df.iterrows()
    }

    # Align sequence arrays with summary_df row indices
    N = len(all_seq)
    if len(cycle_index_map) != N:
        raise RuntimeError("cycle_index_map length mismatch")

    seq_len = all_seq[0].shape[0]
    feat_dim = all_seq[0].shape[1]
    X_raw = np.zeros((N, seq_len, feat_dim), dtype=np.float32)
    times_norm_all = np.zeros((N, seq_len), dtype=np.float32)
    cycle_indices = np.zeros(N, dtype=np.int64)

    for idx, ((bid, cyc), seq, tnorm) in enumerate(zip(cycle_index_map, all_seq, all_tnorm)):
        if seq.shape != (seq_len, feat_dim):
            raise RuntimeError(f"Inconsistent seq shape at idx {idx}: {seq.shape}")
        X_raw[idx] = seq
        times_norm_all[idx] = tnorm
        cycle_indices[idx] = key_to_idx[(bid, cyc)]

    npz_path = MASTER_DIR / "discharge_sequences_raw.npz"
    np.savez_compressed(
        npz_path,
        X_raw=X_raw,
        times_norm=times_norm_all,
        cycle_indices=cycle_indices,
    )
    print(f"Saved raw sequences to {npz_path} with shape {X_raw.shape}.")


if __name__ == "__main__":
    main()
