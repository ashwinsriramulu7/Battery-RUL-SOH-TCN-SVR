"""
Feature engineering for discharge cycles.

Key decisions:
- Time is represented as normalized cycle progress in [0, 1]
  rather than in physical seconds.
- Additional simple features such as sample index and first-order
  differences are included.
"""

import pandas as pd


def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sequence-level and cycle-level features:
    - sample_index: index within each (battery, cycle)
    - time_norm: normalized position in cycle [0, 1]
    - dV, dI: first-order differences of voltage and current
    - cycle_max_temp, cycle_min_voltage: simple cycle aggregates

    Parameters
    ----------
    df : pd.DataFrame
        Discharge dataset after scaling.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features.
    """
    df = df.copy()

    required_cols = ["battery_id", "cycle_number"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

    # Ensure stable ordering inside each cycle:
    # Prefer time_s if present, else existing index order.
    if "time_s" in df.columns:
        df = df.sort_values(["battery_id", "cycle_number", "time_s"]).reset_index(drop=True)
    else:
        df = df.sort_values(["battery_id", "cycle_number"]).reset_index(drop=True)

    # Sample index within each cycle
    df["sample_index"] = df.groupby(["battery_id", "cycle_number"]).cumcount()

    # Normalized time in cycle [0, 1]
    max_idx = df.groupby(["battery_id", "cycle_number"])["sample_index"].transform("max")
    df["time_norm"] = 0.0
    non_zero_mask = max_idx > 0
    df.loc[non_zero_mask, "time_norm"] = df.loc[non_zero_mask, "sample_index"] / max_idx[non_zero_mask]

    # First-order differences of voltage and current
    if "voltage" in df.columns:
        df["dV"] = df.groupby(["battery_id", "cycle_number"])["voltage"].diff().fillna(0.0)
    else:
        df["dV"] = 0.0

    if "current" in df.columns:
        df["dI"] = df.groupby(["battery_id", "cycle_number"])["current"].diff().fillna(0.0)
    else:
        df["dI"] = 0.0

    # Simple cycle-level aggregates
    if "temperature" in df.columns:
        df["cycle_max_temp"] = df.groupby(["battery_id", "cycle_number"])["temperature"].transform("max")
    else:
        df["cycle_max_temp"] = None

    if "voltage" in df.columns:
        df["cycle_min_voltage"] = df.groupby(["battery_id", "cycle_number"])["voltage"].transform("min")
    else:
        df["cycle_min_voltage"] = None

    return df
