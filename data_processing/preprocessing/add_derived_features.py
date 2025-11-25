"""
Post-processing script:
Adds SOH, RUL, and normalized features to an already-built master dataset.
"""

import pandas as pd
from pathlib import Path


# -------------------------------------------------------
# Config
# -------------------------------------------------------

MASTER_PATH = Path("processed-data/master_datasets/discharge_master_normalized.csv")
OUTPUT_PATH = Path("processed-data/master_datasets/discharge_master_normalized_v2.csv")


def add_soh_rul_norm(df: pd.DataFrame) -> pd.DataFrame:
    # -----------------------------------------
    # SOH = capacity_ahr / initial_capacity
    # -----------------------------------------
    if "capacity_ahr" not in df.columns:
        raise ValueError("Column 'capacity_ahr' missing from master dataset.")

    df["initial_capacity"] = df.groupby("battery_id")["capacity_ahr"].transform(
        lambda x: x.iloc[0]
    )

    df["soh"] = df["capacity_ahr"] / df["initial_capacity"]
    df["soh"] = df["soh"].clip(upper=1.0)

    # -----------------------------------------
    # RUL (Remaining Useful Life in cycles)
    # -----------------------------------------
    if "cycle_number" not in df.columns:
        raise ValueError("Column 'cycle_number' missing from master dataset.")

    df["max_cycle"] = df.groupby("battery_id")["cycle_number"].transform("max")
    df["rul_cycles"] = df["max_cycle"] - df["cycle_number"]

    # -----------------------------------------
    # Normalized features: voltage/current/temp
    # -----------------------------------------
    for col in ["voltage", "current", "temperature"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from master dataset.")

    df["voltage_norm"] = (df["voltage"] - df["voltage"].mean()) / df["voltage"].std()
    df["current_norm"] = (df["current"] - df["current"].mean()) / df["current"].std()
    df["temperature_norm"] = (df["temperature"] - df["temperature"].mean()) / df["temperature"].std()

    return df


def main():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master dataset not found at: {MASTER_PATH}")

    print(f"Loading master dataset from: {MASTER_PATH}")
    df = pd.read_csv(MASTER_PATH)

    print("Computing SOH, RUL, and normalization features...")
    df = add_soh_rul_norm(df)

    print(f"Writing updated dataset to: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)

    print("Completed.")


if __name__ == "__main__":
    main()
