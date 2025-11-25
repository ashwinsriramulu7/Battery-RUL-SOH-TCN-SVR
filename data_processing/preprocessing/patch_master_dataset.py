"""
Patch the existing discharge_master_normalized.csv to add:

- cycle_index (same as cycle_number)
- soh
- rul_cycles
- voltage_norm
- current_norm
- temperature_norm

Output -> discharge_master_normalized_v3.csv
"""

import pandas as pd
from pathlib import Path

MASTER_PATH = Path("processed-data/master_datasets/discharge_master_normalized.csv")
OUTPUT_PATH = Path("processed-data/master_datasets/discharge_master_normalized_v3.csv")


def patch(df: pd.DataFrame) -> pd.DataFrame:
    # --------------------------------------------------------
    # 1. cycle_index  (TCN expects this)
    # --------------------------------------------------------
    if "cycle_index" not in df.columns:
        if "cycle_number" not in df.columns:
            raise ValueError("Neither cycle_index nor cycle_number exists.")
        df["cycle_index"] = df["cycle_number"]

    # --------------------------------------------------------
    # 2. SOH = capacity_ahr / initial capacity
    # --------------------------------------------------------
    if "soh" not in df.columns:
        if "capacity_ahr" not in df.columns:
            raise ValueError("capacity_ahr missing — cannot compute SOH")

        df["initial_capacity"] = df.groupby("battery_id")["capacity_ahr"].transform(lambda x: x.iloc[0])
        df["soh"] = df["capacity_ahr"] / df["initial_capacity"]
        df["soh"] = df["soh"].clip(upper=1.0)

    # --------------------------------------------------------
    # 3. RUL in cycles
    # --------------------------------------------------------
    if "rul_cycles" not in df.columns:
        df["max_cycle"] = df.groupby("battery_id")["cycle_index"].transform("max")
        df["rul_cycles"] = df["max_cycle"] - df["cycle_index"]

    # --------------------------------------------------------
    # 4. Normalized features
    # --------------------------------------------------------
    def add_norm(col_name):
        df[f"{col_name}_norm"] = (df[col_name] - df[col_name].mean()) / df[col_name].std()

    if "voltage_norm" not in df.columns:
        add_norm("voltage")

    if "current_norm" not in df.columns:
        add_norm("current")

    if "temperature_norm" not in df.columns:
        add_norm("temperature")

    return df


def main():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master dataset not found at {MASTER_PATH}")

    print(f"Loading: {MASTER_PATH}")
    df = pd.read_csv(MASTER_PATH)

    print("Applying patches...")
    df = patch(df)

    print(f"Saving patched dataset to: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
