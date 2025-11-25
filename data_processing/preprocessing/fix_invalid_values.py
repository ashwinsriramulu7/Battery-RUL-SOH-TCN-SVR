import pandas as pd
import numpy as np
from pathlib import Path

MASTER = Path("processed-data/master_datasets/discharge_master_normalized_v3.csv")

df = pd.read_csv(MASTER)

# Replace inf with nan first
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Forward fill inside each (battery_id, cycle_index)
df = df.groupby(["battery_id", "cycle_index"]).apply(lambda g: g.ffill().bfill()).reset_index(drop=True)

# If still NaN after ffill/bfill (entire sequence invalid) → interpolate globally
df.interpolate(method="linear", limit_direction="both", inplace=True)

# As a final safety measure: clip extreme numeric outliers
for col in ["voltage_norm", "current_norm", "temperature_norm"]:
    df[col] = df[col].clip(-10, 10)

df.to_csv(MASTER, index=False)
print("✔ Fixed NaNs, Infs, clipping applied.")
