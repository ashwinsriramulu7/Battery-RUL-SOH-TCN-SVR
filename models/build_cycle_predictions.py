import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

MASTER_PATH = "processed-data/master_datasets/discharge_master_normalized_v3.csv"
TCN_PRED = "artifacts/results/tcn_cycle_predictions.csv"

def build_cycle_level_outputs():
    master = pd.read_csv(MASTER_PATH)
    preds = pd.read_csv(TCN_PRED)

    # initial capacity per battery
    init_cap = (
        master.groupby("battery_id")["capacity_ahr"]
        .max()
        .reset_index()
        .rename(columns={"capacity_ahr": "init_cap"})
    )

    df = preds.merge(init_cap, on="battery_id")

    df["capacity_pred"] = df["soh_pred"] * df["init_cap"]

    # smooth per battery
    final_list = []

    for bid, sub in df.groupby("battery_id"):
        s = sub.sort_values("cycle_index")
        cap = savgol_filter(s["capacity_pred"].values, 7, 2)
        s["capacity_pred_smooth"] = cap
        final_list.append(s)

    final = pd.concat(final_list)
    final.to_csv("artifacts/results/cycle_level_predictions.csv", index=False)
    print("Saved:", "artifacts/results/cycle_level_predictions.csv")


if __name__ == "__main__":
    build_cycle_level_outputs()
