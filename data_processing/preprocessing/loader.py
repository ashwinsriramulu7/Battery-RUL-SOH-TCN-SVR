"""
Loading utilities for processed discharge CSVs.

This module aggregates all BXXXX_discharge.csv files from
processed-data/discharge_csv/ and attaches experiment metadata for
each battery (ambient temperature, discharge current, cutoff voltage,
experiment group).
"""

import glob
import os
import pandas as pd

from .metadata import BATTERY_METADATA


def load_all_discharge_cycles(
    discharge_dir: str = "processed-data/discharge_csv",
    drop_missing_meta: bool = True,
) -> pd.DataFrame:
    """
    Load all discharge CSVs, attach battery-level metadata, and
    return a single concatenated DataFrame.

    Parameters
    ----------
    discharge_dir : str
        Directory containing BXXXX_discharge.csv files.

    drop_missing_meta : bool
        If True, skip batteries that are not defined in BATTERY_METADATA.

    Returns
    -------
    pd.DataFrame
        Combined discharge dataset across all batteries.
    """
    pattern = os.path.join(discharge_dir, "B*_discharge.csv")
    files = sorted(glob.glob(pattern))

    frames = []

    for path in files:
        fname = os.path.basename(path)
        battery_id = fname[1:5]  # "0005" from "B0005_discharge.csv"

        df = pd.read_csv(path)
        df["battery_id"] = battery_id

        meta = BATTERY_METADATA.get(battery_id)
        if meta is None:
            msg = f"No metadata entry found for battery {battery_id}."
            if drop_missing_meta:
                print(f"[load_all_discharge_cycles] {msg} Skipping.")
                continue
            else:
                print(f"[load_all_discharge_cycles] {msg} Using NaNs.")
                df["experiment_group"] = None
                df["ambient_temp"] = None
                df["discharge_current"] = None
                df["cutoff_voltage"] = None
        else:
            df["experiment_group"] = meta.exp_group
            df["ambient_temp"] = meta.ambient
            df["discharge_current"] = meta.current
            df["cutoff_voltage"] = meta.cutoff


        frames.append(df)

    if not frames:
        raise RuntimeError("No discharge CSV files found or all missing metadata.")

    combined = pd.concat(frames, ignore_index=True)
    return combined
