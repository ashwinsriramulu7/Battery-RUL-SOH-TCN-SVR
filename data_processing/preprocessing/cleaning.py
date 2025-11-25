"""
Cleaning and sanity checks for the merged discharge dataset.
"""

import pandas as pd


def clean_discharge_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning steps:
    - Drop duplicate rows
    - Drop rows with NaNs in core sensor fields
    - Remove obviously invalid sensor readings

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged discharge DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.drop_duplicates()

    core_cols = ["voltage", "current", "temperature", "time_s"]
    existing_core = [c for c in core_cols if c in df.columns]
    df = df.dropna(subset=existing_core)

    # Physical sanity filters
    if "voltage" in df.columns:
        df = df[df["voltage"] > 0.0]
        df = df[df["voltage"] < 5.5]  # just a loose upper bound

    if "current" in df.columns:
        df = df[(df["current"] >= -50.0) & (df["current"] <= 50.0)]

    if "temperature" in df.columns:
        df = df[(df["temperature"] > -40.0) & (df["temperature"] < 100.0)]

    return df.reset_index(drop=True)
    