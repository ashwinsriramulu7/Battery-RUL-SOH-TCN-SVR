"""
Build the unified, normalized master discharge dataset.

Pipeline:
- Load all BXXXX_discharge.csv files
- Attach experiment metadata
- Clean data
- Compute domain-wise statistics
- Apply domain-aware normalization
- Add engineered features (including normalized time)
- Write a single master CSV for model training
"""

import os
import pandas as pd

from .loader import load_all_discharge_cycles
from .cleaning import clean_discharge_dataframe
from .scaler import compute_domain_stats, apply_domain_normalization
from .features import add_cycle_features


def build_master_discharge_dataset(
    discharge_dir: str = "processed-data/discharge_csv",
    output_path: str = "processed-data/master_datasets/discharge_master_normalized.csv",
) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline for discharge data.

    Parameters
    ----------
    discharge_dir : str
        Directory containing per-battery discharge CSVs.

    output_path : str
        Output path for the unified, normalized master CSV.

    Returns
    -------
    pd.DataFrame
        Final prepared DataFrame.
    """
    print("[build_master_discharge_dataset] Loading discharge cycles...")
    df = load_all_discharge_cycles(discharge_dir=discharge_dir)

    print(f"[build_master_discharge_dataset] Loaded {len(df)} rows.")
    df = clean_discharge_dataframe(df)
    print(f"[build_master_discharge_dataset] After cleaning: {len(df)} rows.")

    print("[build_master_discharge_dataset] Computing domain statistics...")
    domain_stats = compute_domain_stats(df)

    print("[build_master_discharge_dataset] Applying domain-aware normalization...")
    df = apply_domain_normalization(df, domain_stats)

    print("[build_master_discharge_dataset] Adding engineered features...")
    df = add_cycle_features(df)

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"[build_master_discharge_dataset] Master dataset written to: {output_path}")

    return df


if __name__ == "__main__":
    build_master_discharge_dataset()
