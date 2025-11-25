"""
Runner script to process all NASA *.mat files and export CSVs.
"""

import os
from .parser import parse_single_mat


def convert_all(root_dir, output_base="processed-data"):
    """
    Recursively traverse root_dir for *.mat files, parse them, and write CSVs.

    Parameters
    ----------
    root_dir : str
        Directory containing raw NASA *.mat experiment folders.

    output_base : str
        Base directory where CSV outputs will be written.
    """

    charge_dir = os.path.join(output_base, "charge_csv")
    discharge_dir = os.path.join(output_base, "discharge_csv")
    impedance_dir = os.path.join(output_base, "impedance_csv")

    os.makedirs(charge_dir, exist_ok=True)
    os.makedirs(discharge_dir, exist_ok=True)
    os.makedirs(impedance_dir, exist_ok=True)

    for root, _, files in os.walk(root_dir):
        for f in files:
            if not f.endswith(".mat"):
                continue

            mat_path = os.path.join(root, f)
            battery_id = f.split(".")[0]  # B0005

            charge_df, discharge_df, imp_df = parse_single_mat(mat_path)

            if charge_df is not None:
                charge_df.to_csv(os.path.join(charge_dir, f"{battery_id}_charge.csv"), index=False)

            if discharge_df is not None:
                discharge_df.to_csv(os.path.join(discharge_dir, f"{battery_id}_discharge.csv"), index=False)

            if imp_df is not None:
                imp_df.to_csv(os.path.join(impedance_dir, f"{battery_id}_impedance.csv"), index=False)


if __name__ == "__main__":
    convert_all("raw-data")
