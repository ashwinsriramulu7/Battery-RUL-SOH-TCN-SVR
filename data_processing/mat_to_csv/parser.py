"""
NASA Battery Dataset MATLAB Parser
----------------------------------
This module provides functionality to parse *.mat files from the NASA Ames
battery aging experiments. It extracts charge, discharge, and impedance cycles
and exports them as consistent, analysis-ready pandas DataFrames.

This parser intentionally handles:
- inconsistent MATLAB struct formats
- inconsistent naming between experiments
- variable-length time series
- missing and optional fields
"""

import os
import scipy.io as sio
import numpy as np
import pandas as pd

from .utils_datetime import matlab_datevec_to_datetime
from .utils_fieldmap import safe_get, get_any
from .utils_alignment import truncate_fields_to_min_length


def parse_single_mat(mat_path):
    """
    Parse a single NASA *.mat file into charge, discharge, and impedance DataFrames.

    Parameters
    ----------
    mat_path : str
        Path to a NASA *.mat file.

    Returns
    -------
    (charge_df, discharge_df, impedance_df)
    """

    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    root_key = [k for k in data.keys() if k.startswith("B")][0]
    battery = data[root_key]

    cycles = np.atleast_1d(battery.cycle)

    charge_rows = []
    discharge_rows = []
    imp_rows = []

    for idx, cy in enumerate(cycles):

        ctype = safe_get(cy, "type")
        if isinstance(ctype, np.ndarray):
            ctype = ctype.item()

        if not isinstance(ctype, str):
            continue

        d = safe_get(cy, "data")
        if d is None:
            continue

        time_vec = safe_get(d, "Time")
        if time_vec is None:
            continue

        cycle_number = safe_get(cy, "cycle", idx + 1)
        start_time_raw = safe_get(cy, "time")
        ambient_temp = safe_get(cy, "ambient_temperature")

        start_time = matlab_datevec_to_datetime(start_time_raw)

        voltage = get_any(d, ["Voltage_measured", "Voltage_load", "Voltage_charge"])
        current = get_any(d, ["Current_measured", "Current_load", "Current_charge"])
        temperature = get_any(d, ["Temperature_measured", "Temperature", "T"])
        capacity = safe_get(d, "Capacity")

        # Build raw fields
        fields = {
            "time_s": time_vec,
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
        }

        if ctype == "discharge" and capacity is not None:
            fields["capacity_ahr"] = np.array([capacity] * len(time_vec))

        fields = truncate_fields_to_min_length(fields)
        df = pd.DataFrame(fields)
        N = len(df)

        df["cycle_number"] = [cycle_number] * N
        df["start_time"] = [start_time.isoformat() if start_time else None] * N
        df["ambient_temperature"] = [ambient_temp] * N

        if ctype == "charge":
            charge_rows.append(df)

        elif ctype == "discharge":
            discharge_rows.append(df)

        elif ctype == "impedance":
            imp_dict = {
                "cycle_number": cycle_number,
                "start_time": start_time.isoformat() if start_time else None,
                "ambient_temperature": ambient_temp,
                "sense_current": safe_get(d, "Sense_current"),
                "battery_current": safe_get(d, "Battery_current"),
                "current_ratio": safe_get(d, "Current_ratio"),
                "impedance_raw": safe_get(d, "Battery_impedance"),
                "impedance_rectified": safe_get(d, "Rectified_impedance"),
                "Re": safe_get(d, "Re"),
                "Rct": safe_get(d, "Rct"),
            }
            imp_rows.append(pd.DataFrame([imp_dict]))

    charge_df = pd.concat(charge_rows, ignore_index=True) if charge_rows else None
    discharge_df = pd.concat(discharge_rows, ignore_index=True) if discharge_rows else None
    impedance_df = pd.concat(imp_rows, ignore_index=True) if imp_rows else None

    return charge_df, discharge_df, impedance_df
