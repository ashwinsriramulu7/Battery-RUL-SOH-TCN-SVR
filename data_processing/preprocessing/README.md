# Preprocessing: Normalization and Feature Engineering

This module builds a unified, normalized discharge dataset for
model training (TCN + SVR) from the per-battery CSV files
produced by the `mat_to_csv` parser.

## Pipeline Overview

1. **Load**
   - Read all `BXXXX_discharge.csv` files from `processed-data/discharge_csv/`.
   - Attach experiment metadata (temperature, discharge current, cutoff, group)
     using the `BATTERY_METADATA` mapping.

2. **Clean**
   - Drop duplicates and NaNs in core sensor fields.
   - Filter out physically invalid readings (e.g., negative voltage).

3. **Domain-Aware Normalization**
   - Define domains as:
     `(experiment_group, ambient_temp, discharge_current, cutoff_voltage)`.
   - Compute per-domain mean and standard deviation for:
     `voltage`, `current`, `temperature`.
   - Apply z-score normalization per domain.

4. **Feature Engineering**
   - Add:
     - `sample_index`: index within each (battery, cycle)
     - `time_norm`: normalized cycle progress in [0, 1]
     - `dV`, `dI`: first-order differences
     - `cycle_max_temp`, `cycle_min_voltage`: simple cycle aggregates

5. **Export**
   - Write a unified master CSV:
     `processed-data/master_datasets/discharge_master_normalized.csv`.

## Usage

From the project root:


python -m data_processing.preprocessing.build_master_dataset
    