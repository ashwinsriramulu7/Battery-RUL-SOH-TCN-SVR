# MATLAB to CSV Conversion Module

This directory contains the research-grade preprocessing pipeline used to convert
the NASA Ames Battery Aging `.mat` files into standardized, analysis-ready CSV files.

## Features

- Compatible with all six NASA battery aging experiments
- Automatically extracts charge, discharge, and impedance cycles
- Handles inconsistent MATLAB struct fields across experiments
- Truncates misaligned time-series fields to a consistent length
- Converts MATLAB `datevec` timestamps into ISO-8601 formatted strings
- Writes CSV files to a reproducible directory structure
- Fully modular and unit-testable

## Output Structure

processed-data/
charge_csv/
discharge_csv/
impedance_csv/

Each CSV file is named:

- `BXXXX_charge.csv`
- `BXXXX_discharge.csv`
- `BXXXX_impedance.csv`

## Usage

python -m data_processing.mat_to_csv.runner_convert_all

By default, this scans the `raw-data/` directory and writes outputs to `processed-data/`.

## Notes

This module is intentionally minimal, deterministic, and free of external dependencies
beyond NumPy, Pandas, and SciPy. It is suitable for long-term research reproducibility.
