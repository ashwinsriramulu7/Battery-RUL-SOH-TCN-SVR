"""
Preprocessing package for preparing NASA battery discharge data
for model training (TCN + SVR).

This package provides:
- Loading and merging discharge CSVs across all batteries
- Attaching experiment metadata (temperature, current, cutoff, group)
- Cleaning and sanity checking
- Domain-aware normalization and scaling
- Feature engineering (cycle index, normalized time, simple derivatives)
"""

from .build_master_dataset import build_master_discharge_dataset

__all__ = ["build_master_discharge_dataset"]
    