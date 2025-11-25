"""
MATLAB-to-CSV conversion package for the NASA battery aging dataset.

This package exposes two main entry points:

- parse_single_mat(mat_path): parse a single .mat file into charge/discharge/impedance DataFrames
- convert_all(root_dir, output_base="processed-data"): batch-convert all .mat files under root_dir
"""

from .parser import parse_single_mat
from .runner_convert_all import convert_all

__all__ = ["parse_single_mat", "convert_all"]
