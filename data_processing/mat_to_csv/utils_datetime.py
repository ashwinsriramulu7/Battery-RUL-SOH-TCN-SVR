"""
Utility functions for handling MATLAB datetime formats.
"""

from datetime import datetime


def matlab_datevec_to_datetime(vec):
    """
    Convert a MATLAB datevec [Y M D h m s] into a Python datetime.
    Fractional seconds, if present, are truncated.

    Parameters
    ----------
    vec : array-like
        MATLAB date vector containing 6 entries.

    Returns
    -------
    datetime or None
    """
    if vec is None:
        return None

    try:
        y, m, d, H, M, S = vec
        return datetime(int(y), int(m), int(d), int(H), int(M), int(S))
    except Exception:
        return None
