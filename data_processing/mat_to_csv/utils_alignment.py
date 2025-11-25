"""
Functions for aligning MATLAB arrays and normalizing their shapes
into clean, 1-dimensional numeric vectors suitable for DataFrames.
"""

import numpy as np


def sanitize_element(x):
    """
    Ensure that a single element inside a vector is scalar.
    If x is an array, list, or nested container, extract the first value.

    Examples that become valid scalars:
        array([3.95])      -> 3.95
        array([[4.2]])     -> 4.2
        [5.1]              -> 5.1
        [[2.7]]            -> 2.7
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.array(x).flatten()
        if arr.size > 0:
            return arr.item()   # return first scalar
        else:
            return np.nan
    return x


def sanitize_vector(v):
    """
    Convert MATLAB vectors (possibly nested, object-dtype, or ragged)
    into clean 1-D float numpy arrays.

    Handles:
        - object arrays
        - arrays of lists
        - arrays of arrays
        - nested matrices
        - cell arrays
    """
    if v is None:
        return None

    arr = np.array(v, dtype=object)

    # Flatten if 2D with single dimension
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)

    # Fully flatten if more than 2D
    if arr.ndim > 2:
        arr = arr.flatten()

    # Sanitize each element
    clean = [sanitize_element(el) for el in arr]

    return np.array(clean)


def truncate_fields_to_min_length(fields):
    """
    Normalize all fields to 1-D sanitized vectors,
    then truncate to the minimum length across all fields.
    """

    # Step 1: sanitize all vectors
    clean = {}
    for k, v in fields.items():
        clean[k] = sanitize_vector(v)

    # Step 2: determine minimum usable length
    lengths = []
    for val in clean.values():
        if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
            try:
                lengths.append(len(val))
            except Exception:
                pass

    if not lengths:
        return clean

    L = min(lengths)

    # Step 3: truncate everything
    for k, v in clean.items():
        if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
            try:
                clean[k] = v[:L]
            except Exception:
                pass

    return clean
