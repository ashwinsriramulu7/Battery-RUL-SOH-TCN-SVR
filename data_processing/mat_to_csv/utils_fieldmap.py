"""
Utility functions for safely retrieving fields from MATLAB structs.
"""


def safe_get(obj, name, default=None):
    """
    Return obj.name if it exists, else return default.
    """
    return getattr(obj, name, default)


def get_any(obj, name_list, default=None):
    """
    Return the first attribute in name_list found on obj.
    Useful because NASA experiments use inconsistent naming.
    """
    for n in name_list:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default
