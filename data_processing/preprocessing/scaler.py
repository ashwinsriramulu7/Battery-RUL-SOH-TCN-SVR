"""
Domain-aware normalization and scaling for discharge data.

Scaling is performed per domain defined by:
    (experiment_group, ambient_temp, discharge_current, cutoff_voltage)
so that models see condition-normalized features.
"""

from typing import Dict, Tuple

import pandas as pd


FEATURE_COLS = ["voltage", "current", "temperature"]
DOMAIN_COLS = ["experiment_group", "ambient_temp", "discharge_current", "cutoff_voltage"]


def compute_domain_stats(df: pd.DataFrame) -> Dict[Tuple, Dict[str, pd.Series]]:
    """
    Compute mean and standard deviation for each feature within each domain.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned discharge dataset.

    Returns
    -------
    dict
        Mapping: domain_key -> {"mu": Series, "sigma": Series}
    """
    domain_stats: Dict[Tuple, Dict[str, pd.Series]] = {}

    grouped = df.groupby(DOMAIN_COLS)

    for domain_key, group in grouped:
        mu = group[FEATURE_COLS].mean()
        sigma = group[FEATURE_COLS].std().replace(0.0, 1e-6)
        domain_stats[domain_key] = {"mu": mu, "sigma": sigma}

    return domain_stats


def apply_domain_normalization(df: pd.DataFrame, domain_stats: Dict[Tuple, Dict[str, pd.Series]]) -> pd.DataFrame:
    """
    Apply domain-specific z-score normalization using precomputed statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned discharge dataset.

    domain_stats : dict
        Output of compute_domain_stats.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature columns replaced by normalized values.
    """
    df_norm = df.copy()

    for domain_key, stats in domain_stats.items():
        mask = (df_norm[DOMAIN_COLS] == pd.Series(domain_key, index=DOMAIN_COLS)).all(axis=1)

        if not mask.any():
            continue

        mu = stats["mu"]
        sigma = stats["sigma"]

        df_norm.loc[mask, FEATURE_COLS] = (df_norm.loc[mask, FEATURE_COLS] - mu) / sigma

    return df_norm
