"""
Experiment metadata for the NASA Battery Aging Dataset.

This module provides experiment-level metadata for every battery in the
dataset (B0005–B0056). These metadata items represent *nominal* test
conditions extracted from the official NASA documentation and the
accompanying READMEs in each subfolder.

The metadata are used during preprocessing for:
- augmenting each record with experiment conditions,
- supporting group-wise domain normalization,
- enabling statistical analysis of degradation under varied conditions.

Fields:
    exp_group : int
        Experiment group index (1–6).
    ambient : float
        Nominal ambient temperature in °C.
    current : float or None
        Nominal discharge current in A, if defined.
        Some experiments used variable / fixed-load profiles → None.
    cutoff : float
        Discharge cutoff voltage in V.
    notes : str
        Any additional context useful for documentation or domain-specific logic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryMeta:
    exp_group: int
    ambient: float
    current: float | None
    cutoff: float
    notes: str = ""


BATTERY_METADATA: dict[str, BatteryMeta] = {

    # ---------------------------------------------------------
    # Group 1 — FY08Q4 — 24°C — CC 2A
    # ---------------------------------------------------------
    "0005": BatteryMeta(1, 24, 2, 2.7),
    "0006": BatteryMeta(1, 24, 2, 2.5),
    "0007": BatteryMeta(1, 24, 2, 2.2),
    "0018": BatteryMeta(1, 24, 2, 2.5),

    # ---------------------------------------------------------
    # Group 2 — Batteries 25–28 — 24°C — 0.05 Hz square-wave 4A
    # ---------------------------------------------------------
    "0025": BatteryMeta(2, 24, 4, 2.0, notes="0.05 Hz square-wave load"),
    "0026": BatteryMeta(2, 24, 4, 2.2, notes="0.05 Hz square-wave load"),
    "0027": BatteryMeta(2, 24, 4, 2.5, notes="0.05 Hz square-wave load"),
    "0028": BatteryMeta(2, 24, 4, 2.7, notes="0.05 Hz square-wave load"),

    # ---------------------------------------------------------
    # Group 3a — Batteries 29–32 — High temperature 43°C — CC 4A
    # ---------------------------------------------------------
    "0029": BatteryMeta(3, 43, 4, 2.0),
    "0030": BatteryMeta(3, 43, 4, 2.2),
    "0031": BatteryMeta(3, 43, 4, 2.5),
    "0032": BatteryMeta(3, 43, 4, 2.7),

    # ---------------------------------------------------------
    # Group 3b — Batteries 33, 34, 36 — 24°C — CC 4A except B0036=2A
    # ---------------------------------------------------------
    "0033": BatteryMeta(3, 24, 4, 2.0),
    "0034": BatteryMeta(3, 24, 4, 2.2),
    "0036": BatteryMeta(3, 24, 2, 2.7, notes="Lower discharge current (2 A)"),

    # ---------------------------------------------------------
    # Group 3c — Batteries 38, 39, 40 — mixed-temperature experiments
    # ---------------------------------------------------------
    "0038": BatteryMeta(3, 24, 1, 2.2, notes="Low-current experiment"),
    "0039": BatteryMeta(3, 44, 2, 2.5, notes="High-temperature experiment"),
    "0040": BatteryMeta(3, 24, 4, 2.7),

    # ---------------------------------------------------------
    # Group 4 — Batteries 41–44 — 4°C — CC 1A / CC 4A
    # ---------------------------------------------------------
    "0041": BatteryMeta(4, 4, 4, 2.0),
    "0042": BatteryMeta(4, 4, 1, 2.2),
    "0043": BatteryMeta(4, 4, 1, 2.5),
    "0044": BatteryMeta(4, 4, 4, 2.7),

    # ---------------------------------------------------------
    # Group 5 — Batteries 49–52 — 4°C — CC 2A — noisy/irregular experiments
    # ---------------------------------------------------------
    "0049": BatteryMeta(5, 4, 2, 2.0, notes="Noisy/crashed experiment"),
    "0050": BatteryMeta(5, 4, 2, 2.2, notes="Noisy/crashed experiment"),
    "0051": BatteryMeta(5, 4, 2, 2.5, notes="Noisy/crashed experiment"),
    "0052": BatteryMeta(5, 4, 2, 2.7, notes="Noisy/crashed experiment"),

    # ---------------------------------------------------------
    # Group 6 — Batteries 53–56 — 4°C — CC 2A
    # ---------------------------------------------------------
    "0053": BatteryMeta(6, 4, 2, 2.0),
    "0054": BatteryMeta(6, 4, 2, 2.2),
    "0055": BatteryMeta(6, 4, 2, 2.5),
    "0056": BatteryMeta(6, 4, 2, 2.7),
}
