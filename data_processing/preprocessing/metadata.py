"""
Experiment metadata for the NASA battery aging dataset.

This module encodes experiment-level conditions for each battery:
- experiment group (1..6)
- ambient temperature (deg C)
- nominal discharge current (A)
- discharge cutoff voltage (V)
"""

BATTERY_METADATA = {
    # Group 1 — FY08Q4 — 24°C — CC 2A
    "0005": {"exp_group": 1, "ambient": 24, "current": 2, "cutoff": 2.7},
    "0006": {"exp_group": 1, "ambient": 24, "current": 2, "cutoff": 2.5},
    "0007": {"exp_group": 1, "ambient": 24, "current": 2, "cutoff": 2.2},
    "0018": {"exp_group": 1, "ambient": 24, "current": 2, "cutoff": 2.5},

    # Group 2 — 25-28 — square-wave 4A — 24°C
    "0025": {"exp_group": 2, "ambient": 24, "current": 4, "cutoff": 2.0},
    "0026": {"exp_group": 2, "ambient": 24, "current": 4, "cutoff": 2.2},
    "0027": {"exp_group": 2, "ambient": 24, "current": 4, "cutoff": 2.5},
    "0028": {"exp_group": 2, "ambient": 24, "current": 4, "cutoff": 2.7},

    # Group 3a — 29-32 — 43°C — CC 4A
    "0029": {"exp_group": 3, "ambient": 43, "current": 4, "cutoff": 2.0},
    "0030": {"exp_group": 3, "ambient": 43, "current": 4, "cutoff": 2.2},
    "0031": {"exp_group": 3, "ambient": 43, "current": 4, "cutoff": 2.5},
    "0032": {"exp_group": 3, "ambient": 43, "current": 4, "cutoff": 2.7},

    # Group 3b — 33-34-36 — 24°C — CC 4A and CC 2A (B0036)
    "0033": {"exp_group": 3, "ambient": 24, "current": 4, "cutoff": 2.0},
    "0034": {"exp_group": 3, "ambient": 24, "current": 4, "cutoff": 2.2},
    "0036": {"exp_group": 3, "ambient": 24, "current": 2, "cutoff": 2.7},

    # Group 3c — 38-40 — 24°C + 44°C — mixed currents
    "0038": {"exp_group": 3, "ambient": 24, "current": 1, "cutoff": 2.2},
    "0039": {"exp_group": 3, "ambient": 44, "current": 2, "cutoff": 2.5},
    "0040": {"exp_group": 3, "ambient": 24, "current": 4, "cutoff": 2.7},

    # Group 4 — 41-44 — 4°C — CC 1A / CC 4A
    "0041": {"exp_group": 4, "ambient": 4, "current": 4, "cutoff": 2.0},
    "0042": {"exp_group": 4, "ambient": 4, "current": 1, "cutoff": 2.2},
    "0043": {"exp_group": 4, "ambient": 4, "current": 1, "cutoff": 2.5},
    "0044": {"exp_group": 4, "ambient": 4, "current": 4, "cutoff": 2.7},

    # Group 5 — 49-52 — 4°C — CC 2A — noisy/crashed experiments
    "0049": {"exp_group": 5, "ambient": 4, "current": 2, "cutoff": 2.0},
    "0050": {"exp_group": 5, "ambient": 4, "current": 2, "cutoff": 2.2},
    "0051": {"exp_group": 5, "ambient": 4, "current": 2, "cutoff": 2.5},
    "0052": {"exp_group": 5, "ambient": 4, "current": 2, "cutoff": 2.7},

    # Group 6 — 53-56 — 4°C — CC 2A
    "0053": {"exp_group": 6, "ambient": 4, "current": 2, "cutoff": 2.0},
    "0054": {"exp_group": 6, "ambient": 4, "current": 2, "cutoff": 2.2},
    "0055": {"exp_group": 6, "ambient": 4, "current": 2, "cutoff": 2.5},
    "0056": {"exp_group": 6, "ambient": 4, "current": 2, "cutoff": 2.7},
}
