# models/config.py

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_DATA_PATH = PROJECT_ROOT / "processed-data" / "master_datasets" / "discharge_master_normalized_v3.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots"

# Create dirs lazily from scripts; we don't mkdir here.

# Input features to feed into TCN (normalized columns)
INPUT_FEATURES = [
    "voltage_norm",
    "current_norm",
    "temperature_norm",
    # add/remove features here as needed
]

# Target column for TCN (per time-step SOH)
TARGET_SOH_COL = "soh"          # 0–1, relative capacity
TARGET_RUL_COL = "rul_cycles"   # RUL in cycles, used for SVR

# Sequence/window config
WINDOW_LENGTH = 256       # number of time steps per sequence window
WINDOW_STRIDE = 64        # stride when sliding windows over a cycle
MIN_CYCLE_LENGTH = 300    # drop super-short cycles for modeling

# TCN architecture
TCN_NUM_CHANNELS = [64, 64, 64]  # hidden channels per layer
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1

# Training hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Train/val/test split
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_SEED = 42

# SVR configuration
SVR_KERNEL = "rbf"
SVR_C = 10.0
SVR_GAMMA = "scale"
SVR_EPSILON = 0.1

# Piecewise SVR segmentation thresholds in terms of SOH (or life fraction)
# For example: SOH >= 0.9 → early, 0.7–0.9 → mid, < 0.7 → late
SOH_SEGMENT_THRESHOLDS = (0.9, 0.7)

# Rated / EOL capacity (NASA dataset: 2 Ah, EOL ~ 70%)
RATED_CAPACITY_AH = 2.0
EOL_CAPACITY_AH = 1.4