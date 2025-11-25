# models/datasets.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .config import (
    MASTER_DATA_PATH,
    INPUT_FEATURES,
    TARGET_SOH_COL,
    TARGET_RUL_COL,
    WINDOW_LENGTH,
    WINDOW_STRIDE,
    MIN_CYCLE_LENGTH,
    VAL_FRACTION,
    TEST_FRACTION,
    BATCH_SIZE,
    RANDOM_SEED,
)


@dataclass
class SequenceIndex:
    """Identify a particular window inside a (battery, cycle) time series."""
    battery_id: str
    cycle_index: int
    start_idx: int
    end_idx: int


class DischargeSequenceDataset(Dataset):
    """
    Loads the master discharge dataset and returns sliding windows of
    time-series features + corresponding SOH targets for TCN training.

    Each item:
        X: [window_len, num_features] float32
        y: [window_len] float32  (SOH trajectory)
        meta: dict with battery_id, cycle_index, start_idx, end_idx
    """

    def __init__(
        self,
        csv_path: Path | str = MASTER_DATA_PATH,
        window_length: int = WINDOW_LENGTH,
        window_stride: int = WINDOW_STRIDE,
        min_cycle_length: int = MIN_CYCLE_LENGTH,
        input_features: List[str] | None = None,
        target_col: str = TARGET_SOH_COL,
    ) -> None:
        super().__init__()

        self.csv_path = Path(csv_path)
        self.window_length = window_length
        self.window_stride = window_stride
        self.min_cycle_length = min_cycle_length
        self.input_features = input_features or INPUT_FEATURES
        self.target_col = target_col

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Master dataset not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        required_cols = {"battery_id", "cycle_index"} | set(self.input_features) | {self.target_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in master dataset: {missing}")

        # Sort for reproducibility
        df = df.sort_values(["battery_id", "cycle_index", "time_s"]).reset_index(drop=True)

        self.df = df
        self.sequence_index: List[SequenceIndex] = []
        self._build_index()

    def _build_index(self) -> None:
        grouped = self.df.groupby(["battery_id", "cycle_index"], sort=True)

        for (bid, cid), g in grouped:
            n = len(g)
            if n < self.min_cycle_length:
                continue

            start = 0
            while start + self.window_length <= n:
                end = start + self.window_length
                self.sequence_index.append(
                    SequenceIndex(
                        battery_id=str(bid),
                        cycle_index=int(cid),
                        start_idx=int(g.index[start]),
                        end_idx=int(g.index[end - 1]),
                    )
                )
                start += self.window_stride

    def __len__(self) -> int:
        return len(self.sequence_index)

    def __getitem__(self, idx: int):
        si = self.sequence_index[idx]

        # Slice original df between those index positions
        mask = (self.df.index >= si.start_idx) & (self.df.index <= si.end_idx)
        g = self.df.loc[mask]

        X = g[self.input_features].to_numpy(dtype=np.float32)
        y = g[self.target_col].to_numpy(dtype=np.float32)

        # Ensure shapes
        if X.shape[0] != self.window_length:
            # This should not happen given how we build the index, but be defensive
            raise RuntimeError(
                f"Window length mismatch for {si.battery_id} cycle {si.cycle_index}: "
                f"expected {self.window_length}, got {X.shape[0]}"
            )

        meta = {
            "battery_id": si.battery_id,
            "cycle_index": si.cycle_index,
            "start_idx": si.start_idx,
            "end_idx": si.end_idx,
        }

        return torch.from_numpy(X), torch.from_numpy(y), meta


def make_train_val_test_loaders(
    dataset: DischargeSequenceDataset,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split dataset into train/val/test and return DataLoaders.
    Splits are random but reproducible via RANDOM_SEED.
    """
    total_len = len(dataset)
    val_len = int(total_len * VAL_FRACTION)
    test_len = int(total_len * TEST_FRACTION)
    train_len = total_len - val_len - test_len

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    def _loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)

    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)


# -------------------------------------------------------------------------
# SVR dataset: one feature vector per (battery, cycle), built from TCN outputs
# -------------------------------------------------------------------------

class RULSVRDataset:
    """
    Lightweight, in-memory dataset for SVR:
    - X: representations from trained TCN (e.g., last hidden state per window/cycle)
    - y: scalar RUL targets
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, soh: np.ndarray) -> None:
        assert features.shape[0] == targets.shape[0] == soh.shape[0]
        self.X = features.astype(np.float32)
        self.y = targets.astype(np.float32)
        self.soh = soh.astype(np.float32)  # used to segment into early/mid/late

    def segment_by_soh(self, thresholds) -> Dict[str, "RULSVRDataset"]:
        hi_thr, mid_thr = thresholds  # e.g. (0.9, 0.7)

        soh = self.soh
        X, y = self.X, self.y

        early_mask = soh >= hi_thr
        mid_mask = (soh < hi_thr) & (soh >= mid_thr)
        late_mask = soh < mid_thr

        def sub(mask):
            return RULSVRDataset(X[mask], y[mask], soh[mask])

        return {
            "early": sub(early_mask),
            "mid": sub(mid_mask),
            "late": sub(late_mask),
        }
