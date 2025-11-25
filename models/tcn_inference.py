import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from .tcn import BatteryTCN
from .config import Config

MASTER_PATH = "processed-data/master_datasets/discharge_master_normalized_v3.csv"
CHECKPOINT_PATH = "artifacts/checkpoints/tcn_best.pth"

class FullDatasetTCNInference(Dataset):
    """
    Uses the SAME feature normalization as training dataset.py
    Assumes dataset columns:
        voltage_norm, current_norm, temperature_norm
        soh (target)
        cycle_index, battery_id
    """

    def __init__(self, df, seq_len=Config.SEQ_LEN):
        self.seq_len = seq_len
        self.df = df

        # group by battery + cycle
        self.groups = df.groupby(["battery_id", "cycle_index"])

        self.keys = list(self.groups.groups.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        bid, cycle = self.keys[idx]
        sub = self.groups.get_group((bid, cycle))

        x = sub[["voltage_norm", "current_norm", "temperature_norm"]].values

        # pad or trim
        if len(x) >= self.seq_len:
            x = x[:self.seq_len]
        else:
            pad = np.zeros((self.seq_len - len(x), 3))
            x = np.vstack([x, pad])

        x = torch.tensor(x, dtype=torch.float32).T  # (3, seq)
        return x, bid, cycle


def run_tcn_inference():
    print("Loading master dataset...")
    df = pd.read_csv(MASTER_PATH)

    ds = FullDatasetTCNInference(df)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BatteryTCN().to(device)

    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds = []
    keys = []

    print("Running inference over ALL cycles...")
    with torch.no_grad():
        for x, bid, cycle in loader:
            x = x.to(device)
            out = model(x).cpu().numpy()
            preds.extend(out)
            keys.extend(list(zip(bid.numpy(), cycle.numpy())))

    out_df = pd.DataFrame({
        "battery_id": [k[0] for k in keys],
        "cycle_index": [k[1] for k in keys],
        "soh_pred": preds
    })

    out_df.to_csv("artifacts/results/tcn_cycle_predictions.csv", index=False)
    print("Saved:", "artifacts/results/tcn_cycle_predictions.csv")


if __name__ == "__main__":
    run_tcn_inference()
