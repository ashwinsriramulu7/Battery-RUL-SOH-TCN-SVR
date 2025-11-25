# models/eval.py

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .config import RESULTS_DIR, SOH_SEGMENT_THRESHOLDS
from .train_svr import _load_best_tcn, _build_cycle_representations
from .datasets import RULSVRDataset
from . import plots


def eval_svr_and_plot() -> None:
    model = _load_best_tcn()
    X, soh, rul = _build_cycle_representations(model)

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(soh) & ~np.isnan(rul)
    X, soh, rul = X[mask], soh[mask], rul[mask]

    full_ds = RULSVRDataset(X, rul, soh)
    segments = full_ds.segment_by_soh(SOH_SEGMENT_THRESHOLDS)

    preds_all = np.zeros_like(rul)

    for seg_name, seg_ds in segments.items():
        if seg_ds.X.shape[0] == 0:
            continue

        model_path = RESULTS_DIR / f"svr_{seg_name}.joblib"
        if not model_path.exists():
            print(f"SVR model for segment '{seg_name}' not found at {model_path}, skipping.")
            continue

        svr = joblib.load(model_path)
        seg_pred = svr.predict(seg_ds.X)

        mask_seg = (full_ds.soh >= SOH_SEGMENT_THRESHOLDS[0]) if seg_name == "early" else \
                   ((full_ds.soh < SOH_SEGMENT_THRESHOLDS[0]) & (full_ds.soh >= SOH_SEGMENT_THRESHOLDS[1])) if seg_name == "mid" else \
                   (full_ds.soh < SOH_SEGMENT_THRESHOLDS[1])

        preds_all[mask_seg] = seg_pred

    rmse = float(np.sqrt(np.mean((preds_all - rul) ** 2)))
    print(f"Overall piecewise-SVR RMSE (cycles): {rmse:.3f}")

    plots.plot_rul_predictions(
        true_rul=rul,
        pred_rul=preds_all,
        save_path=RESULTS_DIR / "svr_rul_scatter.png",
    )


if __name__ == "__main__":
    eval_svr_and_plot()
