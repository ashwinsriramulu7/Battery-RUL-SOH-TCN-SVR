import numpy as np
import pandas as pd
import pwlf
from sklearn.svm import SVR
import json

DATA = "artifacts/results/cycle_level_predictions.csv"

def fit_piecewise_svr():
    df = pd.read_csv(DATA)

    all_info = {}

    for bid, sub in df.groupby("battery_id"):
        s = sub.sort_values("cycle_index")
        x = s["cycle_index"].values.astype(float)
        y = s["capacity_pred_smooth"].values.astype(float)

        plr = pwlf.PiecewiseLinFit(x, y)
        bk = plr.fit(3)

        segs = [(int(bk[i]), int(bk[i+1])) for i in range(len(bk)-1)]
        seg_results = []

        for xs, xe in segs:
            mask = (x >= xs) & (x <= xe)
            xs_seg = x[mask].reshape(-1,1)
            ys_seg = y[mask]

            svr = SVR(kernel="rbf", C=100, epsilon=0.003)
            svr.fit(xs_seg, ys_seg)

            sv_x = svr.support_vectors_.flatten()
            sv_y = ys_seg[svr.support_]
            m, b = np.polyfit(sv_x, sv_y, 1)

            seg_results.append({
                "x_start": xs, "x_end": xe,
                "slope": float(m),
                "intercept": float(b)
            })

        all_info[str(bid)] = {
            "breakpoints": list(map(float, bk)),
            "segments": seg_results
        }

    with open("artifacts/results/svr_piecewise_all.json", "w") as f:
        json.dump(all_info, f, indent=2)

    print("Saved:", "artifacts/results/svr_piecewise_all.json")


if __name__ == "__main__":
    fit_piecewise_svr()
