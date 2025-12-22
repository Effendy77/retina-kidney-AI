import pandas as pd
from pathlib import Path

BASE_DIR = Path("experiments/ablation")

experiments = [
    "image_tabular",                 # baseline
    "image_tabular_no_qrisk",
    "image_tabularONLY_no_qrisk",
    "image",             # if exists
    "tabular",
    "mask", 
    "tabular_no_retinal", # if exists
]

rows = []

for exp in experiments:
    csv_path = BASE_DIR / exp / "cv_summary.csv"
    if not csv_path.exists():
        print(f"[WARN] Missing: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    row = {
        "experiment": exp,
        "rmse_mean": df["rmse"].mean(),
        "rmse_sd":   df["rmse"].std(),
        "mae_mean":  df["mae"].mean(),
        "mae_sd":    df["mae"].std(),
        "r2_mean":   df["r2"].mean(),
        "r2_sd":     df["r2"].std(),
    }
    rows.append(row)

summary = pd.DataFrame(rows)
summary = summary.sort_values("r2_mean", ascending=False)

summary.to_csv("experiments/ablation/ALL_EXPERIMENTS_SUMMARY.csv", index=False)
print(summary.round(4))
