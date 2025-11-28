import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# scikit-survival imports
from sksurv.metrics import (
    cumulative_dynamic_auc,
)
from sksurv.util import Surv


# ------------------------------------------------------------
# Time-dependent ROC Utility
# ------------------------------------------------------------

def compute_tdROC(df, out_csv, out_plot, time_grid=None):
    """
    Computes time-dependent AUC using cumulative_dynamic_auc.

    df must contain:
        time_to_event
        event_occurred
        risk_score

    time_grid:
        list or numpy array of time points (years) at which to compute AUC.
    """

    # Prepare survival data for scikit-survival
    y = Surv.from_arrays(
        event=df["event_occurred"].astype(bool).values,
        time=df["time_to_event"].values
    )

    # Standardize risk scores
    risk = df["risk_score"].values.reshape(-1, 1)
    risk = StandardScaler().fit_transform(risk).reshape(-1)

    times = df["time_to_event"].values

    # Default time grid: 1st to 95th percentile
    if time_grid is None:
        t_min = max(1, int(np.percentile(times, 5)))
        t_max = int(np.percentile(times, 95))
        time_grid = np.arange(t_min, t_max + 1)

    # Compute cumulative dynamic AUC
    aucs, mean_auc = cumulative_dynamic_auc(
        y,
        y,
        risk,
        time_grid,
    )

    # Save CSV
    out_df = pd.DataFrame({
        "time": time_grid,
        "auc": aucs,
    })
    out_df.to_csv(out_csv, index=False)
    print(f">>> Time-dependent AUC saved: {out_csv}")

    # Plot AUC vs Time
    plt.figure(figsize=(9, 6))
    plt.plot(time_grid, aucs, label="AUC(t)", linewidth=2)
    plt.title("Time-Dependent AUC (Survival_v3)", fontsize=14)
    plt.xlabel("Time (years)", fontsize=12)
    plt.ylabel("AUC(t)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.ylim(0.5, 1.0)
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Time-dependent ROC curve saved: {out_plot}")


# ------------------------------------------------------------
# CLI ENTRY
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="survival_v3/experiments/single_run_5fold/eval_v3/all_folds_risk_scores_v3.csv",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="survival_v3/experiments/single_run_5fold/eval_v3/tdROC_auc_values_v3.csv",
    )
    parser.add_argument(
        "--out_plot",
        type=str,
        default="survival_v3/experiments/single_run_5fold/eval_v3/tdROC_v3.png",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Risk score CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    required = ["time_to_event", "event_occurred", "risk_score"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'")

    compute_tdROC(
        df=df,
        out_csv=args.out_csv,
        out_plot=args.out_plot,
    )


if __name__ == "__main__":
    main()
