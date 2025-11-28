import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


# ------------------------------------------------------------
# Plot Kaplan–Meier curves for tertiles of risk
# ------------------------------------------------------------
def plot_km_tertiles(df, out_path):
    """
    df must contain:
        time_to_event
        event_occurred
        risk_score
    """

    # --------------------------------------------------------
    # Assign tertiles
    # --------------------------------------------------------
    df = df.copy()
    df["risk_group"] = pd.qcut(df["risk_score"], q=3, labels=["Low", "Medium", "High"])

    kmf = KaplanMeierFitter()

    plt.figure(figsize=(10, 7))
    colors = {"Low": "green", "Medium": "orange", "High": "red"}

    # --------------------------------------------------------
    # Plot each group
    # --------------------------------------------------------
    for group in ["Low", "Medium", "High"]:
        subset = df[df["risk_group"] == group]
        kmf.fit(
            durations=subset["time_to_event"],
            event_observed=subset["event_occurred"],
            label=f"{group} Risk (n={len(subset)})",
        )
        kmf.plot(ci_show=True, linewidth=2, color=colors[group])

    plt.title("Kaplan–Meier Curves by Risk Tertile (Survival_v3)", fontsize=15)
    plt.xlabel("Time to ESRD (years)", fontsize=13)
    plt.ylabel("Survival Probability", fontsize=13)
    plt.grid(alpha=0.3)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> KM plot saved: {out_path}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="survival_v3/experiments/single_run_5fold/eval_v3/all_folds_risk_scores_v3.csv",
        help="Path to combined fold risk score CSV",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="survival_v3/experiments/single_run_5fold/eval_v3/km_tertiles_v3.png",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Risk score CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    if not all(col in df.columns for col in ["time_to_event", "event_occurred", "risk_score"]):
        raise ValueError("Missing required columns for KM plotting.")

    plot_km_tertiles(df, args.out_path)


if __name__ == "__main__":
    main()
