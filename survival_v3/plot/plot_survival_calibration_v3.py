import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Compute observed event rate at time horizon T
# ------------------------------------------------------------
def observed_event_rate_at_time(df, T):
    """
    Observed probability of ESRD by time T.
    event_occurred == 1 and time_to_event <= T
    """
    return np.mean((df["event_occurred"] == 1) & (df["time_to_event"] <= T))


# ------------------------------------------------------------
# MAIN calibration function
# ------------------------------------------------------------
def calibration_curve_survival(df, out_plot, out_csv, T=5, n_bins=10):
    """
    df must contain:
        risk_score
        time_to_event
        event_occurred

    T     = time horizon (years)
    bins  = number of calibration bins
    """

    df = df.copy()

    # Normalize risk to 0–1 (surrogate predicted probability)
    # This is common practice for survival calibration.
    preds = df["risk_score"].values
    preds_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
    df["pred_norm"] = preds_norm

    # Create calibration bins
    df["bin"] = pd.qcut(df["pred_norm"], q=n_bins, labels=False)

    rows = []
    for b in range(n_bins):
        group = df[df["bin"] == b]
        if len(group) == 0:
            continue

        pred_mean = group["pred_norm"].mean()
        obs_rate = observed_event_rate_at_time(group, T)

        rows.append([b, pred_mean, obs_rate, len(group)])

    cal_df = pd.DataFrame(
        rows,
        columns=["bin", "mean_predicted", "observed_event_rate", "n"],
    )

    # Save CSV
    cal_df.to_csv(out_csv, index=False)
    print(f">>> Calibration bins saved: {out_csv}")

    # Plot calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(cal_df["mean_predicted"], cal_df["observed_event_rate"], "o-", label="Observed vs Predicted")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    plt.title(f"Survival Calibration at {T} Years (v3)", fontsize=14)
    plt.xlabel("Predicted Risk (normalized 0–1)", fontsize=12)
    plt.ylabel("Observed ESRD Probability", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Calibration plot saved: {out_plot}")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="survival_v3/experiments/single_run_5fold/eval_v3/all_folds_risk_scores_v3.csv",
    )
    parser.add_argument(
        "--out_plot",
        type=str,
        default="survival_v3/experiments/single_run_5fold/eval_v3/calibration_curve_v3.png",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="survival_v3/experiments/single_run_5fold/eval_v3/calibration_bins_v3.csv",
    )
    parser.add_argument(
        "--time_horizon",
        type=float,
        default=5.0,
        help="Time horizon (years) for calibration curve",
    )
    parser.add_argument("--bins", type=int, default=10)

    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    required = ["risk_score", "time_to_event", "event_occurred"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    calibration_curve_survival(
        df=df,
        out_plot=args.out_plot,
        out_csv=args.out_csv,
        T=args.time_horizon,
        n_bins=args.bins,
    )


if __name__ == "__main__":
    main()
