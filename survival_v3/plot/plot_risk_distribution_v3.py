import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# Plot KDE risk distribution
# ------------------------------------------------------------
def plot_kde(df, out_path):
    plt.figure(figsize=(9, 6))
    sns.kdeplot(df["risk_score"], fill=True, linewidth=2, color="royalblue")
    plt.title("Predicted Risk Distribution (Survival_v3)", fontsize=14)
    plt.xlabel("Predicted Risk Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">>> KDE plot saved: {out_path}")


# ------------------------------------------------------------
# Plot histogram: event vs non-event
# ------------------------------------------------------------
def plot_hist_event_split(df, out_path):
    plt.figure(figsize=(9, 6))

    events = df[df["event_occurred"] == 1]["risk_score"]
    censored = df[df["event_occurred"] == 0]["risk_score"]

    plt.hist(events, bins=30, alpha=0.6, label="ESRD Event", color="red")
    plt.hist(censored, bins=30, alpha=0.6, label="Censored", color="green")

    plt.title("Risk Score Histogram: Event vs Censored (v3)", fontsize=14)
    plt.xlabel("Predicted Risk Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Histogram saved: {out_path}")


# ------------------------------------------------------------
# Boxplot (risk by event status)
# ------------------------------------------------------------
def plot_box(df, out_path):
    plt.figure(figsize=(7, 6))
    sns.boxplot(
        data=df,
        x="event_occurred",
        y="risk_score",
        palette=["green", "red"],
    )
    plt.xticks([0,1], ["Censored", "Event"])
    plt.title("Risk Scores by Event Status (v3)", fontsize=14)
    plt.xlabel("Status", fontsize=12)
    plt.ylabel("Predicted Risk Score", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">>> Boxplot saved: {out_path}")


# ------------------------------------------------------------
# Violin plot (optional, publication-ready)
# ------------------------------------------------------------
def plot_violin(df, out_path):
    plt.figure(figsize=(7, 6))
    sns.violinplot(
        data=df,
        x="event_occurred",
        y="risk_score",
        palette=["green", "red"],
        inner="quartile",
    )
    plt.xticks([0,1], ["Censored", "Event"])
    plt.title("Risk Score Distribution (v3 Violin)", fontsize=14)
    plt.xlabel("Status", fontsize=12)
    plt.ylabel("Predicted Risk Score", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">>> Violin plot saved: {out_path}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="survival_v3/experiments/single_run_5fold/eval_v3/all_folds_risk_scores_v3.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="survival_v3/experiments/single_run_5fold/eval_v3/",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Risk score CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    required = ["risk_score", "event_occurred"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Generate all risk distribution plots
    plot_kde(df, os.path.join(args.out_dir, "risk_kde_v3.png"))
    plot_hist_event_split(df, os.path.join(args.out_dir, "risk_hist_event_v3.png"))
    plot_box(df, os.path.join(args.out_dir, "risk_boxplot_v3.png"))
    plot_violin(df, os.path.join(args.out_dir, "risk_violin_v3.png"))


if __name__ == "__main__":
    main()
