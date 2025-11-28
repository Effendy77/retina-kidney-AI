import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter


# ------------------------------------------------------------
# Build tertiles for HR computation
# ------------------------------------------------------------
def add_risk_tertiles(df):
    df = df.copy()
    df["risk_group"] = pd.qcut(
        df["risk_score"], q=3, labels=["Low", "Medium", "High"]
    )
    return df


# ------------------------------------------------------------
# Fit Cox model using lifelines
# ------------------------------------------------------------
def fit_cox(df, formula, duration_col="time_to_event", event_col="event_occurred"):
    """
    Fits Cox regression and returns:
        HR, CI, p-value
    """
    cph = CoxPHFitter()
    cph.fit(df, duration_col=duration_col, event_col=event_col, formula=formula)
    return cph


# ------------------------------------------------------------
# Generate forest plot
# ------------------------------------------------------------
def plot_forest(hr_df, out_path):
    plt.figure(figsize=(8, 5))

    y = np.arange(len(hr_df))
    hr = hr_df["HR"].values
    low = hr_df["CI_lower"].values
    high = hr_df["CI_upper"].values
    labels = hr_df["Variable"].values

    plt.errorbar(
        hr, y, xerr=[hr - low, high - hr],
        fmt="o", color="darkblue", ecolor="gray", capsize=4
    )

    plt.axvline(1.0, color="red", linestyle="--", linewidth=1)

    plt.yticks(y, labels)
    plt.xlabel("Hazard Ratio (HR)", fontsize=12)
    plt.title("Hazard Ratios (Survival_v3)", fontsize=14)
    plt.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f">>> Forest plot saved: {out_path}")


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
        "--out_csv",
        type=str,
        default="survival_v3/experiments/single_run_5fold/eval_v3/hazard_ratios_v3.csv",
    )
    parser.add_argument(
        "--out_fig",
        type=str,
        default="survival_v3/experiments/single_run_5fold/eval_v3/hazard_ratio_forest_plot_v3.png",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Risk scores CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    required = ["risk_score", "time_to_event", "event_occurred"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'")

    # ============================================================
    # 1. HR for continuous risk score
    # ============================================================
    df_cont = df.copy()
    cont_model = fit_cox(df_cont, formula="risk_score")

    cont_summary = cont_model.summary.loc["risk_score"]

    hr_cont = cont_summary["exp(coef)"]
    ci95_low = cont_summary["exp(coef) lower 95%"]
    ci95_high = cont_summary["exp(coef) upper 95%"]
    pval = cont_summary["p"]

    # ============================================================
    # 2. HR for tertiles (Medium vs Low, High vs Low)
    # ============================================================
    df_t = add_risk_tertiles(df)
    df_t = pd.get_dummies(df_t, columns=["risk_group"], drop_first=True)

    # This creates:
    #   risk_group_Medium (1 = Medium, 0 = Low)
    #   risk_group_High   (1 = High,   0 = Low)

    tertile_formula = "risk_group_Medium + risk_group_High"
    tertile_model = fit_cox(df_t, formula=tertile_formula)

    ter_summary = tertile_model.summary

    hr_med = ter_summary.loc["risk_group_Medium", "exp(coef)"]
    ci_med_low = ter_summary.loc["risk_group_Medium", "exp(coef) lower 95%"]
    ci_med_high = ter_summary.loc["risk_group_Medium", "exp(coef) upper 95%"]
    p_med = ter_summary.loc["risk_group_Medium", "p"]

    hr_high = ter_summary.loc["risk_group_High", "exp(coef)"]
    ci_high_low = ter_summary.loc["risk_group_High", "exp(coef) lower 95%"]
    ci_high_high = ter_summary.loc["risk_group_High", "exp(coef) upper 95%"]
    p_high = ter_summary.loc["risk_group_High", "p"]

    # ============================================================
    # COMPILE ALL RESULTS
    # ============================================================
    out_df = pd.DataFrame([
        ["Risk Score (continuous)", hr_cont, ci95_low, ci95_high, pval],
        ["Medium vs Low risk", hr_med, ci_med_low, ci_med_high, p_med],
        ["High vs Low risk", hr_high, ci_high_low, ci_high_high, p_high],
    ], columns=["Variable", "HR", "CI_lower", "CI_upper", "p_value"])

    out_df.to_csv(args.out_csv, index=False)
    print(f">>> Hazard ratios saved: {args.out_csv}")

    # ============================================================
    # FOREST PLOT
    # ============================================================
    plot_forest(out_df, args.out_fig)


if __name__ == "__main__":
    main()
