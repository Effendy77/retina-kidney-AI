import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter


def compute_hazard_ratios(root_dir="survival_v2/checkpoints_single_v2_5fold"):
    """
    Computes hazard ratios between pooled risk tertiles:
        - High vs Low
        - Medium vs Low
        - High vs Medium

    Produces CSV and a forest plot.
    """

    all_folds = []

    # ---------------------------------------------------
    # Load all folds
    # ---------------------------------------------------
    for fold in range(1, 6):
        risk_path = os.path.join(root_dir, f"fold{fold}_risk_scores.csv")
        if not os.path.exists(risk_path):
            print(f"[WARNING] Missing: {risk_path}")
            continue

        df = pd.read_csv(risk_path)
        df["fold"] = fold
        all_folds.append(df)

    if len(all_folds) == 0:
        print("[ERROR] No fold data found.")
        return

    pooled = pd.concat(all_folds, ignore_index=True)

    # ---------------------------------------------------
    # Create global tertiles for risk groups
    # ---------------------------------------------------
    pooled["risk_group"] = pd.qcut(
        pooled["risk_score"], q=3, labels=["Low", "Medium", "High"]
    )

    # Encode risk groups numerically for Cox model
    pooled["risk_numeric"] = pooled["risk_group"].map({"Low": 0, "Medium": 1, "High": 2})

    # ---------------------------------------------------
    # Cox model for hazard ratio estimation
    # ---------------------------------------------------
    cph = CoxPHFitter()
    cph_df = pooled[["time_to_event", "event_occurred", "risk_numeric"]].copy()

    cph.fit(
        cph_df,
        duration_col="time_to_event",
        event_col="event_occurred"
    )

    summary = cph.summary.loc["risk_numeric"]

    hr = np.exp(summary["coef"])
    ci_lower = np.exp(summary["coef"] - 1.96 * summary["se(coef)"])
    ci_upper = np.exp(summary["coef"] + 1.96 * summary["se(coef)"])

    # Store in dataframe
    out_df = pd.DataFrame({
        "contrast": ["Per step in risk tertile (Low→Med→High)"],
        "hazard_ratio": [hr],
        "CI_lower": [ci_lower],
        "CI_upper": [ci_upper]
    })

    out_csv = os.path.join(root_dir, "hazard_ratios_v2.csv")
    out_df.to_csv(out_csv, index=False)

    print(">>> Hazard ratio CSV saved to:", out_csv)
    print(out_df)

    # ---------------------------------------------------
    # Forest plot
    # ---------------------------------------------------
    plt.figure(figsize=(9, 4))

    plt.errorbar(
        hr,
        0,
        xerr=[[hr - ci_lower], [ci_upper - hr]],
        fmt="o",
        color="#d9534f",
        ecolor="#555555",
        capsize=5,
        markersize=8,
    )

    plt.axvline(x=1.0, linestyle="--", color="gray")

    plt.yticks([0], ["Risk Tertile (per step)"])
    plt.xlabel("Hazard Ratio")
    plt.title("Hazard Ratio per Tertile Increase (Low → Medium → High)", fontsize=14)
    plt.grid(alpha=0.3)

    out_png = os.path.join(root_dir, "hazard_ratio_forest_plot_v2.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(">>> Forest plot saved:", out_png)


if __name__ == "__main__":
    compute_hazard_ratios()
