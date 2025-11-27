import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def plot_km_combined(root_dir="survival_v2/checkpoints_single_v2_5fold"):
    """
    Creates one combined Kaplan–Meier curve using pooled tertile risk groups
    from all folds. This reflects the global survival separation quality.
    """

    # ------------------------------
    # Load all fold risk score files
    # ------------------------------
    all_folds = []

    for fold in range(1, 6):
        risk_path = os.path.join(root_dir, f"fold{fold}_risk_scores.csv")
        if not os.path.exists(risk_path):
            print(f"[WARNING] Missing: {risk_path}")
            continue

        df = pd.read_csv(risk_path)
        df["fold"] = fold
        all_folds.append(df)

    if len(all_folds) == 0:
        print("[ERROR] No folds found to combine.")
        return

    pooled = pd.concat(all_folds, ignore_index=True)

    # ------------------------------
    # Create global tertiles
    # ------------------------------
    pooled["risk_group"] = pd.qcut(
        pooled["risk_score"],
        q=3,
        labels=["Low", "Medium", "High"]
    )

    # ------------------------------
    # Plot KM curves
    # ------------------------------
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 7))

    color_map = {
        "Low": "#3CB371",        # green
        "Medium": "#f0ad4e",     # yellow
        "High": "#d9534f"        # red
    }

    for group in ["Low", "Medium", "High"]:
        mask = pooled["risk_group"] == group

        if mask.sum() == 0:
            continue

        kmf.fit(
            durations=pooled.loc[mask, "time_to_event"],
            event_observed=pooled.loc[mask, "event_occurred"],
            label=group
        )
        kmf.plot_survival_function(
            ci_show=False,
            linewidth=2.5,
            color=color_map[group]
        )

    plt.title("Combined Kaplan–Meier Curves (All 5 Folds)", fontsize=18)
    plt.xlabel("Time (years)", fontsize=15)
    plt.ylabel("Survival Probability", fontsize=15)
    plt.grid(alpha=0.3)

    out_path = os.path.join(root_dir, "KM_combined_v2.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Combined KM plot saved: {out_path}")


if __name__ == "__main__":
    plot_km_combined()
