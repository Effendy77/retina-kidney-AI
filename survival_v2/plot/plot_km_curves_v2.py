import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def plot_km_folds(root_dir="survival_v2/checkpoints_single_v2_5fold"):
    """
    Generate KM curves for each fold, grouping subjects by risk tertiles.
    """

    for fold in range(1, 6):
        risk_path = os.path.join(root_dir, f"fold{fold}_risk_scores.csv")
        if not os.path.exists(risk_path):
            print(f"[WARNING] Missing: {risk_path}")
            continue

        df = pd.read_csv(risk_path)

        # Sort into low / mid / high tertiles
        df["risk_group"] = pd.qcut(df["risk_score"], q=3, labels=["Low", "Medium", "High"])

        kmf = KaplanMeierFitter()

        plt.figure(figsize=(9, 7))

        for group, color in zip(["Low", "Medium", "High"], ["#3CB371", "#f0ad4e", "#d9534f"]):
            mask = df["risk_group"] == group

            if mask.sum() == 0:
                continue

            kmf.fit(
                durations=df.loc[mask, "time_to_event"],
                event_observed=df.loc[mask, "event_occurred"],
                label=group
            )
            kmf.plot_survival_function(ci_show=False, color=color, linewidth=2)

        plt.title(f"Kaplan–Meier Survival Curves (Fold {fold})", fontsize=16)
        plt.xlabel("Time (years)", fontsize=14)
        plt.ylabel("Survival Probability", fontsize=14)
        plt.grid(alpha=0.3)

        out_path = os.path.join(root_dir, f"fold{fold}_KM_curves_v2.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f">>> KM curves saved: {out_path}")


if __name__ == "__main__":
    plot_km_folds()
