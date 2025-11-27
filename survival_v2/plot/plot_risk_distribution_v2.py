import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_risk_distribution(root_dir="survival_v2/checkpoints_single_v2_5fold"):
    """
    Plots histogram + KDE distribution of predicted risks per fold.
    """

    for fold in range(1, 6):
        risk_path = os.path.join(root_dir, f"fold{fold}_risk_scores.csv")
        if not os.path.exists(risk_path):
            print(f"[WARNING] Missing: {risk_path}")
            continue

        df = pd.read_csv(risk_path)
        risks = df["risk_score"].values

        plt.figure(figsize=(9, 7))
        sns.histplot(risks, kde=True, bins=30, color="#4F81BD", alpha=0.7)

        plt.title(f"Risk Score Distribution (Fold {fold})", fontsize=16)
        plt.xlabel("Predicted Risk Score", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.grid(alpha=0.3)

        out_path = os.path.join(root_dir, f"fold{fold}_risk_distribution_v2.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f">>> Risk distribution saved: {out_path}")


if __name__ == "__main__":
    plot_risk_distribution()
