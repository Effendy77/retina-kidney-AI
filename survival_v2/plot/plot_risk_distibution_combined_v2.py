import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_combined_risk_distribution(
    root_dir="survival_v2/checkpoints_single_v2_5fold"
):
    """
    Creates a combined pooled risk distribution (histogram + KDE)
    from all folds of the survival V2 model.
    """

    risk_scores = []

    # ---------------------------------------------------
    # Collect all risk score files from folds 1–5
    # ---------------------------------------------------
    for fold in range(1, 6):
        risk_path = os.path.join(root_dir, f"fold{fold}_risk_scores.csv")
        if not os.path.exists(risk_path):
            print(f"[WARNING] Missing: {risk_path}")
            continue

        df = pd.read_csv(risk_path)
        risk_scores.extend(df["risk_score"].values)

    if len(risk_scores) == 0:
        print("[ERROR] No risk score files found. Aborting.")
        return

    # Convert to DataFrame
    all_risk_df = pd.DataFrame({"risk_score": risk_scores})

    # ---------------------------------------------------
    # Plot distribution
    # ---------------------------------------------------
    sns.set(style="whitegrid", font_scale=1.3)

    plt.figure(figsize=(10, 7))
    sns.histplot(
        all_risk_df["risk_score"],
        kde=True,
        bins=40,
        color="#4F81BD",
        alpha=0.7,
        linewidth=1,
    )

    plt.title("Combined Risk Score Distribution (All 5 Folds)", fontsize=18)
    plt.xlabel("Predicted Risk Score", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.grid(alpha=0.3)

    out_path = os.path.join(root_dir, "risk_distribution_combined_v2.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Combined risk distribution saved: {out_path}")


if __name__ == "__main__":
    plot_combined_risk_distribution()
