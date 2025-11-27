import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_5fold_cindex(root_dir="survival_v2/checkpoints_single_v2_5fold"):
    """
    Plot a box-and-scatter plot for 5-fold C-index results.
    """

    fold_values = []

    for fold in range(1, 6):
        metrics_path = os.path.join(root_dir, f"fold{fold}_metrics.csv")
        if not os.path.exists(metrics_path):
            print(f"[Warning] Missing {metrics_path}")
            continue

        df = pd.read_csv(metrics_path)
        best_c = df["val_cindex"].max()
        fold_values.append(best_c)

        print(f"Fold {fold} C-index = {best_c:.4f}")

    if len(fold_values) == 0:
        print("[Error] No folds found.")
        return

    # -----------------------------
    # Plot
    # -----------------------------
    sns.set(style="whitegrid", font_scale=1.2)

    plt.figure(figsize=(8, 6))
    plt.boxplot(fold_values, widths=0.4, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#B3D1FF"))

    # Scatter individual points
    for i, val in enumerate(fold_values, start=1):
        plt.scatter(1, val, color="#003C8F", s=80)

    plt.title("5-Fold C-index (Survival V2)", fontsize=16)
    plt.ylabel("C-index", fontsize=14)
    plt.xticks([1], ["Survival V2"])

    out_path = os.path.join(root_dir, "cindex_boxplot_v2.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n>>> Saved plot → {out_path}")


if __name__ == "__main__":
    plot_5fold_cindex()
