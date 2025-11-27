import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", font_scale=1.2)

df = pd.read_csv("/home/fendy77/projects/retina-kidney-AI/checkpoints_5fold/cv_summary.csv")

# -----------------------------
# 1. Boxplot (MAE, RMSE, MSE)
# -----------------------------
metrics = ["mae", "rmse", "mse"]

plt.figure(figsize=(10,6))
sns.boxplot(data=df[metrics])
plt.title("5-Fold Cross-Validation Performance (Boxplot)")
plt.savefig("fig_5fold_boxplot.png", dpi=300)
plt.close()

# -----------------------------
# 2. Violin plot
# -----------------------------
plt.figure(figsize=(10,6))
sns.violinplot(data=df[metrics], inner="point")
plt.title("5-Fold Performance (Violin Plot)")
plt.savefig("fig_5fold_violin.png", dpi=300)
plt.close()

# -----------------------------
# 3. Fold-wise line plot
# -----------------------------
plt.figure(figsize=(8,5))
sns.lineplot(x="fold", y="mae", data=df, marker="o", label="MAE")
sns.lineplot(x="fold", y="rmse", data=df, marker="o", label="RMSE")
plt.title("Fold-by-Fold Performance Trend")
plt.xlabel("Fold")
plt.ylabel("Error")
plt.savefig("fig_5fold_lineplot.png", dpi=300)
plt.close()

# -----------------------------
# 4. KDE Density (Distribution)
# -----------------------------
plt.figure(figsize=(8,5))
for m in metrics:
    sns.kdeplot(df[m], label=m, fill=True, alpha=0.3)
plt.title("Distribution of Error Metrics (KDE)")
plt.legend()
plt.savefig("fig_5fold_kde.png", dpi=300)
plt.close()


print("All plots saved:")
print(" - fig_5fold_boxplot.png")
print(" - fig_5fold_violin.png")
print(" - fig_5fold_lineplot.png")
print(" - fig_5fold_kde.png")
