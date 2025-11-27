# survival_v2/shap/plot_tabular_shap_v2.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():

    shap_dir = "survival_v2/checkpoints_single_v2/shap_tabular_v2"

    shap_values_fp = os.path.join(shap_dir, "tabular_shap_values.npy")
    feature_vals_fp = os.path.join(shap_dir, "tabular_shap_samples.npy")

    # --------------------------------------------------
    # Validate files exist
    # --------------------------------------------------
    if not os.path.exists(shap_values_fp):
        print(f"[ERROR] Missing file: {shap_values_fp}")
        return
    if not os.path.exists(feature_vals_fp):
        print(f"[ERROR] Missing file: {feature_vals_fp}")
        return

    # --------------------------------------------------
    # Load files
    # --------------------------------------------------
    shap_values = np.load(shap_values_fp)         # [N, F]
    feature_vals = np.load(feature_vals_fp)       # [N, F]

    # Hardcode feature names based on metadata in your dataset
    feature_names = [
        "egfr",
        "qrisk3",
        "age",
        "diabetes",
        "sex",
        "dm_htn_combined",
        "hypertension"
    ]

    if shap_values.shape[1] != len(feature_names):
        print("[ERROR] Feature count mismatch.")
        print("SHAP array shape:", shap_values.shape)
        print("Expected features:", feature_names)
        return

    # --------------------------------------------------
    # 1. Mean |SHAP| Bar Plot
    # --------------------------------------------------
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    plt.figure(figsize=(12,6))
    plt.bar(feature_names, mean_abs)
    plt.xticks(rotation=45)
    plt.ylabel("Mean |SHAP|")
    plt.title("Mean |SHAP| for Tabular Features")
    plt.tight_layout()
    bar_path = os.path.join(shap_dir, "tabular_shap_barplot_v2.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(">>> Saved:", bar_path)

    # --------------------------------------------------
    # 2. Beeswarm Plot
    # --------------------------------------------------
    plt.figure(figsize=(12,6))
    try:
        import shap
        shap.summary_plot(
            shap_values,
            feature_vals,
            feature_names=feature_names,
            show=False
        )
        beeswarm_path = os.path.join(shap_dir, "tabular_shap_beeswarm_v2.png")
        plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(">>> Saved:", beeswarm_path)
    except Exception as e:
        print("[WARNING] Could not generate beeswarm plot:", e)

    # --------------------------------------------------
    # 3. Save summary CSV
    # --------------------------------------------------
    import pandas as pd
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    })
    df = df.sort_values("mean_abs_shap", ascending=False)

    csv_path = os.path.join(shap_dir, "tabular_shap_summary_v2.csv")
    df.to_csv(csv_path, index=False)
    print(">>> Saved:", csv_path)

    print("\n>>> Tabular SHAP standalone plots completed successfully.")


if __name__ == "__main__":
    main()
