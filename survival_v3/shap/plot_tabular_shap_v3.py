import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():

    # --------------------------------------------------
    # SHAP directory (consistent with Survival_v3 folder)
    # --------------------------------------------------
    shap_dir = "survival_v3/checkpoints_single_v3/shap_tabular_v3"

    shap_values_fp = os.path.join(shap_dir, "tabular_shap_values.npy")
    feature_vals_fp = os.path.join(shap_dir, "tabular_shap_samples.npy")

    # --------------------------------------------------
    # Validate SHAP files exist
    # --------------------------------------------------
    if not os.path.exists(shap_values_fp):
        print(f"[ERROR] Missing file: {shap_values_fp}")
        return
    if not os.path.exists(feature_vals_fp):
        print(f"[ERROR] Missing file: {feature_vals_fp}")
        return

    # --------------------------------------------------
    # Load SHAP values + input feature values
    # --------------------------------------------------
    shap_values = np.load(shap_values_fp)      # [N, F]
    feature_vals = np.load(feature_vals_fp)    # [N, F]

    # --------------------------------------------------
    # Correct full list of 11 tabular features for Survival_v3
    # (MUST match MultimodalSurvivalDatasetV3.tabular_features)
    # --------------------------------------------------
    feature_names = [
        "age",
        "sex",
        "diabetes",
        "hypertension",
        "egfr",
        "qrisk3",
        "dm_htn_combined",
        "fractal_dim",
        "vessel_density",
        "eccentricity",
        "mean_width_px"
    ]

    # --------------------------------------------------
    # Validate feature dimension matches
    # --------------------------------------------------
    if shap_values.shape[1] != len(feature_names):
        print("[ERROR] Feature count mismatch.")
        print("SHAP array shape:", shap_values.shape)
        print("Expected features:", feature_names)
        return

    # --------------------------------------------------
    # 1. Mean |SHAP| Bar Plot
    # --------------------------------------------------
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    plt.figure(figsize=(14, 6))
    colors = ['steelblue' if val < np.mean(mean_abs) else 'crimson' for val in mean_abs]
    plt.bar(feature_names, mean_abs, color=colors)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean |SHAP|")
    plt.title("Mean |SHAP| for Tabular Features (Survival_v3)")
    plt.tight_layout()

    bar_path = os.path.join(shap_dir, "tabular_shap_barplot_v3.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(">>> Saved:", bar_path)

    # --------------------------------------------------
    # 2. Beeswarm Plot
    # --------------------------------------------------
    plt.figure(figsize=(14, 6))
    try:
        import shap
        shap.summary_plot(
            shap_values,
            feature_vals,
            feature_names=feature_names,
            show=False,
        )
        beeswarm_path = os.path.join(shap_dir, "tabular_shap_beeswarm_v3.png")
        plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(">>> Saved:", beeswarm_path)
    except Exception as e:
        print("[WARNING] Could not generate beeswarm plot:", e)

    # --------------------------------------------------
    # 3. Save summary CSV
    # --------------------------------------------------
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    })
    df = df.sort_values("mean_abs_shap", ascending=False)

    csv_path = os.path.join(shap_dir, "tabular_shap_summary_v3.csv")
    df.to_csv(csv_path, index=False)
    print(">>> Saved:", csv_path)

    print("\n>>> Tabular SHAP plots (Survival_v3) completed successfully.\n")


if __name__ == "__main__":
    main()
