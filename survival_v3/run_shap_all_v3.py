import os
import subprocess

def run(cmd):
    print("\n====================================================")
    print(f"Running: {cmd}")
    print("====================================================\n")
    status = subprocess.call(cmd, shell=True)
    if status != 0:
        print(f"[ERROR] Command failed: {cmd}")
        exit(1)

def main():

    # Ensure the SHAP folders exist
    os.makedirs("survival_v3/checkpoints_single_v3/shap_tabular_v3", exist_ok=True)
    os.makedirs("survival_v3/checkpoints_single_v3/shap_image_v3", exist_ok=True)
    os.makedirs("survival_v3/checkpoints_single_v3/shap_mask_v3", exist_ok=True)
    os.makedirs("survival_v3/checkpoints_single_v3/shap_combined_v3", exist_ok=True)
    os.makedirs("survival_v3/checkpoints_single_v3/shap_multimodal_panel_v3", exist_ok=True)

    print("\n====================================================")
    print("      SURVIVAL_V3 — FULL SHAP PIPELINE")
    print("====================================================\n")

    # ------------------------------------------------------
    # 1. TABULAR SHAP (generate .npy raw SHAP values)
    # ------------------------------------------------------
    run("python survival_v3/shap/shap_tabular_v3.py")

    # ------------------------------------------------------
    # 2. TABULAR SHAP PLOTS (barplot + beeswarm)
    # ------------------------------------------------------
    run("python survival_v3/shap/plot_tabular_shap_v3.py")

    # ------------------------------------------------------
    # 3. IMAGE SHAP (fundus)
    # ------------------------------------------------------
    run("python survival_v3/shap/shap_image_v3.py")

    # ------------------------------------------------------
    # 4. MASK SHAP (vessel segmentation mask)
    # ------------------------------------------------------
    run("python survival_v3/shap/shap_mask_v3.py")

    # ------------------------------------------------------
    # 5. COMBINED SHAP (2×2: image + mask only)
    # ------------------------------------------------------
    run("python survival_v3/shap/combined_shap_v3.py")

    # ------------------------------------------------------
    # 6. MULTIMODAL PANEL (3×2: image + mask + tabular)
    # ------------------------------------------------------
    run("python survival_v3/shap/multimodal_panel_v3.py")

    print("\n====================================================")
    print("  ALL SHAP MODULES COMPLETED SUCCESSFULLY 🎉")
    print("  Results saved under survival_v3/checkpoints_single_v3/")
    print("====================================================\n")

if __name__ == "__main__":
    main()
