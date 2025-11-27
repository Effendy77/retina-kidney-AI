#!/usr/bin/env python3

import os
import time
import subprocess
import sys

# -------------------------------------------------------------
# STEP REGISTRY
# -------------------------------------------------------------
STEPS = [
    # 1) Image SHAP (Captum)
    ("Image SHAP", "survival_v2.shap.shap_image_v2"),

    # 2) Mask SHAP (Captum)
    ("Mask SHAP", "survival_v2.shap.shap_mask_v2"),

    # 3) Tabular SHAP (KernelExplainer)
    ("Tabular SHAP", "survival_v2.shap.shap_tabular_v2"),

    # 4) Combined SHAP (image + mask)
    ("Combined SHAP", "survival_v2.shap.combined_shap_v2"),

    # 5) Multimodal SHAP panel (3×2)
    ("Multimodal Panel", "survival_v2.shap.multimodal_panel_v2"),
]


def run_step(name, module):
    print("\n==============================")
    print(f">>> STEP: {name}")
    print(f">>> MODULE: {module}")
    print("==============================\n")

    start = time.time()

    cmd = [sys.executable, "-m", module]
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"\n[ERROR] SHAP step failed with return code {result.returncode}: {name}")
            print("Command was:", " ".join(cmd))
            return False
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] SHAP step failed: {name}")
        print("Command was:", " ".join(cmd))
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error running {name}: {e}")
        return False

    elapsed = time.time() - start
    print(f">>> Completed: {name} in {elapsed:.1f} sec")

    return True


def main():
    project_root = os.getcwd()
    print(">>> Running SHAP pipeline from project root:")
    print(project_root, "\n")

    # Validate project structure
    required_dirs = ["survival_v2", "src"]
    for d in required_dirs:
        if not os.path.isdir(os.path.join(project_root, d)):
            print(f"[ERROR] Missing directory: {d}")
            print(f">>> Please run from project root: {project_root}")
            return

    # Ensure Python sees project modules
    os.environ["PYTHONPATH"] = project_root
    sys.path.insert(0, project_root)

    for step_name, module in STEPS:
        ok = run_step(step_name, module)
        if not ok:
            print(f"\n>>> Pipeline stopped at step: {step_name}")
            print(">>> To resume, fix the error and run this script again.\n")
            return

    print("\n=====================================")
    print(">>> ALL SHAP STEPS COMPLETED SUCCESSFULLY")
    print("=====================================\n")


if __name__ == "__main__":
    main()
