import os
import sys
import subprocess

"""
Run the full survival_v2 evaluation pipeline in sequence.

Assumes:
- Training has finished for single-run + 5-fold CV
- esrd_survival_v2.yaml is in survival_v2/configs/
- 5-fold outputs live in survival_v2/checkpoints_single_v2_5fold/
- All eval/plot scripts use the same config path internally
"""

STEPS = [
    # 1) Aggregate 5-fold C-index results
    ("Aggregate 5-fold C-index", "survival_v2.eval.agregate_5fold_results_v2"),

    # 2) Dynamic C-index (Uno)
    ("Compute dynamic C-index", "survival_v2.eval.compute_dynamic_cindex_v2"),

    # 3) Integrated Brier Score / IVS
    ("Compute IBS / IVS", "survival_v2.eval.compute_IVS_v2"),

    # 4) Hazard ratios by risk tertiles
    ("Compute hazard ratios", "survival_v2.eval.compute_hazard_ratio_v2"),

    # 5) 5-fold C-index boxplot
    ("Plot 5-fold C-index", "survival_v2.plot.plot_5fold_cindex_v2"),

    # 6) KM curves by tertiles
    ("Plot combined KM curves", "survival_v2.plot.plot_km_combined_v2"),

    # 7) Per-fold risk distributions
    ("Plot per-fold risk distributions", "survival_v2.plot.plot_risk_distribution_v2"),

    # 8) Combined risk distribution
    ("Plot combined risk distribution", "survival_v2.plot.plot_risk_distibution_combined_v2"),

    # 9) Calibration curve + Brier
    ("Plot survival calibration", "survival_v2.plot.plot_survival_calibration_v2"),

    # 10) Time-dependent ROC / AUC
    ("Plot time-dependent ROC", "survival_v2.plot.plot_time_dependent_roc_v2"),
]


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f">>> Running evaluation from project root: {project_root}\n")

    python_exe = sys.executable

    for label, module_path in STEPS:
        print(f"\n==============================")
        print(f">>> STEP: {label}")
        print(f">>> MODULE: {module_path}")
        print(f"==============================\n")

        cmd = [python_exe, "-m", module_path]
        ret = subprocess.run(cmd)

        if ret.returncode != 0:
            print(f"\n[ERROR] Step failed: {label}")
            print("Command was:", " ".join(cmd))
            sys.exit(ret.returncode)

    print("\n>>> Full evaluation pipeline (v2) completed successfully.\n")


if __name__ == "__main__":
    main()
