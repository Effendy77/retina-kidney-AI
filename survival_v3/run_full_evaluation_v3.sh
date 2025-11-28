#!/usr/bin/env bash
set -e

echo "==============================================================="
echo "         RETINA–KIDNEY–AI SURVIVAL_V3 — FULL PIPELINE"
echo "==============================================================="
echo ""

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
ENV_NAME="retina-renal-ai"
CONFIG="survival_v3/configs/esrd_survival_v3.yaml"
EXP_DIR="survival_v3/experiments/single_run_5fold"
EVAL_DIR="${EXP_DIR}/eval_v3"

# ---------------------------------------------------------------
# ACTIVATE CONDA ENVIRONMENT
# ---------------------------------------------------------------
echo "[INFO] Activating conda environment: ${ENV_NAME}"
# If conda.sh exists use it, else fallback to mamba
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate ${ENV_NAME}
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ${ENV_NAME}
else
    echo "[WARNING] Could not find conda.sh, using 'mamba activate'"
    mamba activate ${ENV_NAME}
fi

echo ""
echo "---------------------------------------------------------------"
echo " STEP 1: 5-FOLD CROSS-VALIDATION TRAINING"
echo "---------------------------------------------------------------"

python survival_v3/scripts/run_5fold_v3.py --config ${CONFIG}

echo ""
echo "---------------------------------------------------------------"
echo " STEP 2: AGGREGATE + SUMMARY EVALUATION"
echo "---------------------------------------------------------------"

python survival_v3/scripts/eval_5fold_v3.py \
    --exp_root ${EXP_DIR} \
    --out_dir ${EVAL_DIR}

echo ""
echo "---------------------------------------------------------------"
echo " STEP 3: KAPLAN–MEIER PLOTS"
echo "---------------------------------------------------------------"

python survival_v3/plot/plot_km_curves_v3.py \
    --input_csv ${EVAL_DIR}/all_folds_risk_scores_v3.csv \
    --out_path ${EVAL_DIR}/km_tertiles_v3.png

echo ""
echo "---------------------------------------------------------------"
echo " STEP 4: CALIBRATION CURVES"
echo "---------------------------------------------------------------"

python survival_v3/plot/plot_survival_calibration_v3.py \
    --input_csv ${EVAL_DIR}/all_folds_risk_scores_v3.csv \
    --out_plot ${EVAL_DIR}/calibration_curve_v3.png \
    --out_csv ${EVAL_DIR}/calibration_bins_v3.csv \
    --time_horizon 5 \
    --bins 10

echo ""
echo "---------------------------------------------------------------"
echo " STEP 5: TIME-DEPENDENT ROC CURVES"
echo "---------------------------------------------------------------"

python survival_v3/plot/plot_time_dependent_roc_v3.py \
    --input_csv ${EVAL_DIR}/all_folds_risk_scores_v3.csv \
    --out_csv ${EVAL_DIR}/tdROC_auc_values_v3.csv \
    --out_plot ${EVAL_DIR}/tdROC_v3.png

echo ""
echo "---------------------------------------------------------------"
echo " STEP 6: RISK DISTRIBUTION PLOTS"
echo "---------------------------------------------------------------"

python survival_v3/plot/plot_risk_distribution_v3.py \
    --input_csv ${EVAL_DIR}/all_folds_risk_scores_v3.csv \
    --out_dir ${EVAL_DIR}

echo ""
echo "---------------------------------------------------------------"
echo " STEP 7: HAZARD RATIO ANALYSIS + FOREST PLOT"
echo "---------------------------------------------------------------"

python survival_v3/eval/compute_hazard_ratio_v3.py \
    --input_csv ${EVAL_DIR}/all_folds_risk_scores_v3.csv \
    --out_csv  ${EVAL_DIR}/hazard_ratios_v3.csv \
    --out_fig  ${EVAL_DIR}/hazard_ratio_forest_plot_v3.png

echo ""
echo "---------------------------------------------------------------"
echo " STEP 8: (OPTIONAL) SHAP ANALYSIS — COPY FROM V2"
echo "---------------------------------------------------------------"
echo "[INFO] SHAP analysis not auto-run. Run manually:"
echo "       python survival_v3/shap/shap_tabular_v3.py"
echo "       python survival_v3/shap/shap_image_v3.py"
echo "       python survival_v3/shap/shap_mask_v3.py"
echo "       python survival_v3/shap/combined_shap_v3.py"

echo ""
echo "==============================================================="
echo "      SURVIVAL_V3 FULL EVALUATION — PIPELINE COMPLETE"
echo "==============================================================="
echo ""
