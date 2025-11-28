#!/bin/bash

###############################################################################
# eGFR Multimodal V2 – Full Pipeline Runner
# - Runs 5-fold multimodal training
# - Runs V2 evaluation across all folds
# - Does NOT modify or touch V1 experiments
###############################################################################

echo "========================================================="
echo "   RETINA-KIDNEY-AI — eGFR MULTIMODAL V2 FULL PIPELINE"
echo "========================================================="

# -------------------------
# Activate Conda (modify if needed)
# -------------------------
if command -v conda &> /dev/null; then
    echo "[INFO] Activating conda environment: retina-renal-ai"
    source /home/fendy77/miniconda3/etc/profile.d/conda.sh
    conda activate retina-renal-ai
else
    echo "[WARN] Conda not detected — using system Python."
fi

# -------------------------
# Paths
# -------------------------
CSV_PATH="data/multimodal_master_CLEANv2.csv"
IMAGE_ROOT="data/images"
MASK_ROOT="data/masks"
RETF_PATH="../retfound/RETFound_mae_natureCFP.pth"
OUTDIR="experiments/egfr_v2"

echo "[INFO] Using CSV:         $CSV_PATH"
echo "[INFO] Using image root:  $IMAGE_ROOT"
echo "[INFO] Using mask root:   $MASK_ROOT"
echo "[INFO] RETFound weights:  $RETF_PATH"
echo "[INFO] Output directory:  $OUTDIR"

# -------------------------
# Step 1 — 5-Fold Training
# -------------------------
echo "---------------------------------------------------------"
echo "   STEP 1: TRAINING MULTIMODAL EGFR V2 (5 FOLDS)"
echo "---------------------------------------------------------"

python main_multimodal_egfr_v2_5fold.py \
    --csv $CSV_PATH \
    --images $IMAGE_ROOT \
    --masks $MASK_ROOT \
    --retfound_weights $RETF_PATH \
    --outdir $OUTDIR \
    --batch_size 16 \
    --lr 1e-4 \
    --epochs 20 \
    --num_workers 4

# -------------------------
# Step 2 — Evaluation
# -------------------------
echo "---------------------------------------------------------"
echo "   STEP 2: EVALUATING ALL FOLDS (V2)"
echo "---------------------------------------------------------"

python -m src.eval.eval_multimodal_egfr_v2 \
    --root_dir $OUTDIR \
    --images $IMAGE_ROOT \
    --masks $MASK_ROOT \
    --retfound_weights $RETF_PATH

echo "---------------------------------------------------------"
echo "  DONE! RESULTS AVAILABLE UNDER: $OUTDIR/results"
echo "---------------------------------------------------------"

