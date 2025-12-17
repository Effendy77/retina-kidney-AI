#!/bin/bash
set -e

# ============================================
# Multimodal eGFR ablation: QRISK3 silenced
# ============================================

echo "[INFO] Starting image + tabular (no QRISK3) experiment"

cd ~/projects/retina-kidney-AI/egfr_ablation

python main_multimodal_egfr_ablation.py \
  --images /home/fendy77/data/retina_images \
  --masks  /home/fendy77/projects/retina-kidney-AI/data/masks_raw_binary \
  --use_image \
  --use_mask \
  --use_tabular \
  --use_retinal_features \
  --silence_qrisk \
  --outdir experiments/ablation/image_tabular_no_qrisk

echo "[INFO] Experiment completed successfully"
