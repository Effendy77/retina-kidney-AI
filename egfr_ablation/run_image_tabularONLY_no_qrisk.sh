#!/bin/bash
set -e

echo "[INFO] Starting image + tabular ONLY (no mask, no retinal scalars) with QRISK3 silenced"

cd ~/projects/retina-kidney-AI/egfr_ablation

python main_multimodal_egfr_ablation.py \
  --use_image \
  --no_mask \
  --use_tabular \
  --no_retinal_features \
  --silence_qrisk \
  --outdir experiments/ablation/image_tabularONLY_no_qrisk

echo "[INFO] Done"
