#!/bin/bash
set -e

echo "[INFO] Dry-run (1 epoch): image + tabular only, QRISK silenced"
cd ~/projects/retina-kidney-AI
export PYTHONPATH=$PWD

python egfr_ablation_v2/main_multimodal_egfr_ablation.py \
  --epochs 1 \
  --batch_size 8 \
  --num_workers 2 \
  --images /home/fendy77/data/retina_images \
  --masks  /home/fendy77/projects/retina-kidney-AI/data/masks_raw_binary \
  --retfound_weights retfound/RETFound_mae_natureCFP.pth \
  --use_image \
  --no_mask \
  --use_tabular \
  --no_retinal_features \
  --tabular_mode baseline_plus_qrisk \
  --silence_qrisk \
  --outdir egfr_ablation_v2/experiments/dryrun_image_tabularONLY_no_qrisk

echo "[INFO] Done"
