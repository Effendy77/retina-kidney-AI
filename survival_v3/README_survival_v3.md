📄 README — Survival_v3 (RETINA–KIDNEY–AI)

Deep Learning Survival Model for Predicting ESRD from Retinal Images, Vessel Masks, and Tabular Clinical Data

📘 Overview

Survival_v3 is a fully reproducible deep-learning pipeline designed to predict time-to-end-stage renal disease (ESRD) using:

Color fundus photographs (left eye)

Binary vessel segmentation masks

Clinical tabular variables (age, sex, diabetes, hypertension, eGFR, QRISK3, dm_htn_combined)

Retinal vascular features (fractal_dim, vessel_density, eccentricity, mean_width_px)

The model is based on a multimodal DeepSurv architecture, combining:

RETFound ViT-L image encoder

Light CNN vessel-mask encoder

Tabular encoder (11 variables)

Fusion MLP → Cox proportional hazards head

The pipeline supports:

Single-run training

5-fold cross-validation

Full evaluation suite

SHAP explainability

Publication-ready figures

This project mirrors the architecture and quality of our eGFR regression pipeline and is intended for a journal submission (e.g., The Lancet Digital Health).

📂 Project Structure
survival_v3/
│
├── configs/
│   └── esrd_survival_v3.yaml
│
├── scripts/
│   ├── run_single_fold_v3.py
│   ├── run_5fold_v3.py
│   └── eval_5fold_v3.py
│
├── src/
│   ├── datasets/
│   │   └── multimodal_survival_dataset_v3.py
│   ├── model/
│   │   └── multimodal_deepsurv_v3.py
│   ├── train/
│   │   └── train_survival_v3.py
│   └── utils/
│
├── plot/
│   ├── plot_km_curves_v3.py
│   ├── plot_survival_calibration_v3.py
│   ├── plot_time_dependent_roc_v3.py
│   └── plot_risk_distribution_v3.py
│
├── eval/
│   └── compute_hazard_ratio_v3.py
│
├── shap/                # Copied from survival_v2 (rename imports to _v3)
│   ├── shap_tabular_v3.py
│   ├── shap_image_v3.py
│   ├── shap_mask_v3.py
│   ├── combined_shap_v3.py
│   └── multimodal_panel_v3.py
│
├── run_full_evaluation_v3.sh
└── experiments/
    └── single_run_5fold/

🧬 Data Requirements

Input CSV must contain (already provided as survival_multimodal_master_v3.csv):

Column	Description
eid	UKB participant ID
fundus_image	path to left-eye RGB image
vessel_mask	path to binary mask
time_to_event	survival time (years)
event_occurred	1=ESRD, 0=censored
age, sex	demographics
diabetes, hypertension	comorbidities
egfr	baseline eGFR
qrisk3	QRISK3 (%)
dm_htn_combined	diabetes + hypertension indicator
fractal_dim, vessel_density, eccentricity, mean_width_px	retinal vascular metrics
⚙️ Configuration

Edit the YAML file:

survival_v3/configs/esrd_survival_v3.yaml


Key parameters:

csv_path: "data/survival_multimodal_master_v3.csv"
retfound_weights: "retfound/RETFound_mae_natureCFP.pth"

batch_size: 16
epochs: 50
lr: 1e-5
patience: 10

fusion_dim: 512
dropout: 0.2

🚀 Quick Start
1. Activate environment
conda activate retina-renal-ai

2. Run full 5-fold training + evaluation
bash survival_v3/run_full_evaluation_v3.sh


This runs:

✓ 5-fold training
✓ Risk aggregation
✓ KM curves
✓ Calibration
✓ Time-dependent ROC
✓ Risk distribution
✓ Hazard ratios
✓ SHAP (manual)

All results saved to:

survival_v3/experiments/single_run_5fold/eval_v3/

📈 Outputs
Key files produced:
File	Description
km_tertiles_v3.png	Kaplan–Meier survival curves across risk tertiles
calibration_curve_v3.png	Time-horizon survival calibration
tdROC_v3.png	Time-dependent ROC curve
risk_kde_v3.png	KDE distribution of predicted risk
hazard_ratios_v3.csv	Hazard ratios (continuous + tertiles)
hazard_ratio_forest_plot_v3.png	Forest plot
brier_curve_v3.csv	Brier score curve
summary_v3.csv	Overall evaluation metrics
all_folds_risk_scores_v3.csv	Combined risk table

Plus all SHAP figures if run manually.

🧠 SHAP Explainability

SHAP folder copied from Survival_v2.

Run manually:

python survival_v3/shap/shap_tabular_v3.py
python survival_v3/shap/shap_image_v3.py
python survival_v3/shap/shap_mask_v3.py
python survival_v3/shap/combined_shap_v3.py


Outputs include:

SHAP barplot

Beeswarm

Raw and overlay Grad-CAM

Multimodal interpretability panel

🧪 Reproducibility

This pipeline is:

Fully deterministic

Cross-platform

Compatible with GPU clusters (Barkla, SLURM)

Ready for GitHub publication + supplementary materials

Versioning notes:

Python ≥ 3.10

PyTorch ≥ 2.0

scikit-survival required for tdROC

RETFound weights must be present in /retfound/

✍️ Citation

I will include this once your manuscript is accepted. For now:

“This Survival_v3 model is part of the RETINA–KIDNEY–AI project for developing multimodal deep learning survival models predicting ESRD from retinal imaging.”

📬 Contact

Effendy Bin Hashim
Postgraduate Researcher, PhD Candidate
Department of Eye and Vision Science
University of Liverpool
GitHub: https://github.com/Effendy77
