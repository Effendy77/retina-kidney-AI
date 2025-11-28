📘 SHAP Explainability — Survival_v3 (RETINA–KIDNEY–AI)

Multimodal DeepSurv SHAP System for ESRD Prediction

This folder contains all SHAP-based explainability modules for the multimodal DeepSurv ESRD prediction model (Survival_v3).
The SHAP pipeline quantifies how:

Retinal fundus images

Vessel segmentation masks

Tabular clinical variables

contribute to the predicted ESRD risk for each individual.

The resulting SHAP visualizations form the interpretability component of the Survival_v3 publication and supplementary materials.

📂 Directory Structure
survival_v3/
│
├── shap/
│   ├── shap_tabular_v3.py          # Kernel SHAP for tabular features
│   ├── plot_tabular_shap_v3.py     # Barplot & beeswarm for tabular features
│   ├── shap_image_v3.py            # Saliency-based SHAP for fundus images
│   ├── shap_mask_v3.py             # Saliency-based SHAP for vessel masks
│   ├── combined_shap_v3.py         # 2×2 panel: image + mask SHAP
│   ├── multimodal_panel_v3.py      # 3×2 panel: image + mask + tabular
│   └── README.md                   # You are here
│
└── run_shap_all_v3.py              # One-click SHAP pipeline launcher

🎯 SHAP Pipeline Goals

The SHAP system answers three core interpretability questions:

1. Where in the retinal image does the model attend when predicting ESRD risk?

▶ via image SHAP (gradients → heatmaps)

2. Which vessel-mask regions influence risk prediction?

▶ via mask SHAP (1-channel gradient heatmaps)

3. Which clinical & retinal features shape the risk score?

▶ via tabular Kernel SHAP for 11 variables

4. How do all modalities interact together?

▶ via combined 2×2 panels
▶ via full multimodal 3×2 panels (image + mask + tabular)

These outputs are suitable for:

Manuscript figures

Supplementary materials

Clinical interpretability reporting

Ethics reviews

Supervisor review presentations

🚀 Running the Full SHAP Pipeline

From the project root:

conda activate retina-renal-ai
python survival_v3/run_shap_all_v3.py


This single command generates all SHAP outputs:

survival_v3/checkpoints_single_v3/
│
├── shap_tabular_v3/
│   ├── tabular_shap_values.npy
│   ├── tabular_shap_samples.npy
│   ├── tabular_shap_barplot_v3.png
│   ├── tabular_shap_beeswarm_v3.png
│   └── tabular_shap_summary_v3.csv
│
├── shap_image_v3/
│   ├── eid_XXXXX_image_shap_overlay_v3.png
│   └── eid_XXXXX_image_shap_raw_v3.png
│
├── shap_mask_v3/
│   ├── eid_XXXXX_mask_shap_overlay_v3.png
│   └── eid_XXXXX_mask_shap_raw_v3.png
│
├── shap_combined_v3/
│   └── eid_XXXXX_combined_shap_v3.png
│
└── shap_multimodal_panel_v3/
    └── eid_XXXXX_multimodal_panel_v3.png

🧠 Methodology
1️⃣ Tabular SHAP (Global Feature Importance)

shap_tabular_v3.py uses:

KernelExplainer (model-agnostic SHAP)

Input dimension: 11 tabular features

Baseline: mean tabular vector across the dataset

Image + mask inputs replaced with model-compatible mean baseline images

Outputs:

raw .npy SHAP values

summary barplot

beeswarm plot

Suitable for:

Understanding relative importance of age, eGFR, vessel_density, fractal_dim, etc.

Global feature ranking in publications

2️⃣ Image SHAP (Fundus)

shap_image_v3.py performs:

Captum Saliency on the image branch

Mask + tabular inputs set to mean baselines

Heatmaps show spatial attention regions relevant to ESRD risk

Outputs:

✨ image_shap_overlay_v3.png

🔥 image_shap_raw_v3.png

Used to interpret retinal regions contributing to risk.

3️⃣ Mask SHAP (Vessel Segmentation)

shap_mask_v3.py performs:

Saliency on the vessel-mask branch

Image + tabular replaced with mean baselines

Explains influence of vascular geometry on risk

Outputs:

Vessel-mask SHAP overlay

Vessel-mask SHAP heatmap

4️⃣ Combined SHAP Panel (Image + Mask)

combined_shap_v3.py creates:

2×2 Multimodal Panel

Fundus Overlay	Fundus Heatmap
Mask Overlay	Mask Heatmap

Used for publication-ready multimodal interpretations.

5️⃣ Full Multimodal Panel (3×2)

multimodal_panel_v3.py creates:

3×2 Multimodal Panel

Fundus Overlay	Fundus Heatmap
Mask Overlay	Mask Heatmap
Tabular Barplot	Tabular Beeswarm

This is the recommended main interpretability figure for the manuscript.

🔗 Dependencies

Your environment must include:

torch

captum

shap

pandas

numpy

matplotlib

RETFound weights:

retfound/RETFound_mae_natureCFP.pth


No duplicated weight files are needed — one copy in retfound/ is sufficient.

📝 Citation

This SHAP system is part of the RETINA–KIDNEY–AI Survival_v3 pipeline for multimodal ESRD prediction.

(Citation statement will be added upon manuscript acceptance.)

📬 Contact

Effendy Bin Hashim
Postgraduate Researcher, PhD Candidate
University of Liverpool
GitHub: https://github.com/Effendy77