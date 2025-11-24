# Retina-Kidney-AI

A multimodal deep learning framework for predicting kidney function (eGFR), CKD risk, and ESRD progression using **retinal fundus images**, **vessel segmentation masks**, and **clinical tabular data**.

This repository integrates:

- RETFound ViT-Large backbone for retinal feature encoding
- Vessel segmentation masks (AutoMorph raw_binary)
- Tabular risk factors (age, sex, diabetes, hypertension, QRISK3, etc.)
- Multimodal fusion for regression, binary classification, or survival modelling

The system is designed for **scalable training** on local GPU or HPC clusters (e.g., Barkla) and will later support **5-fold cross-validation**, **survival analysis**, and **CKD staging**.

---

## 📌 Project Structure
```
retina-kidney-AI/
├── main_egfr.py                 # simple eGFR regression baseline
├── main_multimodal_egfr.py      # multimodal ViT-L + vessels + tabular model
├── main_multimodal_egfr_5fold.py# planned 5-fold CV training
├── main_survival.py             # DeepSurv / CoxPH multimodal model
├── configs/                     # configuration templates
├── data/                        # multimodal CSV, masks, fundus images
├── retfound/                    # RETFound weights (not included)
├── src/
│   ├── model/                   # encoders + fusion model
│   ├── datasets/                # multimodal dataloaders
│   ├── train/                   # training loops
│   ├── eval/                    # metrics, calibration, DCA, etc.
│   └── utils/                   # helpers
├── scripts/                     # preprocessing utilities
└── notebooks/                   # optional research notebooks
```

---

## 🔧 Installation
```bash
conda create -n retina-renal-ai python=3.10 -y
conda activate retina-renal-ai
pip install -r requirements.txt
```

---

## 🧠 Model Architecture
### 1. **Image Encoder** (RETFound ViT-Large)
- Loads ViT-Large Patch16 224
- Uses CLS token as embedding → **1024-dim vector**

### 2. **Vessel Mask Encoder**
- Lightweight CNN → **128-dim vector**

### 3. **Tabular Encoder**
- MLP → **128-dim vector**

### 4. **Fusion Layer**
- Concatenate → 1024 + 128 + 128 = **1280-dim**
- Fully connected + ReLU + dropout

### 5. **Task Head**
- Regression: predict eGFR
- Binary: CKD stage
- Survival: CoxPH risk score

---

## 📊 Training (Multimodal eGFR)
```bash
python main_multimodal_egfr.py \
    --csv data/multimodal_master_CLEAN.csv \
    --image_root /path/to/fundus_images \
    --mask_root /path/to/masks \
    --weights retfound/RETFound_mae_natureCFP.pth \
    --epochs 20 \
    --batch_size 16
```

---

## 🚀 Performance (Local GPU)
Initial results show:
- **Best Validation MAE ≈ 7.6** after 20 epochs
- Stable convergence
- Good learning of both fundus and vessel features

---

## 📁 Data Requirements
The multimodal CSV must contain:

| Column | Description |
|--------|-------------|
| `fundus_image` | full path to left-eye image |
| `vessel_mask` | full path to AutoMorph vessel mask |
| `age` | numeric |
| `sex` | 0 = female, 1 = male |
| `diabetes` | binary |
| `hypertension` | binary |
| `qrisk3` | cardiovascular risk score |
| `egfr` | outcome for regression |

---

## 🧪 Future Work
- ✔ Add 5-fold CV (ongoing)
- ✔ Add survival DeepSurv pipeline
- ✔ Add ESRD prediction
- ☐ Add calibration & DCA plots
- ☐ External validation on Barkla HPC
- ☐ Model card + documentation
- ☐ Inference pipeline + Grad-CAM

---

## 📄 License
MIT License

---

## 🙌 Acknowledgements
- RETFound team (Yukun Zhou et al.)
- AutoMorph vessel segmentation pipeline
- UK Biobank data access through approved project

---

## 👤 Author
**Effendy Bin Hashim**  
PhD Researcher  
University of Liverpool

