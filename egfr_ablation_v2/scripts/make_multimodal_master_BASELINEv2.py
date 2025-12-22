import pandas as pd
import numpy as np

CLEANV2_PATH = "egfr_ablation/data/multimodal_master_CLEANv2.csv"
BASELINE_PATH = "data/cleaned_left_eye_only.csv"
OUT_PATH = "egfr_ablation_v2/data/multimodal_master_BASELINEv2.csv"

# Load
mm = pd.read_csv(CLEANV2_PATH)
base = pd.read_csv(BASELINE_PATH)

# Keep only columns we need from baseline
need_base = ["eid", "Sex_0F_1M", "Age_at_baseline", "diabetes_prevalent", "hypertension_prevalent", "qrisk3"]
missing = [c for c in need_base if c not in base.columns]
if missing:
    raise ValueError(f"Missing required columns in baseline file: {missing}")

base_small = base[need_base].drop_duplicates("eid").copy()

# Merge onto modelling cohort (left-join to preserve exact analytic cohort)
out = mm.merge(base_small, on="eid", how="left", suffixes=("", "_baseline"))

print("[INFO] CLEANv2 rows:", len(mm))
print("[INFO] BASELINE unique eids:", base_small["eid"].nunique())
print("[INFO] After merge rows:", len(out))

# Overwrite/define baseline-anchored clinical predictors
# Age: use Age_at_baseline (baseline anchored)
out["age"] = out["Age_at_baseline"]

# Sex: ensure 0/1 with 1=male
# baseline column Sex_0F_1M is already 0F 1M
out["sex"] = out["Sex_0F_1M"]

# Diabetes/Hypertension: baseline prevalent
out["diabetes"] = out["diabetes_prevalent"].fillna(0).astype(int)
out["hypertension"] = out["hypertension_prevalent"].fillna(0).astype(int)

# Combined indicator
out["dm_htn_combined"] = ((out["diabetes"] == 1) | (out["hypertension"] == 1)).astype(int)

# QRISK3: baseline
out["qrisk3"] = pd.to_numeric(out["qrisk3"], errors="coerce")

# Clean up intermediate columns (optional)
drop_cols = ["Sex_0F_1M", "Age_at_baseline", "diabetes_prevalent", "hypertension_prevalent"]
for c in drop_cols:
    if c in out.columns:
        out.drop(columns=c, inplace=True)

# Final sanity: required columns for the existing pipeline
required = [
    "eid","age","sex","diabetes","hypertension","dm_htn_combined","qrisk3","egfr",
    "fundus_image","vessel_mask","fractal_dim","vessel_density","eccentricity","mean_width_px"
]
missing_req = [c for c in required if c not in out.columns]
if missing_req:
    raise ValueError(f"Output missing required modelling columns: {missing_req}")

# Report prevalence sanity
print("[SANITY] diabetes prevalence:", out["diabetes"].mean())
print("[SANITY] hypertension prevalence:", out["hypertension"].mean())
print("[SANITY] dm_htn_combined prevalence:", out["dm_htn_combined"].mean())

# Save
out.to_csv(OUT_PATH, index=False)
print("[DONE] Saved:", OUT_PATH)
print("[DONE] Rows:", len(out), "Cols:", len(out.columns))
