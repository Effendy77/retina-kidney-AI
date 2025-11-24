import pandas as pd
import os

CSV_IN = "data/multimodal_master_with_paths.csv"
CSV_OUT = "data/multimodal_master_CLEAN.csv"

TAB_COLS = ["age","sex","diabetes","hypertension","qrisk3","dm_htn_combined"]

df = pd.read_csv(CSV_IN)

print("Original rows:", len(df))

# -------------------------------------------------------
# 1) Drop rows with missing eGFR
# -------------------------------------------------------
df = df.dropna(subset=["egfr"])
print("After dropping missing eGFR:", len(df))

# -------------------------------------------------------
# 2) Drop rows with missing tabular predictors
# -------------------------------------------------------
df = df.dropna(subset=TAB_COLS)
print("After dropping tabular NaNs:", len(df))

# -------------------------------------------------------
# 3) OPTIONAL sanity check: ensure files exist
# (We already know they do—but this confirms.)
# -------------------------------------------------------
def ok(p): return os.path.exists(p)

mask_ok = df["vessel_mask"].apply(ok)
img_ok = df["fundus_image"].apply(ok)

df = df[mask_ok & img_ok]

print("After checking image/mask paths:", len(df))

# -------------------------------------------------------
# 4) Save cleaned file
# -------------------------------------------------------
df.to_csv(CSV_OUT, index=False)
print("Saved cleaned CSV →", CSV_OUT)

print("\nPreview:")
print(df.head())
