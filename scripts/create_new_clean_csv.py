import pandas as pd
import os
import sys

# Use relative path from script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

input_csv = os.path.join(project_root, "data", "multimodal_master_with_paths.csv")
output_csv = os.path.join(project_root, "data", "multimodal_master_clean.csv")

# Load your dataset
if not os.path.exists(input_csv):
    print(f"Error: Input CSV not found at {input_csv}")
    sys.exit(1)

df = pd.read_csv(input_csv)

# Required columns for model
required_cols = [
    "age", "sex", "diabetes", "hypertension",
    "qrisk3", "dm_htn_combined", "egfr",
    "fundus_image", "vessel_mask"
]

# Validate required columns exist
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"Error: Missing columns in CSV: {missing_cols}")
    sys.exit(1)

# Drop rows with missing eGFR
df_clean = df[df["egfr"].notna()].copy()
print(f"After removing missing eGFR: {len(df_clean)} rows (dropped {len(df) - len(df_clean)})")

# Drop NaN tabular values
df_clean = df_clean.dropna(subset=[
    "age", "sex", "diabetes", "hypertension",
    "qrisk3", "dm_htn_combined"
])
print(f"After removing NaN tabular values: {len(df_clean)} rows")

# Drop rows where images or masks do not exist on disk
original_len = len(df_clean)
df_clean = df_clean[
    df_clean["fundus_image"].apply(os.path.exists) &
    df_clean["vessel_mask"].apply(os.path.exists)
]
print(f"Removed {original_len - len(df_clean)} rows due to missing images/masks")

# Save final cleaned version
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df_clean.to_csv(output_csv, index=False)

print(f"\nCleaned dataset summary:")
print(f"  Original rows: {len(df)}")
print(f"  Final rows: {len(df_clean)}")
print(f"  Saved to: {output_csv}")
print(df_clean.head())
