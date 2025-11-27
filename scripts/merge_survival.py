import pandas as pd

# -----------------------------
# 1. Load both datasets
# -----------------------------
df_left = pd.read_csv("/home/fendy77/projects/retina-kidney-AI/data/cleaned_left_eye_only.csv")
df_master = pd.read_csv("/home/fendy77/projects/retina-kidney-AI/data/multimodal_master_CLEAN.csv")

print("Left-eye rows:", len(df_left))
print("Master rows:", len(df_master))

# -----------------------------
# 2. Standardize EID
# -----------------------------
if "eid_x" in df_master.columns:
    df_master.rename(columns={"eid_x": "eid"}, inplace=True)

# -----------------------------
# 3. Drop *_y leftover columns
# -----------------------------
y_cols = [c for c in df_master.columns if c.endswith("_y")]
df_master.drop(columns=y_cols, inplace=True, errors="ignore")

# -----------------------------
# 4. Rename ESRD columns in left CSV to match final names
# -----------------------------
rename_map = {
    "esrd_date_combined": "esrd_date",
    "esrd_2yr": "esrd_2yr",
    "esrd_5yr": "esrd_5yr",
    "time_to_event": "time_to_event"
}

for old, new in rename_map.items():
    if old in df_left.columns:
        df_left.rename(columns={old: new}, inplace=True)

# -----------------------------
# 5. Merge datasets (INNER JOIN)
# -----------------------------
df_merge = pd.merge(
    df_master,
    df_left[["eid", "left_image_filename", "esrd_date", "esrd_2yr", "esrd_5yr", "time_to_event"]],
    on="eid",
    how="inner"
)

print("\nMerged rows:", len(df_merge))

# -----------------------------
# 6. Rename *_x columns
# -----------------------------
for col in list(df_merge.columns):
    if col.endswith("_x") and col != "eid":
        df_merge.rename(columns={col: col.replace("_x", "")}, inplace=True)

# -----------------------------
# 7. Save final merged file
# -----------------------------
out_path = "/home/fendy77/projects/retina-kidney-AI/data/survival_multimodal_master.csv"
df_merge.to_csv(out_path, index=False)

print(f"\nSaved merged dataset to: {out_path}")
print(df_merge.head())
