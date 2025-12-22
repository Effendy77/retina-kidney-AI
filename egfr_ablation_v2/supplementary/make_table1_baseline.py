import pandas as pd
import numpy as np

# -----------------------------
# Paths
# -----------------------------
BASELINE_PATH = "/home/fendy77/projects/retina-kidney-AI/data/cleaned_left_eye_only.csv"
MODEL_COHORT_PATH = "/home/fendy77/projects/retina-kidney-AI/egfr_ablation/data/multimodal_master_CLEANv2.csv"
OUT_PATH = "/home/fendy77/projects/retina-kidney-AI/egfr_ablation/supplementary/Table1_baseline_characteristics.csv"

# -----------------------------
# Load data
# -----------------------------
df_base = pd.read_csv(BASELINE_PATH)
df_model = pd.read_csv(MODEL_COHORT_PATH)

# -----------------------------
# Filter baseline to analytic cohort used in modelling
# -----------------------------
if "eid" not in df_base.columns or "eid" not in df_model.columns:
    raise ValueError("Both files must contain 'eid' column for linkage.")

model_eids = set(df_model["eid"].unique())
df = df_base[df_base["eid"].isin(model_eids)].copy()

print(f"[INFO] Baseline total rows: {len(df_base)}")
print(f"[INFO] Model cohort rows: {len(df_model)}")
print(f"[INFO] Intersection (Table 1 N): {len(df)}")

# -----------------------------
# Helpers
# -----------------------------
def mean_sd(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return f"{x.mean():.1f} ± {x.std():.1f}" if len(x) else "NA"

def median_iqr(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "NA"
    q1, q3 = np.percentile(x, [25, 75])
    return f"{np.median(x):.1f} [{q1:.1f}, {q3:.1f}]"

def n_pct_from_binary(x):
    x = pd.to_numeric(x, errors="coerce")
    x = x.dropna()
    if len(x) == 0:
        return "NA"
    n = int((x == 1).sum())
    pct = 100 * (x == 1).mean()
    return f"{n} ({pct:.1f}%)"

# -----------------------------
# Column mapping (from your baseline file)
# -----------------------------
AGE_COL = "Age_at_baseline"
SEX_COL = "Sex_0F_1M"          # 0=F, 1=M
EGFR_COL = "egfr"              # (also has 'eGFR' column; keep consistent with model)
QRISK_COL = "qrisk3"

# For diabetes/hypertension, prefer prevalent/binary if available
DIAB_COL_CANDIDATES = ["diabetes_prevalent"]
HTN_COL_CANDIDATES  = ["hypertension_prevalent"]

def pick_first_existing(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

DIAB_COL = pick_first_existing(DIAB_COL_CANDIDATES)
HTN_COL  = pick_first_existing(HTN_COL_CANDIDATES)

if AGE_COL not in df.columns:
    raise KeyError(f"Missing age column: {AGE_COL}")
if SEX_COL not in df.columns:
    raise KeyError(f"Missing sex column: {SEX_COL}")
if EGFR_COL not in df.columns:
    raise KeyError(f"Missing eGFR column: {EGFR_COL}")

# Convert sex to binary male indicator (1=male)
male = pd.to_numeric(df[SEX_COL], errors="coerce")

# Diabetes / HTN binary
diab = pd.to_numeric(df[DIAB_COL], errors="coerce") if DIAB_COL else None
htn  = pd.to_numeric(df[HTN_COL], errors="coerce") if HTN_COL else None

# -----------------------------
# Build Table 1
# -----------------------------
rows = []

rows.append(("Number of participants", f"{len(df)}"))
rows.append(("Age, years (mean ± SD)", mean_sd(df[AGE_COL])))
rows.append(("Male sex, n (%)", n_pct_from_binary(male)))

if diab is not None:
    rows.append(("Diabetes, n (%)", n_pct_from_binary(diab)))
else:
    rows.append(("Diabetes, n (%)", "NA"))

if htn is not None:
    rows.append(("Hypertension, n (%)", n_pct_from_binary(htn)))
else:
    rows.append(("Hypertension, n (%)", "NA"))

rows.append(("eGFR, mL/min/1.73m² (mean ± SD)", mean_sd(df[EGFR_COL])))
rows.append(("eGFR, mL/min/1.73m² (median [IQR])", median_iqr(df[EGFR_COL])))

if QRISK_COL in df.columns:
    rows.append(("QRISK3 score (mean ± SD)", mean_sd(df[QRISK_COL])))

# Optional: ethnicity (if you want it in Table 1)
if "Ethnicity_Label_QRISK3" in df.columns:
    eth = df["Ethnicity_Label_QRISK3"].value_counts(dropna=False)
    # keep concise: White vs Non-White if needed; otherwise omit
    # rows.append(("Ethnicity (categories)", "See Supplementary"))  # optional

table1 = pd.DataFrame(rows, columns=["Characteristic", "Value"])
table1.to_csv(OUT_PATH, index=False)

print(f"[DONE] Saved Table 1 → {OUT_PATH}")
print(table1)
