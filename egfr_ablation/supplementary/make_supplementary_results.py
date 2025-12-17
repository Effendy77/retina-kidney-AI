from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ABLA_DIR = Path("experiments/ablation")
OUT_DIR = Path("supplementary/egfr_ablation_results")
TABLES = OUT_DIR / "tables"
FIGS = OUT_DIR / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# Update if you add new experiments
EXPERIMENTS = [
    "image_tabularONLY_no_qrisk",
    "image_tabular",
    "image_tabular_no_qrisk",
    "tabular",
    "tabular_no_retinal",
    "image",
    "mask",
]

def load_cv(exp: str) -> pd.DataFrame:
    p = ABLA_DIR / exp / "cv_summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing cv_summary.csv for {exp}: {p}")
    df = pd.read_csv(p)
    df["experiment"] = exp
    return df

def mean_sd(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("experiment")[["rmse", "mae", "r2"]]
    out = g.agg(["mean", "std"]).reset_index()
    # flatten columns
    out.columns = ["experiment"] + [f"{m}_{s}" for m, s in out.columns[1:]]
    return out

def format_mean_sd_table(ms: pd.DataFrame) -> pd.DataFrame:
    # Pretty formatting: "mean ± sd"
    rows = []
    for _, r in ms.iterrows():
        rows.append({
            "experiment": r["experiment"],
            "RMSE (mean ± SD)": f"{r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}",
            "MAE (mean ± SD)":  f"{r['mae_mean']:.4f} ± {r['mae_std']:.4f}",
            "R² (mean ± SD)":   f"{r['r2_mean']:.4f} ± {r['r2_std']:.4f}",
        })
    return pd.DataFrame(rows)

def save_barplot(ms: pd.DataFrame, metric: str, ylabel: str, filename: str):
    # Sort by metric mean (descending for r2, ascending for errors)
    ascending = metric in ["rmse", "mae"]
    ms2 = ms.sort_values(f"{metric}_mean", ascending=ascending).copy()

    x = np.arange(len(ms2))
    y = ms2[f"{metric}_mean"].values
    e = ms2[f"{metric}_std"].values

    plt.figure(figsize=(10, 4.8))
    plt.bar(x, y, yerr=e, capsize=4)
    plt.xticks(x, ms2["experiment"].values, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIGS / filename, dpi=300)
    plt.close()

def load_fold_metrics(exp: str) -> pd.DataFrame:
    # Reads fold*/metrics.csv (last row) to ensure per-fold values align with cv_summary
    exp_dir = ABLA_DIR / exp
    rows = []
    for fold in range(5):
        mpath = exp_dir / f"fold{fold}" / "metrics.csv"
        if not mpath.exists():
            continue
        m = pd.read_csv(mpath).iloc[-1]
        rows.append({
            "experiment": exp,
            "fold": fold,
            "rmse": float(m["rmse"]),
            "mae": float(m["mae"]),
            "r2": float(m["r2"]),
        })
    if len(rows) == 0:
        # fallback to cv_summary.csv if needed
        df = load_cv(exp).copy()
        return df[["experiment", "fold", "rmse", "mae", "r2"]]
    return pd.DataFrame(rows)

def paired_delta(exp_a: str, exp_b: str) -> pd.DataFrame:
    # delta = A - B per fold
    A = load_fold_metrics(exp_a).sort_values("fold")
    B = load_fold_metrics(exp_b).sort_values("fold")

    # sanity check
    assert list(A["fold"]) == list(B["fold"]), "Fold indices do not align."

    d = pd.DataFrame({
        "fold": A["fold"].values,
        "delta_rmse": A["rmse"].values - B["rmse"].values,
        "delta_mae":  A["mae"].values - B["mae"].values,
        "delta_r2":   A["r2"].values  - B["r2"].values,
    })
    return d

def plot_paired_delta(d: pd.DataFrame, filename: str):
    # Fold-wise paired deltas (each fold a point; connect to show consistency)
    plt.figure(figsize=(9, 4.6))
    x = d["fold"].values

    plt.plot(x, d["delta_r2"].values, marker="o")
    plt.axhline(0, linewidth=1)
    plt.xticks(x)
    plt.ylabel("Δ R² (A - B)")
    plt.xlabel("Fold")
    plt.tight_layout()
    plt.savefig(FIGS / filename, dpi=300)
    plt.close()

def pooled_predictions(exp: str) -> pd.DataFrame:
    # Concatenate fold predictions.csv
    exp_dir = ABLA_DIR / exp
    parts = []
    for fold in range(5):
        p = exp_dir / f"fold{fold}" / "predictions.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["fold"] = fold
            df["experiment"] = exp
            parts.append(df)
    if len(parts) == 0:
        raise FileNotFoundError(f"No predictions.csv found for {exp}")
    return pd.concat(parts, ignore_index=True)

def plot_pred_vs_true(pred_df: pd.DataFrame, y_col: str, yhat_col: str, filename: str):
    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(pred_df[y_col], pred_df[yhat_col], s=6, alpha=0.35)
    mn = min(pred_df[y_col].min(), pred_df[yhat_col].min())
    mx = max(pred_df[y_col].max(), pred_df[yhat_col].max())
    plt.plot([mn, mx], [mn, mx], linewidth=1)
    plt.xlabel("Observed eGFR")
    plt.ylabel("Predicted eGFR")
    plt.tight_layout()
    plt.savefig(FIGS / filename, dpi=300)
    plt.close()

def main():
    # -------------------------
    # Supplementary Table S1
    # -------------------------
    cv_all = pd.concat([load_cv(e) for e in EXPERIMENTS], ignore_index=True)
    ms = mean_sd(cv_all)
    ms_fmt = format_mean_sd_table(ms)

    ms.to_csv(TABLES / "Table_S1_ablation_mean_sd_raw.csv", index=False)
    ms_fmt.to_csv(TABLES / "Table_S1_ablation_mean_sd_formatted.csv", index=False)

    # -------------------------
    # Supplementary Table S2
    # -------------------------
    fold_all = pd.concat([load_fold_metrics(e) for e in EXPERIMENTS], ignore_index=True)
    fold_all = fold_all.sort_values(["experiment", "fold"])
    fold_all.to_csv(TABLES / "Table_S2_per_fold_metrics_all_experiments.csv", index=False)

    # -------------------------
    # Supplementary Table S3
    # Paired deltas: no_qrisk vs baseline (image_tabular)
    # -------------------------
    d = paired_delta("image_tabularONLY_no_qrisk", "image_tabular")
    d.to_csv(TABLES / "Table_S3_paired_fold_deltas_no_qrisk_vs_baseline.csv", index=False)

    # Also save summary of deltas
    d_summary = pd.DataFrame([{
        "delta_rmse_mean": d["delta_rmse"].mean(),
        "delta_rmse_sd": d["delta_rmse"].std(),
        "delta_mae_mean": d["delta_mae"].mean(),
        "delta_mae_sd": d["delta_mae"].std(),
        "delta_r2_mean": d["delta_r2"].mean(),
        "delta_r2_sd": d["delta_r2"].std(),
    }])
    d_summary.to_csv(TABLES / "Table_S3b_paired_delta_summary.csv", index=False)

    # -------------------------
    # Supplementary Figures
    # -------------------------
    save_barplot(ms, "r2", "R² (mean ± SD)", "Fig_S1_R2_mean_sd.png")
    save_barplot(ms, "mae", "MAE (mean ± SD)", "Fig_S2_MAE_mean_sd.png")
    save_barplot(ms, "rmse", "RMSE (mean ± SD)", "Fig_S2b_RMSE_mean_sd.png")

    plot_paired_delta(d, "Fig_S3_paired_delta_R2_no_qrisk_vs_baseline.png")

    # Optional: pooled prediction scatter for best model
    try:
        pred_best = pooled_predictions("image_tabularONLY_no_qrisk")
        # Try common column names
        cols = [c.lower() for c in pred_best.columns]
        # Heuristic mapping
        y_col = "target"
        yhat_col = "pred" 
        if y_col and yhat_col:
            plot_pred_vs_true(pred_best, y_col, yhat_col, "Fig_S4_pred_vs_true_best_model.png")
        else:
            print("[WARN] Could not infer prediction column names for Fig_S4.")
    except Exception as e:
        print(f"[WARN] Skipping Fig_S4: {e}")

    print("[DONE] Supplementary tables and figures saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
