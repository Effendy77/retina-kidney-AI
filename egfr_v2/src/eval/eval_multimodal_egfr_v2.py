import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.datasets.multimodal_dataset_v2 import MultimodalKidneyDatasetV2
from src.model.multimodal_fusion_v2 import MultimodalKidneyModelV2
from src.utils.metrics_regression_v2 import rmse, mae, r2_score


# ============================================================
# EVALUATE ONE FOLD
# ============================================================

def evaluate_one_fold(
    fold_dir,
    csv_path,
    image_root,
    mask_root,
    retfound_weights,
    num_workers=4,
    batch_size=32,
):
    """
    Evaluate a single fold using its best checkpoint.
    """

    print(f"\n[INFO] Evaluating fold: {fold_dir}")
    print(f"[INFO] Using RETFound weights: {retfound_weights}")

    # Paths
    best_model_path = os.path.join(fold_dir, "best_model.pth")
    pred_output_path = os.path.join(fold_dir, "fold_predictions.csv")
    metric_output_path = os.path.join(fold_dir, "fold_metrics_final.csv")

    # Dataset
    ds = MultimodalKidneyDatasetV2(csv_path, image_root, mask_root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalKidneyModelV2(
        weight_path=retfound_weights,
        num_tabular_features=10,
        fusion_dim=1024,
        dropout=0.2,
    )
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Inference
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            tab = batch["tabular"].to(device)
            target = batch["egfr"].to(device)

            pred = model(img, mask, tab).squeeze(1)

            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())

    # Compute metrics
    m = {
        "rmse": rmse(targets, preds),
        "mae": mae(targets, preds),
        "r2": r2_score(targets, preds),
    }

    # Save predictions
    pred_df = pd.DataFrame({"target": targets, "pred": preds})
    pred_df.to_csv(pred_output_path, index=False)

    # Save metrics
    metric_df = pd.DataFrame([m])
    metric_df.to_csv(metric_output_path, index=False)

    print(f"[INFO] Fold metrics: {m}")
    print(f"[INFO] Saved predictions → {pred_output_path}")
    print(f"[INFO] Saved metrics     → {metric_output_path}")

    return pred_df, m


# ============================================================
# EVALUATE ALL FOLDS (5-FOLD)
# ============================================================

def evaluate_5fold(
    root_dir="experiments/egfr_v2",
    image_root="data/images",
    mask_root="data/masks_raw_binary",
    retfound_weights="../retfound/RETFound_mae_natureCFP.pth",
    num_workers=4,
    batch_size=32,
):
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_preds = []
    all_metrics = []

    for fold_idx in range(5):
        fold_dir = os.path.join(root_dir, f"fold{fold_idx}")
        val_csv = os.path.join(fold_dir, "val.csv")

        fold_pred_df, fold_metrics = evaluate_one_fold(
            fold_dir=fold_dir,
            csv_path=val_csv,
            image_root=image_root,
            mask_root=mask_root,
            retfound_weights=retfound_weights,
            num_workers=num_workers,
            batch_size=batch_size,
        )

        fold_pred_df["fold"] = fold_idx
        all_preds.append(fold_pred_df)

        fold_metrics["fold"] = fold_idx
        all_metrics.append(fold_metrics)

    # ------------------------------------------------------------
    # SAVE COMBINED CSVs
    # ------------------------------------------------------------
    combined_pred = pd.concat(all_preds, ignore_index=True)
    combined_pred_path = os.path.join(results_dir, "all_predictions.csv")
    combined_pred.to_csv(combined_pred_path, index=False)

    summary_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(results_dir, "summary_final.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n[INFO] Saved combined predictions → {combined_pred_path}")
    print(f"[INFO] Saved summary metrics     → {summary_path}")

    # ------------------------------------------------------------
    # VISUALIZATIONS
    # ------------------------------------------------------------

    # Scatter: true vs pred
    plt.figure(figsize=(7, 7))
    plt.scatter(combined_pred["target"], combined_pred["pred"], alpha=0.4)
    plt.xlabel("True eGFR")
    plt.ylabel("Predicted eGFR")
    plt.title("Predicted vs True eGFR (All Folds)")
    plt.plot(
        [combined_pred["target"].min(), combined_pred["target"].max()],
        [combined_pred["target"].min(), combined_pred["target"].max()],
        "r--",
        label="Ideal Line",
    )
    plt.legend()
    plt.grid(True)
    scatter_path = os.path.join(results_dir, "scatter_pred_vs_true.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()

    # Residual plot
    plt.figure(figsize=(7, 7))
    residuals = combined_pred["pred"] - combined_pred["target"]
    plt.scatter(combined_pred["target"], residuals, alpha=0.4)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("True eGFR")
    plt.ylabel("Residual (Pred - True)")
    plt.title("Residual Plot (All Folds)")
    plt.grid(True)
    residuals_path = os.path.join(results_dir, "residuals_plot.png")
    plt.savefig(residuals_path, dpi=150)
    plt.close()

    print(f"[INFO] Saved plot → {scatter_path}")
    print(f"[INFO] Saved plot → {residuals_path}")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    evaluate_5fold()
