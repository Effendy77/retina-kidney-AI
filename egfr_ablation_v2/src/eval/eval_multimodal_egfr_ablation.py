import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

# Correct imports
from src.datasets.multimodal_dataset_ablation import MultimodalKidneyDatasetV2
from src.model.multimodal_fusion_ablation import MultimodalKidneyModelV2
from src.utils.metrics_regression_ablation import rmse, mae, r2_score


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
    use_image=True,
    use_mask=True,
    use_tabular=True,
    use_retinal_features=True,
):
    print(f"\n[INFO] Evaluating fold: {fold_dir}")
    print(f"[INFO] Using RETFound weights: {retfound_weights}")

    # Validate paths
    if not os.path.exists(fold_dir):
        print(f"[ERROR] Fold directory not found: {fold_dir}")
        return None, None

    if not os.path.exists(csv_path):
        print(f"[ERROR] Val CSV not found: {csv_path}")
        return None, None

    best_model_path = os.path.join(fold_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        print(f"[ERROR] Missing checkpoint: {best_model_path}")
        return None, None

    pred_output_path = os.path.join(fold_dir, "fold_predictions.csv")
    metric_output_path = os.path.join(fold_dir, "fold_metrics_final.csv")

    # Dataset ----------------------------------------------------
    try:
        ds = MultimodalKidneyDatasetV2(
            csv_path,
            image_root,
            mask_root,
            use_image=use_image,
            use_mask=use_mask,
            use_tabular=use_tabular,
            use_retinal_features=use_retinal_features,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return None, None

    if len(ds) == 0:
        print(f"[ERROR] Dataset empty: {csv_path}")
        return None, None

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Model ------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = MultimodalKidneyModelV2(
            weight_path=retfound_weights,
            num_tabular_features=len(ds.tabular_features),
            fusion_dim=1024,
            dropout=0.2,
        )
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None, None

    model = model.to(device)
    model.eval()

    # Inference --------------------------------------------------
    preds, targets = [], []

    try:
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(device)
                mask = batch["mask"].to(device)
                tab = batch["tabular"].to(device)
                target = batch["egfr"].to(device)

                pred = model(img, mask, tab).squeeze(1)

                preds.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return None, None

    # Metrics ----------------------------------------------------
    metrics = {
        "rmse": rmse(targets, preds),
        "mae": mae(targets, preds),
        "r2": r2_score(targets, preds),
    }

    # Save outputs
    pd.DataFrame({"target": targets, "pred": preds}).to_csv(pred_output_path, index=False)
    pd.DataFrame([metrics]).to_csv(metric_output_path, index=False)

    print(f"[INFO] Fold metrics: {metrics}")
    print(f"[INFO] Saved predictions → {pred_output_path}")
    print(f"[INFO] Saved metrics     → {metric_output_path}")

    return pd.DataFrame({"target": targets, "pred": preds}), metrics


# ============================================================
# EVALUATE ALL FOLDS
# ============================================================

def evaluate_5fold(
    root_dir,
    image_root,
    mask_root,
    retfound_weights,
    num_workers,
    batch_size,
    use_image,
    use_mask,
    use_tabular,
    use_retinal_features,
    log_file=None,
):
    def log_msg(msg):
        print(msg)
        if log_file:
            with open(log_file, "a") as f:
                f.write(msg + "\n")

    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    log_msg(f"[INFO] Starting 5-fold evaluation...")
    log_msg(f"[INFO] Using root: {root_dir}")
    log_msg(f"[INFO] Output: {results_dir}\n")

    all_preds, all_metrics = [], []
    completed = 0

    for fold_idx in range(5):
        fold_dir = os.path.join(root_dir, f"fold{fold_idx}")
        val_csv = os.path.join(fold_dir, "val.csv")

        pred_df, metrics = evaluate_one_fold(
            fold_dir,
            val_csv,
            image_root,
            mask_root,
            retfound_weights,
            num_workers,
            batch_size,
            use_image,
            use_mask,
            use_tabular,
            use_retinal_features,
        )

        if pred_df is None:
            log_msg(f"[WARN] Fold {fold_idx} skipped.")
            continue

        pred_df["fold"] = fold_idx
        all_preds.append(pred_df)

        metrics["fold"] = fold_idx
        all_metrics.append(metrics)
        completed += 1

    if completed == 0:
        log_msg("[ERROR] No folds completed.")
        return

    # Save all results -----------------------------------------
    combined = pd.concat(all_preds, ignore_index=True)
    combined.to_csv(os.path.join(results_dir, "all_predictions.csv"), index=False)

    summary = pd.DataFrame(all_metrics)
    summary.to_csv(os.path.join(results_dir, "summary_final.csv"), index=False)

    log_msg(f"[INFO] Completed folds: {completed}/5")
    log_msg(f"[INFO] Summary saved.")

    # Plots -----------------------------------------------------
    plt.figure(figsize=(7,7))
    plt.scatter(combined["target"], combined["pred"], alpha=0.4)
    plt.plot(
        [combined["target"].min(), combined["target"].max()],
        [combined["target"].min(), combined["target"].max()],
        "r--"
    )
    plt.xlabel("True eGFR")
    plt.ylabel("Predicted eGFR")
    plt.title("Predicted vs True eGFR")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "scatter_pred_vs_true.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7,7))
    residuals = combined["pred"] - combined["target"]
    plt.scatter(combined["target"], residuals, alpha=0.4)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("True eGFR")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "residuals_plot.png"), dpi=150)
    plt.close()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", required=True,
                        choices=["image", "mask", "tabular", "tabular_no_retinal"])

    parser.add_argument("--root_dir", default="experiments/ablation")
    parser.add_argument("--image_root", default="data/images")
    parser.add_argument("--mask_root", default="data/masks")
    parser.add_argument("--retfound_weights", default="../retfound/RETFound_mae_natureCFP.pth")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    configs = {
        "image": {"use_image": True, "use_mask": False, "use_tabular": False, "use_retinal_features": False},
        "mask": {"use_image": False, "use_mask": True, "use_tabular": False, "use_retinal_features": False},
        "tabular": {"use_image": False, "use_mask": False, "use_tabular": True, "use_retinal_features": True},
        "tabular_no_retinal": {"use_image": False, "use_mask": False, "use_tabular": True, "use_retinal_features": False},
    }

    cfg = configs[args.mode]

    mode_dir = os.path.join(args.root_dir, args.mode)
    log_file = os.path.join(mode_dir, f"eval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    evaluate_5fold(
        root_dir=mode_dir,
        image_root=args.image_root,
        mask_root=args.mask_root,
        retfound_weights=args.retfound_weights,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        use_image=cfg["use_image"],
        use_mask=cfg["use_mask"],
        use_tabular=cfg["use_tabular"],
        use_retinal_features=cfg["use_retinal_features"],
        log_file=log_file,
    )
