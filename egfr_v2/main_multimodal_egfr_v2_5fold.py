import os
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader

# -----------------------------
# V2 DATASET & MODEL
# -----------------------------
from src.datasets.multimodal_dataset_v2 import MultimodalKidneyDatasetV2
from src.model.multimodal_fusion_v2 import MultimodalKidneyModelV2

# -----------------------------
# TRAINING UTILITIES
# -----------------------------
from src.train.train_multimodal_egfr_v2 import train_one_fold
from src.utils.metrics_regression_v2 import rmse, mae, r2_score



# ============================================================
# GLOBAL SEED (REPRODUCIBLE)
# ============================================================

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# MAIN EXECUTION — 5-FOLD CV
# ============================================================

def run_5fold_training(
    csv_path,
    image_root,
    mask_root,
    weight_path,
    output_root,
    batch_size=16,
    lr=1e-4,
    epochs=20,
    num_workers=4,
    seed=123,
):
    set_seed(seed)

    # Make output directory
    os.makedirs(output_root, exist_ok=True)

    # Validate input paths exist
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image root not found: {image_root}")
    if not os.path.isdir(mask_root):
        raise FileNotFoundError(f"Mask root not found: {mask_root}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"RETFound weights not found: {weight_path}")

    # Load entire CSV
    df = pd.read_csv(csv_path)
    num_rows = len(df)
    print(f"[INFO] Loaded dataset with {num_rows} rows.")

    # Determine device once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Prepare 5-fold splitter
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
         print(f"\n========================")
         print(f" STARTING FOLD {fold_idx}")
         print(f"========================")

         fold_dir = os.path.join(output_root, f"fold{fold_idx}")
         os.makedirs(fold_dir, exist_ok=True)

         # Create train/val split CSVs
         train_df = df.iloc[train_idx].reset_index(drop=True)
         val_df   = df.iloc[val_idx].reset_index(drop=True)

         train_csv = os.path.join(fold_dir, "train.csv")
         val_csv   = os.path.join(fold_dir, "val.csv")

         train_df.to_csv(train_csv, index=False)
         val_df.to_csv(val_csv, index=False)

         # Dataset objects
         train_ds = MultimodalKidneyDatasetV2(train_csv, image_root, mask_root)
         val_ds   = MultimodalKidneyDatasetV2(val_csv,   image_root, mask_root)

         # Derive num_tabular_features from dataset
         num_tabular = len(train_ds.tabular_features)
         print(f"[INFO] Fold {fold_idx}: num_tabular_features = {num_tabular}")

         train_loader = DataLoader(
             train_ds,
             batch_size=batch_size,
             shuffle=True,
             num_workers=num_workers,
             pin_memory=True,
         )

         val_loader = DataLoader(
             val_ds,
             batch_size=batch_size,
             shuffle=False,
             num_workers=num_workers,
             pin_memory=True,
         )

         # Build V2 Model
         model = MultimodalKidneyModelV2(
             weight_path=weight_path,
             num_tabular_features=num_tabular,
             fusion_dim=1024,
             dropout=0.2,
         )
         model = model.to(device)

         # Train this fold
         best_model_path  = os.path.join(fold_dir, "best_model.pth")
         pred_csv_path    = os.path.join(fold_dir, "predictions.csv")
         metrics_csv_path = os.path.join(fold_dir, "metrics.csv")

         try:
             train_one_fold(
                 model=model,
                 train_loader=train_loader,
                 val_loader=val_loader,
                 lr=lr,
                 epochs=epochs,
                 save_path=best_model_path,
                 pred_output_path=pred_csv_path,
                 metric_output_path=metrics_csv_path,
             )
         except Exception as e:
             print(f"[ERROR] Fold {fold_idx} training failed: {e}")
             continue

         # Load metrics for summary
         if not os.path.exists(metrics_csv_path):
             print(f"[WARN] Metrics file not found for fold {fold_idx}. Skipping.")
             continue

         m = pd.read_csv(metrics_csv_path).iloc[-1]
         fold_results.append({
             "fold": fold_idx,
             "rmse": m["rmse"],
             "mae":  m["mae"],
             "r2":   m["r2"],
         })

    # SAVE FINAL SUMMARY (outside loop, inside function)
    if len(fold_results) == 0:
        print("[ERROR] No folds completed successfully. Exiting.")
        return

    summary_path = os.path.join(output_root, "cv_summary.csv")
    summary_df = pd.DataFrame(fold_results)
    summary_df.to_csv(summary_path, index=False)

    # Print aggregate stats
    print(f"\n===========================================")
    print(f" 5-FOLD TRAINING COMPLETE!")
    print(f" Completed {len(fold_results)} / 5 folds")
    print(f" Mean RMSE: {summary_df['rmse'].mean():.4f} ± {summary_df['rmse'].std():.4f}")
    print(f" Mean MAE:  {summary_df['mae'].mean():.4f} ± {summary_df['mae'].std():.4f}")
    print(f" Mean R²:   {summary_df['r2'].mean():.4f} ± {summary_df['r2'].std():.4f}")
    print(f" Summary saved at: {summary_path}")
    print(f"===========================================\n")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="5-Fold Cross-Validation for Multimodal eGFR Prediction (V2)"
    )

    parser.add_argument("--csv", type=str,
                        default="data/multimodal_master_CLEANv2.csv",
                        help="Path to CSV with eGFR labels and image/mask paths")

    parser.add_argument("--images", type=str,
                        default="data/images",
                        help="Root directory containing fundus images")

    parser.add_argument("--masks", type=str,
                        default="data/masks",
                        help="Root directory containing vessel masks")

    parser.add_argument("--retfound_weights", type=str,
                        default="../retfound/RETFound_mae_natureCFP.pth",
                        help="Path to RETFound pretrained weights")

    parser.add_argument("--outdir", type=str,
                        default="experiments/egfr_v2",
                        help="Output directory for fold results and summary")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

     # Setup output directory and logging
os.makedirs(args.outdir, exist_ok=True)
from datetime import datetime
log_file = os.path.join(args.outdir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def log_msg(msg):
    """Print to stdout and append to log file."""
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

# Validate input paths at CLI entry point
log_msg("[INFO] Validating input paths...")

# Define file vs directory checks
path_checks = {
    "--csv": (args.csv, "file"),
    "--images": (args.images, "dir"),
    "--masks": (args.masks, "dir"),
    "--retfound_weights": (args.retfound_weights, "file"),
}

for path_arg, (path_val, check_type) in path_checks.items():
    if check_type == "file":
        exists = os.path.exists(path_val)
    else:
        exists = os.path.isdir(path_val)
    
    if not exists:
        log_msg(f"[ERROR] {check_type.upper()} not found: {path_arg} = {path_val}")
        exit(1)
log_msg("[INFO] All input paths validated.\n")

# Validate CSV is not empty
try:
    df_check = pd.read_csv(args.csv)
    if len(df_check) == 0:
        log_msg("[ERROR] CSV is empty. Exiting.")
        exit(1)
    log_msg(f"[INFO] CSV contains {len(df_check)} rows.\n")
except Exception as e:
    log_msg(f"[ERROR] Failed to read CSV: {e}")
    exit(1)

# Log system info
log_msg(f"[INFO] PyTorch version: {torch.__version__}")
log_msg(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log_msg(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
log_msg(f"[INFO] Output directory: {args.outdir}")
log_msg(f"[INFO] Log file: {log_file}\n")

# Run training with error handling
try:
    run_5fold_training(
        csv_path=args.csv,
        image_root=args.images,
        mask_root=args.masks,
        weight_path=args.retfound_weights,
        output_root=args.outdir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        num_workers=args.num_workers,
        seed=args.seed,
    )
except Exception as e:
    log_msg(f"\n[FATAL] Training failed with exception:")
    log_msg(f"  {type(e).__name__}: {e}")
    import traceback
    log_msg(traceback.format_exc())
    exit(1)

log_msg("[INFO] 5-Fold training script completed successfully.")
log_msg(f"[INFO] Summary saved at: {os.path.join(args.outdir, 'cv_summary.csv')}")
