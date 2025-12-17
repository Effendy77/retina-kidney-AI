import os
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader

# -----------------------------
# ABLATION DATASET & MODEL
# -----------------------------
from src.datasets.multimodal_dataset_ablation import MultimodalKidneyDatasetV2
from src.model.multimodal_fusion_ablation import MultimodalKidneyModelV2

# -----------------------------
# TRAINING UTILITIES
# -----------------------------
from src.train.train_multimodal_egfr_ablation import train_one_fold
from src.utils.metrics_regression_ablation import rmse, mae, r2_score


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
    use_image=True,
    use_mask=True,
    use_tabular=True,
    use_retinal_features=True,
    silence_qrisk=False,     # ← ADDED PARAMETER
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

        # Dataset objects WITH ablation flags
        train_ds = MultimodalKidneyDatasetV2(
            train_csv, image_root, mask_root,
            use_image=use_image,
            use_mask=use_mask,
            use_tabular=use_tabular,
            use_retinal_features=use_retinal_features,
            silence_qrisk=silence_qrisk,
        )

        val_ds = MultimodalKidneyDatasetV2(
            val_csv, image_root, mask_root,
            use_image=use_image,
            use_mask=use_mask,
            use_tabular=use_tabular,
            use_retinal_features=use_retinal_features,
            silence_qrisk=silence_qrisk,
        )

        # Derive num_tabular_features
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

        # Build Model WITH ablation flags
        model = MultimodalKidneyModelV2(
            weight_path=weight_path,
            num_tabular_features=num_tabular,
            fusion_dim=1024,
            dropout=0.2,
            use_image=use_image,
            use_mask=use_mask,
            use_tabular=use_tabular,
            use_retinal_features=use_retinal_features,
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

        # Load metrics
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

    # SAVE FINAL SUMMARY
    if len(fold_results) == 0:
        print("[ERROR] No folds completed successfully. Exiting.")
        return

    summary_path = os.path.join(output_root, "cv_summary.csv")
    summary_df = pd.DataFrame(fold_results)
    summary_df.to_csv(summary_path, index=False)

    print("\n===========================================")
    print(" 5-FOLD TRAINING COMPLETE!")
    print(f" Completed {len(fold_results)} / 5 folds")
    print(f" Mean RMSE: {summary_df['rmse'].mean():.4f} ± {summary_df['rmse'].std():.4f}")
    print(f" Mean MAE:  {summary_df['mae'].mean():.4f} ± {summary_df['mae'].std():.4f}")
    print(f" Mean R²:   {summary_df['r2'].mean():.4f} ± {summary_df['r2'].std():.4f}")
    print(f" Summary saved at: {summary_path}")
    print("===========================================\n")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation 5-Fold CV for Multimodal eGFR Prediction"
    )

    parser.add_argument("--csv", type=str, default="data/multimodal_master_CLEANv2.csv")
    parser.add_argument("--images", type=str, default="data/images")
    parser.add_argument("--masks", type=str, default="data/masks")
    parser.add_argument("--retfound_weights", type=str,
                        default="../retfound/RETFound_mae_natureCFP.pth")
    parser.add_argument("--outdir", type=str, default="experiments/ablation")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)

    # =====================================================
    # CORRECT BOOLEAN FLAGS (store_true / store_false)
    # =====================================================

    # IMAGE branch
    parser.add_argument("--use_image", action="store_true", dest="use_image")
    parser.add_argument("--no_image", action="store_false", dest="use_image")
    parser.set_defaults(use_image=True)

    # MASK branch
    parser.add_argument("--use_mask", action="store_true", dest="use_mask")
    parser.add_argument("--no_mask", action="store_false", dest="use_mask")
    parser.set_defaults(use_mask=True)

    # TABULAR branch
    parser.add_argument("--use_tabular", action="store_true", dest="use_tabular")
    parser.add_argument("--no_tabular", action="store_false", dest="use_tabular")
    parser.set_defaults(use_tabular=True)

    # RETINAL SCALAR FEATURES (within tabular)
    parser.add_argument("--use_retinal_features", action="store_true", dest="use_retinal_features")
    parser.add_argument("--no_retinal_features", action="store_false", dest="use_retinal_features")
    parser.set_defaults(use_retinal_features=True)
    
    # QRISK3 ABLATION
    parser.add_argument(
        "--silence_qrisk",
        action="store_true",
        help="Silence QRISK3 feature (set to zero) for ablation study"
    )

    

    args = parser.parse_args()
    print("[INFO] Parsed arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")    
    print("")   
        
    # Logging Setup
    os.makedirs(args.outdir, exist_ok=True)
    from datetime import datetime
    log_file = os.path.join(args.outdir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    def log_msg(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    # Path validation
    log_msg("[INFO] Validating input paths...")

    checks = {
        "--csv": (args.csv, "file"),
        "--images": (args.images, "dir"),
        "--masks": (args.masks, "dir"),
        "--retfound_weights": (args.retfound_weights, "file"),
    }

    for arg, (path, typ) in checks.items():
        ok = os.path.exists(path) if typ == "file" else os.path.isdir(path)
        if not ok:
            log_msg(f"[ERROR] {typ} not found: {arg} = {path}")
            exit(1)

    log_msg("[INFO] All paths validated.\n")

    # Run training
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
            use_image=args.use_image,
            use_mask=args.use_mask,
            use_tabular=args.use_tabular,
            use_retinal_features=args.use_retinal_features,
            silence_qrisk=args.silence_qrisk,
        )
    except Exception as e:
        log_msg(f"[FATAL] Training failed: {e}")
        import traceback
        log_msg(traceback.format_exc())
        exit(1)

    log_msg("[INFO] 5-Fold ablation training completed.")
    log_msg(f"[INFO] Summary at: {os.path.join(args.outdir, 'cv_summary.csv')}")
