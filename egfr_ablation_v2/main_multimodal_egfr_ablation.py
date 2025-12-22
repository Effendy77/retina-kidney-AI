# egfr_ablation_v2/main_multimodal_egfr_ablation.py

import os
import argparse
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GroupKFold

import torch
from torch.utils.data import DataLoader

# -----------------------------
# ABLATION DATASET & MODEL
# -----------------------------
from egfr_ablation_v2.src.datasets.multimodal_dataset_ablation import MultimodalKidneyDatasetV2
from egfr_ablation_v2.src.model.multimodal_fusion_ablation import MultimodalKidneyModelV2

# -----------------------------
# TRAINING UTILITIES
# -----------------------------
from egfr_ablation_v2.src.train.train_multimodal_egfr_ablation import train_one_fold


# ============================================================
# GLOBAL SEED (REPRODUCIBLE)
# ============================================================

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    tabular_mode="baseline_plus_qrisk",
    silence_qrisk=False,
):
    set_seed(seed)

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

    df = pd.read_csv(csv_path)
    num_rows = len(df)
    print(f"[INFO] Loaded dataset with {num_rows} rows.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Use GroupKFold if eid exists (prevents leakage if multiple rows per participant)
    if "eid" in df.columns:
        splitter = GroupKFold(n_splits=5)
        split_iter = splitter.split(df, groups=df["eid"].values)
        print("[INFO] Using GroupKFold split by eid.")
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        split_iter = splitter.split(df)
        print("[INFO] Using KFold split by rows (no eid column found).")

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        print(f"\n========================")
        print(f" STARTING FOLD {fold_idx}")
        print(f"========================")

        fold_dir = os.path.join(output_root, f"fold{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        train_csv = os.path.join(fold_dir, "train.csv")
        val_csv   = os.path.join(fold_dir, "val.csv")
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        # Dataset objects WITH ablation flags + tabular_mode
        train_ds = MultimodalKidneyDatasetV2(
            csv_path=train_csv,
            image_root=image_root,
            mask_root=mask_root,
            use_image=use_image,
            use_mask=use_mask,
            use_tabular=use_tabular,
            use_retinal_features=use_retinal_features,
            tabular_mode=tabular_mode,
            silence_qrisk=silence_qrisk,
        )

        val_ds = MultimodalKidneyDatasetV2(
            csv_path=val_csv,
            image_root=image_root,
            mask_root=mask_root,
            use_image=use_image,
            use_mask=use_mask,
            use_tabular=use_tabular,
            use_retinal_features=use_retinal_features,
            tabular_mode=tabular_mode,
            silence_qrisk=silence_qrisk,
        )

        num_tabular = len(train_ds.tabular_features)
        print(f"[INFO] Fold {fold_idx}: num_tabular_features = {num_tabular}")
        print(f"[INFO] Fold {fold_idx}: tabular_features = {train_ds.tabular_features}")

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

        model = MultimodalKidneyModelV2(
            weight_path=weight_path,
            num_tabular_features=num_tabular,
            fusion_dim=1024,
            dropout=0.2,
            use_image=use_image,
            use_mask=use_mask,
            use_tabular=use_tabular,
            use_retinal_features=use_retinal_features,
        ).to(device)

        best_model_path  = os.path.join(fold_dir, "best_model.pth")
        pred_csv_path    = os.path.join(fold_dir, "predictions.csv")
        metrics_csv_path = os.path.join(fold_dir, "metrics.csv")

        try:
            best_info = train_one_fold(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=lr,
                epochs=epochs,
                save_path=best_model_path,
                pred_output_path=pred_csv_path,
                metric_output_path=metrics_csv_path,
                seed=seed,
            )
        except Exception as e:
            print(f"[ERROR] Fold {fold_idx} training failed: {e}")
            continue

        # Prefer trainer return (best epoch metrics)
        if isinstance(best_info, dict) and ("best_rmse" in best_info):
            fold_results.append({
                "fold": fold_idx,
                "best_epoch": best_info.get("best_epoch", None),
                "rmse": best_info["best_rmse"],
                "mae":  best_info["best_mae"],
                "r2":   best_info["best_r2"],
            })
        else:
            # Fallback: compute best epoch from metrics.csv
            if not os.path.exists(metrics_csv_path):
                print(f"[WARN] Metrics file not found for fold {fold_idx}. Skipping.")
                continue
            metrics_df = pd.read_csv(metrics_csv_path)
            best_row = metrics_df.loc[metrics_df["rmse"].idxmin()]
            fold_results.append({
                "fold": fold_idx,
                "best_epoch": int(best_row["epoch"]),
                "rmse": float(best_row["rmse"]),
                "mae":  float(best_row["mae"]),
                "r2":   float(best_row["r2"]),
            })

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

    parser.add_argument("--csv", type=str, default="egfr_ablation_v2/data/multimodal_master_BASELINEv2.csv")
    parser.add_argument("--images", type=str, default="/home/fendy77/data/retina_images")
    parser.add_argument("--masks", type=str, default="/home/fendy77/projects/retina-kidney-AI/data/masks_raw_binary")
    parser.add_argument("--retfound_weights", type=str, default="../retfound/RETFound_mae_natureCFP.pth")
    parser.add_argument("--outdir", type=str, default="egfr_ablation_v2/experiments/ablation")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)

    # Tabular mode (NEW, IMPORTANT)
    parser.add_argument(
        "--tabular_mode",
        type=str,
        default="baseline_plus_qrisk",
        choices=["qrisk_only", "baseline", "baseline_plus_qrisk"],
        help="Which tabular feature set to use."
    )

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

    # Logging setup
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
            raise SystemExit(1)

    log_msg("[INFO] All paths validated.\n")

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
            tabular_mode=args.tabular_mode,
            silence_qrisk=args.silence_qrisk,
        )
    except Exception as e:
        log_msg(f"[FATAL] Training failed: {e}")
        import traceback
        log_msg(traceback.format_exc())
        raise SystemExit(1)

    log_msg("[INFO] 5-Fold ablation training completed.")
    log_msg(f"[INFO] Summary at: {os.path.join(args.outdir, 'cv_summary.csv')}")
