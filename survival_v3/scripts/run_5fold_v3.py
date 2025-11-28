import argparse
import yaml
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# V3 imports
from survival_v3.src.datasets.multimodal_survival_dataset_v3 import (
    MultimodalSurvivalDatasetV3,
)
from survival_v3.src.model.multimodal_deepsurv_v3 import MultimodalDeepSurvV3
from survival_v3.src.train.train_survival_v3 import (
    train_one_epoch_v3,
    validate_v3,
)


# ------------------------------------------------------------
# Load config YAML
# ------------------------------------------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="survival_v3/configs/esrd_survival_v3.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    csv_path = os.path.abspath(cfg["csv_path"])
    weight_path = os.path.abspath(cfg["retfound_weights"])
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    max_epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])
    num_workers = int(cfg["num_workers"])
    fusion_dim = int(cfg["fusion_dim"])
    dropout = float(cfg["dropout"])

    save_root = os.path.abspath(cfg["save_dir"] + "_5fold")
    os.makedirs(save_root, exist_ok=True)

    # Load full CSV
    df = pd.read_csv(csv_path)
    event_col = "event_occurred"

    # Sanity check required columns
    required_cols = ["fundus_image", "vessel_mask", "time_to_event", event_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Stratify by ESRD event
    y = df[event_col].values

    # Setup 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    fold_idx = 0
    fold_results = []

    # --------------------------------------------------------
    # FOLD LOOP
    # --------------------------------------------------------
    for train_index, val_index in skf.split(df, y):
        fold_idx += 1
        print(f"\n========== Fold {fold_idx} ==========\n")

        train_df = df.iloc[train_index].reset_index(drop=True)
        val_df = df.iloc[val_index].reset_index(drop=True)

        # Save fold-level CSVs
        train_path = f"{save_root}/fold{fold_idx}_train.csv"
        val_path = f"{save_root}/fold{fold_idx}_val.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        # Create datasets
        train_ds = MultimodalSurvivalDatasetV3(train_path)
        val_ds = MultimodalSurvivalDatasetV3(val_path)

        # Dataloaders
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

        # Initialise model
        model = MultimodalDeepSurvV3(
            weight_path=weight_path,
            num_tabular_features=len(train_ds.tabular_features),
            fusion_dim=fusion_dim,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        best_c = -1.0
        bad_epochs = 0

        fold_ckpt = f"{save_root}/fold{fold_idx}_best.pth"
        fold_metrics_path = f"{save_root}/fold{fold_idx}_metrics.csv"
        fold_risk_path = f"{save_root}/fold{fold_idx}_risk_scores.csv"

        history = []

        # ----------------------------------------------------
        # EPOCH LOOP
        # ----------------------------------------------------
        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch_v3(model, train_loader, optimizer, device)
            val_loss, val_c = validate_v3(model, val_loader, device)

            print(
                f"[Fold {fold_idx}] Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_cindex={val_c:.4f}"
            )

            history.append([epoch, train_loss, val_loss, val_c])

            # Checkpoint if improved
            if val_c > best_c:
                best_c = val_c
                bad_epochs = 0
                torch.save(model.state_dict(), fold_ckpt)
            else:
                bad_epochs += 1

            # Early stopping
            if bad_epochs >= patience:
                print(f">>> Early stopping on Fold {fold_idx}")
                break

        # Save metrics
        hist_df = pd.DataFrame(
            history,
            columns=["epoch", "train_loss", "val_loss", "val_cindex"],
        )
        hist_df.to_csv(fold_metrics_path, index=False)

        # ----------------------------------------------------
        # Compute RISK SCORES for validation set
        # ----------------------------------------------------
        print(f">>> Computing risk scores for Fold {fold_idx}")

        try:
            model.load_state_dict(torch.load(fold_ckpt, map_location=device))
        except Exception as e:
            print(f"Warning: Could not load best checkpoint: {e}")

        model.eval()

        all_eids, all_risks, all_times, all_events = [], [], [], []

        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                mask = batch["mask"].to(device)
                tab = batch["tabular"].to(device)
                time = batch["time"]
                event = batch["event"]

                risk = model(img, mask, tab).cpu()

                # EID support
                if "eid" in batch:
                    all_eids.extend(batch["eid"])
                else:
                    all_eids.extend([None] * len(risk))

                all_risks.extend(risk.numpy())
                all_times.extend(time.numpy())
                all_events.extend(event.numpy())

        out = pd.DataFrame({
            "eid": all_eids,
            "risk_score": all_risks,
            "time_to_event": all_times,
            "event_occurred": all_events,
        })
        out.to_csv(fold_risk_path, index=False)

        print(f"[Fold {fold_idx}] Best C-index = {best_c:.4f}")
        fold_results.append(best_c)

    # --------------------------------------------------------
    # Summary of CV results
    # --------------------------------------------------------
    print("\n===== 5-Fold Summary =====")
    for i, c in enumerate(fold_results):
        print(f"Fold {i+1}: C-index = {c:.4f}")

    print(
        f"Mean C-index = {pd.Series(fold_results).mean():.4f} ± "
        f"{pd.Series(fold_results).std():.4f}"
    )


if __name__ == "__main__":
    main()
