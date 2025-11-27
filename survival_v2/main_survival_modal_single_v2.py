import argparse
import yaml
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split

from survival_v2.src.datasets.multimodal_survival_dataset_v2 import (
    MultimodalSurvivalDatasetV2,
)
from survival_v2.src.model.multimodal_deepsurv_v2 import (
    MultimodalDeepSurvV2,
)
from survival_v2.src.train.train_survival_v2 import (
    train_one_epoch_v2,
    validate_v2,
)


def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="survival_v2/configs/esrd_survival_v2.yaml"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------
    cfg = load_config(args.config)

    # Type-cast and validate config values
    csv_path = os.path.abspath(str(cfg["csv_path"]))
    weight_path = os.path.abspath(str(cfg["retfound_weights"]))
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    max_epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])
    val_split = float(cfg["val_split"])
    num_workers = int(cfg["num_workers"])
    save_dir = os.path.abspath(str(cfg["save_dir"]))
    fusion_dim = int(cfg.get("fusion_dim", 512))
    dropout = float(cfg.get("dropout", 0.2))

    # Validate paths exist
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"RETFound weights not found: {weight_path}")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # ------------------------------------------------------------
    # Dataset + Dataloaders
    # ------------------------------------------------------------
    dataset = MultimodalSurvivalDatasetV2(csv_path=csv_path)

    n = len(dataset)
    val_n = int(val_split * n)
    train_n = n - val_n
    train_set, val_set = random_split(dataset, [train_n, val_n])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = MultimodalDeepSurvV2(
        weight_path=weight_path,
        num_tabular_features=len(dataset.tabular_features),
        fusion_dim=fusion_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # ------------------------------------------------------------
    # Training with Early Stopping
    # ------------------------------------------------------------
    best_c = -1.0
    bad_epochs = 0

    print("\n===== Starting Training (v2 DeepSurv) =====\n")

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch_v2(
            model, train_loader, optimizer, device
        )
        val_loss, val_c = validate_v2(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_cindex={val_c:.4f}"
        )

        # Early-stopping logic
        if val_c > best_c:
            best_c = val_c
            bad_epochs = 0
            try:
                torch.save(model.state_dict(), save_path)
                print(f">>> Saved best model (C-index = {best_c:.4f})")
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("\n>>> Early stopping triggered.")
            break

    print("\n===== Training Complete =====")
    print(f"Best Validation C-index: {best_c:.4f}\n")


if __name__ == "__main__":
    main()
