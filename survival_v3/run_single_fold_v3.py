import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader, random_split

from survival_v3.src.datasets.multimodal_survival_dataset_v3 import (
    MultimodalSurvivalDatasetV3,
)
from survival_v3.src.model.multimodal_deepsurv_v3 import (
    MultimodalDeepSurvV3,
)
from survival_v3.src.train.train_survival_v3 import (
    train_one_epoch_v3,
    validate_v3,
)


# ------------------------------------------------------------
# Load YAML config
# ------------------------------------------------------------
def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

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

    # Load config
    cfg = load_config(args.config)

    csv_path = cfg["csv_path"]
    weight_path = cfg["retfound_weights"]

    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    max_epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])
    num_workers = int(cfg["num_workers"])
    val_split = float(cfg["val_split"])
    fusion_dim = int(cfg["fusion_dim"])
    dropout = float(cfg["dropout"])
    save_dir = cfg["save_dir"]

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # --------------------------------------------------------
    # Dataset + Dataloaders
    # --------------------------------------------------------
    dataset = MultimodalSurvivalDatasetV3(csv_path=csv_path)
    N = len(dataset)
    val_n = int(val_split * N)
    train_n = N - val_n

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    num_tab_features = len(dataset.tabular_features)

    model = MultimodalDeepSurvV3(
        weight_path=weight_path,
        num_tabular_features=num_tab_features,
        fusion_dim=fusion_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    best_c = -1
    bad_epochs = 0

    print("\n===== Starting Survival_v3 Single-Run Training =====\n")

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch_v3(model, train_loader, optimizer, device)
        val_loss, val_c = validate_v3(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_cindex={val_c:.4f}"
        )

        # Checkpoint on improvement
        if val_c > best_c:
            best_c = val_c
            bad_epochs = 0
            torch.save(model.state_dict(), save_path)
            print(f">>> Saved best model (C-index = {best_c:.4f})")
        else:
            bad_epochs += 1

        # Early stopping
        if bad_epochs >= patience:
            print("\n>>> Early stopping triggered.")
            break

    print("\n===== Training Complete =====")
    print(f"Best Validation C-index: {best_c:.4f}\n")


if __name__ == "__main__":
    main()
