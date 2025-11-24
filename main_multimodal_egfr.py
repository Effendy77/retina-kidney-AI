import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from src.datasets.multimodal_dataset import MultimodalKidneyDataset
from src.model.multimodal_fusion import MultimodalKidneyModel


# -------------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)
        tab = batch["tabular"].to(device)
        target = batch["egfr"].to(device).unsqueeze(1)

        optimizer.zero_grad()

        with autocast():
            pred = model(img, mask, tab)
            loss = criterion(pred, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * img.size(0)

    return total_loss / len(dataloader.dataset)


# -------------------------------------------------------
# VALIDATION FUNCTION
# -------------------------------------------------------
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_mae = 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            tab = batch["tabular"].to(device)
            target = batch["egfr"].to(device).unsqueeze(1)

            with autocast():
                pred = model(img, mask, tab)
                loss = criterion(pred, target)
                mae = torch.mean(torch.abs(pred - target))

            total_loss += loss.item() * img.size(0)
            total_mae += mae.item() * img.size(0)

    return (
        total_loss / len(dataloader.dataset),
        total_mae / len(dataloader.dataset),
    )


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--mask_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="checkpoint_egfr.pth")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # ------------------------------
    # LOAD DATA
    # ------------------------------
    dataset = MultimodalKidneyDataset(
        csv_path=args.csv,
        image_root=args.image_root,
        mask_root=args.mask_root,
    )

    # Train/Val split
    n = len(dataset)
    split = int(0.8 * n)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [split, n - split],
        generator=torch.Generator().manual_seed(42)  # reproducible split
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ------------------------------
    # MODEL
    # ------------------------------
    model = MultimodalKidneyModel(
        weight_path=args.weights,
        num_tabular_features=len(dataset.tabular_features),
        fusion_dim=1024,
        output_type="regression"
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    best_val_mae = float("inf")

    # ------------------------------
    # TRAINING LOOP
    # ------------------------------
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, scaler, device)
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_mae={val_mae:.4f}"
        )

        # Save best checkpoint based on MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), args.save_path)
            print(f">>> Saved new best model with MAE {best_val_mae:.4f}")

    print("Training complete.")
    print(f"Best Validation MAE: {best_val_mae:.4f}")


if __name__ == "__main__":
    main()
