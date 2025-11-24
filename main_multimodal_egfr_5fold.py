import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import KFold

from src.datasets.multimodal_dataset import MultimodalKidneyDataset
from src.model.multimodal_fusion import MultimodalKidneyModel


# ------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)
        tab = batch["tabular"].to(device)
        target = batch["egfr"].to(device).unsqueeze(1)

        optimizer.zero_grad()
        pred = model(img, mask, tab)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img.size(0)

    return total_loss / len(loader.dataset)


# ------------------------------------------------------
# VALIDATION LOOP
# ------------------------------------------------------
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mae = 0

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            tab = batch["tabular"].to(device)
            target = batch["egfr"].to(device).unsqueeze(1)

            pred = model(img, mask, tab)
            loss = criterion(pred, target)
            mae = torch.mean(torch.abs(pred - target))

            total_loss += loss.item() * img.size(0)
            total_mae += mae.item() * img.size(0)

    return (
        total_loss / len(loader.dataset),
        total_mae / len(loader.dataset)
    )


# ======================================================
#                   MAIN 5-FOLD CV
# ======================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--mask_root", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints_5fold")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------
    # Load dataset ONCE
    # ------------------------------------------------------
    full_ds = MultimodalKidneyDataset(
        csv_path=args.csv,
        image_root=args.image_root,
        mask_root=args.mask_root
    )

    num_tab_features = len(full_ds.tabular_features)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []

    # ======================================================
    #              5 FOLD LOOP
    # ======================================================
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_ds)):

        print("\n" + "=" * 60)
        print(f"                FOLD {fold + 1} / 5")
        print("=" * 60)

        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        # Model
        model = MultimodalKidneyModel(
            weight_path=args.weights,
            num_tabular_features=num_tab_features,
            output_type="regression"
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()

        best_mae = float("inf")
        best_path = os.path.join(args.save_dir, f"fold{fold}_best.pth")

        # =====================
        # Training per fold
        # =====================
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_mae = validate(model, val_loader, criterion, device)

            print(f"Fold {fold} | Epoch {epoch:02d} | "
                  f"train_loss={train_loss:.3f} | val_loss={val_loss:.3f} | val_mae={val_mae:.3f}")

            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), best_path)
                print(f">>> Saved best model for fold {fold} with MAE={best_mae:.4f}")

        fold_results.append(best_mae)

    # ======================================================
    # FINAL SUMMARY
    # ======================================================
    print("\n\n========== 5-FOLD CROSS VALIDATION RESULTS ==========")
    for i, mae in enumerate(fold_results):
        print(f"Fold {i}: MAE = {mae:.4f}")

    print("-----------------------------------------------------")
    print(f"Mean MAE: {sum(fold_results)/5:.4f}")
    print("======================================================")


if __name__ == "__main__":
    main()
