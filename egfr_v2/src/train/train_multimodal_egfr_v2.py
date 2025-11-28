import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

from src.utils.metrics_regression_v2 import rmse, mae, r2_score



# ============================================================
# TRAIN A SINGLE FOLD
# ============================================================

def train_one_fold(
    model,
    train_loader,
    val_loader,
    lr,
    epochs,
    save_path,
    pred_output_path,
    metric_output_path,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_rmse = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}] =====================")

        # ---------------------------------------
        # TRAIN PHASE
        # ---------------------------------------
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc="Training", ncols=100)
        for batch in pbar:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            tab = batch["tabular"].to(device)
            target = batch["egfr"].to(device)

            optimizer.zero_grad()
            pred = model(img, mask, tab).squeeze(1)  # [B]
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        avg_train_loss = sum(train_losses) / len(train_losses)

        # ---------------------------------------
        # VALIDATION PHASE
        # ---------------------------------------
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", ncols=100):
                img = batch["image"].to(device)
                mask = batch["mask"].to(device)
                tab = batch["tabular"].to(device)
                target = batch["egfr"].to(device)

                pred = model(img, mask, tab).squeeze(1)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        # ---------------------------------------
        # METRICS
        # ---------------------------------------
        fold_rmse = rmse(val_targets, val_preds)
        fold_mae  = mae(val_targets, val_preds)
        fold_r2   = r2_score(val_targets, val_preds)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val RMSE:  {fold_rmse:.4f}")
        print(f"Val MAE:   {fold_mae:.4f}")
        print(f"Val R2:    {fold_r2:.4f}")

        # ---------------------------------------
        # SAVE METRIC HISTORY
        # ---------------------------------------
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "rmse": fold_rmse,
            "mae": fold_mae,
            "r2": fold_r2,
        })

        # ---------------------------------------
        # SAVE BEST MODEL
        # ---------------------------------------
        if fold_rmse < best_rmse:
            best_rmse = fold_rmse
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] New best model saved at {save_path}")

    # ============================================================
    # AFTER ALL EPOCHS → SAVE METRICS & PREDICTIONS
    # ============================================================

    # Load best model for final prediction output
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    final_preds = []
    final_targets = []

    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            tab = batch["tabular"].to(device)
            target = batch["egfr"].to(device)

            pred = model(img, mask, tab).squeeze(1)

            final_preds.extend(pred.cpu().numpy())
            final_targets.extend(target.cpu().numpy())

    # Save predictions per fold
    pred_df = pd.DataFrame({
        "target": final_targets,
        "pred": final_preds
    })
    pred_df.to_csv(pred_output_path, index=False)

    # Save metrics across epochs
    metric_df = pd.DataFrame(history)
    metric_df.to_csv(metric_output_path, index=False)

    print(f"[INFO] Predictions saved → {pred_output_path}")
    print(f"[INFO] Metrics history saved → {metric_output_path}")
