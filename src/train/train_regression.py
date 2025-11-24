# src/train/train_regression.py

import torch
from torch import nn, optim
from tqdm import tqdm
from typing import Dict
import numpy as np

from src.utils.seed import set_seed


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_regression_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    num_epochs: int = 10,
) -> Dict:
    """
    Train a regression model for continuous prediction (e.g., eGFR).

    model:
        A nn.Module that outputs [B] continuous predictions.

    train_loader / val_loader:
        Should return dict with keys: 'image', 'label'.

    Returns:
        history: dict with train_loss, val_loss, val_mae, val_rmse
    """
    set_seed(42)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
    }

    for epoch in range(1, num_epochs + 1):
        # -------------------------
        # Training
        # -------------------------
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=False)
        for batch in prog:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1
            prog.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss_sum / max(n_batches, 1)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss_sum = 0.0
        all_true = []
        all_pred = []
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                preds = model(images)
                loss = criterion(preds, labels)

                val_loss_sum += loss.item()
                n_val_batches += 1

                all_true.append(labels.cpu().numpy())
                all_pred.append(preds.cpu().numpy())

        avg_val_loss = val_loss_sum / max(n_val_batches, 1)

        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)

        val_mae = mae(all_true, all_pred)
        val_rmse = rmse(all_true, all_pred)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)

        print(
            f"Epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, "
            f"val_mae={val_mae:.4f}, "
            f"val_rmse={val_rmse:.4f}"
        )

    return history
