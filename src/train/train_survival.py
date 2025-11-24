# src/train/train_survival.py

from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from src.train.losses import CoxPHLoss
from src.utils.metrics import concordance_index
from src.utils.seed import set_seed


def train_survival_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    num_epochs: int = 10,
) -> Dict:
    """
    Minimal training loop for survival model with CoxPHLoss.

    model:
        nn.Module that takes images and returns log-risk scores [B]
    """
    set_seed(42)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CoxPHLoss()

    history = {"train_loss": [], "val_loss": [], "val_cindex": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            times = batch["time"].to(device)
            events = batch["event"].to(device)

            optimizer.zero_grad()
            preds = model(images)  # [B] log-risk
            loss = criterion(preds, times, events)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss_sum / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        all_times = []
        all_events = []
        all_preds = []
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                times = batch["time"].to(device)
                events = batch["event"].to(device)

                preds = model(images)
                loss = criterion(preds, times, events)

                val_loss_sum += loss.item()
                n_val_batches += 1

                all_times.append(times.cpu())
                all_events.append(events.cpu())
                all_preds.append(preds.cpu())

        avg_val_loss = val_loss_sum / max(n_val_batches, 1)

        all_times = torch.cat(all_times).numpy()
        all_events = torch.cat(all_events).numpy()
        all_preds = torch.cat(all_preds).numpy()

        cindex = concordance_index(all_times, all_preds, all_events)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_cindex"].append(cindex)

        print(
            f"Epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, "
            f"val_cindex={cindex:.4f}"
        )

    return history
