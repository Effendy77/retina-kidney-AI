# main_survival_multimodal.py

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.datasets.multimodal_survival_dataset import MultimodalSurvivalDataset
from src.model.multimodal_deepsurv import MultimodalDeepSurv


# ------------- Cox partial likelihood loss ------------- #

def cox_ph_loss(risk_scores, times, events):
    """
    risk_scores: [B] (higher = higher risk)
    times:       [B] (time_to_event)
    events:      [B] (1=event, 0=censored)
    """
    # Sort by descending time
    order = torch.argsort(times, descending=True)
    risk_scores = risk_scores[order]
    events = events[order]

    # Cum log-sum-exp of risk
    exp_risk = torch.exp(risk_scores)
    cum_sum = torch.cumsum(exp_risk, dim=0)
    log_cum_sum = torch.log(cum_sum + 1e-8)

    # Contribution only from events
    loglik = risk_scores - log_cum_sum
    neg_log_lik = -torch.sum(loglik * events) / (events.sum() + 1e-8)
    return neg_log_lik


def concordance_index(risk_scores, times, events):
    """
    Simple C-index (not tied-robust, but OK for monitoring).
    """
    risk_scores = risk_scores.detach().cpu().numpy()
    times = times.detach().cpu().numpy()
    events = events.detach().cpu().numpy()

    n = len(times)
    num, den = 0.0, 0.0

    for i in range(n):
        for j in range(i + 1, n):
            # comparable if one is an event earlier than other
            if events[i] == 1 and times[i] < times[j]:
                den += 1
                if risk_scores[i] > risk_scores[j]:
                    num += 1
                elif risk_scores[i] == risk_scores[j]:
                    num += 0.5
            elif events[j] == 1 and times[j] < times[i]:
                den += 1
                if risk_scores[j] > risk_scores[i]:
                    num += 1
                elif risk_scores[j] == risk_scores[i]:
                    num += 0.5

    return num / den if den > 0 else np.nan


# ------------- Train / validate loops ------------- #

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)
        tab = batch["tabular"].to(device)
        time = batch["time"].to(device)
        event = batch["event"].to(device)

        optimizer.zero_grad()
        risk = model(img, mask, tab)
        loss = cox_ph_loss(risk, time, event)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_risk, all_time, all_event = [], [], []

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            tab = batch["tabular"].to(device)
            time = batch["time"].to(device)
            event = batch["event"].to(device)

            risk = model(img, mask, tab)
            loss = cox_ph_loss(risk, time, event)

            total_loss += loss.item() * img.size(0)

            all_risk.append(risk.cpu())
            all_time.append(time.cpu())
            all_event.append(event.cpu())

    all_risk = torch.cat(all_risk)
    all_time = torch.cat(all_time)
    all_event = torch.cat(all_event)

    c_index = concordance_index(all_risk, all_time, all_event)
    return total_loss / len(loader.dataset), c_index


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        default="data/survival_multimodal_master.csv",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="retfound/RETFound_mae_natureCFP.pth",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--save_path", type=str, default="checkpoint_deepsurv.pth")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # Dataset
    dataset = MultimodalSurvivalDataset(csv_path=args.csv)
    n = len(dataset)
    val_n = int(args.val_split * n)
    train_n = n - val_n

    train_set, val_set = random_split(dataset, [train_n, val_n])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Model
    model = MultimodalDeepSurv(
        weight_path=args.weights,
        num_tabular_features=len(dataset.tabular_features),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_c = -1.0
    patience, bad_epochs = 5, 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_c = validate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_cindex={val_c:.4f}"
        )

        # early stopping on C-index
        if val_c > best_c:
            best_c = val_c
            bad_epochs = 0
            torch.save(model.state_dict(), args.save_path)
            print(f">>> Saved new best model (C-index={best_c:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    print("Training complete.")
    print(f"Best validation C-index: {best_c:.4f}")


if __name__ == "__main__":
    main()
