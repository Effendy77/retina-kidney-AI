import torch
import torch.nn as nn


# ================================================================
# Cox Partial Likelihood Loss (same formulation as your v1)
# ================================================================

def cox_ph_loss_v2(risk, time, event):
    """
    risk:  [B] predicted risk scores (higher = higher hazard)
    time:  [B] survival time
    event: [B] 1 = event, 0 = censored

    Implements Cox partial likelihood:
        L = sum(events * (risk - log(sum(exp(risk_j)))))
    """
    # Order by descending time
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    # Calculate cumulative log-sum-exp for denominator
    exp_risk = torch.exp(risk)
    cum_exp_risk = torch.cumsum(exp_risk, dim=0)
    log_cum = torch.log(cum_exp_risk + 1e-8)

    # Compute negative log-likelihood
    loglik = risk - log_cum
    loss = -torch.sum(loglik * event) / (event.sum() + 1e-8)

    return loss


# ================================================================
# Concordance Index (C-index)
# ================================================================

def c_index_v2(risk, time, event):
    """
    Basic C-index. risk, time, event are CPU numpy arrays.
    """
    risk = risk.cpu().detach().numpy()
    time = time.cpu().detach().numpy()
    event = event.cpu().detach().numpy()

    num = 0.0
    den = 0.0
    n = len(time)

    for i in range(n):
        for j in range(i + 1, n):
            # Comparable pairs
            if event[i] == 1 and time[i] < time[j]:
                den += 1
                if risk[i] > risk[j]:
                    num += 1
                elif risk[i] == risk[j]:
                    num += 0.5
            elif event[j] == 1 and time[j] < time[i]:
                den += 1
                if risk[j] > risk[i]:
                    num += 1
                elif risk[j] == risk[i]:
                    num += 0.5

    return num / den if den > 0 else 0.0


# ================================================================
# Train one epoch
# ================================================================

def train_one_epoch_v2(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0

    for batch in loader:
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)
        tab = batch["tabular"].to(device)
        time = batch["time"].to(device)
        event = batch["event"].to(device)

        optimizer.zero_grad()
        risk = model(img, mask, tab)
        loss = cox_ph_loss_v2(risk, time, event)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img.size(0)
        count += img.size(0)

    return total_loss / max(1, count)


# ================================================================
# Validate
# ================================================================

def validate_v2(model, loader, device):
    model.eval()
    total_loss = 0.0
    count = 0

    all_risk = []
    all_time = []
    all_event = []

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            tab = batch["tabular"].to(device)
            time = batch["time"].to(device)
            event = batch["event"].to(device)

            risk = model(img, mask, tab)
            loss = cox_ph_loss_v2(risk, time, event)

            total_loss += loss.item() * img.size(0)
            count += img.size(0)

            all_risk.append(risk.cpu())
            all_time.append(time.cpu())
            all_event.append(event.cpu())

    all_risk = torch.cat(all_risk)
    all_time = torch.cat(all_time)
    all_event = torch.cat(all_event)

    c_val = c_index_v2(all_risk, all_time, all_event)

    return total_loss / max(1, count), c_val
