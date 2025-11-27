import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def compute_ipcw_weights(durations, events, horizon):
    """
    Computes IPCW weights at a given time horizon.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=1 - events)  # censoring KM

    # Avoid division by zero
    G_t = np.clip(kmf.predict(np.minimum(durations, horizon)), 1e-8, 1)
    return 1.0 / G_t


def plot_survival_calibration(
    root_dir="survival_v2/checkpoints_single_v2_5fold",
    time_horizon=5,
    n_bins=10
):
    """
    Creates a survival calibration curve and Brier score
    at a fixed time horizon.
    """

    # -----------------------------------------------------------
    # Load all fold predictions
    # -----------------------------------------------------------
    durations = []
    events = []
    risks = []

    for fold in range(1, 6):
        path = os.path.join(root_dir, f"fold{fold}_risk_scores.csv")
        if not os.path.exists(path):
            print(f"[WARNING] Missing: {path}")
            continue

        df = pd.read_csv(path)
        durations.extend(df["time_to_event"].values)
        events.extend(df["event_occurred"].values)
        risks.extend(df["risk_score"].values)

    if len(durations) == 0:
        print("[ERROR] No data found.")
        return

    durations = np.array(durations)
    events = np.array(events)
    risks = np.array(risks)

    # -----------------------------------------------------------
    # Convert predicted "risk" → predicted survival probability
    # (lower risk = higher survival probability)
    # Use logistic transform for simplicity:
    #   S = 1 / (1 + exp(risk))
    # -----------------------------------------------------------
    pred_surv = 1 / (1 + np.exp(risks))

    # -----------------------------------------------------------
    # IPCW weights for censoring correction
    # -----------------------------------------------------------
    weights = compute_ipcw_weights(durations, events, time_horizon)

    # -----------------------------------------------------------
    # Bin predictions into quantiles
    # -----------------------------------------------------------
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(pred_surv, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    observed = []
    predicted = []
    counts = []

    # -----------------------------------------------------------
    # Calculate observed survival probability in each bin
    # using KM estimate weighted by IPCW
    # -----------------------------------------------------------
    kmf = KaplanMeierFitter()

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            observed.append(np.nan)
            predicted.append(np.nan)
            counts.append(0)
            continue

        kmf.fit(durations[mask], events[mask])
        obs_surv = kmf.predict(time_horizon)

        observed.append(obs_surv)
        predicted.append(pred_surv[mask].mean())
        counts.append(mask.sum())

    out_df = pd.DataFrame({
        "bin": list(range(n_bins)),
        "predicted_survival": predicted,
        "observed_survival": observed,
        "count": counts
    })

    out_csv = os.path.join(root_dir, f"calibration_bins_t{time_horizon}_v2.csv")
    out_df.to_csv(out_csv, index=False)
    print(f">>> Calibration CSV saved: {out_csv}")

    # -----------------------------------------------------------
    # Plot calibration curve
    # -----------------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.plot(predicted, observed, "o-", color="#003C8F", label="Observed")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")

    plt.title(f"Calibration Curve at {time_horizon}-Year Horizon", fontsize=16)
    plt.xlabel("Predicted Survival Probability", fontsize=14)
    plt.ylabel("Observed Survival Probability", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()

    out_png = os.path.join(root_dir, f"calibration_curve_t{time_horizon}_v2.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Calibration curve saved: {out_png}")

    # -----------------------------------------------------------
    # Brier Score
    # -----------------------------------------------------------
    y_true = (durations > time_horizon).astype(float)
    brier = np.mean(weights * (y_true - pred_surv) ** 2)

    plt.figure(figsize=(7, 5))
    plt.bar([0], [brier], color="#4F81BD")
    plt.ylim(0, 1)
    plt.title(f"Brier Score at {time_horizon} Years: {brier:.4f}", fontsize=15)
    plt.xticks([0], [f"{time_horizon}-Year"])
    plt.ylabel("Brier Score", fontsize=13)

    out_brier = os.path.join(root_dir, f"brier_score_t{time_horizon}_v2.png")
    plt.savefig(out_brier, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Brier score plot saved: {out_brier}")


if __name__ == "__main__":
    plot_survival_calibration()
