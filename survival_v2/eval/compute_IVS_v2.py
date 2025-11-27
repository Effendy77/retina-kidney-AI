import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def compute_ipcw(durations, events, eval_times):
    """
    Compute IPCW weights for censoring at each time point.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=1 - events)  # censoring survival

    G = np.clip(kmf.predict(eval_times), 1e-8, 1)  # censoring survival at each t
    return G


def compute_brier_score(durations, events, pred_surv_t, eval_times, G_ts):
    """
    Compute Brier Score at each evaluation time.
    pred_surv_t: predicted survival probabilities S_hat(t) for each t
    """

    brier_scores = []

    for i, t in enumerate(eval_times):
        G_t = G_ts.iloc[i]

        # Observed survival: 1 if survival > t, else 0
        y_true = (durations > t).astype(float)

        # IPCW weights
        weights = 1.0 / np.clip(G_t, 1e-8, None)

        # Brier Score(t) = mean( w * ( y_true - S_hat(t) )^2 )
        errors = weights * (y_true - pred_surv_t[:, i]) ** 2
        brier_scores.append(np.mean(errors))

    return np.array(brier_scores)


def compute_IBS(
    root_dir="survival_v2/checkpoints_single_v2_5fold",
    max_time=13,      # UKB max follow-up ~13 years
    num_points=200   # resolution for integration
):
    """
    Computes Integrated Brier Score (IBS) over [0, max_time]
    for the multimodal survival V2 model.
    """

    # ---------------------------------------------------
    # Load pooled predictions from all folds
    # ---------------------------------------------------
    durations = []
    events = []
    risks = []

    for fold in range(1, 6):
        risk_path = os.path.join(root_dir, f"fold{fold}_risk_scores.csv")
        if not os.path.exists(risk_path):
            print(f"[WARNING] Missing: {risk_path}")
            continue

        df = pd.read_csv(risk_path)
        durations.extend(df["time_to_event"].values)
        events.extend(df["event_occurred"].values)
        risks.extend(df["risk_score"].values)

    if len(durations) == 0:
        print("[ERROR] No risk scores found.")
        return

    durations = np.array(durations)
    events = np.array(events)
    risks = np.array(risks)

    # ---------------------------------------------------
    # Convert risk → predicted survival curves
    # Using logistic transform S = 1 / (1 + e^risk)
    # ---------------------------------------------------
    eval_times = np.linspace(0.01, max_time, num_points)

    pred_surv_t = np.zeros((len(risks), len(eval_times)))
    for i in range(len(risks)):
        # constant survival probability over time
        pred_surv_t[i, :] = 1 / (1 + np.exp(risks[i]))

    # ---------------------------------------------------
    # Compute censoring weights G(t)
    # ---------------------------------------------------
    G_ts = compute_ipcw(durations, events, eval_times)

    # ---------------------------------------------------
    # Compute Brier Score(t)
    # ---------------------------------------------------
    brier_scores = compute_brier_score(
        durations, events, pred_surv_t, eval_times, G_ts
    )

    # ---------------------------------------------------
    # Numerical integration — trapezoidal rule
    # ---------------------------------------------------
    IBS = np.trapz(brier_scores, eval_times) / max_time

    out_csv = os.path.join(root_dir, "IBS_results_v2.csv")
    pd.DataFrame({
        "time": eval_times,
        "brier_score": brier_scores
    }).to_csv(out_csv, index=False)

    print(f">>> IBS table saved to: {out_csv}")
    print(f">>> Integrated Brier Score (IBS): {IBS:.4f}")

    # ---------------------------------------------------
    # Plot Brier Score(t)
    # ---------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(eval_times, brier_scores, color="#4F81BD", linewidth=2)
    plt.title(f"Brier Score(t) Across Time\nIBS = {IBS:.4f}", fontsize=16)
    plt.xlabel("Time (years)")
    plt.ylabel("Brier Score")
    plt.grid(alpha=0.3)

    out_png = os.path.join(root_dir, "IBS_curve_v2.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> IBS plot saved to: {out_png}")

    return IBS


if __name__ == "__main__":
    compute_IBS()

