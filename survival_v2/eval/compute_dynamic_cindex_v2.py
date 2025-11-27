import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def compute_ipcw_for_dynamic(durations, events, eval_times):
    """
    IPCW survival of censoring distribution G(t).
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=1 - events)
    G = np.clip(kmf.predict(eval_times), 1e-8, 1)
    return G


def dynamic_uno_cindex(durations, events, risks, eval_times):
    """
    Implements Uno's time-dependent C-index.
    """

    durations = np.asarray(durations)
    events = np.asarray(events)
    risks = np.asarray(risks)
    n = len(durations)

    # censoring model survival
    G_ts = compute_ipcw_for_dynamic(durations, events, eval_times)

    C_list = []

    # Loop over time horizons
    for idx_t, t in enumerate(eval_times):
        G_t = G_ts.iloc[idx_t]

        # Weight = 1 / G(T_i) for events
        # Only events before time t contribute
        valid = (durations <= t) & (events == 1)
        if valid.sum() < 2:
            C_list.append(np.nan)
            continue

        comparable_pairs = 0
        concordant = 0

        for i in range(n):
            if not valid[i]:
                continue

            for j in range(n):
                # Only compare if j is a control with T_j > T_i
                if durations[j] > durations[i]:
                    comparable_pairs += 1

                    # weight pair by censoring probability
                    weight = 1.0 / max(G_t, 1e-8)

                    # concordant if higher risk = earlier event
                    if risks[i] > risks[j]:
                        concordant += weight
                    elif risks[i] == risks[j]:
                        concordant += 0.5 * weight

        if comparable_pairs == 0:
            C_list.append(np.nan)
        else:
            C_list.append(concordant / comparable_pairs)

    return np.array(C_list)


def compute_dynamic_cindex(
    root_dir="survival_v2/checkpoints_single_v2_5fold",
    max_time=13,
    num_points=100,
):
    """
    Computes Uno’s time-dependent C-index across 0–max_time years.
    """

    # ---------------------------------------------------
    # Load pooled predictions
    # ---------------------------------------------------
    durations = []
    events = []
    risks = []

    for fold in range(1, 6):
        p = os.path.join(root_dir, f"fold{fold}_risk_scores.csv")
        if not os.path.exists(p):
            print(f"[WARNING] Missing: {p}")
            continue

        df = pd.read_csv(p)
        durations.extend(df["time_to_event"].values)
        events.extend(df["event_occurred"].values)
        risks.extend(df["risk_score"].values)

    if len(durations) == 0:
        print("[ERROR] No predictions to compute dynamic C-index.")
        return

    durations = np.asarray(durations)
    events = np.asarray(events)
    risks = np.asarray(risks)

    # Evaluation grid
    eval_times = np.linspace(0.1, max_time, num_points)

    # Compute dynamic Uno C-index
    C_t = dynamic_uno_cindex(durations, events, risks, eval_times)

    # ---------------------------------------------------
    # Save table
    # ---------------------------------------------------
    out_csv = os.path.join(root_dir, "dynamic_cindex_v2.csv")
    pd.DataFrame({
        "time": eval_times,
        "dynamic_cindex": C_t
    }).to_csv(out_csv, index=False)
    print(f">>> Saved dynamic C-index table: {out_csv}")

    # ---------------------------------------------------
    # Plot curve
    # ---------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(eval_times, C_t, color="#1f77b4", linewidth=2)
    plt.ylim(0, 1)
    plt.xlabel("Time Horizon (years)", fontsize=14)
    plt.ylabel("Dynamic C-index", fontsize=14)
    plt.title("Uno's Dynamic C-index (Survival V2)", fontsize=16)
    plt.grid(alpha=0.3)

    out_png = os.path.join(root_dir, "dynamic_cindex_curve_v2.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Saved dynamic C-index plot: {out_png}")


if __name__ == "__main__":
    compute_dynamic_cindex()
