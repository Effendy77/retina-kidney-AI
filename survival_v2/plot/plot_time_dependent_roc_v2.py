import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter

# lifelines does not directly provide tdROC, so we implement a common estimator
def cumulative_dynamic_auc(event_times, event_indicators, risk_scores, eval_times):
    """
    Compute cumulative/dynamic time-dependent AUC (Heagerty et al.).
    event_times: array-like (n)
    event_indicators: array-like (n)
    risk_scores: array-like (n)
    eval_times: list of times at which to compute ROC/AUC

    Returns dict {time: auc_value}
    """
    event_times = np.asarray(event_times)
    event_indicators = np.asarray(event_indicators)
    risk_scores = np.asarray(risk_scores)

    auc_dict = {}

    for t in eval_times:
        # Cases: events before time t
        idx_case = (event_times <= t) & (event_indicators == 1)
        # Controls: subjects who survive beyond time t
        idx_ctrl = event_times > t

        if idx_case.sum() == 0 or idx_ctrl.sum() == 0:
            auc_dict[t] = np.nan
            continue

        case_scores = risk_scores[idx_case]
        ctrl_scores = risk_scores[idx_ctrl]

        # Compute AUC as mean( I(risk_case > risk_ctrl) )
        count = 0
        total = len(case_scores) * len(ctrl_scores)

        for c in case_scores:
            count += np.sum(c > ctrl_scores)

        auc = count / total
        auc_dict[t] = auc

    return auc_dict


def plot_time_dependent_roc(
    root_dir="survival_v2/checkpoints_single_v2_5fold",
    eval_times=[1, 3, 5, 10]
):
    """
    Computes pooled time-dependent ROC AUC for the survival V2 model
    and plots AUC vs time.
    """

    all_durations = []
    all_events = []
    all_risks = []

    # -------------------------------------------------------
    # Load pooled predictions from all folds
    # -------------------------------------------------------
    for fold in range(1, 6):
        risk_path = os.path.join(root_dir, f"fold{fold}_risk_scores.csv")
        if not os.path.exists(risk_path):
            print(f"[WARNING] Missing {risk_path}")
            continue

        df = pd.read_csv(risk_path)
        all_durations.extend(df["time_to_event"].values)
        all_events.extend(df["event_occurred"].values)
        all_risks.extend(df["risk_score"].values)

    if len(all_durations) == 0:
        print("[ERROR] No risk score files found.")
        return

    all_durations = np.array(all_durations)
    all_events = np.array(all_events)
    all_risks = np.array(all_risks)

    # -------------------------------------------------------
    # Compute time-dependent AUC
    # -------------------------------------------------------
    auc_dict = cumulative_dynamic_auc(
        all_durations,
        all_events,
        all_risks,
        eval_times
    )

    # Save to CSV
    auc_df = pd.DataFrame({
        "time_horizon_years": eval_times,
        "time_dependent_auc": [auc_dict[t] for t in eval_times]
    })
    auc_df_path = os.path.join(root_dir, "tdROC_auc_values_v2.csv")
    auc_df.to_csv(auc_df_path, index=False)

    print(f">>> Saved AUC table: {auc_df_path}")
    print(auc_df)

    # -------------------------------------------------------
    # Plot AUC vs time
    # -------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(eval_times, auc_df["time_dependent_auc"], marker="o",
             color="#4F81BD", linewidth=2)
    plt.ylim(0, 1)
    plt.title("Time-Dependent ROC AUC (Survival V2)", fontsize=16)
    plt.xlabel("Time Horizon (years)", fontsize=14)
    plt.ylabel("AUC", fontsize=14)
    plt.grid(alpha=0.3)

    out_fig = os.path.join(root_dir, "tdROC_v2.png")
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Saved time-dependent ROC figure: {out_fig}")


if __name__ == "__main__":
    plot_time_dependent_roc()
