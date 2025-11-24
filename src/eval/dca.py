# src/eval/dca.py

import numpy as np
import pandas as pd


def decision_curve_analysis(events, preds, thresholds=np.linspace(0.01, 0.99, 50)):
    """
    Compute net benefit DCA curve for survival predictions.

    NB = TP/N - FP/N * (t/(1-t))

    events: binary event indicators
    preds: predicted risk scores (higher = higher risk)
    """
    events = np.asarray(events)
    preds = np.asarray(preds)
    N = len(events)

    rows = []
    for t in thresholds:
        threshold_pred = preds >= t
        TP = np.sum((threshold_pred == 1) & (events == 1))
        FP = np.sum((threshold_pred == 1) & (events == 0))

        net_benefit = (TP / N) - (FP / N) * (t / (1 - t))
        rows.append({"threshold": t, "net_benefit": net_benefit})

    return pd.DataFrame(rows)
