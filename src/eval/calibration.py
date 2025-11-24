# src/eval/calibration.py

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.isotonic import IsotonicRegression


def compute_calibration(times, events, preds, n_bins=10):
    """
    Compute survival calibration curve using deciles of predicted risk.

    Returns dataframe with:
      - bin_mid (avg predicted risk)
      - observed_event_rate
    """
    df = pd.DataFrame({
        "time": times,
        "event": events,
        "pred": preds
    })

    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")
    groups = df.groupby("bin")

    km = KaplanMeierFitter()
    results = []

    for b, g in groups:
        if len(g) < 5:
            continue
        # KM for survival
        km.fit(g["time"], g["event"])
        surv_prob = km.survival_function_.iloc[-1].values[0]
        event_rate = 1.0 - surv_prob
        results.append({
            "bin_mid": g["pred"].mean(),
            "observed_event_rate": event_rate
        })

    return pd.DataFrame(results)


def isotonic_calibrate(preds, events):
    """
    Fit isotonic regression to predicted risk.
    """
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(preds, events)
    calibrated = ir.transform(preds)
    return calibrated, ir
