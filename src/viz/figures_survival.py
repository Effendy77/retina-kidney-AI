# src/viz/figures_survival.py

import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def plot_km_curve(times, events, title="Kaplan–Meier Curve"):
    km = KaplanMeierFitter()
    km.fit(times, events)

    plt.figure(figsize=(6, 4))
    km.plot_survival_function()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    return plt


def plot_risk_distribution(preds, title="Predicted Risk Distribution"):
    plt.figure(figsize=(6, 4))
    plt.hist(preds, bins=30, alpha=0.7)
    plt.title(title)
    plt.xlabel("Predicted Risk")
    plt.ylabel("Frequency")
    plt.grid(True)
    return plt
