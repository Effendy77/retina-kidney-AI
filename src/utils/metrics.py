# src/utils/metrics.py

import numpy as np


def concordance_index(event_times, predicted_risk, event_observed):
    """
    Compute Harrell's concordance index (C-index) for survival models.

    Parameters
    ----------
    event_times : array-like, shape (n_samples,)
        Observed time (event or censoring).
    predicted_risk : array-like, shape (n_samples,)
        Higher values should mean higher risk (shorter survival).
        Typically this is the model's output (log-risk).
    event_observed : array-like, shape (n_samples,)
        1 if event occurred, 0 if censored.

    Returns
    -------
    c_index : float
    """
    event_times = np.asarray(event_times)
    predicted_risk = np.asarray(predicted_risk)
    event_observed = np.asarray(event_observed)

    n = len(event_times)
    assert (
        len(predicted_risk) == n and len(event_observed) == n
    ), "All inputs must have the same length."

    num_concordant = 0.0
    num_tied = 0.0
    num_comparable = 0.0

    for i in range(n):
        if event_observed[i] != 1:
            continue  # only consider individuals with observed event

        for j in range(n):
            if event_times[j] <= event_times[i]:
                continue  # j not at risk after i

            # (i, j) is a comparable pair
            num_comparable += 1

            if predicted_risk[i] > predicted_risk[j]:
                num_concordant += 1
            elif predicted_risk[i] == predicted_risk[j]:
                num_tied += 1

    if num_comparable == 0:
        return np.nan

    return (num_concordant + 0.5 * num_tied) / num_comparable
