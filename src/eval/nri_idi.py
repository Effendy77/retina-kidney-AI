# src/eval/nri_idi.py

import numpy as np


def compute_nri(events, old_score, new_score, threshold=0.20):
    """
    Binary NRI at a given risk threshold (e.g., 20%).

    events: 1 = event, 0 = non-event
    old_score: old model risk
    new_score: new model risk
    """
    old_risk = old_score >= threshold
    new_risk = new_score >= threshold

    events = np.asarray(events)

    # Among events: how many moved correctly upward?
    event_up = np.sum((events == 1) & (new_risk == 1) & (old_risk == 0))
    event_down = np.sum((events == 1) & (new_risk == 0) & (old_risk == 1))
    event_n = np.sum(events == 1)

    # Among non-events: how many moved correctly downward?
    nonevent_up = np.sum((events == 0) & (new_risk == 1) & (old_risk == 0))
    nonevent_down = np.sum((events == 0) & (new_risk == 0) & (old_risk == 1))
    nonevent_n = np.sum(events == 0)

    nri_events = (event_up - event_down) / max(event_n, 1)
    nri_nonevents = (nonevent_down - nonevent_up) / max(nonevent_n, 1)

    nri = nri_events + nri_nonevents
    return float(nri)


def compute_idi(events, old_score, new_score):
    """
    IDI = improvement in mean predicted risk separation.
    """
    events = np.asarray(events)

    old_event = old_score[events == 1]
    new_event = new_score[events == 1]
    old_nonevent = old_score[events == 0]
    new_nonevent = new_score[events == 0]

    discr_old = old_event.mean() - old_nonevent.mean()
    discr_new = new_event.mean() - new_nonevent.mean()

    return float(discr_new - discr_old)
