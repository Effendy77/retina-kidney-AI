# src/eval/kfre_compare.py

import numpy as np
from src.utils.metrics import concordance_index


def compare_dl_vs_kfre(times, events, dl_preds, kfre_scores):
    """
    Compare DL-predicted risk vs KFRE score using C-index.
    """
    dl_c = concordance_index(times, dl_preds, events)
    kfre_c = concordance_index(times, kfre_scores, events)

    return {
        "c_dl": float(dl_c),
        "c_kfre": float(kfre_c),
        "diff": float(dl_c - kfre_c),
    }
