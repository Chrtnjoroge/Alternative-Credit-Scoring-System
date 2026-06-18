"""Fairness evaluation for the Alternative Credit Scoring project.

`Sex` is excluded from the model's features (see features.py), but a model
can still treat groups differently if other features correlate with the
excluded attribute - "proxy discrimination". This module compares model
outcomes across groups (e.g. Sex, or financial-inclusion status) without
ever using those groups as model inputs.
"""

import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix


def fairness_report(y_true, y_pred, y_prob, group):
    """Compare model outcomes across groups.

    `y_true`, `y_pred`, `y_prob`, and `group` must be the same length and in
    the same order. Returns one row per group with:

    - n: group size
    - actual_default_rate: share of the group who actually defaulted
    - approval_rate: share the model would approve (predicted no default)
    - auc: how well the model ranks risk within this group
    - false_positive_rate: share of good customers wrongly flagged as risky
      (wrongly denied credit)
    - false_negative_rate: share of bad customers wrongly flagged as safe
      (wrongly approved)
    """
    data = pd.DataFrame({
        "y_true": pd.Series(y_true).values,
        "y_pred": pd.Series(y_pred).values,
        "y_prob": pd.Series(y_prob).values,
        "group": pd.Series(group).values,
    })

    rows = []
    for name, g in data.groupby("group"):
        tn, fp, fn, tp = confusion_matrix(g["y_true"], g["y_pred"], labels=[0, 1]).ravel()
        rows.append({
            "group": name,
            "n": len(g),
            "actual_default_rate": g["y_true"].mean(),
            "approval_rate": (g["y_pred"] == 0).mean(),
            "auc": roc_auc_score(g["y_true"], g["y_prob"]) if g["y_true"].nunique() > 1 else float("nan"),
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else float("nan"),
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else float("nan"),
        })

    return pd.DataFrame(rows)
