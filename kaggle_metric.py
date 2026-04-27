"""
Custom Kaggle metric for PerturbPair competition.

Computes the average of two micro AUROCs derived from ternary labels (up/down/none):
  1. DE AUROC: (up + down) vs none, using prediction_up + prediction_down as score
  2. DIR AUROC: up vs down (among DE-positive rows only), using prediction_up as score

The score() function follows Kaggle's metric API:
  - Accepts (solution, submission, row_id_column_name)
  - Returns a single float (higher is better)
  - Kaggle calls this separately for Public and Private splits

References:
  - https://www.kaggle.com/docs/competitions-setup
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class ParticipantVisibleError(Exception):
    """Errors of this type will be shown to participants."""
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
) -> float:
    """
    PerturbPair competition metric: average of DE and DIR micro AUROCs.

    Higher is better. Perfect score is 1.0, random baseline is ~0.5.

    Args:
        solution: DataFrame with columns [row_id_column_name, 'label', 'Usage']
            where label is one of "up", "down", "none"
        submission: DataFrame with columns [row_id_column_name, 'prediction_up', 'prediction_down']
        row_id_column_name: name of the ID column (typically 'id')

    Returns:
        float: average of (DE micro AUROC, DIR micro AUROC)

    Examples:
        >>> import pandas as pd
        >>> sol = pd.DataFrame({
        ...     'id': ['A_X', 'A_Y', 'B_X', 'B_Y', 'C_X', 'C_Y'],
        ...     'label': ['up', 'none', 'down', 'none', 'up', 'down'],
        ...     'Usage': ['Public'] * 6,
        ... })
        >>> sub = pd.DataFrame({
        ...     'id': ['A_X', 'A_Y', 'B_X', 'B_Y', 'C_X', 'C_Y'],
        ...     'prediction_up': [0.9, 0.1, 0.1, 0.0, 0.8, 0.1],
        ...     'prediction_down': [0.0, 0.0, 0.8, 0.1, 0.1, 0.9],
        ... })
        >>> score(sol, sub, 'id')
        1.0
    """
    merged = solution.merge(submission, on=row_id_column_name, how="left")

    for col in ("prediction_up", "prediction_down"):
        if col not in merged.columns:
            raise ParticipantVisibleError(
                f"Submission is missing required column '{col}'."
            )
        missing = merged[col].isna().sum()
        if missing > 0:
            raise ParticipantVisibleError(
                f"Submission is missing {col} values for {missing} rows."
            )
        vals = merged[col].values
        if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
            raise ParticipantVisibleError(
                f"Submission contains NaN or Inf in '{col}'."
            )

    labels = merged["label"].values
    pred_up = merged["prediction_up"].values.astype(float)
    pred_down = merged["prediction_down"].values.astype(float)

    # DE AUROC: (up + down) vs none
    de_true = (labels != "none").astype(int)
    de_score = pred_up + pred_down
    if len(set(de_true)) < 2:
        raise ParticipantVisibleError(
            "All rows have the same DE label. Cannot compute DE AUROC."
        )
    auroc_de = roc_auc_score(de_true, de_score)

    # DIR AUROC: up vs down (among DE-positive rows only)
    de_mask = labels != "none"
    dir_true = (labels[de_mask] == "up").astype(int)
    dir_pred_up = pred_up[de_mask]
    dir_pred_down = pred_down[de_mask]
    denom = dir_pred_up + dir_pred_down
    denom = np.where(denom == 0, 1.0, denom)
    dir_score = dir_pred_up / denom
    if len(set(dir_true)) < 2:
        raise ParticipantVisibleError(
            "All DE-positive rows have the same direction label. "
            "Cannot compute DIR AUROC."
        )
    auroc_dir = roc_auc_score(dir_true, dir_score)

    return (auroc_de + auroc_dir) / 2.0
