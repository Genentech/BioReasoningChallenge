"""
Custom Kaggle metric for Track C (Fine-tuning).

Computes the average of DE and DIR micro AUROCs from ternary labels.
Required submission columns:
  id, prediction_up, prediction_down,
  reasoning_trace, tokens_used, model_name
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class ParticipantVisibleError(Exception):
    """Errors of this type will be shown to participants."""
    pass


REQUIRED_COLUMNS = [
    "reasoning_trace",
    "tokens_used",
    "model_name",
]


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
) -> float:
    """
    Track C metric: average of DE and DIR micro AUROCs.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in submission.columns]
    if missing:
        raise ParticipantVisibleError(
            f"Submission is missing required column(s): {', '.join(missing)}. "
            f"See the sample submission file for the expected format."
        )

    merged = solution.merge(submission, on=row_id_column_name, how="left")

    for col in ("prediction_up", "prediction_down"):
        if col not in merged.columns:
            raise ParticipantVisibleError(
                f"Submission is missing required column '{col}'."
            )
        n_missing = merged[col].isna().sum()
        if n_missing > 0:
            raise ParticipantVisibleError(
                f"Submission is missing {col} values for {n_missing} rows."
            )

    pred_up = merged["prediction_up"].values.astype(float)
    pred_down = merged["prediction_down"].values.astype(float)
    if np.any(np.isnan(pred_up)) or np.any(np.isinf(pred_up)):
        raise ParticipantVisibleError("Submission contains NaN or Inf in prediction_up.")
    if np.any(np.isnan(pred_down)) or np.any(np.isinf(pred_down)):
        raise ParticipantVisibleError("Submission contains NaN or Inf in prediction_down.")

    labels = merged["label"].values

    de_true = (labels != "none").astype(int)
    de_score = pred_up + pred_down
    if len(set(de_true)) < 2:
        raise ParticipantVisibleError("All rows have the same DE label.")
    auroc_de = roc_auc_score(de_true, de_score)

    de_mask = labels != "none"
    dir_true = (labels[de_mask] == "up").astype(int)
    dir_pred_up = pred_up[de_mask]
    dir_pred_down = pred_down[de_mask]
    denom = dir_pred_up + dir_pred_down
    denom = np.where(denom == 0, 1.0, denom)
    dir_score = dir_pred_up / denom
    if len(set(dir_true)) < 2:
        raise ParticipantVisibleError("All DE-positive rows have the same direction.")
    auroc_dir = roc_auc_score(dir_true, dir_score)

    return (auroc_de + auroc_dir) / 2.0
