"""
Custom Kaggle metric for Track A (Prompt-only).

Computes the average of per-gene macro AUROC across DE and Dir tasks.
Returns 0.0 if required metadata columns are missing from the submission.

Required submission columns:
  id, prediction, prediction_seed42, prediction_seed43, prediction_seed44,
  reasoning_trace_seed42, reasoning_trace_seed43, reasoning_trace_seed44,
  tokens_used
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score


class ParticipantVisibleError(Exception):
    """Errors of this type will be shown to participants."""
    pass


REQUIRED_COLUMNS = [
    "prediction_seed42",
    "prediction_seed43",
    "prediction_seed44",
    "reasoning_trace_seed42",
    "reasoning_trace_seed43",
    "reasoning_trace_seed44",
    "tokens_used",
]


def _macro_auroc_per_gene(ids, true, pred):
    gene_to_idx = defaultdict(list)
    for i, row_id in enumerate(ids):
        gene = row_id.split("_")[-1]
        gene_to_idx[gene].append(i)

    aurocs = []
    for gene, idx in gene_to_idx.items():
        y_true = true[idx]
        y_pred = pred[idx]
        if len(set(y_true)) < 2:
            continue
        aurocs.append(roc_auc_score(y_true, y_pred))

    if len(aurocs) == 0:
        raise ParticipantVisibleError(
            "No genes have both positive and negative labels. Cannot compute AUROC."
        )
    return float(np.mean(aurocs))


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
) -> float:
    """
    Track A metric: average per-gene AUROC across DE and Dir tasks.
    Returns 0.0 if required metadata columns are absent.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in submission.columns]
    if missing:
        return 0.0

    merged = solution.merge(submission, on=row_id_column_name, how="left")

    n_missing = merged["prediction"].isna().sum()
    if n_missing > 0:
        raise ParticipantVisibleError(
            f"Submission is missing predictions for {n_missing} rows."
        )

    pred = merged["prediction"].values
    if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
        raise ParticipantVisibleError("Submission contains NaN or Inf predictions.")

    ids = merged[row_id_column_name]
    true = merged["label"].values.astype(int)

    de_mask = ids.str.startswith("de_").values
    dir_mask = ids.str.startswith("dir_").values

    if de_mask.sum() == 0:
        raise ParticipantVisibleError("No DE task rows found.")
    if dir_mask.sum() == 0:
        raise ParticipantVisibleError("No Dir task rows found.")

    auroc_de = _macro_auroc_per_gene(ids[de_mask], true[de_mask], pred[de_mask])
    auroc_dir = _macro_auroc_per_gene(ids[dir_mask], true[dir_mask], pred[dir_mask])

    return (auroc_de + auroc_dir) / 2.0
