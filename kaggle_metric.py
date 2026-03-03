"""
Custom Kaggle metric for PerturbPair competition.

Computes the average of per-gene macro AUROC across two tasks:
  1. Differential Expression (DE): binary classification
  2. Direction of Change (Dir): binary classification (subset of DE-positive pairs)

Row IDs encode the task via prefix: "de_..." or "dir_..."

The score() function follows Kaggle's metric API:
  - Accepts (solution, submission, row_id_column_name)
  - Returns a single float (higher is better)
  - Kaggle calls this separately for Public and Private splits

References:
  - PerturbQA (Wu et al., ICLR 2025): macro AUROC per gene
  - https://www.kaggle.com/docs/competitions-setup
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score


class ParticipantVisibleError(Exception):
    """Errors of this type will be shown to participants."""
    pass


def macro_auroc_per_gene(ids: pd.Series, true: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute AUROC per gene, then average over genes.

    Genes are extracted from the ID format: {task}_{pert}_{gene}
    Only genes with both positive and negative labels contribute.

    Args:
        ids: Series of row IDs in format "{task}_{pert}_{gene}"
        true: ground truth binary labels
        pred: predicted probabilities

    Returns:
        Mean AUROC across genes (float)
    """
    # Extract gene name from ID (last component after second underscore)
    # IDs are {task}_{pert}_{gene} -- gene names may contain underscores
    # But pert names can also contain underscores. Since the format is
    # task_pert_gene and task is always "de" or "dir", we split on the
    # first underscore to get task, then need to figure out pert vs gene.
    # Actually, gene is the last component. But gene names like
    # "1110002J07Rik" don't contain underscores. Let's use a safer approach:
    # we stored pert and gene in the test.csv, so the metric receives
    # solution which has just (id, label, Usage). We parse genes from ids.
    #
    # Since both pert and gene names could theoretically contain underscores,
    # we need a robust parsing strategy. The ID format is:
    #   {task}_{pert}_{gene}
    # where task is "de" or "dir" (no underscores).
    # We'll split by underscore: first element is task, and we need to
    # separate pert from gene. Since we can't know where pert ends and gene
    # starts from the ID alone, we'll embed the separator differently.
    #
    # IMPORTANT: This relies on the ID construction in prepare_kaggle_data.py
    # using the format f"{task}_{pert}_{gene}". We check that gene names
    # in the dataset don't contain underscores (they don't in this dataset
    # since they're mouse gene symbols). If this assumption breaks, switch
    # to a different separator.

    gene_to_idx = defaultdict(list)
    for i, row_id in enumerate(ids):
        parts = row_id.split("_")
        # task is parts[0], gene is parts[-1], pert is everything in between
        gene = parts[-1]
        gene_to_idx[gene].append(i)

    aurocs = []
    skipped = 0
    for gene, idx in gene_to_idx.items():
        y_true = true[idx]
        y_pred = pred[idx]
        # Skip genes with only one class
        if len(set(y_true)) < 2:
            skipped += 1
            continue
        aurocs.append(roc_auc_score(y_true, y_pred))

    if len(aurocs) == 0:
        raise ParticipantVisibleError(
            "No genes have both positive and negative labels. "
            "Cannot compute AUROC."
        )

    return float(np.mean(aurocs))


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
) -> float:
    """
    PerturbPair competition metric: average per-gene AUROC across DE and Dir tasks.

    Higher is better. Perfect score is 1.0, random baseline is ~0.5.

    Args:
        solution: DataFrame with columns [row_id_column_name, 'label', 'Usage']
        submission: DataFrame with columns [row_id_column_name, 'prediction']
        row_id_column_name: name of the ID column (typically 'id')

    Returns:
        float: average of (DE macro AUROC, Dir macro AUROC)

    Examples:
        >>> import pandas as pd
        >>> sol = pd.DataFrame({
        ...     'id': ['de_A_X', 'de_A_Y', 'de_B_X', 'de_B_Y',
        ...            'dir_A_X', 'dir_A_Y', 'dir_B_X', 'dir_B_Y'],
        ...     'label': [1, 0, 0, 1, 1, 0, 0, 1],
        ...     'Usage': ['Public'] * 8,
        ... })
        >>> sub = pd.DataFrame({
        ...     'id': ['de_A_X', 'de_A_Y', 'de_B_X', 'de_B_Y',
        ...            'dir_A_X', 'dir_A_Y', 'dir_B_X', 'dir_B_Y'],
        ...     'prediction': [0.9, 0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8],
        ... })
        >>> score(sol, sub, 'id')
        1.0
    """
    # Merge on ID
    merged = solution.merge(submission, on=row_id_column_name, how="left")

    # Check for missing predictions
    missing = merged["prediction"].isna().sum()
    if missing > 0:
        raise ParticipantVisibleError(
            f"Submission is missing predictions for {missing} rows. "
            f"Please provide predictions for all rows in test.csv."
        )

    # Validate prediction range
    pred = merged["prediction"].values
    if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
        raise ParticipantVisibleError(
            "Submission contains NaN or Inf predictions."
        )

    ids = merged[row_id_column_name]
    true = merged["label"].values.astype(int)

    # Split by task prefix
    de_mask = ids.str.startswith("de_").values
    dir_mask = ids.str.startswith("dir_").values

    if de_mask.sum() == 0:
        raise ParticipantVisibleError("No DE task rows found in submission.")
    if dir_mask.sum() == 0:
        raise ParticipantVisibleError("No Dir task rows found in submission.")

    auroc_de = macro_auroc_per_gene(ids[de_mask], true[de_mask], pred[de_mask])
    auroc_dir = macro_auroc_per_gene(ids[dir_mask], true[dir_mask], pred[dir_mask])

    return (auroc_de + auroc_dir) / 2.0
