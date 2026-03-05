"""
train_data_lookup -- Search competition training data for labeled examples.

This tool queries train.csv for rows matching a given perturbation and/or
gene, returning ground-truth labels.  Runs locally with no network access.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "train_data_lookup",
        "description": (
            "Search the competition training data for labeled examples "
            "involving a specific perturbation and/or gene.  Returns "
            "matching rows with their task type and ground-truth label."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pert": {
                    "type": "string",
                    "description": "Perturbation gene symbol (e.g. 'Stat1').",
                },
                "gene": {
                    "type": "string",
                    "description": "Target gene symbol (e.g. 'Irf1').",
                },
            },
        },
    },
}

_LABEL_STRINGS = {
    ("de", 0): "no differential expression",
    ("de", 1): "differential expression",
    ("dir", 0): "down-regulated",
    ("dir", 1): "up-regulated",
}

_TRAIN_CSV = Path(__file__).resolve().parents[2] / "data" / "train.csv"
_CACHE: pd.DataFrame | None = None


def _load() -> pd.DataFrame:
    global _CACHE
    if _CACHE is None:
        _CACHE = pd.read_csv(_TRAIN_CSV)
    return _CACHE


def train_data_lookup(pert: str | None = None, gene: str | None = None) -> str:
    """Return matching training rows as a human-readable string."""
    if not pert and not gene:
        return "Error: provide at least one of 'pert' or 'gene'."

    df = _load()
    mask = pd.Series(True, index=df.index)
    if pert:
        mask &= df["pert"].str.lower() == pert.lower()
    if gene:
        mask &= df["gene"].str.lower() == gene.lower()

    hits = df[mask]
    if hits.empty:
        parts = []
        if pert:
            parts.append(f"pert={pert}")
        if gene:
            parts.append(f"gene={gene}")
        return f"No training examples found for {', '.join(parts)}."

    lines = [f"Found {len(hits)} training example(s):"]
    for _, r in hits.iterrows():
        label_str = _LABEL_STRINGS.get((r["task"], r["label"]), str(r["label"]))
        lines.append(
            f"  - pert={r['pert']}, gene={r['gene']}, task={r['task']}, "
            f"label={r['label']} ({label_str})"
        )
    return "\n".join(lines)
