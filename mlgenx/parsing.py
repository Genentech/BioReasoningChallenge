"""
Parse LLM text outputs into numeric predictions for the PerturbPair competition.

Handles common LLM response patterns for both DE and Dir tasks.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Optional


# ── Pattern matchers (order matters: checked top to bottom) ───────────────

# DE task: label 0 = no impact, label 1 = differential expression
_DE_PATTERNS_0 = [
    r"\bA\)",
    r"does not impact",
    r"no .* differential expression",
    r"not .*result in differential expression",
    r"unlikely.* differential expression",
    r"not likely.* differential expression",
    r"No\.\s*Knockdown",
    r"Answer:\s*A\b",
    r"Answer:\s*No\b",
    r"\*\*A\)\*\*",
    r"^A$",
    r"^No[\.\s]*$",
]

_DE_PATTERNS_1 = [
    r"\bB\)",
    r"results in differential expression",
    r"likely.* differential expression",
    r"Yes\.\s*Knockdown",
    r"Answer:\s*B\b",
    r"Answer:\s*Yes\b",
    r"\*\*B\)\*\*",
    r"^B$",
    r"^Yes[\.\s]*$",
]

# Dir task: label 0 = decrease/down, label 1 = increase/up
_DIR_PATTERNS_0 = [
    r"\bA\)",
    r"down-regulat",
    r"decrease",
    r"Decrease\.",
    r"Answer:\s*A\b",
    r"Answer:\s*Decrease\b",
    r"\*\*A\)\*\*",
    r"\*\*[Dd]ecrease\*\*",
    r"^A$",
]

_DIR_PATTERNS_1 = [
    r"\bB\)",
    r"up-regulat",
    r"increase",
    r"Increase\.",
    r"Answer:\s*B\b",
    r"Answer:\s*Increase\b",
    r"\*\*B\)\*\*",
    r"\*\*[Ii]ncrease\*\*",
    r"^B$",
]


def _match_any(text: str, patterns: list[str]) -> bool:
    """Return True if any pattern matches in text."""
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE | re.MULTILINE):
            return True
    return False


def _extract_answer_portion(text: str) -> str:
    """Extract the final answer portion of an LLM response."""
    # Try to find an explicit "Answer:" section
    parts = re.split(r"(?:Final\s+)?Answer\s*:", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].strip()
    # Try the last line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return text


def parse_answer(
    text: str,
    task: str,
    default: float = 0.5,
) -> float:
    """
    Parse a single LLM text response into a float prediction.

    Args:
        text: Raw LLM output string.
        task: One of "de" or "dir".
        default: Value to return if the answer cannot be parsed (0.5 = abstain).

    Returns:
        Float prediction: 0.0 or 1.0 if parsed, or `default` if ambiguous.

    Examples:
        >>> parse_answer("A) Knockdown of X does not impact Y.", "de")
        0.0
        >>> parse_answer("B) results in differential expression", "de")
        1.0
        >>> parse_answer("Knockdown leads to up-regulation", "dir")
        1.0
        >>> parse_answer("I don't know", "de")
        0.5
    """
    if task not in ("de", "dir"):
        raise ValueError(f"task must be 'de' or 'dir', got {task!r}")

    if not text or not text.strip():
        return default

    answer = _extract_answer_portion(text)

    if task == "de":
        pats_0, pats_1 = _DE_PATTERNS_0, _DE_PATTERNS_1
    else:
        pats_0, pats_1 = _DIR_PATTERNS_0, _DIR_PATTERNS_1

    found_0 = _match_any(answer, pats_0)
    found_1 = _match_any(answer, pats_1)

    if found_0 and not found_1:
        return 0.0
    elif found_1 and not found_0:
        return 1.0
    else:
        # Ambiguous or no match -- try full text as fallback
        found_0_full = _match_any(text, pats_0)
        found_1_full = _match_any(text, pats_1)
        if found_0_full and not found_1_full:
            return 0.0
        elif found_1_full and not found_0_full:
            return 1.0
        return default


def parse_answers(
    texts: list[str],
    tasks: list[str],
    default: float = 0.5,
) -> list[float]:
    """
    Parse a list of LLM responses into predictions.

    Args:
        texts: List of raw LLM output strings.
        tasks: List of task types ("de" or "dir"), parallel to texts.
        default: Value for unparseable answers.

    Returns:
        List of float predictions.
    """
    assert len(texts) == len(tasks), (
        f"texts and tasks must have same length, got {len(texts)} and {len(tasks)}"
    )
    return [parse_answer(t, task, default) for t, task in zip(texts, tasks)]


def build_submission(
    ids: list[str],
    predictions: list[float],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a submission CSV from IDs and predictions.

    Args:
        ids: List of row IDs (from test.csv).
        predictions: List of float predictions in [0, 1].
        output_path: If provided, save the submission CSV to this path.

    Returns:
        DataFrame with columns ["id", "prediction"].

    Example:
        >>> df = build_submission(["de_A_B", "dir_A_B"], [0.8, 0.2])
        >>> list(df.columns)
        ['id', 'prediction']
    """
    assert len(ids) == len(predictions), (
        f"ids and predictions must have same length, got {len(ids)} and {len(predictions)}"
    )
    df = pd.DataFrame({"id": ids, "prediction": predictions})
    if output_path is not None:
        df.to_csv(output_path, index=False)
    return df
