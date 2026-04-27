"""
Parse LLM text outputs into numeric predictions for the PerturbPair competition.

Handles ternary responses: A) Up, B) Down, C) None.
Returns (prediction_up, prediction_down) tuples.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Optional


# ── Pattern matchers (order matters: checked top to bottom) ───────────────

_PATTERNS_UP = [
    r"\bA\)",
    r"up-regulat",
    r"upregulat",
    r"increase",
    r"Increase\.",
    r"Answer:\s*A\b",
    r"\*\*A\)\*\*",
    r"^A$",
]

_PATTERNS_DOWN = [
    r"\bB\)",
    r"down-regulat",
    r"downregulat",
    r"decrease",
    r"Decrease\.",
    r"Answer:\s*B\b",
    r"\*\*B\)\*\*",
    r"^B$",
]

_PATTERNS_NONE = [
    r"\bC\)",
    r"does not significantly",
    r"no significant",
    r"does not impact",
    r"not .*result in differential",
    r"not .*significantly affect",
    r"Answer:\s*C\b",
    r"\*\*C\)\*\*",
    r"^C$",
    r"^No[\.\s]*$",
]


def _match_any(text: str, patterns: list[str]) -> bool:
    """Return True if any pattern matches in text."""
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE | re.MULTILINE):
            return True
    return False


def _extract_answer_portion(text: str) -> str:
    """Extract the final answer portion of an LLM response."""
    parts = re.split(r"(?:Final\s+)?Answer\s*:", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return text


def _classify(text: str) -> str | None:
    """Classify text as 'up', 'down', 'none', or None if ambiguous."""
    answer = _extract_answer_portion(text)

    found_up = _match_any(answer, _PATTERNS_UP)
    found_down = _match_any(answer, _PATTERNS_DOWN)
    found_none = _match_any(answer, _PATTERNS_NONE)

    matches = sum([found_up, found_down, found_none])
    if matches == 1:
        if found_up:
            return "up"
        if found_down:
            return "down"
        return "none"

    # Ambiguous in answer portion — try full text
    found_up = _match_any(text, _PATTERNS_UP)
    found_down = _match_any(text, _PATTERNS_DOWN)
    found_none = _match_any(text, _PATTERNS_NONE)

    matches = sum([found_up, found_down, found_none])
    if matches == 1:
        if found_up:
            return "up"
        if found_down:
            return "down"
        return "none"

    return None


# Default prediction for unparseable responses (uniform over 3 classes)
_DEFAULT_UP = round(1 / 3, 3)
_DEFAULT_DOWN = round(1 / 3, 3)

_LABEL_TO_PRED = {
    "up": (1.0, 0.0),
    "down": (0.0, 1.0),
    "none": (0.0, 0.0),
}


def parse_answer(
    text: str,
    default: tuple[float, float] = (_DEFAULT_UP, _DEFAULT_DOWN),
) -> tuple[float, float]:
    """
    Parse a single LLM text response into (prediction_up, prediction_down).

    Args:
        text: Raw LLM output string.
        default: Value pair to return if the answer cannot be parsed.

    Returns:
        Tuple of (prediction_up, prediction_down).

    Examples:
        >>> parse_answer("A) Knockdown of X results in up-regulation of Y.")
        (1.0, 0.0)
        >>> parse_answer("B) down-regulation")
        (0.0, 1.0)
        >>> parse_answer("C) does not significantly affect")
        (0.0, 0.0)
        >>> parse_answer("I don't know")
        (0.333, 0.333)
    """
    if not text or not text.strip():
        return default

    label = _classify(text)
    if label is not None:
        return _LABEL_TO_PRED[label]
    return default


def parse_answers(
    texts: list[str],
    default: tuple[float, float] = (_DEFAULT_UP, _DEFAULT_DOWN),
) -> list[tuple[float, float]]:
    """
    Parse a list of LLM responses into (prediction_up, prediction_down) tuples.

    Args:
        texts: List of raw LLM output strings.
        default: Value pair for unparseable answers.

    Returns:
        List of (prediction_up, prediction_down) tuples.
    """
    return [parse_answer(t, default) for t in texts]


def build_submission(
    ids: list[str],
    predictions_up: list[float],
    predictions_down: list[float],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a submission CSV from IDs and prediction pairs.

    Args:
        ids: List of row IDs (from test.csv).
        predictions_up: List of float predictions for P(upregulated).
        predictions_down: List of float predictions for P(downregulated).
        output_path: If provided, save the submission CSV to this path.

    Returns:
        DataFrame with columns ["id", "prediction_up", "prediction_down"].

    Example:
        >>> df = build_submission(["A_B", "C_D"], [0.8, 0.1], [0.1, 0.7])
        >>> list(df.columns)
        ['id', 'prediction_up', 'prediction_down']
    """
    assert len(ids) == len(predictions_up) == len(predictions_down), (
        f"All inputs must have same length, got {len(ids)}, "
        f"{len(predictions_up)}, {len(predictions_down)}"
    )
    df = pd.DataFrame({
        "id": ids,
        "prediction_up": predictions_up,
        "prediction_down": predictions_down,
    })
    if output_path is not None:
        df.to_csv(output_path, index=False)
    return df
