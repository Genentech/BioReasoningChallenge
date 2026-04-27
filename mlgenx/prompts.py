"""
Generate text prompts for the PerturbPair competition.

Prompt templates adapted from PerturbQA (https://github.com/Genentech/PerturbQA).

Each (pert, gene) pair receives a single ternary question:
  A) Upregulated  B) Downregulated  C) No significant effect
"""

from __future__ import annotations

import pandas as pd
from typing import Optional


# ── Cell line context ─────────────────────────────────────────────────────

CELL_DESC = (
    "Mouse bone marrow-derived macrophages (BMDMs) are primary immune cells "
    "differentiated from bone marrow precursors using M-CSF."
)

# ── Prompt templates ──────────────────────────────────────────────────────

_PROMPT_ZERO = """You are an expert molecular biologist who studies how genes are related using Perturb-seq.

Context: {cell_desc}

Question: If you knockdown {pert} using CRISPRi in mouse BMDMs, what is the effect on {gene}?

Your answer must be one of:
A) Knockdown of {pert} results in up-regulation of {gene}.
B) Knockdown of {pert} results in down-regulation of {gene}.
C) Knockdown of {pert} does not significantly affect {gene}.

Answer:"""

# ── Few-shot templates ────────────────────────────────────────────────────

_EXAMPLE_BLOCK = """Example:
- Perturbed gene: {pert}
- Gene of interest: {gene}
Answer: {answer}"""

_PROMPT_FEWSHOT = """You are an expert molecular biologist who studies how genes are related using Perturb-seq.

You are given as Input:
- Perturbed gene: the gene that is perturbed via CRISPRi knockdown
- Gene of interest: the gene whose expression change you wish to predict

Context: {cell_desc}

Question: If you knockdown the perturbed gene using CRISPRi, what is the effect on the gene of interest?

Your answer must end with one of these three choices and nothing else.
A) Knockdown of the perturbed gene results in up-regulation of the gene of interest.
B) Knockdown of the perturbed gene results in down-regulation of the gene of interest.
C) Knockdown of the perturbed gene does not significantly affect the gene of interest.

{examples_block}

Query:
- Perturbed gene: {pert}
- Gene of interest: {gene}
Answer:"""


# ── Answer strings for few-shot examples ──────────────────────────────────

ANSWERS = {
    "up": "A) Knockdown of the perturbed gene results in up-regulation of the gene of interest.",
    "down": "B) Knockdown of the perturbed gene results in down-regulation of the gene of interest.",
    "none": "C) Knockdown of the perturbed gene does not significantly affect the gene of interest.",
}


# ── Public API ────────────────────────────────────────────────────────────

def format_prompt(
    pert: str,
    gene: str,
    examples: Optional[list[dict]] = None,
    cell_desc: str = CELL_DESC,
) -> str:
    """
    Generate a text prompt for a single (pert, gene) query.

    Args:
        pert: Name of the perturbed gene (e.g., "Aars").
        gene: Name of the gene of interest (e.g., "Actb").
        examples: Optional list of few-shot examples, each a dict with keys
            "pert", "gene", "label" (one of "up", "down", "none").
            If None, uses zero-shot.
        cell_desc: Cell line description string.

    Returns:
        Formatted prompt string ready to send to an LLM.

    Example:
        >>> prompt = format_prompt("Aars", "Actb")
        >>> "Aars" in prompt and "Actb" in prompt
        True
        >>> prompt = format_prompt("Aars", "Actb",
        ...     examples=[{"pert": "X", "gene": "Y", "label": "none"}])
        >>> "X" in prompt and "Y" in prompt
        True
    """
    if examples is None:
        return _PROMPT_ZERO.format(pert=pert, gene=gene, cell_desc=cell_desc)

    blocks = []
    for ex in examples:
        blocks.append(_EXAMPLE_BLOCK.format(
            pert=ex["pert"],
            gene=ex["gene"],
            answer=ANSWERS[ex["label"]],
        ))
    examples_block = "\n\n".join(blocks)

    return _PROMPT_FEWSHOT.format(
        pert=pert,
        gene=gene,
        examples_block=examples_block,
        cell_desc=cell_desc,
    )


def format_prompts_from_csv(
    csv_path: str,
    examples: Optional[list[dict]] = None,
    cell_desc: str = CELL_DESC,
) -> pd.DataFrame:
    """
    Generate prompts for every row in a competition CSV (train.csv or test.csv).

    Args:
        csv_path: Path to train.csv or test.csv.
        examples: Optional few-shot examples (applied to all rows).
        cell_desc: Cell line description string.

    Returns:
        DataFrame with columns ["id", "prompt"] for each row in the CSV.

    Example:
        >>> import tempfile, os
        >>> csv = "id,pert,gene,label\\nA_B,A,B,up\\nC_D,C,D,none\\n"
        >>> path = os.path.join(tempfile.mkdtemp(), "test.csv")
        >>> with open(path, "w") as f: _ = f.write(csv)
        >>> df = format_prompts_from_csv(path)
        >>> len(df) == 2
        True
    """
    df = pd.read_csv(csv_path)
    prompts = []
    for _, row in df.iterrows():
        prompts.append({
            "id": row["id"],
            "prompt": format_prompt(
                pert=row["pert"],
                gene=row["gene"],
                examples=examples,
                cell_desc=cell_desc,
            ),
        })
    return pd.DataFrame(prompts)
