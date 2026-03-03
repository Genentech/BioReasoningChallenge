"""
Generate text prompts in the right format for the PerturbPair competition.

Prompt templates adapted from PerturbQA (https://github.com/Genentech/PerturbQA).
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
# These are "zero-shot" templates that only need (pert, gene).
# For few-shot, use format_prompt(..., examples=...).

_PROMPT_DE_ZERO = """You are an expert molecular biologist who studies how genes are related using Perturb-seq.

Context: {cell_desc}

Question: If you knockdown {pert} using CRISPRi in mouse BMDMs, does it result in differential expression of {gene}?

Your answer must be one of:
A) Knockdown of {pert} does not impact {gene}.
B) Knockdown of {pert} results in differential expression of {gene}.

Answer:"""

_PROMPT_DIR_ZERO = """You are an expert molecular biologist who studies how genes are related using Perturb-seq.

Context: {cell_desc}

Question: When you knockdown {pert} using CRISPRi in mouse BMDMs, {gene} is differentially expressed. Is {gene} down-regulated or up-regulated?

Your answer must be one of:
A) Knockdown of {pert} results in down-regulation of {gene}.
B) Knockdown of {pert} results in up-regulation of {gene}.

Answer:"""

# ── Few-shot templates ────────────────────────────────────────────────────

_EXAMPLE_BLOCK_DE = """Example:
- Perturbed gene: {pert}
- Gene of interest: {gene}
Answer: {answer}"""

_EXAMPLE_BLOCK_DIR = """Example:
- Perturbed gene: {pert}
- Gene of interest: {gene}
Answer: {answer}"""

_PROMPT_DE_FEWSHOT = """You are an expert molecular biologist who studies how genes are related using Perturb-seq.

You are given as Input:
- Perturbed gene: the gene that is perturbed via CRISPRi knockdown
- Gene of interest: the gene whose expression change you wish to predict

Context: {cell_desc}

Question: If you knockdown the perturbed gene using CRISPRi, does it result in differential expression of the gene of interest?

Your answer must end with one of these two choices and nothing else.
A) Knockdown of the perturbed gene does not impact the gene of interest.
B) Knockdown of the perturbed gene results in differential expression of the gene of interest.

{examples_block}

Query:
- Perturbed gene: {pert}
- Gene of interest: {gene}
Answer:"""

_PROMPT_DIR_FEWSHOT = """You are an expert molecular biologist who studies how genes are related using Perturb-seq.

You are given as Input:
- Perturbed gene: the gene that is perturbed via CRISPRi knockdown
- Gene of interest: the gene whose expression change you wish to predict

Context: {cell_desc}

Question: When you knockdown the perturbed gene using CRISPRi, the gene of interest is differentially expressed. Is the gene of interest down-regulated or up-regulated?

Your answer must end with one of these two choices and nothing else.
A) Knockdown of the perturbed gene results in down-regulation of the gene of interest.
B) Knockdown of the perturbed gene results in up-regulation of the gene of interest.

{examples_block}

Query:
- Perturbed gene: {pert}
- Gene of interest: {gene}
Answer:"""


# ── Answer strings for few-shot examples ──────────────────────────────────

ANSWERS_DE = {
    0: "A) Knockdown of the perturbed gene does not impact the gene of interest.",
    1: "B) Knockdown of the perturbed gene results in differential expression of the gene of interest.",
}

ANSWERS_DIR = {
    0: "A) Knockdown of the perturbed gene results in down-regulation of the gene of interest.",
    1: "B) Knockdown of the perturbed gene results in up-regulation of the gene of interest.",
}


# ── Public API ────────────────────────────────────────────────────────────

def format_prompt(
    pert: str,
    gene: str,
    task: str,
    examples: Optional[list[dict]] = None,
    cell_desc: str = CELL_DESC,
) -> str:
    """
    Generate a text prompt for a single (pert, gene, task) query.

    Args:
        pert: Name of the perturbed gene (e.g., "Aars").
        gene: Name of the gene of interest (e.g., "Actb").
        task: One of "de" (differential expression) or "dir" (direction).
        examples: Optional list of few-shot examples, each a dict with keys
            "pert", "gene", "label" (int 0 or 1). If None, uses zero-shot.
        cell_desc: Cell line description string.

    Returns:
        Formatted prompt string ready to send to an LLM.

    Example:
        >>> prompt = format_prompt("Aars", "Actb", "de")
        >>> "Aars" in prompt and "Actb" in prompt
        True
        >>> prompt = format_prompt("Aars", "Actb", "de",
        ...     examples=[{"pert": "X", "gene": "Y", "label": 0}])
        >>> "X" in prompt and "Y" in prompt
        True
    """
    if task not in ("de", "dir"):
        raise ValueError(f"task must be 'de' or 'dir', got {task!r}")

    if examples is None:
        # Zero-shot
        template = _PROMPT_DE_ZERO if task == "de" else _PROMPT_DIR_ZERO
        return template.format(pert=pert, gene=gene, cell_desc=cell_desc)

    # Few-shot
    answers = ANSWERS_DE if task == "de" else ANSWERS_DIR
    example_template = _EXAMPLE_BLOCK_DE if task == "de" else _EXAMPLE_BLOCK_DIR
    blocks = []
    for ex in examples:
        blocks.append(example_template.format(
            pert=ex["pert"],
            gene=ex["gene"],
            answer=answers[ex["label"]],
        ))
    examples_block = "\n\n".join(blocks)

    template = _PROMPT_DE_FEWSHOT if task == "de" else _PROMPT_DIR_FEWSHOT
    return template.format(
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
        >>> csv = "id,pert,gene,task,label\\nde_A_B,A,B,de,1\\ndir_A_B,A,B,dir,0\\n"
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
                task=row["task"],
                examples=examples,
                cell_desc=cell_desc,
            ),
        })
    return pd.DataFrame(prompts)
