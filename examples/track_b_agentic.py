"""
Track B -- Agentic tool-use baseline using DSPy ReAct.

Runs a ReAct agent loop where the LLM reasons in text, decides which tools
to call, and iterates until it produces a final answer.  Uses DSPy's
text-based tool calling which works with any instruction-following model
(no native function-calling API support required).

Ships with three working tools:

  1. train_data_lookup    -- query competition training data (local)
  2. gene_info            -- fetch gene annotations from mygene.info
  3. protein_interactions -- fetch protein interactions from STRING DB

Participants should extend or replace these with their own tools.

Usage:
    pip install -e .          # from repo root -- installs mlgenx
    pip install dspy
    python examples/track_b_agentic.py \\
        --api-base http://your-endpoint/v1 \\
        --api-key YOUR_KEY \\
        --model openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import dspy
import pandas as pd

from mlgenx import format_prompt, parse_answer
from mlgenx.prompts import CELL_DESC, _PROMPT_DE_ZERO, _PROMPT_DIR_ZERO

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data" / "test.csv"
TRAIN_CSV = ROOT / "data" / "train.csv"

_TRAIN_DF: pd.DataFrame | None = None


def _get_train_df() -> pd.DataFrame:
    global _TRAIN_DF
    if _TRAIN_DF is None:
        _TRAIN_DF = pd.read_csv(TRAIN_CSV)
    return _TRAIN_DF


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def train_data_lookup(pert: str = "", gene: str = "") -> str:
    """Search the competition training data for labeled examples involving
    a specific perturbation gene and/or target gene.  Returns matching rows
    with their task type and ground-truth label.  Provide at least one of
    pert or gene."""
    if not pert and not gene:
        return "Error: provide at least one of 'pert' or 'gene'."

    df = _get_train_df()
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
        label_str = {
            ("de", 0): "no differential expression",
            ("de", 1): "differential expression",
            ("dir", 0): "down-regulated",
            ("dir", 1): "up-regulated",
        }.get((r["task"], r["label"]), str(r["label"]))
        lines.append(
            f"  - pert={r['pert']}, gene={r['gene']}, task={r['task']}, "
            f"label={r['label']} ({label_str})"
        )
    return "\n".join(lines)


def gene_info(gene_symbol: str) -> str:
    """Look up annotations for a mouse gene from mygene.info, including
    gene name, summary, Gene Ontology biological process terms, and KEGG
    pathways."""
    url = (
        f"https://mygene.info/v3/query?"
        f"q=symbol:{gene_symbol}&species=mouse"
        f"&fields=symbol,name,summary,go.BP,pathway.kegg&size=1"
    )
    try:
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return f"Error querying mygene.info for {gene_symbol}: {e}"

    hits = data.get("hits", [])
    if not hits:
        return f"No results found for gene symbol '{gene_symbol}' in mouse."

    hit = hits[0]
    lines = [f"Gene: {hit.get('symbol', gene_symbol)}"]

    name = hit.get("name")
    if name:
        lines.append(f"Full name: {name}")

    summary = hit.get("summary")
    if summary:
        lines.append(f"Summary: {summary}")

    go_bp = hit.get("go", {}).get("BP", [])
    if isinstance(go_bp, dict):
        go_bp = [go_bp]
    if go_bp:
        terms = list({t["term"] for t in go_bp if "term" in t})[:8]
        if terms:
            lines.append(
                f"GO Biological Process ({len(terms)} shown): "
                + "; ".join(terms)
            )

    pathways = hit.get("pathway", {}).get("kegg", [])
    if isinstance(pathways, dict):
        pathways = [pathways]
    if pathways:
        pnames = [p.get("name", p.get("id", "?")) for p in pathways][:5]
        lines.append("KEGG Pathways: " + "; ".join(pnames))

    return "\n".join(lines)


def protein_interactions(gene_symbol: str, limit: int = 10) -> str:
    """Fetch known protein-protein interactions for a mouse gene from
    STRING DB.  Returns up to 10 interaction partners with combined
    confidence scores."""
    limit = min(max(1, limit), 50)
    url = (
        f"https://string-db.org/api/json/interaction_partners?"
        f"identifiers={gene_symbol}&species=10090&limit={limit}"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return f"Error querying STRING DB for {gene_symbol}: {e}"

    if not data:
        return (
            f"No protein interactions found for '{gene_symbol}' "
            f"in mouse (STRING DB)."
        )

    lines = [f"Protein interactions for {gene_symbol} (mouse, STRING DB):"]
    for entry in data[:limit]:
        partner = entry.get("preferredName_B", entry.get("stringId_B", "?"))
        score = entry.get("score", 0)
        lines.append(f"  - {partner} (combined score: {score:.3f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_answer_tag(text: str) -> str | None:
    m = re.search(r"<answer>\s*([ABab])\s*</answer>", text)
    return m.group(1).upper() if m else None


def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def save_cache(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


# ---------------------------------------------------------------------------
# DSPy signature
# ---------------------------------------------------------------------------

class BioPredict(dspy.Signature):
    """You are an expert molecular biologist who studies gene expression
    using Perturb-seq.  Use the available tools to look up training data
    and gene annotations, then answer the gene expression question."""

    question: str = dspy.InputField(
        desc="Gene expression prediction question with answer choices"
    )
    answer: str = dspy.OutputField(
        desc="Your answer: A or B, with brief justification"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert molecular biologist participating in a gene expression "
    "prediction challenge. You have access to tools that can:\n"
    "  1) Look up labeled training examples for perturbation/gene pairs\n"
    "  2) Fetch gene annotations (function, GO terms, pathways) from mygene.info\n"
    "  3) Fetch known protein-protein interactions from STRING DB\n\n"
    "Use these tools to gather evidence, reason about the biological "
    "relationship, then provide your final answer.\n\n"
    "Return your final choice in this exact format:\n"
    "<answer>A</answer> or <answer>B</answer>"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track B: Agentic tool-use baseline (DSPy ReAct)"
    )
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--max-iters", type=int, default=5,
        help="Max ReAct iterations (tool-call rounds per row)",
    )
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "track_b"
    )
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Delete cached API responses and start fresh.",
    )
    args = parser.parse_args()

    # ── Configure DSPy ────────────────────────────────────────────────
    # litellm model format: "openai/<model_name_on_server>"
    # The "openai/" prefix selects the OpenAI-compatible provider and is
    # stripped before the request is sent.  Since the vLLM server
    # registers the model as "openai/gpt-oss-120b", we need the full
    # name in the request body, so we always prepend "openai/".
    litellm_model = f"openai/{args.model}"
    lm = dspy.LM(
        model=litellm_model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=1.0,
        num_retries=args.max_retries,
    )
    dspy.configure(
        lm=lm,
        adapter=dspy.ChatAdapter(use_native_function_calling=False),
    )

    react = dspy.ReAct(
        BioPredict,
        tools=[train_data_lookup, gene_info, protein_interactions],
        max_iters=args.max_iters,
    )

    # ── Load data and cache ───────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "responses_cache.json"
    if args.clear_cache and cache_path.exists():
        cache_path.unlink()
        print("Cleared response cache.")
    cache = load_cache(cache_path)
    if "rows" not in cache:
        cache["rows"] = {}

    test_df = pd.read_csv(args.test_csv)
    total = len(test_df)
    new_count = 0

    for idx, row in test_df.iterrows():
        rid = row["id"]
        task = row["task"]

        if rid in cache["rows"] and "prediction" in cache["rows"][rid]:
            print(f"[{idx+1}/{total}] {rid} cache_hit")
            continue

        user_prompt = format_prompt(row["pert"], row["gene"], task)

        tool_calls_count = 0
        trace: Any = {}

        try:
            result = react(question=user_prompt)
            final_text = result.answer or ""
            trajectory = getattr(result, "trajectory", {}) or {}
            trace = trajectory
            tool_calls_count = sum(
                1 for k in trajectory
                if isinstance(k, str) and k.startswith("tool_name")
            )
        except Exception as e:
            print(f"  [error] ReAct failed: {e}")
            final_text = ""
            trace = {"error": str(e)}

        tag = extract_answer_tag(final_text)
        source = tag if tag else final_text
        pred = float(parse_answer(source, task=task, default=0.5))

        cache["rows"][rid] = {
            "task": task,
            "prediction": pred,
            "reasoning_trace": json.dumps(trace, default=str),
            "tokens_used": 0,
            "num_tool_calls": tool_calls_count,
        }

        new_count += 1
        print(
            f"[{idx+1}/{total}] {rid} task={task} pred={pred:.3f} "
            f"tools={tool_calls_count}"
        )
        if new_count % args.save_every == 0:
            save_cache(cache_path, cache)

    save_cache(cache_path, cache)
    print(f"Collected {total} rows ({new_count} new API calls)")

    # ── Build submission CSV ──────────────────────────────────────────
    rows_out = []
    for _, row in test_df.iterrows():
        rid = row["id"]
        c = cache["rows"].get(rid, {})
        rows_out.append({
            "id": rid,
            "prediction": c.get("prediction", 0.5),
            "reasoning_trace": c.get("reasoning_trace", ""),
            "tokens_used": int(c.get("tokens_used", 0)),
            "num_tool_calls": int(c.get("num_tool_calls", 0)),
        })

    sub_df = pd.DataFrame(rows_out)
    sub_path = args.output_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    prompt_path = args.output_dir / "prompt.txt"
    prompt_path.write_text(
        "# System prompt used for Track B (DSPy ReAct)\n\n" + SYSTEM_PROMPT
        + "\n\n# User prompt templates (zero-shot)\n\n"
        "## DE task\n\n"
        + _PROMPT_DE_ZERO.format(pert="{pert}", gene="{gene}", cell_desc=CELL_DESC)
        + "\n\n## Dir task\n\n"
        + _PROMPT_DIR_ZERO.format(pert="{pert}", gene="{gene}", cell_desc=CELL_DESC)
    )

    out_tools = args.output_dir / "tools"
    if out_tools.exists():
        shutil.rmtree(out_tools)
    src_tools = Path(__file__).resolve().parent / "tools"
    if src_tools.exists():
        shutil.copytree(src_tools, out_tools)
    else:
        out_tools.mkdir()
        (out_tools / "__init__.py").write_text("")

    zip_path = args.output_dir / "submission_track_b.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_path, "submission.csv")
        zf.write(prompt_path, "prompt.txt")
        for tool_file in out_tools.rglob("*.py"):
            zf.write(tool_file, f"tools/{tool_file.name}")

    print(f"Wrote {sub_path}")
    print(f"Wrote {prompt_path}")
    print(f"Wrote {zip_path}  <-- upload this to Kaggle")


if __name__ == "__main__":
    main()
