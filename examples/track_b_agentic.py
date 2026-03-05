"""
Track B -- Agentic tool-use baseline using DSPy ReAct.

Runs a ReAct agent loop where the LLM reasons in text, decides which tools
to call, and iterates until it produces a final answer.  Uses DSPy's
text-based tool calling which works with any instruction-following model
(no native function-calling API support required).

Ships with five working tools:

  1. lookup_pert          -- what genes are affected by a perturbation?
  2. lookup_gene          -- what perturbations affect a gene?
  3. gene_info            -- fetch gene annotations from mygene.info
  4. protein_interactions -- fetch protein interactions from STRING DB
  5. submit_answer        -- submit the final A/B answer

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

_LABEL_STR = {
    ("de", 0): "not differentially expressed",
    ("de", 1): "differentially expressed",
    ("dir", 0): "down-regulated",
    ("dir", 1): "up-regulated",
}


def lookup_pert(pert: str) -> str:
    """Look up all training examples where a given gene was knocked down
    (CRISPRi perturbation).  Returns a summary of which target genes were
    differentially expressed and which were not, plus direction info if
    available.  Use this to understand the downstream effects of perturbing
    a gene."""
    df = _get_train_df()
    hits = df[df["pert"].str.lower() == pert.lower()]
    if hits.empty:
        return f"No training examples found for perturbation '{pert}'."

    de = hits[hits["task"] == "de"]
    de_yes = de[de["label"] == 1]["gene"].tolist()
    de_no = de[de["label"] == 0]["gene"].tolist()
    dir_hits = hits[hits["task"] == "dir"]
    dir_up = dir_hits[dir_hits["label"] == 1]["gene"].tolist()
    dir_down = dir_hits[dir_hits["label"] == 0]["gene"].tolist()

    lines = [f"Training data for perturbation '{pert}' ({len(hits)} examples):"]
    if de_yes or de_no:
        lines.append(
            f"  DE: {len(de_yes)} differentially expressed, "
            f"{len(de_no)} not impacted"
        )
        if de_yes:
            lines.append(f"    DE genes: {', '.join(de_yes[:30])}")
        if de_no:
            lines.append(f"    Not impacted: {', '.join(de_no[:30])}")
    if dir_up or dir_down:
        lines.append(
            f"  Direction: {len(dir_up)} up-regulated, "
            f"{len(dir_down)} down-regulated"
        )
        if dir_up:
            lines.append(f"    Up-regulated: {', '.join(dir_up[:30])}")
        if dir_down:
            lines.append(f"    Down-regulated: {', '.join(dir_down[:30])}")
    return "\n".join(lines)


def lookup_gene(gene: str) -> str:
    """Look up all training examples where a given gene was the measurement
    target (its expression was checked after some perturbation).  Returns
    which perturbations caused differential expression of this gene and
    which did not.  Use this to understand what regulates a gene."""
    df = _get_train_df()
    hits = df[df["gene"].str.lower() == gene.lower()]
    if hits.empty:
        return f"No training examples found for target gene '{gene}'."

    lines = [f"Training data for target gene '{gene}' ({len(hits)} examples):"]
    for _, r in hits.iterrows():
        label_str = _LABEL_STR.get((r["task"], r["label"]), str(r["label"]))
        lines.append(
            f"  - pert={r['pert']}, task={r['task']}: {label_str}"
        )
    return "\n".join(lines)


def submit_answer(answer: str, reasoning: str = "") -> str:
    """Submit your final answer to the gene expression question.  You MUST
    call this tool when you have reached a conclusion.  The answer parameter
    must be exactly 'A' or 'B'."""
    answer = answer.strip().upper()
    if answer not in ("A", "B"):
        return f"Error: answer must be 'A' or 'B', got '{answer}'."
    return f"Answer recorded: {answer}"


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

def _tokens_from_history(lm: dspy.LM, start_idx: int) -> int:
    """Sum total_tokens from LM history entries added since start_idx."""
    total = 0
    for entry in lm.history[start_idx:]:
        if isinstance(entry, dict):
            usage = entry.get("usage") or {}
            if isinstance(usage, dict):
                total += usage.get("total_tokens", 0)
            else:
                total += getattr(usage, "total_tokens", 0) or 0
        elif hasattr(entry, "usage"):
            u = entry.usage
            total += getattr(u, "total_tokens", 0) if u else 0
    return total


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
    and gene annotations, then call submit_answer with your final choice
    (A or B)."""

    question: str = dspy.InputField(
        desc="Gene expression prediction question with answer choices"
    )
    answer: str = dspy.OutputField(
        desc="Your answer: A or B, with brief justification"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT_PATH = ROOT / "examples" / "b_system_prompt.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track B: Agentic tool-use baseline (DSPy ReAct)"
    )
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT_PATH,
        help="Path to system prompt file (contents used as-is).",
    )
    parser.add_argument(
        "--max-iters", type=int, default=250,
        help="Max ReAct iterations (tool-call rounds per row)",
    )
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "track_b"
    )
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Delete cached API responses and start fresh.",
    )
    args = parser.parse_args()

    system_prompt = args.system_prompt.read_text().strip()
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(enc.encode(system_prompt))
    except Exception:
        prompt_tokens = len(system_prompt) // 4  # rough fallback
    print(
        f"System prompt loaded from {args.system_prompt} "
        f"({len(system_prompt)} chars, ~{prompt_tokens} tokens)"
    )
    if prompt_tokens > 4096:
        parser.error(
            f"System prompt is ~{prompt_tokens} tokens, exceeds 4096 limit. "
            f"Shorten {args.system_prompt} and retry."
        )

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

    tool_list = [
        lookup_pert, lookup_gene, gene_info,
        protein_interactions, submit_answer,
    ]
    if len(tool_list) > 100:
        parser.error(
            f"Too many tools ({len(tool_list)}), competition limit is 100."
        )
    print(f"Tools: {len(tool_list)}, max_iters: {args.max_iters}")

    react = dspy.ReAct(
        BioPredict,
        tools=tool_list,
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
        tokens = 0
        trace: Any = {}

        history_before = len(lm.history)
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
        tokens = _tokens_from_history(lm, history_before)

        # Extract answer: prefer explicit submit_answer call, then fall
        # back to parsing the ReAct answer text.
        submitted = None
        for k in sorted(trace.keys() if isinstance(trace, dict) else []):
            if k.startswith("tool_name_") and trace[k] == "submit_answer":
                step = k.split("_")[-1]
                args_d = trace.get(f"tool_args_{step}", {})
                submitted = args_d.get("answer", "").strip().upper()

        if submitted in ("A", "B"):
            source = submitted
        else:
            source = "" #No answer submitted

        #    tag = extract_answer_tag(final_text)
        #    source = tag if tag else final_text
        pred = float(parse_answer(source, task=task, default=0.5))

        cache["rows"][rid] = {
            "task": task,
            "prediction": pred,
            "reasoning_trace": json.dumps(trace, default=str),
            "tokens_used": tokens,
            "num_tool_calls": tool_calls_count,
        }

        new_count += 1
        print(
            f"[{idx+1}/{total}] {rid} task={task} pred={pred:.3f} "
            f"tools={tool_calls_count} tokens={tokens}"
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
        "# System prompt used for Track B (DSPy ReAct)\n\n" + system_prompt
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
        zf.write(args.system_prompt, args.system_prompt.name)
        for tool_file in out_tools.rglob("*.py"):
            zf.write(tool_file, f"tools/{tool_file.name}")

    print(f"Wrote {sub_path}")
    print(f"Wrote {prompt_path}")
    print(f"Wrote {zip_path}  <-- upload this to Kaggle")


if __name__ == "__main__":
    main()
