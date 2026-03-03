"""
Track B -- Agentic tool-use baseline.

Runs an agent loop where the LLM can call biological tools before producing
a final answer.  Ships with three working tools:

  1. train_data_lookup  -- query competition training data (local, no network)
  2. gene_info          -- fetch gene annotations from mygene.info (public API)
  3. protein_interactions -- fetch protein-protein interactions from STRING DB

Participants should extend or replace these with their own tools.

Usage:
    pip install -e .   # from repo root -- installs mlgenx
    python examples/track_b_agentic.py \\
        --api-base http://your-endpoint/v1 \\
        --api-key YOUR_KEY \\
        --model GPT-OSS-120B
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from mlgenx import format_prompt, parse_answer
from mlgenx.prompts import CELL_DESC, _PROMPT_DE_ZERO, _PROMPT_DIR_ZERO

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data" / "test.csv"
TRAIN_CSV = ROOT / "data" / "train.csv"

MAX_TOOL_CALLS_PER_ROW = 250

# Cache the training data in-memory once (lazy-loaded)
_TRAIN_DF: pd.DataFrame | None = None


def _get_train_df() -> pd.DataFrame:
    global _TRAIN_DF
    if _TRAIN_DF is None:
        _TRAIN_DF = pd.read_csv(TRAIN_CSV)
    return _TRAIN_DF


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
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
                        "description": (
                            "Perturbation gene symbol to search for "
                            "(e.g. 'Stat1'). Optional if gene is provided."
                        ),
                    },
                    "gene": {
                        "type": "string",
                        "description": (
                            "Target gene symbol to search for "
                            "(e.g. 'Irf1'). Optional if pert is provided."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gene_info",
            "description": (
                "Look up annotations for a mouse gene from mygene.info, "
                "including gene name, summary, Gene Ontology biological "
                "process terms, and KEGG pathways."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_symbol": {
                        "type": "string",
                        "description": "Mouse gene symbol (e.g. 'Stat1').",
                    },
                },
                "required": ["gene_symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "protein_interactions",
            "description": (
                "Fetch known protein-protein interactions for a mouse gene "
                "from STRING DB.  Returns up to 10 interaction partners "
                "with combined confidence scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_symbol": {
                        "type": "string",
                        "description": "Mouse gene symbol (e.g. 'Stat1').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of interaction partners (default 10).",
                    },
                },
                "required": ["gene_symbol"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_train_data_lookup(pert: str | None = None, gene: str | None = None) -> str:
    """Query competition training data for matching rows."""
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


def tool_gene_info(gene_symbol: str) -> str:
    """Fetch gene annotations from mygene.info."""
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
            lines.append(f"GO Biological Process ({len(terms)} shown): " + "; ".join(terms))

    pathways = hit.get("pathway", {}).get("kegg", [])
    if isinstance(pathways, dict):
        pathways = [pathways]
    if pathways:
        pnames = [p.get("name", p.get("id", "?")) for p in pathways][:5]
        lines.append(f"KEGG Pathways: " + "; ".join(pnames))

    return "\n".join(lines)


def tool_protein_interactions(gene_symbol: str, limit: int = 10) -> str:
    """Fetch protein-protein interactions from STRING DB."""
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
        return f"No protein interactions found for '{gene_symbol}' in mouse (STRING DB)."

    lines = [f"Protein interactions for {gene_symbol} (mouse, STRING DB):"]
    for entry in data[:limit]:
        partner = entry.get("preferredName_B", entry.get("stringId_B", "?"))
        score = entry.get("score", 0)
        lines.append(f"  - {partner} (combined score: {score:.3f})")

    return "\n".join(lines)


def execute_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call and return the result string."""
    if name == "train_data_lookup":
        return tool_train_data_lookup(
            pert=arguments.get("pert"),
            gene=arguments.get("gene"),
        )
    elif name == "gene_info":
        return tool_gene_info(arguments.get("gene_symbol", ""))
    elif name == "protein_interactions":
        return tool_protein_interactions(
            gene_symbol=arguments.get("gene_symbol", ""),
            limit=arguments.get("limit", 10),
        )
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def post_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None,
    max_tokens: int,
    timeout_s: int,
) -> dict:
    """Raw OpenAI-compatible chat completion request; returns full JSON."""
    url = api_base.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode())


def extract_answer_tag(text: str) -> str | None:
    m = re.search(r"<answer>\s*([ABab])\s*</answer>", text)
    return m.group(1).upper() if m else None


def run_agent_loop(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: List[Dict[str, Any]],
    max_tokens: int,
    timeout_s: int,
    max_retries: int,
) -> Tuple[str, int, float, List[dict]]:
    """
    Run a tool-use agent loop until the model produces a final text answer
    or the tool-call budget is exhausted.

    Returns (final_text, num_tool_calls, total_tokens, trace).
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    trace: List[dict] = []
    num_tool_calls = 0
    total_tokens = 0.0

    while num_tool_calls < MAX_TOOL_CALLS_PER_ROW:
        out = None
        for attempt in range(max_retries + 1):
            try:
                out = post_chat_completion(
                    api_base, api_key, model, messages,
                    tools=tools if num_tool_calls < MAX_TOOL_CALLS_PER_ROW else None,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                )
                break
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(2**attempt)
                else:
                    trace.append({"error": str(e)})
                    return "", num_tool_calls, total_tokens, trace

        usage = (out or {}).get("usage", {}) or {}
        total_tokens += float(usage.get("total_tokens", 0))

        choice = ((out or {}).get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        finish_reason = choice.get("finish_reason", "")

        messages.append(msg)
        trace.append({
            "role": "assistant",
            "content": msg.get("content"),
            "tool_calls": msg.get("tool_calls"),
            "finish_reason": finish_reason,
        })

        tool_calls = msg.get("tool_calls")
        if not tool_calls or finish_reason == "stop":
            return (msg.get("content") or "").strip(), num_tool_calls, total_tokens, trace

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                arguments = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}

            result = execute_tool(name, arguments)
            num_tool_calls += 1

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result,
            }
            messages.append(tool_msg)
            trace.append({
                "role": "tool",
                "name": name,
                "arguments": arguments,
                "result": result,
            })

    return (msg.get("content") or "").strip(), num_tool_calls, total_tokens, trace


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

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
        description="Track B: Agentic tool-use baseline"
    )
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--model", default="GPT-OSS-120B")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "track_b"
    )
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "responses_cache.json"
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

        final_text, tool_calls_count, tokens, trace = run_agent_loop(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=TOOL_DEFINITIONS,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
        )

        tag = extract_answer_tag(final_text)
        source = tag if tag else final_text
        pred = float(parse_answer(source, task=task, default=0.5))

        cache["rows"][rid] = {
            "task": task,
            "prediction": pred,
            "reasoning_trace": json.dumps(trace),
            "tokens_used": int(tokens),
            "num_tool_calls": tool_calls_count,
        }

        new_count += 1
        print(
            f"[{idx+1}/{total}] {rid} task={task} pred={pred:.3f} "
            f"tools={tool_calls_count} tokens={int(tokens)}"
        )
        if new_count % args.save_every == 0:
            save_cache(cache_path, cache)

    save_cache(cache_path, cache)
    print(f"Collected {total} rows ({new_count} new API calls)")

    # Build submission CSV
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

    # Write prompt.txt
    prompt_path = args.output_dir / "prompt.txt"
    prompt_path.write_text(
        "# System prompt used for Track B\n\n" + SYSTEM_PROMPT
        + "\n\n# User prompt templates (zero-shot)\n\n"
        "## DE task\n\n"
        + _PROMPT_DE_ZERO.format(pert="{pert}", gene="{gene}", cell_desc=CELL_DESC)
        + "\n\n## Dir task\n\n"
        + _PROMPT_DIR_ZERO.format(pert="{pert}", gene="{gene}", cell_desc=CELL_DESC)
    )

    # Copy tools/ into output dir as standalone .py files
    out_tools = args.output_dir / "tools"
    if out_tools.exists():
        shutil.rmtree(out_tools)
    src_tools = Path(__file__).resolve().parent / "tools"
    if src_tools.exists():
        shutil.copytree(src_tools, out_tools)
    else:
        out_tools.mkdir()
        (out_tools / "__init__.py").write_text("")

    # Package zip
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
