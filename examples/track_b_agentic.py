"""
Track B -- Agentic tool-use baseline.

Demonstrates a simple agent loop where the LLM can call tools before
producing a final answer. Ships with a placeholder ``gene_lookup`` tool
that participants should replace with their own.

Usage:
    pip install -e .   # from repo root -- installs mlgenx
    python examples/track_b_agentic.py \
        --api-base http://your-endpoint/v1 \
        --api-key YOUR_KEY \
        --model GPT-OSS-120B
"""

from __future__ import annotations

import argparse
import json
import os
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
TOOLS_DIR = ROOT / "examples" / "tools"

MAX_TOOL_CALLS_PER_ROW = 250


# ---------------------------------------------------------------------------
# Tool definitions -- replace / extend with your own
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "gene_lookup",
            "description": (
                "Look up basic information about a mouse gene, including known "
                "pathways and Gene Ontology terms. Returns a short text summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_symbol": {
                        "type": "string",
                        "description": "Mouse gene symbol, e.g. 'Aars'",
                    }
                },
                "required": ["gene_symbol"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    """
    Dispatch a tool call and return the result as a string.

    Participants: add your own tools here. Each tool should accept a dict of
    arguments and return a plain-text result string.
    """
    if name == "gene_lookup":
        symbol = arguments.get("gene_symbol", "unknown")
        return (
            f"{symbol}: This is a placeholder response. Replace the "
            f"gene_lookup tool with a real implementation that queries "
            f"your gene annotation database."
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
    "prediction challenge. You have access to tools that can look up gene "
    "information. Use them if helpful, then provide your final answer.\n\n"
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

    # Copy tools/ into output dir
    out_tools = args.output_dir / "tools"
    if out_tools.exists():
        shutil.rmtree(out_tools)
    out_tools.mkdir()
    # Write the tool definitions as a .py file for auditability
    tool_py = out_tools / "gene_lookup.py"
    tool_py.write_text(
        '"""Placeholder gene_lookup tool -- replace with your implementation."""\n\n'
        "TOOL_SCHEMA = " + json.dumps(TOOL_DEFINITIONS[0], indent=2) + "\n\n\n"
        "def gene_lookup(gene_symbol: str) -> str:\n"
        '    return f"{gene_symbol}: placeholder -- add real implementation"\n'
    )

    # Package zip
    zip_path = args.output_dir / "submission_track_b.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_path, "submission.csv")
        zf.write(prompt_path, "prompt.txt")
        for tool_file in out_tools.glob("*.py"):
            zf.write(tool_file, f"tools/{tool_file.name}")

    print(f"Wrote {sub_path}")
    print(f"Wrote {prompt_path}")
    print(f"Wrote {zip_path}  <-- upload this to Kaggle")


if __name__ == "__main__":
    main()
