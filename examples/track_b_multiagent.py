"""
Track B -- Multi-agent tool-use baseline using DSPy ReAct.

A coordinator agent delegates to specialist sub-agents, each of which is
itself a DSPy ReAct loop backed by the same LLM.  Sub-agents have access
to different tool subsets:

  - biology_expert : gene_info, protein_interactions
  - data_analyst   : lookup_pert, lookup_gene

The coordinator sees the original question, consults one or both
specialists, synthesizes their findings, and calls submit_answer.

All traces (coordinator + nested sub-agents) are captured in a
hierarchical JSON structure written to the reasoning_trace column.
Token and tool-call counts aggregate across all agents.

Usage:
    pip install -e .          # from repo root -- installs mlgenx
    pip install dspy

    python examples/track_b_multiagent.py \\
        --api-base http://your-endpoint/v1 \\
        --api-key YOUR_KEY \\
        --model openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import threading
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import dspy
import pandas as pd

from mlgenx import format_prompt, parse_answer
from mlgenx.prompts import CELL_DESC, _PROMPT_ZERO

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
# Leaf-level tool implementations (used by sub-agents)
# ---------------------------------------------------------------------------

_LABEL_STR = {
    "up": "up-regulated",
    "down": "down-regulated",
    "none": "not differentially expressed",
}


def lookup_pert(pert: str) -> str:
    """Look up all training examples where a given gene was knocked down
    (CRISPRi perturbation).  Returns a summary of target genes grouped by
    label (up-regulated, down-regulated, or not differentially expressed).
    Use this to understand the downstream effects of perturbing a gene."""
    df = _get_train_df()
    hits = df[df["pert"].str.lower() == pert.lower()]
    if hits.empty:
        return f"No training examples found for perturbation '{pert}'."

    lab = hits["label"].astype(str).str.lower()
    up_genes = hits[lab == "up"]["gene"].tolist()
    down_genes = hits[lab == "down"]["gene"].tolist()
    none_genes = hits[lab == "none"]["gene"].tolist()

    lines = [f"Training data for perturbation '{pert}' ({len(hits)} examples):"]
    lines.append(
        f"  Summary: {len(up_genes)} up-regulated, "
        f"{len(down_genes)} down-regulated, "
        f"{len(none_genes)} not differentially expressed"
    )
    if up_genes:
        lines.append(f"    Up-regulated: {', '.join(up_genes[:30])}")
    if down_genes:
        lines.append(f"    Down-regulated: {', '.join(down_genes[:30])}")
    if none_genes:
        lines.append(
            f"    Not differentially expressed: {', '.join(none_genes[:30])}"
        )
    return "\n".join(lines)


def lookup_gene(gene: str) -> str:
    """Look up all training examples where a given gene was the measurement
    target (its expression was checked after some perturbation).  Returns
    which perturbations led to up-, down-, or no regulation of this gene.
    Use this to understand what regulates a gene."""
    df = _get_train_df()
    hits = df[df["gene"].str.lower() == gene.lower()]
    if hits.empty:
        return f"No training examples found for target gene '{gene}'."

    lines = [f"Training data for target gene '{gene}' ({len(hits)} examples):"]
    for _, r in hits.iterrows():
        lab = str(r["label"]).lower()
        label_str = _LABEL_STR.get(lab, str(r["label"]))
        lines.append(f"  - pert={r['pert']}: {label_str}")
    return "\n".join(lines)


def submit_answer(answer: str, reasoning: str = "") -> str:
    """Submit your final answer to the gene expression question.  You MUST
    call this tool when you have reached a conclusion.  The answer parameter
    must be exactly 'A', 'B', or 'C'."""
    answer = answer.strip().upper()
    if answer not in ("A", "B", "C"):
        return f"Error: answer must be 'A', 'B', or 'C', got '{answer}'."
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
    """Sum total_tokens from LM history entries added since start_idx.

    DSPy caps lm.history at max_history_size (default 10 000) using a deque.
    Once the cap is hit, new entries evict old ones from the front while
    len(history) stays constant, so an absolute start_idx would overshoot.
    Instead we compute how many *new* entries were appended and slice from
    the tail.
    """
    new_count = len(lm.history) - start_idx
    if new_count <= 0:
        return 0
    total = 0
    for entry in list(lm.history)[-new_count:]:
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


def _count_tool_calls(trajectory: dict) -> int:
    return sum(
        1 for k in trajectory
        if isinstance(k, str) and k.startswith("tool_name")
    )


def extract_answer_tag(text: str) -> str | None:
    m = re.search(r"<answer>\s*([ABCabc])\s*</answer>", text)
    return m.group(1).upper() if m else None


def _safe_trajectory(trajectory: Any) -> dict:
    """Convert a trajectory to a JSON-safe dict."""
    if isinstance(trajectory, dict):
        return {k: str(v) for k, v in trajectory.items()}
    return {"raw": str(trajectory)}


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
# DSPy signatures
# ---------------------------------------------------------------------------

class BiologyAnalysis(dspy.Signature):
    """You are a biology expert who studies gene regulation in mouse
    macrophages.  Use gene_info and protein_interactions tools to gather
    biological evidence, then summarize your findings."""

    question: str = dspy.InputField(
        desc="A biology question about gene regulation or expression"
    )
    answer: str = dspy.OutputField(
        desc="Your biological analysis and findings"
    )


class DataAnalysis(dspy.Signature):
    """You are a data analyst with access to Perturb-seq training data.
    Use lookup_pert and lookup_gene tools to search for relevant patterns,
    then summarize what the data shows."""

    question: str = dspy.InputField(
        desc="A question about perturbation effects in the training data"
    )
    answer: str = dspy.OutputField(
        desc="Your data analysis findings"
    )


class CoordinatorPredict(dspy.Signature):
    """You are a coordinator for a gene expression prediction challenge.
    You lead a team of specialist agents: biology_expert (for gene
    annotations, pathways, protein interactions) and data_analyst (for
    searching training data patterns).  Consult them with targeted
    questions, synthesize findings, and call submit_answer with A, B, or C."""

    question: str = dspy.InputField(
        desc="Gene expression prediction question with answer choices"
    )
    answer: str = dspy.OutputField(
        desc="Your answer: A, B, or C, with brief justification"
    )


# ---------------------------------------------------------------------------
# Sub-agent tool factories
# ---------------------------------------------------------------------------

def make_sub_agent_tools(
    lm: dspy.LM,
    sub_agent_traces: List[dict],
    max_sub_iters: int,
) -> tuple:
    """Create biology_expert and data_analyst tool functions.

    Each tool wraps a DSPy ReAct sub-agent.  Traces from sub-agent runs
    are appended to sub_agent_traces (shared mutable list for the current
    row).
    """
    bio_react = dspy.ReAct(
        BiologyAnalysis,
        tools=[gene_info, protein_interactions],
        max_iters=max_sub_iters,
    )
    data_react = dspy.ReAct(
        DataAnalysis,
        tools=[lookup_pert, lookup_gene],
        max_iters=max_sub_iters,
    )

    def biology_expert(question: str) -> str:
        """Consult a biology expert sub-agent that can look up gene
        annotations from mygene.info and protein interactions from STRING
        DB.  Ask specific biology questions about gene function, pathways,
        or protein-protein interactions relevant to the prediction."""
        history_before = len(lm.history)
        try:
            result = bio_react(question=question)
            trajectory = getattr(result, "trajectory", {}) or {}
            answer_text = result.answer or ""
            tokens = _tokens_from_history(lm, history_before)
            tool_calls = _count_tool_calls(trajectory)
            sub_agent_traces.append({
                "agent": "biology_expert",
                "question": question,
                "answer": answer_text,
                "trajectory": _safe_trajectory(trajectory),
                "tokens": tokens,
                "tool_calls": tool_calls,
            })
            return f"Biology expert findings:\n{answer_text}"
        except Exception as e:
            sub_agent_traces.append({
                "agent": "biology_expert",
                "question": question,
                "error": str(e),
            })
            return f"Biology expert encountered an error: {e}"

    def data_analyst(question: str) -> str:
        """Consult a data analyst sub-agent that can search the training
        data for perturbation effects and expression patterns.  Ask about
        what happens when specific genes are perturbed, or what regulates
        a given gene in the training data."""
        history_before = len(lm.history)
        try:
            result = data_react(question=question)
            trajectory = getattr(result, "trajectory", {}) or {}
            answer_text = result.answer or ""
            tokens = _tokens_from_history(lm, history_before)
            tool_calls = _count_tool_calls(trajectory)
            sub_agent_traces.append({
                "agent": "data_analyst",
                "question": question,
                "answer": answer_text,
                "trajectory": _safe_trajectory(trajectory),
                "tokens": tokens,
                "tool_calls": tool_calls,
            })
            return f"Data analyst findings:\n{answer_text}"
        except Exception as e:
            sub_agent_traces.append({
                "agent": "data_analyst",
                "question": question,
                "error": str(e),
            })
            return f"Data analyst encountered an error: {e}"

    return biology_expert, data_analyst


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT_PATH = (
    ROOT / "examples" / "b_multiagent_system_prompt.txt"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track B: Multi-agent tool-use baseline (DSPy ReAct)"
    )
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--reasoning-effort", default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort level sent to the model (default: low).",
    )
    parser.add_argument(
        "--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT_PATH,
        help="Path to system prompt file (contents used as-is).",
    )
    parser.add_argument(
        "--max-iters", type=int, default=20,
        help="Max ReAct iterations for the coordinator agent.",
    )
    parser.add_argument(
        "--max-sub-iters", type=int, default=5,
        help="Max ReAct iterations for each sub-agent call.",
    )
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "track_b_multi"
    )
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of concurrent rows to process. Increase to speed up.",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Delete cached API responses and start fresh.",
    )
    parser.add_argument(
        "--model-name", default=None,
        help="Override model name recorded in submission (defaults to --model).",
    )
    args = parser.parse_args()

    model_name = args.model_name or args.model

    system_prompt = args.system_prompt.read_text().strip()
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(enc.encode(system_prompt))
    except Exception:
        prompt_tokens = len(system_prompt) // 4
    print(
        f"System prompt loaded from {args.system_prompt} "
        f"({len(system_prompt)} chars, ~{prompt_tokens} tokens)"
    )
    if prompt_tokens > 16384:
        parser.error(
            f"System prompt is ~{prompt_tokens} tokens, exceeds 16,384 limit. "
            f"Shorten {args.system_prompt} and retry."
        )

    # ── Configure DSPy ────────────────────────────────────────────────
    litellm_model = f"openai/{args.model}"
    lm = dspy.LM(
        model=litellm_model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=1.0,
        num_retries=args.max_retries,
        reasoning_effort=args.reasoning_effort,
        allowed_openai_params=["reasoning_effort"],
    )
    dspy.configure(
        lm=lm,
        adapter=dspy.ChatAdapter(use_native_function_calling=False),
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
    cache_lock = threading.Lock()
    new_count = 0

    def process_row(idx: int, row: pd.Series) -> None:
        nonlocal new_count
        rid = row["id"]

        with cache_lock:
            if rid in cache["rows"] and "prediction_up" in cache["rows"][rid]:
                print(f"[{idx+1}/{total}] {rid} cache_hit")
                return

        user_prompt = format_prompt(row["pert"], row["gene"])

        sub_agent_traces: List[dict] = []
        biology_expert, data_analyst = make_sub_agent_tools(
            lm, sub_agent_traces, args.max_sub_iters,
        )

        coordinator_tools = [biology_expert, data_analyst, submit_answer]
        num_distinct_tools = len(set([
            "biology_expert", "data_analyst", "submit_answer",
            "gene_info", "protein_interactions",
            "lookup_pert", "lookup_gene",
        ]))
        coordinator = dspy.ReAct(
            CoordinatorPredict,
            tools=coordinator_tools,
            max_iters=args.max_iters,
        )

        coordinator_trace: Any = {}
        coordinator_tool_calls = 0
        tokens = 0

        history_before = len(lm.history)
        try:
            result = coordinator(question=user_prompt)
            final_text = result.answer or ""
            trajectory = getattr(result, "trajectory", {}) or {}
            coordinator_trace = trajectory
            coordinator_tool_calls = _count_tool_calls(trajectory)
        except Exception as e:
            print(f"  [error] Coordinator failed: {e}")
            final_text = ""
            coordinator_trace = {"error": str(e)}
        tokens = _tokens_from_history(lm, history_before)

        sub_agent_tool_calls = sum(
            t.get("tool_calls", 0) for t in sub_agent_traces
        )
        total_tool_calls = coordinator_tool_calls + sub_agent_tool_calls

        full_trace = {
            "coordinator": _safe_trajectory(coordinator_trace),
            "sub_agents": sub_agent_traces,
        }

        submitted = None
        if isinstance(coordinator_trace, dict):
            for k in sorted(coordinator_trace.keys()):
                if (
                    k.startswith("tool_name_")
                    and coordinator_trace[k] == "submit_answer"
                ):
                    step = k.split("_")[-1]
                    args_d = coordinator_trace.get(f"tool_args_{step}", {})
                    submitted = args_d.get("answer", "").strip().upper()

        if submitted in ("A", "B", "C"):
            source = submitted
        else:
            tag = extract_answer_tag(final_text)
            source = tag if tag else (final_text or "")

        pred_up, pred_down = parse_answer(source)

        with cache_lock:
            cache["rows"][rid] = {
                "prediction_up": pred_up,
                "prediction_down": pred_down,
                "reasoning_trace": json.dumps(full_trace, default=str),
                "tokens_used": tokens,
                "num_tool_calls": total_tool_calls,
                "model_name": model_name,
            }
            new_count += 1
            print(
                f"[{idx+1}/{total}] {rid} pred_up={pred_up:.3f} "
                f"pred_down={pred_down:.3f} "
                f"tools={total_tool_calls} "
                f"(coordinator={coordinator_tool_calls}, "
                f"sub_agents={sub_agent_tool_calls}) "
                f"tokens={tokens}"
            )
            if new_count % args.save_every == 0:
                save_cache(cache_path, cache)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(process_row, idx, row)
            for idx, row in test_df.iterrows()
        ]
        for future in as_completed(futures):
            future.result()

    save_cache(cache_path, cache)
    print(f"Collected {total} rows ({new_count} new API calls)")

    # ── Build submission CSV ──────────────────────────────────────────
    rows_out = []
    for _, row in test_df.iterrows():
        rid = row["id"]
        c = cache["rows"].get(rid, {})
        rows_out.append({
            "id": rid,
            "prediction_up": c.get("prediction_up", round(1 / 3, 3)),
            "prediction_down": c.get("prediction_down", round(1 / 3, 3)),
            "reasoning_trace": c.get("reasoning_trace", ""),
            "tokens_used": int(c.get("tokens_used", 0)),
            "num_tool_calls": int(c.get("num_tool_calls", 0)),
            "prompt_tokens": prompt_tokens,
            "num_distinct_tools": num_distinct_tools,
            "model_name": c.get("model_name", model_name),
        })

    sub_df = pd.DataFrame(rows_out)
    sub_path = args.output_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    prompt_path = args.output_dir / "prompt.txt"
    prompt_path.write_text(
        "# System prompt used for Track B (Multi-agent)\n\n"
        + system_prompt
        + "\n\n# User prompt template (zero-shot)\n\n"
        + _PROMPT_ZERO.format(pert="{pert}", gene="{gene}", cell_desc=CELL_DESC)
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
