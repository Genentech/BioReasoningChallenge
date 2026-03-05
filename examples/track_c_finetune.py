"""
Track C -- Fine-tuning inference baseline.

Runs inference against a locally-served fine-tuned model (e.g. via vLLM)
and captures the full reasoning trace including <think>...</think> blocks.
Packages results into a zip ready for Kaggle upload.

Fine-tune a model first (using the train extra -- see README):
    uv sync --extra train
    uv run --extra train python examples/finetune.py  # -> outputs/finetuned_model/

Then switch to the serve environment and start vLLM:
    uv sync --extra serve
    uv run --extra serve vllm serve outputs/finetuned_model/ --host 0.0.0.0 --port 8000

Then run this script (in a separate terminal, same serve environment):
    uv run --extra serve python examples/track_c_finetune.py \\
        --api-base http://localhost:8000/v1 \\
        --model outputs/finetuned_model/

You can also point --model at any HuggingFace model ID served via vLLM,
e.g. --model Qwen/Qwen3-4B-Thinking-2507 for the un-tuned base model.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from mlgenx import format_prompt, parse_answer
from mlgenx.prompts import CELL_DESC, _PROMPT_DE_ZERO, _PROMPT_DIR_ZERO

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data" / "test.csv"


def post_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: int,
) -> Tuple[str, Dict[str, float], Dict[str, str]]:
    """
    Call an OpenAI-compatible endpoint and return
    (full_text, token_stats, {reasoning_trace, response_trace}).
    """
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        out = json.loads(resp.read().decode())

    usage = out.get("usage", {}) or {}
    c_details = usage.get("completion_tokens_details", {}) or {}
    completion_tokens = float(usage.get("completion_tokens", 0))
    reasoning_tokens = float(
        c_details.get("reasoning_tokens", usage.get("reasoning_tokens", 0))
    )
    token_stats = {
        "prompt_tokens": float(usage.get("prompt_tokens", 0)),
        "completion_tokens": completion_tokens,
        "total_tokens": float(usage.get("total_tokens", 0)),
        "reasoning_tokens": reasoning_tokens,
    }

    traces: Dict[str, str] = {"reasoning_trace": "", "response_trace": ""}

    choices = out.get("choices", [])
    if not choices:
        return "", token_stats, traces

    msg = choices[0].get("message", {}) or {}

    # Some servers expose reasoning in a separate field
    reasoning = msg.get("reasoning", msg.get("reasoning_content"))
    if reasoning:
        traces["reasoning_trace"] = str(reasoning).strip()

    content = msg.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            str(c.get("text", c.get("content", "")))
            for c in content
            if isinstance(c, dict)
        )
    text = str(content).strip()
    traces["response_trace"] = text

    # Extract <think>...</think> blocks (common in reasoning models)
    m = re.search(r"<think>\s*(.*?)\s*</think>\s*(.*)", text, flags=re.DOTALL)
    if m:
        if not traces["reasoning_trace"]:
            traces["reasoning_trace"] = m.group(1).strip()
        traces["response_trace"] = m.group(2).strip()

    return text, token_stats, traces


def append_answer_tag(prompt: str) -> str:
    return (
        f"{prompt.rstrip()}\n\n"
        "Return ONLY the final choice in this exact format:\n"
        "<answer>A</answer> or <answer>B</answer>\n"
        "Do not include any other text."
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track C: Fine-tuning inference baseline"
    )
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument(
        "--model", default="outputs/finetuned_model",
        help="Model name or path. Use the output of finetune.py, or a "
             "HuggingFace model ID (e.g. Qwen/Qwen3-4B-Thinking-2507).",
    )
    parser.add_argument(
        "--base-model", default=None,
        help="If serving a LoRA adapter separately, specify the base model "
             "name here (used only for the model_name field in submission).",
    )
    parser.add_argument("--max-tokens", type=int, default=16000,
                        help="Max new tokens (Track C allows up to 16,000)")
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "track_c"
    )
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    model_name = args.model
    display_name = args.base_model if args.base_model else model_name

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

        prompt_raw = format_prompt(row["pert"], row["gene"], task)
        prompt = append_answer_tag(prompt_raw)

        text = ""
        token_stats: Dict[str, float] = {}
        traces: Dict[str, str] = {"reasoning_trace": "", "response_trace": ""}
        for attempt in range(args.max_retries + 1):
            try:
                text, token_stats, traces = post_chat_completion(
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model=model_name,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout_s,
                )
                break
            except Exception as e:
                print(f"  attempt={attempt+1} error={e}")
                if attempt < args.max_retries:
                    time.sleep(2**attempt)

        tag = extract_answer_tag(text)
        source = tag if tag else traces.get("response_trace", text)
        pred = float(parse_answer(source, task=task, default=0.5))

        # Combine reasoning + response into a single trace string
        reasoning_trace = traces.get("reasoning_trace", "")
        response_trace = traces.get("response_trace", text)
        full_trace = ""
        if reasoning_trace:
            full_trace = f"<think>\n{reasoning_trace}\n</think>\n{response_trace}"
        else:
            full_trace = response_trace

        cache["rows"][rid] = {
            "task": task,
            "prediction": pred,
            "reasoning_trace": full_trace,
            "tokens_used": int(token_stats.get("total_tokens", 0)),
            "model_name": display_name,
        }

        new_count += 1
        print(
            f"[{idx+1}/{total}] {rid} task={task} pred={pred:.3f} "
            f"tokens={int(token_stats.get('total_tokens', 0))}"
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
        trace = c.get("reasoning_trace", "") or "(no reasoning trace)"
        rows_out.append({
            "id": rid,
            "prediction": c.get("prediction", 0.5),
            "reasoning_trace": trace,
            "tokens_used": int(c.get("tokens_used", 0)),
            "model_name": c.get("model_name", display_name),
        })

    sub_df = pd.DataFrame(rows_out)
    sub_path = args.output_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    # Write prompt.txt
    prompt_path = args.output_dir / "prompt.txt"
    prompt_path.write_text(
        f"# Prompt template used for Track C (model: {display_name})\n\n"
        "## DE task\n\n"
        + _PROMPT_DE_ZERO.format(pert="{pert}", gene="{gene}", cell_desc=CELL_DESC)
        + "\n\n## Dir task\n\n"
        + _PROMPT_DIR_ZERO.format(pert="{pert}", gene="{gene}", cell_desc=CELL_DESC)
    )

    # Package zip
    zip_path = args.output_dir / "submission_track_c.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_path, "submission.csv")
        zf.write(prompt_path, "prompt.txt")

    print(f"Wrote {sub_path}")
    print(f"Wrote {prompt_path}")
    print(f"Wrote {zip_path}  <-- upload this to Kaggle")


if __name__ == "__main__":
    main()
