"""
Track A -- Prompt-only baseline.

Calls an OpenAI-compatible API 3 times per question (seeds 42, 43, 44)
with temperature=1.0, top_p=1.0. Averages the per-seed predictions into
a final score and packages everything into a zip ready for Kaggle upload.

Usage:
    pip install -e .   # from repo root -- installs mlgenx
    python examples/track_a_prompt_only.py \
        --api-base http://your-endpoint/v1 \
        --api-key YOUR_KEY \
        --model GPT-OSS-120B
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
SEEDS = [42, 43, 44]


def post_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    seed: int,
    max_tokens: int,
    timeout_s: int,
) -> Tuple[str, Dict[str, float]]:
    """Call an OpenAI-compatible chat endpoint and return (text, token_stats)."""
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        out = json.loads(resp.read().decode())

    usage = out.get("usage", {}) or {}
    token_stats = {
        "prompt_tokens": float(usage.get("prompt_tokens", 0)),
        "completion_tokens": float(usage.get("completion_tokens", 0)),
        "total_tokens": float(usage.get("total_tokens", 0)),
    }

    choices = out.get("choices", [])
    if choices:
        msg = choices[0].get("message", {}) or {}
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(c.get("text", c.get("content", "")))
                for c in content
                if isinstance(c, dict)
            )
        return str(content).strip(), token_stats

    return "", token_stats


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
        description="Track A: Prompt-only baseline (3 seeds)"
    )
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--model", default="GPT-OSS-120B")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "track_a"
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
        prompt_raw = format_prompt(row["pert"], row["gene"], task)
        prompt = append_answer_tag(prompt_raw)

        cached = cache["rows"].get(rid, {})
        if all(f"prediction_seed{s}" in cached for s in SEEDS):
            print(f"[{idx+1}/{total}] {rid} cache_hit")
            continue

        for seed in SEEDS:
            key_pred = f"prediction_seed{seed}"
            key_trace = f"reasoning_trace_seed{seed}"
            if key_pred in cached:
                continue

            text = ""
            token_stats: Dict[str, float] = {}
            for attempt in range(args.max_retries + 1):
                try:
                    text, token_stats = post_chat_completion(
                        api_base=args.api_base,
                        api_key=args.api_key,
                        model=args.model,
                        prompt=prompt,
                        seed=seed,
                        max_tokens=args.max_tokens,
                        timeout_s=args.timeout_s,
                    )
                    break
                except Exception as e:
                    print(f"  seed={seed} attempt={attempt+1} error={e}")
                    if attempt < args.max_retries:
                        time.sleep(2**attempt)

            tag = extract_answer_tag(text)
            source = tag if tag else text
            pred = float(parse_answer(source, task=task, default=0.5))
            cached[key_pred] = pred
            cached[key_trace] = text
            cached[f"tokens_seed{seed}"] = token_stats.get("total_tokens", 0.0)

        cached["tokens_used"] = sum(
            cached.get(f"tokens_seed{s}", 0.0) for s in SEEDS
        )
        cached["prediction"] = sum(
            cached.get(f"prediction_seed{s}", 0.5) for s in SEEDS
        ) / len(SEEDS)
        cached["task"] = task
        cache["rows"][rid] = cached

        new_count += 1
        print(
            f"[{idx+1}/{total}] {rid} task={task} "
            f"pred={cached['prediction']:.3f} tokens={cached['tokens_used']:.0f}"
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
            "prediction_seed42": c.get("prediction_seed42", 0.5),
            "prediction_seed43": c.get("prediction_seed43", 0.5),
            "prediction_seed44": c.get("prediction_seed44", 0.5),
            "reasoning_trace_seed42": c.get("reasoning_trace_seed42", ""),
            "reasoning_trace_seed43": c.get("reasoning_trace_seed43", ""),
            "reasoning_trace_seed44": c.get("reasoning_trace_seed44", ""),
            "tokens_used": int(c.get("tokens_used", 0)),
        })

    sub_df = pd.DataFrame(rows_out)
    sub_path = args.output_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    # Write prompt.txt
    prompt_path = args.output_dir / "prompt.txt"
    prompt_path.write_text(
        "# Prompt template used for Track A (zero-shot)\n\n"
        "## DE task\n\n"
        + _PROMPT_DE_ZERO.format(pert="{pert}", gene="{gene}", cell_desc=CELL_DESC)
        + "\n\n## Dir task\n\n"
        + _PROMPT_DIR_ZERO.format(pert="{pert}", gene="{gene}", cell_desc=CELL_DESC)
    )

    # Package zip
    zip_path = args.output_dir / "submission_track_a.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_path, "submission.csv")
        zf.write(prompt_path, "prompt.txt")

    print(f"Wrote {sub_path}")
    print(f"Wrote {prompt_path}")
    print(f"Wrote {zip_path}  <-- upload this to Kaggle")


if __name__ == "__main__":
    main()
