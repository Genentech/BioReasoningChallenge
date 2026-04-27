"""
Track A -- Prompt-only baseline.

Calls an OpenAI-compatible API 3 times per question (seeds 42, 43, 44)
with temperature=1.0, top_p=1.0. Averages the per-seed predictions into
final ``prediction_up`` and ``prediction_down`` scores and packages
everything into a zip ready for Kaggle upload.

Note on reasoning models:
    GPT-OSS-120B is a reasoning model.  The max_tokens budget covers both
    the internal chain-of-thought and the visible answer.  If the model
    exhausts the budget during reasoning, the response is empty and the
    prediction defaults to the uniform ternary prior (1/3, 1/3).  This is
    expected -- averaging across 3 seeds still yields useful predictions
    when at least one seed succeeds.

Prompt input modes (mutually exclusive):

  --prompts-csv FILE    CSV or JSONL with columns ``id`` and ``prompt``.
                        Each row supplies a ready-to-send prompt.  The file
                        must cover every id in test.csv.

  --prompt-template FILE  A text file containing a prompt template with
                        ``{pert}``, ``{gene}``, and optionally ``{cell_desc}``
                        placeholders.  The script reads test.csv and fills in
                        the template per row.

  (neither)             Falls back to ``mlgenx.format_prompt`` zero-shot
                        templates.

Usage:
    pip install -e .   # from repo root -- installs mlgenx

    # Default (built-in mlgenx prompts):
    python examples/track_a_prompt_only.py \\
        --api-base http://localhost:8000/v1

    # Parallel requests (much faster):
    python examples/track_a_prompt_only.py \\
        --api-base http://localhost:8000/v1 --concurrency 20

    # With a custom prompt template:
    python examples/track_a_prompt_only.py \\
        --prompt-template examples/prompt_template.txt \\
        --api-base http://localhost:8000/v1

    # With pre-built per-row prompts:
    python examples/track_a_prompt_only.py \\
        --prompts-csv my_prompts.csv \\
        --api-base http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from mlgenx import format_prompt, parse_answer
from mlgenx.prompts import CELL_DESC

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data" / "test.csv"
SEEDS = [42, 43, 44]

_DEFAULT_UP, _DEFAULT_DOWN = parse_answer("")


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts_csv(path: Path) -> Dict[str, str]:
    """Load a CSV or JSONL of (id, prompt) pairs into {id: prompt}."""
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        records = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(path)

    missing = {"id", "prompt"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {missing}. "
            f"Expected columns: id, prompt"
        )
    return dict(zip(df["id"].astype(str), df["prompt"].astype(str)))


def load_prompt_template(path: Path) -> str:
    """Read a prompt template file. Must contain at least {pert} and {gene}."""
    text = path.read_text()
    for required in ("{pert}", "{gene}"):
        if required not in text:
            raise ValueError(
                f"Template {path} must contain placeholder {required}"
            )
    return text


def resolve_prompt(
    row: pd.Series,
    *,
    prompts_map: Dict[str, str] | None,
    template: str | None,
) -> str:
    """Return the prompt string for one test row."""
    rid = str(row["id"])

    if prompts_map is not None and rid in prompts_map:
        return prompts_map[rid]

    if template is not None:
        return template.format(
            pert=row["pert"],
            gene=row["gene"],
            cell_desc=CELL_DESC,
        )

    return format_prompt(row["pert"], row["gene"])


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def post_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    seed: int,
    max_tokens: int,
    timeout_s: int,
    reasoning_effort: str = "low",
) -> Tuple[str, Dict[str, float]]:
    """Call an OpenAI-compatible chat endpoint and return (text, token_stats)."""
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": seed,
        "max_completion_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
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
        reasoning = msg.get("reasoning", "") or ""
        content = msg.get("content", "") or ""
        if isinstance(content, list):
            content = "\n".join(
                str(c.get("text", c.get("content", "")))
                for c in content
                if isinstance(c, dict)
            )
        parts = [p for p in (str(reasoning).strip(), str(content).strip()) if p]
        return "\n\n".join(parts), token_stats

    return "", token_stats


def append_answer_tag(prompt: str) -> str:
    return (
        f"{prompt.rstrip()}\n\n"
        "Return ONLY the final choice in this exact format:\n"
        "<answer>A</answer>, <answer>B</answer>, or <answer>C</answer>\n"
        "Do not include any other text."
    )


def extract_answer_tag(text: str) -> str | None:
    m = re.search(r"<answer>\s*([ABCabc])\s*</answer>", text)
    return m.group(1).upper() if m else None


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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track A: Prompt-only baseline (3 seeds)"
    )

    # Prompt sources (can combine: CSV overrides template, template overrides default)
    parser.add_argument(
        "--prompts-csv", type=Path, default=None,
        help="CSV or JSONL with columns (id, prompt). Rows found here use "
             "the provided prompt; remaining rows fall back to --prompt-template "
             "or the built-in default.",
    )
    parser.add_argument(
        "--prompt-template", type=Path, default=None,
        help="Text file with a prompt template containing {pert}, {gene}, and "
             "optionally {cell_desc} placeholders. Used as fallback for "
             "rows not in --prompts-csv.",
    )

    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--reasoning-effort", default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort level sent to the model (default: low).",
    )
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "track_a"
    )
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of concurrent requests. Increase to speed up inference.",
    )
    parser.add_argument(
        "--model-name", default=None,
        help="Override model name recorded in submission (defaults to --model).",
    )
    args = parser.parse_args()

    model_name = args.model_name or args.model

    # Resolve prompt source
    prompts_map: Dict[str, str] | None = None
    template: str | None = None

    if args.prompts_csv is not None:
        prompts_map = load_prompts_csv(args.prompts_csv)
        print(f"Loaded {len(prompts_map)} prompts from {args.prompts_csv}")
    if args.prompt_template is not None:
        template = load_prompt_template(args.prompt_template)
        print(f"Loaded prompt template from {args.prompt_template}")
    if prompts_map is None and template is None:
        print("Using default mlgenx zero-shot prompts")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "responses_cache.json"
    cache = load_cache(cache_path)
    if "rows" not in cache:
        cache["rows"] = {}

    test_df = pd.read_csv(args.test_csv)
    total = len(test_df)
    cache_lock = threading.Lock()
    new_count = 0

    try:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
        def _count_tokens(text: str) -> int:
            return len(_enc.encode(text))
    except Exception:
        def _count_tokens(text: str) -> int:
            return len(text) // 4

    def process_row(idx: int, row: pd.Series) -> None:
        nonlocal new_count
        rid = row["id"]
        prompt_raw = resolve_prompt(
            row, prompts_map=prompts_map, template=template
        )
        prompt = append_answer_tag(prompt_raw)
        row_prompt_tokens = _count_tokens(prompt)

        with cache_lock:
            cached = cache["rows"].get(rid, {})
            if all(
                f"prediction_up_seed{s}" in cached
                and f"prediction_down_seed{s}" in cached
                for s in SEEDS
            ):
                print(f"[{idx+1}/{total}] {rid} cache_hit")
                return

        for seed in SEEDS:
            key_up = f"prediction_up_seed{seed}"
            key_down = f"prediction_down_seed{seed}"
            key_trace = f"reasoning_trace_seed{seed}"
            with cache_lock:
                if key_up in cached and key_down in cached:
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
                        reasoning_effort=args.reasoning_effort,
                    )
                    break
                except Exception as e:
                    print(f"  seed={seed} attempt={attempt+1} error={e}")
                    if attempt < args.max_retries:
                        time.sleep(2**attempt)

            tag = extract_answer_tag(text)
            source = tag if tag else text
            pred_up, pred_down = parse_answer(source)
            cached[key_up] = pred_up
            cached[key_down] = pred_down
            cached[key_trace] = text
            cached[f"tokens_seed{seed}"] = token_stats.get("total_tokens", 0.0)

        cached["tokens_used"] = sum(
            cached.get(f"tokens_seed{s}", 0.0) for s in SEEDS
        )
        cached["prompt_tokens"] = row_prompt_tokens
        cached["prediction_up"] = sum(
            cached.get(f"prediction_up_seed{s}", _DEFAULT_UP) for s in SEEDS
        ) / len(SEEDS)
        cached["prediction_down"] = sum(
            cached.get(f"prediction_down_seed{s}", _DEFAULT_DOWN) for s in SEEDS
        ) / len(SEEDS)

        with cache_lock:
            cache["rows"][rid] = cached
            new_count += 1
            print(
                f"[{idx+1}/{total}] {rid} "
                f"pred_up={cached['prediction_up']:.3f} "
                f"pred_down={cached['prediction_down']:.3f} "
                f"tokens={cached['tokens_used']:.0f}"
            )
            if new_count % args.save_every == 0:
                save_cache(cache_path, cache)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(process_row, idx, row)
            for idx, row in test_df.iterrows()
        ]
        for future in as_completed(futures):
            future.result()  # raise any exceptions

    save_cache(cache_path, cache)
    print(f"Collected {total} rows ({new_count} new API calls)")

    # Build submission CSV
    rows_out = []
    for _, row in test_df.iterrows():
        rid = row["id"]
        c = cache["rows"].get(rid, {})
        rows_out.append({
            "id": rid,
            "prediction_up": c.get("prediction_up", _DEFAULT_UP),
            "prediction_down": c.get("prediction_down", _DEFAULT_DOWN),
            "prediction_up_seed42": c.get("prediction_up_seed42", _DEFAULT_UP),
            "prediction_down_seed42": c.get("prediction_down_seed42", _DEFAULT_DOWN),
            "prediction_up_seed43": c.get("prediction_up_seed43", _DEFAULT_UP),
            "prediction_down_seed43": c.get("prediction_down_seed43", _DEFAULT_DOWN),
            "prediction_up_seed44": c.get("prediction_up_seed44", _DEFAULT_UP),
            "prediction_down_seed44": c.get("prediction_down_seed44", _DEFAULT_DOWN),
            "reasoning_trace_seed42": c.get("reasoning_trace_seed42") or "none",
            "reasoning_trace_seed43": c.get("reasoning_trace_seed43") or "none",
            "reasoning_trace_seed44": c.get("reasoning_trace_seed44") or "none",
            "tokens_used": int(c.get("tokens_used", 0)),
            "prompt_tokens": int(c.get("prompt_tokens", 0)),
            "model_name": c.get("model_name", model_name),
        })

    sub_df = pd.DataFrame(rows_out)
    sub_path = args.output_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    # Write prompt.txt (record whichever prompt source was used)
    prompt_path = args.output_dir / "prompt.txt"
    if args.prompts_csv is not None:
        prompt_path.write_text(
            f"# Track A -- per-row prompts loaded from {args.prompts_csv.name}\n"
            f"# Total prompts: {len(prompts_map)}\n"
        )
    elif args.prompt_template is not None:
        prompt_path.write_text(
            f"# Track A -- prompt template from {args.prompt_template.name}\n\n"
            + template
        )
    else:
        from mlgenx.prompts import _PROMPT_ZERO
        prompt_path.write_text(
            "# Track A -- default mlgenx ternary zero-shot prompt\n\n"
            + _PROMPT_ZERO.format(
                pert="{pert}", gene="{gene}", cell_desc=CELL_DESC
            )
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
