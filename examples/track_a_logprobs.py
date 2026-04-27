"""
Track A -- Logprob-based continuous predictions.

Like the prompt-only baseline (track_a_prompt_only.py), but extracts
continuous ``prediction_up`` and ``prediction_down`` from the
log-probabilities of the A/B/C answer token, rather than hard-parsing
the text alone.  A=up-regulated, B=down-regulated, C=no significant effect.

The model reasons fully (normal max_tokens budget) and is asked to wrap
its final choice in ``<answer>A</answer>``, ``<answer>B</answer>``, or
``<answer>C</answer>`` tags.  The script requests ``logprobs=True,
top_logprobs=20`` and, after generation, finds the A/B/C token inside the
answer tags.  At that token position, it reads the top_logprobs for
``A``, ``B``, and ``C`` and computes a softmax over all three classes,
returning ``(P(A), P(B))`` as ``(prediction_up, prediction_down)``.
``P(C)`` is implied as ``1 - P(A) - P(B)`` when probabilities are
properly normalized over the three classes.

Requires the vLLM server to be started with the -inf sanitisation patch
(``serve_with_logprobs_fix.py``) so that logprob responses serialise
correctly.  Falls back to text parsing when logprobs are unavailable.

Prompt input modes (same as track_a_prompt_only.py):

  --prompts-csv FILE      CSV/JSONL with (id, prompt) pairs.
  --prompt-template FILE  Template with {pert}, {gene}, etc. placeholders.
  (neither)               Falls back to mlgenx zero-shot prompts.

Usage:
    pip install -e .   # from repo root -- installs mlgenx

    # Start vLLM with the logprobs fix:
    python serve_with_logprobs_fix.py openai/gpt-oss-120b --port 8000

    # Then run this script:
    python examples/track_a_logprobs.py \\
        --api-base http://localhost:8000/v1

    # Parallel requests (much faster):
    python examples/track_a_logprobs.py \\
        --api-base http://localhost:8000/v1 --concurrency 20
"""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from mlgenx import format_prompt, parse_answer
from mlgenx.prompts import CELL_DESC

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data" / "test.csv"
SEEDS = [42, 43, 44]

_DEFAULT_UP, _DEFAULT_DOWN = parse_answer("")


# ---------------------------------------------------------------------------
# Prompt loading (identical to track_a_prompt_only.py)
# ---------------------------------------------------------------------------

def load_prompts_csv(path: Path) -> Dict[str, str]:
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

def _do_request(url: str, payload: dict, api_key: str, timeout_s: int) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode())


# Sticky auto-detection: None = untested, True/False = remembered.
_logprobs_ok: bool | None = None
_lock_detect = threading.Lock()


def post_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    seed: int,
    max_tokens: int,
    timeout_s: int,
    top_logprobs: int = 20,
    no_reasoning: bool = False,
) -> Tuple[str, str, Dict[str, float], List[dict]]:
    """Call an OpenAI-compatible chat endpoint with logprobs.

    Requests ``logprobs=True`` so that each generated token includes
    ``top_logprobs`` alternatives.  Requires the vLLM server to sanitise
    -inf values (see ``serve_with_logprobs_fix.py``).

    When *no_reasoning* is True, passes
    ``chat_template_kwargs: {"enable_thinking": false}`` so that the
    model skips its hidden chain-of-thought and generates content
    directly.  This produces genuinely calibrated logprobs at the
    answer token, instead of near-degenerate post-reasoning logprobs.

    On the first call, if the server rejects the logprobs request, all
    future calls fall back to plain requests (no logprobs).

    Returns (content_text, reasoning_text, token_stats, logprobs_content).
    """
    global _logprobs_ok

    url = api_base.rstrip("/") + "/chat/completions"
    base_payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    if no_reasoning:
        base_payload["chat_template_kwargs"] = {"enable_thinking": False}

    out: dict | None = None

    with _lock_detect:
        try_logprobs = _logprobs_ok is not False

    if try_logprobs:
        payload_lp = {
            **base_payload,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }
        try:
            out = _do_request(url, payload_lp, api_key, timeout_s)
            with _lock_detect:
                if _logprobs_ok is None:
                    _logprobs_ok = True
                    print("  [logprobs] Server accepted logprobs request")
        except urllib.error.HTTPError as exc:
            with _lock_detect:
                if _logprobs_ok is None:
                    _logprobs_ok = False
                    print(
                        f"  [logprobs] Server rejected logprobs ({exc}). "
                        f"Falling back to text-parsing for all requests. "
                        f"Use serve_with_logprobs_fix.py to enable logprobs."
                    )
            out = None

    if out is None:
        out = _do_request(url, base_payload, api_key, timeout_s)

    # -- Parse response --------------------------------------------------
    usage = out.get("usage", {}) or {}
    token_stats = {
        "prompt_tokens": float(usage.get("prompt_tokens", 0)),
        "completion_tokens": float(usage.get("completion_tokens", 0)),
        "total_tokens": float(usage.get("total_tokens", 0)),
    }

    choices = out.get("choices", [])
    if not choices:
        return "", "", token_stats, []

    choice = choices[0]
    msg = choice.get("message", {}) or {}

    reasoning = str(msg.get("reasoning", "") or "").strip()

    content = msg.get("content", "") or ""
    if isinstance(content, list):
        content = "\n".join(
            str(c.get("text", c.get("content", "")))
            for c in content
            if isinstance(c, dict)
        )
    content = str(content).strip()

    logprobs_data = choice.get("logprobs") or {}
    logprobs_content = logprobs_data.get("content") or []

    return content, reasoning, token_stats, logprobs_content


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
# Logprob extraction
# ---------------------------------------------------------------------------

def prediction_from_logprobs(
    logprobs_content: List[dict],
    debug: bool = False,
) -> Optional[Tuple[float, float]]:
    """Extract (prediction_up, prediction_down) from logprobs at the A/B/C token.

    Reconstructs the full text from the token stream (so character
    offsets are guaranteed to align), finds ``<answer>X</answer>`` for
    X in {A,B,C}, and reads ``top_logprobs`` at that token to compute a
    softmax over ``A``, ``B``, and ``C``.  Returns ``(P(A), P(B))`` as
    ``(prediction_up, prediction_down)``.

    Returns None if extraction fails.
    """
    if not logprobs_content:
        return None

    # Reconstruct text from tokens so offsets match exactly.
    tokens = [t.get("token", "") for t in logprobs_content]
    reconstructed = "".join(tokens)

    m = re.search(r"<answer>\s*([ABCabc])\s*</answer>", reconstructed)
    if not m:
        if debug:
            tail = reconstructed[-200:] if len(reconstructed) > 200 else reconstructed
            print(f"    [debug] no <answer> tag in token stream (last 200 chars): {tail!r}")
        return None

    answer_char_start = m.start(1)

    # Map character offset → token index.
    char_pos = 0
    answer_token_idx: int | None = None
    for i, tok_text in enumerate(tokens):
        tok_end = char_pos + len(tok_text)
        if char_pos <= answer_char_start < tok_end:
            answer_token_idx = i
            break
        char_pos = tok_end

    if answer_token_idx is None:
        if debug:
            print(f"    [debug] char offset {answer_char_start} outside token range (total chars {char_pos})")
        return None

    top_lps = logprobs_content[answer_token_idx].get("top_logprobs") or []

    if debug:
        chosen = logprobs_content[answer_token_idx]
        print(f"    [debug] answer token idx={answer_token_idx} "
              f"tok={chosen.get('token')!r} lp={chosen.get('logprob')} "
              f"top_lps={[(e.get('token'), e.get('logprob')) for e in top_lps[:5]]}")

    logprob_a: float | None = None
    logprob_b: float | None = None
    logprob_c: float | None = None

    for entry in top_lps:
        tok = entry.get("token", "").strip().upper()
        lp = entry.get("logprob")
        if lp is None:
            continue
        # Tokens may include a prefix from the tag, e.g. ">A", ">B", ">C"
        ends_a = tok == "A" or tok.endswith(">A")
        ends_b = tok == "B" or tok.endswith(">B")
        ends_c = tok == "C" or tok.endswith(">C")
        if ends_a and logprob_a is None:
            logprob_a = float(lp)
        elif ends_b and logprob_b is None:
            logprob_b = float(lp)
        elif ends_c and logprob_c is None:
            logprob_c = float(lp)

    if logprob_a is None and logprob_b is None and logprob_c is None:
        if debug:
            print(f"    [debug] neither A, B, nor C found in top_logprobs")
        return None

    # If some tokens are missing, assume negligible probability vs. the best.
    lps = [lp for lp in (logprob_a, logprob_b, logprob_c) if lp is not None]
    floor = min(lps) - 20.0 if lps else -20.0
    if logprob_a is None:
        logprob_a = floor
    if logprob_b is None:
        logprob_b = floor
    if logprob_c is None:
        logprob_c = floor

    max_lp = max(logprob_a, logprob_b, logprob_c)
    exp_a = math.exp(logprob_a - max_lp)
    exp_b = math.exp(logprob_b - max_lp)
    exp_c = math.exp(logprob_c - max_lp)
    sum_exp = exp_a + exp_b + exp_c
    if sum_exp <= 0:
        return None
    p_a = exp_a / sum_exp
    p_b = exp_b / sum_exp
    return (p_a, p_b)


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
        description="Track A: Logprob-based continuous predictions (3 seeds)"
    )

    parser.add_argument(
        "--prompts-csv", type=Path, default=None,
        help="CSV or JSONL with columns (id, prompt).",
    )
    parser.add_argument(
        "--prompt-template", type=Path, default=None,
        help="Text file with a prompt template containing {pert}, {gene}, and "
             "optionally {cell_desc} placeholders.",
    )

    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--top-logprobs", type=int, default=20,
                        help="Number of top logprobs to request per token.")
    parser.add_argument(
        "--no-reasoning", action="store_true",
        help="Disable the model's hidden reasoning/thinking phase via "
             "chat_template_kwargs.enable_thinking=false. Produces calibrated "
             "logprobs instead of post-reasoning degenerate ones.",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Print debug info for logprob extraction.")
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "track_a_logprobs",
    )
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of concurrent requests.",
    )
    parser.add_argument(
        "--model-name", default=None,
        help="Override model name recorded in submission (defaults to --model).",
    )
    args = parser.parse_args()

    model_name = args.model_name or args.model

    # ── Resolve prompt source ─────────────────────────────────────────
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
    if args.no_reasoning:
        print("Reasoning disabled (enable_thinking=false) for calibrated logprobs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "responses_cache.json"
    cache = load_cache(cache_path)
    if "rows" not in cache:
        cache["rows"] = {}

    test_df = pd.read_csv(args.test_csv)
    total = len(test_df)
    cache_lock = threading.Lock()
    new_count = 0
    logprob_hits = 0
    logprob_misses = 0

    def process_row(idx: int, row: pd.Series) -> None:
        nonlocal new_count, logprob_hits, logprob_misses
        rid = row["id"]
        prompt_raw = resolve_prompt(
            row, prompts_map=prompts_map, template=template
        )
        prompt = append_answer_tag(prompt_raw)

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

            content = ""
            reasoning = ""
            token_stats: Dict[str, float] = {}
            logprobs_content: List[dict] = []
            for attempt in range(args.max_retries + 1):
                try:
                    content, reasoning, token_stats, logprobs_content = (
                        post_chat_completion(
                            api_base=args.api_base,
                            api_key=args.api_key,
                            model=args.model,
                            prompt=prompt,
                            seed=seed,
                            max_tokens=args.max_tokens,
                            timeout_s=args.timeout_s,
                            top_logprobs=args.top_logprobs,
                            no_reasoning=args.no_reasoning,
                        )
                    )
                    break
                except Exception as e:
                    print(f"  seed={seed} attempt={attempt+1} error={e}")
                    if attempt < args.max_retries:
                        time.sleep(2**attempt)

            # Try logprob-based prediction first
            pair = prediction_from_logprobs(logprobs_content, debug=args.debug)
            used_logprobs = pair is not None

            if pair is None:
                # Fall back to text parsing
                combined = "\n\n".join(
                    p for p in (reasoning, content) if p
                )
                tag = extract_answer_tag(combined)
                source = tag if tag else combined
                pair = parse_answer(source)

            pred_up, pred_down = pair

            with cache_lock:
                if used_logprobs:
                    logprob_hits += 1
                else:
                    logprob_misses += 1

            trace_text = "\n\n".join(
                p for p in (reasoning, content) if p
            )

            cached[key_up] = pred_up
            cached[key_down] = pred_down
            cached[key_trace] = trace_text
            cached[f"tokens_seed{seed}"] = token_stats.get("total_tokens", 0.0)
            cached[f"used_logprobs_seed{seed}"] = used_logprobs

        cached["tokens_used"] = sum(
            cached.get(f"tokens_seed{s}", 0.0) for s in SEEDS
        )
        cached["prediction_up"] = sum(
            cached.get(f"prediction_up_seed{s}", _DEFAULT_UP) for s in SEEDS
        ) / len(SEEDS)
        cached["prediction_down"] = sum(
            cached.get(f"prediction_down_seed{s}", _DEFAULT_DOWN) for s in SEEDS
        ) / len(SEEDS)
        cached["model_name"] = model_name

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
            future.result()

    save_cache(cache_path, cache)
    print(
        f"Collected {total} rows ({new_count} new API calls). "
        f"Logprob extraction: {logprob_hits} hit, {logprob_misses} fallback."
    )

    # ── Build submission CSV ──────────────────────────────────────────
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
            "model_name": c.get("model_name", model_name),
        })

    sub_df = pd.DataFrame(rows_out)
    sub_path = args.output_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    # ── Write prompt.txt ──────────────────────────────────────────────
    prompt_path = args.output_dir / "prompt.txt"
    if args.prompts_csv is not None:
        prompt_path.write_text(
            f"# Track A (logprobs) -- per-row prompts from {args.prompts_csv.name}\n"
            f"# Total prompts: {len(prompts_map)}\n"
            "# Prediction: (P(A), P(B)) from softmax over A/B/C at <answer> token\n"
        )
    elif args.prompt_template is not None:
        prompt_path.write_text(
            f"# Track A (logprobs) -- template from {args.prompt_template.name}\n"
            "# Prediction: (P(A), P(B)) from softmax over A/B/C at <answer> token\n\n"
            + template
        )
    else:
        from mlgenx.prompts import _PROMPT_ZERO
        prompt_path.write_text(
            "# Track A (logprobs) -- default mlgenx ternary zero-shot prompt\n"
            "# Prediction: (P(A), P(B)) from softmax over A/B/C at <answer> token\n\n"
            + _PROMPT_ZERO.format(
                pert="{pert}", gene="{gene}", cell_desc=CELL_DESC
            )
        )

    # ── Package zip ───────────────────────────────────────────────────
    zip_path = args.output_dir / "submission_track_a.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_path, "submission.csv")
        zf.write(prompt_path, "prompt.txt")

    print(f"Wrote {sub_path}")
    print(f"Wrote {prompt_path}")
    print(f"Wrote {zip_path}  <-- upload this to Kaggle")


if __name__ == "__main__":
    main()
