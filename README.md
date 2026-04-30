# BioReasoning Challenge -- MLGenX LLM Perturbation Competition

<p align="center">
  <img src="docs/img/challenge_overview.png" alt="BioReasoning Challenge overview" width="800">
</p>

Predict gene expression changes from CRISPRi perturbations in mouse bone marrow-derived macrophages (BMDMs).

## Website
Please checkout [the website](https://genentech.github.io/BioReasoningChallenge/) for full details!


## Overview

Participants are given (perturbation, gene) pairs and must predict a **ternary** effect on the target gene:

- **up** — upregulated
- **down** — downregulated
- **none** — not significantly affected

Ground-truth labels use a **5% FDR** threshold and **|shrunken log2FC| >= log2(1.5)**.

Submissions provide two probabilities per row: `prediction_up` and `prediction_down`. P(none) is implicitly `1 - prediction_up - prediction_down`.

The competition is hosted on Kaggle with three separate tracks:

| Track | Name | Model | Key constraint |
|-------|------|-------|----------------|
| A | Prompt-only | GPT-OSS-120B (fixed) | Single prompt, 3 seeds, no tools |
| B | Agentic tool-use | GPT-OSS-120B (fixed) | Tools allowed, max 250 calls |
| C | Fine-tuning | Open model < 10B parameters | Any fine-tuning, no tools at inference |

## Installation

```bash
git clone https://github.com/genentech/bioreasoningchallenge.git
cd bioreasoningchallenge
uv sync            # core deps only (prompts, parsing, submission)
```

This installs the `mlgenx` helper package, which provides prompt generation and answer parsing.

Track C has separate dependency groups for fine-tuning and serving (they require
incompatible `transformers` versions and cannot be installed together):

```bash
uv sync --extra train   # fine-tuning: torch, transformers 5.x, trl, peft, …
uv sync --extra serve   # serving:     vllm (brings transformers 4.x)
```

## Data

All competition data lives in `data/`:

| File | Description |
|------|-------------|
| `train.csv` | Training data with labels (`id, pert, gene, label`) — label is `up`, `down`, or `none` |
| `test.csv` | Test data without labels (`id, pert, gene`) |
| `sample_submission.csv` | Minimal submission template (`id, prediction_up, prediction_down`) |
| `sample_submission_track_a.csv` | Track A template with per-seed columns |
| `sample_submission_track_b.csv` | Track B template with tool-call columns |
| `sample_submission_track_c.csv` | Track C template with model-name column |

Row IDs are `{perturbation}_{gene}`, e.g. `Aars_Actb` or `Stat1_Irf1`.

See [`kaggle_data_description.md`](kaggle_data_description.md) for full data documentation.

### Dataset size

| Split | Perturbations | Rows | Labels (train) |
|-------|---------------|------|----------------|
| Train | 386 | 7,705 | 2,359 up, 1,086 down, 4,260 none |
| Test (validation + test) | 96 | 1,813 | — |

Splits are disjoint along **both** the perturbation axis (80/10/10) and the gene axis (60/20/20). No gene appears in more than one split.

## Tracks

### Track A -- Prompt-only

- **Model**: GPT-OSS-120B (fixed, no fine-tuning)
- **Sampling**: `temperature=1.0, top_p=1.0`
- **Format**: Single prompt per question, max 4,096 prompt tokens
- **Seeds**: 3 samples per question (seeds 42, 43, 44); final prediction = average of `prediction_up` / `prediction_down` across seeds
- **Submission**: `submission.csv` + `prompt.txt` in a zip

### Track B -- Agentic tool-use

- **Model**: GPT-OSS-120B (fixed, no fine-tuning)
- **Sampling**: `temperature=1.0, top_p=1.0`
- **Format**: Prompt + tools + input question, max 4,096 prompt tokens
- **Limits**: Max 100 distinct tools, max 250 tool calls per question
- **Submission**: `submission.csv` + `tools/` folder + `prompt.txt` in a zip

### Track C -- Fine-tuning

- **Model**: Open model < 10B parameters (e.g., Qwen3-4B-Thinking-2507), any fine-tuning allowed
- **Format**: Prompt + input question, max 16,000 new tokens at inference
- **Allowed**: SFT/LoRA, RL, process reward models, critic reranking, best-of-N
- **Not allowed**: Tools, web access, or external models during inference
- **Submission**: `submission.csv` + `prompt.txt` in a zip

## Serving GPT-OSS-120B (Tracks A & B)

Tracks A and B use a fixed model that you serve locally via vLLM:

```bash
uv sync --extra serve

uv run --extra serve vllm serve openai/gpt-oss-120b \
    --port 8000 \
    --enforce-eager \
    --no-enable-prefix-caching
```

The model is ~120B parameters with mxfp4 quantization (~60 GB of weights).
Use `--tensor-parallel-size <N>` to shard across multiple GPUs if a single GPU
does not have enough memory.  Two GPUs with ~80 GB each (e.g. A100-80G, H100,
B200) are sufficient with `--tensor-parallel-size 2`.

> **Important server flags:**
>
> - **`--enforce-eager`** — Disables CUDA graph capture. Without this flag,
>   GPT-OSS hits a [known vLLM bug](https://github.com/vllm-project/vllm/issues/30498)
>   where the first 1--2 requests succeed but subsequent requests return
>   `content: null` with `finish_reason: "length"` despite tokens being
>   generated server-side. The bug is triggered by CUDA graphs interacting
>   with prefix caching and the attention-sink mechanism.
>
> - **`--no-enable-prefix-caching`** — Recommended by the
>   [vLLM GPT-OSS recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)
>   for consistent behavior.

The first run downloads model weights from Hugging Face.
Set `HF_HOME` to a partition with at least **120 GB of free disk space** before
starting the server.  If the download is interrupted (e.g. disk full), the
cached snapshot may be left in an inconsistent state -- delete the partial cache
directory under `$HF_HOME/hub/models--openai--gpt-oss-120b/` and retry.

### Reasoning model behavior

GPT-OSS-120B is a **reasoning model**.  Use `max_completion_tokens` (not the
deprecated `max_tokens`) in your API requests to set the output budget for
reasoning + visible answer combined. Set `reasoning_effort` to control how
much the model reasons before answering:

| `reasoning_effort` | Behavior | Typical tokens |
|--------------------|----------|----------------|
| `"low"` | Brief reasoning, fast responses | 30--100 |
| `"medium"` | Moderate reasoning | 200--2,000 |
| `"high"` | Extended reasoning, highest quality | 1,000--10,000+ |

**Key parameter: `max_completion_tokens` vs `max_tokens`.**  For reasoning
models, `max_completion_tokens` correctly budgets reasoning and visible output
together.  Using the legacy `max_tokens` parameter causes the model to consume
the entire budget on reasoning without producing a visible answer.

The API response separates reasoning from the final answer:

```json
{
  "choices": [{
    "message": {
      "reasoning": "... internal chain-of-thought ...",
      "content": "... final answer ..."
    },
    "finish_reason": "stop"
  }]
}
```

When the model runs out of tokens during reasoning, both `reasoning` and
`content` will be `null`.

## Example Scripts

### Track A -- `examples/track_a_prompt_only.py`

Calls the LLM with 3 seeds (42, 43, 44), averages the predictions, and packages a zip.
Use `--concurrency N` to send multiple requests in parallel for faster runs.

```bash
# Default: uses mlgenx built-in prompts
uv run python examples/track_a_prompt_only.py --api-base http://localhost:8000/v1

# Parallel requests (much faster)
uv run python examples/track_a_prompt_only.py --api-base http://localhost:8000/v1 --concurrency 20

# Use a custom prompt template (placeholders: {pert}, {gene}, {cell_desc})
uv run python examples/track_a_prompt_only.py --prompt-template examples/prompt_template.txt ...

# Use a CSV/JSONL of pre-written per-row prompts (columns: id, prompt)
uv run python examples/track_a_prompt_only.py --prompts-csv examples/example_prompts.csv ...
```

See `examples/prompt_template.txt` and `examples/example_prompts.csv` for input format examples.

### Track B -- `examples/track_b_agentic.py`

Runs an agentic loop where the LLM can call tools between reasoning steps.

```bash
uv run python examples/track_b_agentic.py --api-base http://localhost:8000/v1
```

Three example tools are provided in `examples/tools/`:

| Tool | Source | Description |
|------|--------|-------------|
| `train_data_lookup` | Local `train.csv` | Look up known labels for a perturbation or gene |
| `gene_info` | [mygene.info](https://mygene.info) API | Retrieve gene annotations (summary, GO terms, pathways) |
| `protein_interactions` | [STRING DB](https://string-db.org) API | Query protein-protein interaction partners |

### Track B (multi-agent variant) -- `examples/track_b_multiagent.py`

A multi-agent version of Track B where a **coordinator** agent delegates to specialist
sub-agents, each backed by the same LLM via DSPy ReAct:

- **`biology_expert`** — sub-agent with `gene_info` and `protein_interactions` tools
- **`data_analyst`** — sub-agent with `lookup_pert` and `lookup_gene` tools

The coordinator consults one or both specialists, synthesizes their findings, and calls
`submit_answer`.  All traces are captured hierarchically:
`{"coordinator": {...}, "sub_agents": [...]}`.  Token and tool-call counts aggregate
across all agents.

```bash
uv run python examples/track_b_multiagent.py --api-base http://localhost:8000/v1

# Tune iteration budgets
uv run python examples/track_b_multiagent.py \
    --api-base http://localhost:8000/v1 \
    --max-iters 20 --max-sub-iters 5
```

### Track C -- `examples/finetune.py` + `examples/track_c_finetune.py`

Track C is a two-step workflow. Fine-tuning and serving require **different
dependency sets** (`train` vs `serve` extras) because `trl` needs
`transformers>=5.3` while vLLM requires `transformers<5`. Switch between them
by re-running `uv sync` with the appropriate extra.

**Step 1: Fine-tune** (run once, needs a GPU)

```bash
uv sync --extra train

uv run --extra train python examples/finetune.py \
    --train-csv data/train.csv \
    --model-id Qwen/Qwen3-4B-Thinking-2507 \
    --output-dir outputs/finetuned_model \
    --epochs 3 --lr 2e-4 --lora-rank 16
```

This produces a merged LoRA model in `outputs/finetuned_model/`.

**Step 1b: Patch tokenizer** (one-time fix after fine-tuning)

The `train` extra uses `transformers>=5.3`, which saves `extra_special_tokens`
in a format incompatible with the `transformers 4.x` bundled by vLLM. Run this
once after fine-tuning to fix the tokenizer config:

```bash
python -c "
import json; from pathlib import Path
p = Path('outputs/finetuned_model/tokenizer_config.json')
cfg = json.loads(p.read_text())
est = cfg.get('extra_special_tokens')
if isinstance(est, list):
    cfg['extra_special_tokens'] = {t: t for t in est} if est else {}
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f'Fixed: converted list of {len(est)} tokens to dict')
else:
    print('No fix needed')
"
```

**Step 2: Serve and run inference** (needs a GPU)

```bash
uv sync --extra serve

# Serve with vLLM
uv run --extra serve vllm serve outputs/finetuned_model --port 8000

# In another terminal -- generate predictions
uv run --extra serve python examples/track_c_finetune.py \
    --api-base http://localhost:8000/v1 \
    --model outputs/finetuned_model \
    --base-model Qwen/Qwen3-4B-Thinking-2507
```

## How to Submit

### Step 1: Generate predictions

Use the example scripts above or write your own. Each script outputs a zip file ready for Kaggle upload.

### Step 2: Verify your submission

Each track requires specific columns in `submission.csv`:

**Track A** columns: `id, prediction_up, prediction_down, prediction_up_seed42, prediction_down_seed42, prediction_up_seed43, prediction_down_seed43, prediction_up_seed44, prediction_down_seed44, reasoning_trace_seed42, reasoning_trace_seed43, reasoning_trace_seed44, tokens_used, model_name`

**Track B** columns: `id, prediction_up, prediction_down, reasoning_trace, tokens_used, num_tool_calls, prompt_tokens, num_distinct_tools, model_name`

**Track C** columns: `id, prediction_up, prediction_down, reasoning_trace, tokens_used, model_name`

The `id` column must match every row in `test.csv` exactly. Only `id`, `prediction_up`, and `prediction_down` are used for scoring; all other columns are required metadata. **Submissions missing required metadata columns will receive a score of 0.**

**No null values allowed.** Every cell must be filled. For rows where the model
returned an empty response, use `"none"` for reasoning traces and `0` for token
counts. The example scripts handle this automatically.

### Step 3: Package into a zip

```
# Track A zip contents:
submission.csv
prompt.txt

# Track B zip contents:
submission.csv
prompt.txt
tools/*.py

# Track C zip contents:
submission.csv
prompt.txt
```

### Step 4: Upload to Kaggle

Go to the competition page on Kaggle and upload your zip file.

## Evaluation

The competition metric is the **average of two micro AUROCs** computed from the ternary labels:

- **DE AUROC**: (up + down) vs none, using score `prediction_up + prediction_down`.
- **DIR AUROC**: up vs down among DE-positive rows, using score `prediction_up / (prediction_up + prediction_down)` (conditional probability of up given DE).

```
score = (DE_AUROC + DIR_AUROC) / 2
```

- Random baseline (reasonable spread across classes): near chance on both components
- Perfect model: 1.0

Submissions that omit required metadata columns (reasoning traces, token counts, etc.) will score **0.0**.

## Quick Start

```python
from mlgenx import format_prompt, parse_answer, build_submission

# Generate a prompt
prompt = format_prompt("Aars", "Actb")

# ... send to LLM, get response_text ...

# Parse the response
prediction_up, prediction_down = parse_answer(response_text)

# Build a submission
df = build_submission(ids, predictions_up, predictions_down, output_path="submission.csv")
```

### Batch prompt generation

```python
from mlgenx import format_prompts_from_csv

prompts_df = format_prompts_from_csv("data/test.csv")
# DataFrame with columns: id, prompt
```

### Few-shot prompting

```python
prompt = format_prompt("Aars", "Actb", examples=[
    {"pert": "Brca1", "gene": "Tp53", "label": "none"},
    {"pert": "Myc", "gene": "Cdkn1a", "label": "up"},
])
```

## API Reference

| Function | Description |
|----------|-------------|
| `format_prompt(pert, gene, examples=None)` | Generate a single LLM prompt (zero-shot or few-shot) |
| `format_prompts_from_csv(csv_path, examples=None)` | Generate prompts for all rows in a CSV |
| `parse_answer(text, default=(0.333, 0.333))` | Parse one LLM response into `(prediction_up, prediction_down)` |
| `parse_answers(texts, default=(0.333, 0.333))` | Parse a list of LLM responses |
| `build_submission(ids, predictions_up, predictions_down, output_path=None)` | Assemble a submission DataFrame/CSV |

## References

- Data format inspired by [PerturbQA](https://github.com/Genentech/PerturbQA) (Wu et al., ICLR 2025)
- Source data: CRISPRi Perturb-seq in mouse BMDMs
