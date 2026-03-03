# BioReasoning Challenge -- MLGenX PerturbPair Competition

Predict gene expression changes from CRISPRi perturbations in mouse bone marrow-derived macrophages (BMDMs).

## Overview

Participants are given (perturbation, gene) pairs and must predict:

1. **Differential Expression (DE)** -- Does CRISPRi knockdown of the perturbation gene cause differential expression of the target gene? (0 = no, 1 = yes)
2. **Direction of Change (Dir)** -- For DE-positive pairs, is the target gene downregulated or upregulated? (0 = down, 1 = up)

Ground-truth labels use a **5% FDR** threshold and **|shrunken log2FC| >= log2(1.5)**.

The competition is hosted on Kaggle with three separate tracks:

| Track | Name | Model | Key constraint |
|-------|------|-------|----------------|
| A | Prompt-only | GPT-OSS-120B (fixed) | Single prompt, 3 seeds, no tools |
| B | Agentic tool-use | GPT-OSS-120B (fixed) | Tools allowed, max 250 calls |
| C | Fine-tuning | Open-source (e.g. Qwen3-4B) | Any fine-tuning, no tools at inference |

## Installation

```bash
git clone https://github.com/genentech/bioreasoningchallenge.git
cd bioreasoningchallenge
pip install -e .
```

This installs the `mlgenx` helper package, which provides prompt generation and answer parsing.

## Data

All competition data lives in `data/`:

| File | Description |
|------|-------------|
| `train.csv` | Training data with labels (`id, pert, gene, task, label`) |
| `test.csv` | Test data without labels (`id, pert, gene, task`) |
| `sample_submission.csv` | Minimal submission template (`id, prediction`) |
| `sample_submission_track_a.csv` | Track A template with per-seed columns |
| `sample_submission_track_b.csv` | Track B template with tool-call columns |
| `sample_submission_track_c.csv` | Track C template with model-name column |

Row IDs encode `{task}_{perturbation}_{gene}`, e.g. `de_Aars_Actb` or `dir_Stat1_Irf1`.

See [`kaggle_data_description.md`](kaggle_data_description.md) for full data documentation.

## Tracks

### Track A -- Prompt-only

- **Model**: GPT-OSS-120B (fixed, no fine-tuning)
- **Sampling**: `temperature=1.0, top_p=1.0`
- **Format**: Single prompt per question, max 4,096 prompt tokens
- **Seeds**: 3 samples per question (seeds 42, 43, 44); final prediction = average
- **Submission**: `submission.csv` + `prompt.txt` in a zip

### Track B -- Agentic tool-use

- **Model**: GPT-OSS-120B (fixed, no fine-tuning)
- **Sampling**: `temperature=1.0, top_p=1.0`
- **Format**: Prompt + tools + input question, max 4,096 prompt tokens
- **Limits**: Max 100 distinct tools, max 250 tool calls per question
- **Submission**: `submission.csv` + `tools/` folder + `prompt.txt` in a zip

### Track C -- Fine-tuning

- **Model**: Open-source LLM (e.g. Qwen3-4B-Thinking-2507), any fine-tuning allowed
- **Format**: Prompt + input question, max 16,000 new tokens at inference
- **Allowed**: SFT/LoRA, RL, process reward models, critic reranking, best-of-N
- **Not allowed**: Tools, web access, or external models during inference
- **Submission**: `submission.csv` + `prompt.txt` in a zip

## Example Scripts

### Track A -- `examples/track_a_prompt_only.py`

Calls the LLM with 3 seeds (42, 43, 44), averages the predictions, and packages a zip.

```bash
# Default: uses mlgenx built-in prompts
python examples/track_a_prompt_only.py --api-base http://your-api/v1 --api-key YOUR_KEY

# Use a custom prompt template (placeholders: {pert}, {gene}, {task}, {cell_desc})
python examples/track_a_prompt_only.py --prompt-template examples/prompt_template.txt ...

# Use a CSV/JSONL of pre-written per-row prompts (columns: id, prompt)
python examples/track_a_prompt_only.py --prompts-csv examples/example_prompts.csv ...
```

See `examples/prompt_template.txt` and `examples/example_prompts.csv` for input format examples.

### Track B -- `examples/track_b_agentic.py`

Runs an agentic loop where the LLM can call tools between reasoning steps.

```bash
python examples/track_b_agentic.py --api-base http://your-api/v1 --api-key YOUR_KEY
```

Three example tools are provided in `examples/tools/`:

| Tool | Source | Description |
|------|--------|-------------|
| `train_data_lookup` | Local `train.csv` | Look up known labels for a perturbation or gene |
| `gene_info` | [mygene.info](https://mygene.info) API | Retrieve gene annotations (summary, GO terms, pathways) |
| `protein_interactions` | [STRING DB](https://string-db.org) API | Query protein-protein interaction partners |

### Track C -- `examples/finetune.py` + `examples/track_c_finetune.py`

Track C is a two-step workflow:

**Step 1: Fine-tune** (run once)

```bash
python examples/finetune.py \
    --train-csv data/train.csv \
    --model Qwen/Qwen3-4B-Thinking-2507 \
    --output outputs/finetuned_model \
    --epochs 3 --lr 2e-4 --lora-r 16
```

This produces a merged LoRA model in `outputs/finetuned_model/`.

**Step 2: Serve and run inference**

```bash
# Serve with vLLM
vllm serve outputs/finetuned_model --port 8000

# Generate predictions
python examples/track_c_finetune.py \
    --api-base http://localhost:8000/v1 \
    --model outputs/finetuned_model \
    --base-model Qwen/Qwen3-4B-Thinking-2507
```

## How to Submit

### Step 1: Generate predictions

Use the example scripts above or write your own. Each script outputs a zip file ready for Kaggle upload.

### Step 2: Verify your submission

Each track requires specific columns in `submission.csv`:

**Track A** columns: `id, prediction, prediction_seed42, prediction_seed43, prediction_seed44, reasoning_trace_seed42, reasoning_trace_seed43, reasoning_trace_seed44, tokens_used`

**Track B** columns: `id, prediction, reasoning_trace, tokens_used, num_tool_calls`

**Track C** columns: `id, prediction, reasoning_trace, tokens_used, model_name`

The `id` column must match every row in `test.csv` exactly. Only `id` and `prediction` are used for scoring; all other columns are required metadata. **Submissions missing required metadata columns will receive a score of 0.**

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

The competition metric is the **mean of per-gene macro AUROC** across both tasks:

```
score = (macro_AUROC_DE + macro_AUROC_Dir) / 2
```

For each target gene, AUROC is computed over all perturbations that include that gene, then averaged across genes. Genes with only one label class are excluded.

- Random baseline (all 0.5): ~0.5
- Perfect model: 1.0

Submissions that omit required metadata columns (reasoning traces, token counts, etc.) will score **0.0**.

## Quick Start

```python
from mlgenx import format_prompt, parse_answer, build_submission

# Generate a prompt
prompt = format_prompt("Aars", "Actb", "de")

# ... send to LLM, get response_text ...

# Parse the response
prediction = parse_answer(response_text, "de")  # -> 0.0, 1.0, or 0.5

# Build a submission
df = build_submission(ids, predictions, output_path="submission.csv")
```

### Batch prompt generation

```python
from mlgenx import format_prompts_from_csv

prompts_df = format_prompts_from_csv("data/test.csv")
# DataFrame with columns: id, prompt
```

### Few-shot prompting

```python
prompt = format_prompt("Aars", "Actb", "de", examples=[
    {"pert": "Brca1", "gene": "Tp53", "label": 0},
    {"pert": "Myc", "gene": "Cdkn1a", "label": 1},
])
```

## API Reference

| Function | Description |
|----------|-------------|
| `format_prompt(pert, gene, task, examples=None)` | Generate a single LLM prompt (zero-shot or few-shot) |
| `format_prompts_from_csv(csv_path, examples=None)` | Generate prompts for all rows in a CSV |
| `parse_answer(text, task, default=0.5)` | Parse one LLM response into a float prediction |
| `parse_answers(texts, tasks, default=0.5)` | Parse a list of LLM responses |
| `build_submission(ids, predictions, output_path=None)` | Assemble a submission DataFrame/CSV |

## References

- Data format inspired by [PerturbQA](https://github.com/Genentech/PerturbQA) (Wu et al., ICLR 2025)
- Source data: CropFlow CRISPRi Perturb-seq in mouse BMDMs
