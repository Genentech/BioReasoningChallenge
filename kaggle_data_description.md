## Dataset Description

This competition uses data derived from a genome-wide CRISPRi Perturb-seq screen in **mouse bone marrow-derived macrophages (BMDMs)**, processed through the CropFlow differential expression pipeline. Each row represents a (perturbation, target gene) pair. The task is **ternary classification**: predict whether the target gene is **upregulated**, **downregulated**, or **unaffected** (not significantly differentially expressed) when the perturbation gene is knocked down.

Ground-truth labels combine a **5% FDR** threshold with **|shrunken log2 fold-change|** cutoffs (see label definitions below).

### Splits

Data is split along **both the perturbation axis and the gene axis**: perturbations and genes are each partitioned into disjoint pools. No gene appears in more than one split, and no perturbation appears in more than one split. This prevents models from exploiting gene-identity shortcuts (e.g. gene-frequency or direction biases).

| Split | Perturbations | Unique Genes | Total Rows |
|-------|--------------|------------|------------|
| Train | 386 | ~6,000 | 7,705 |
| Test (Public + Private) | 96 | ~4,000 | 1,813 |

The gene pool is split 60/20/20 (train / validation / test) across all 10,000 genes. Perturbations are split 80/10/10. Each split samples only from its own gene pool.

You submit predictions for all rows in `test.csv`; Kaggle handles the Public/Private split automatically.

---

## Files

### train.csv

Training data **with** ground-truth labels. Use this to build and validate your model.

| Column | Description |
|--------|-------------|
| `id` | Unique row identifier: `{perturbation}_{gene}` |
| `pert` | Name of the perturbed (knocked-down) gene |
| `gene` | Name of the target gene whose expression may change |
| `label` | Ground-truth ternary label (see below) |

**Label definitions:**

- **up**: The target gene is significantly **upregulated** (FDR < 5%, logFC >= log2(1.5)).
- **down**: The target gene is significantly **downregulated** (FDR < 5%, logFC <= -log2(1.5)).
- **none**: The target gene is **not** significantly differentially expressed (does not meet the criteria above).

### test.csv

Test data **without** labels. Submit your predictions for every row in this file.

| Column | Description |
|--------|-------------|
| `id` | Unique row identifier (same format as train) |
| `pert` | Perturbed gene |
| `gene` | Target gene |

### Submission Files (per track)

Each track has its own sample submission file with track-specific metadata columns. The **`prediction_up`** and **`prediction_down`** columns are used for scoring; all other columns are metadata for auditability and leaderboard display.

Submissions are uploaded as a **zip file** containing `submission.csv` (the filled-in sample submission) plus any additional required files (see per-track details below).

#### Track A -- Prompt-only (`sample_submission_track_a.csv`)

Three calls per question using seeds 42, 43, and 44. The scored `prediction_up` and `prediction_down` values are typically the averages of the three per-seed columns (see your sample submission for the exact aggregation).

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | string | -- | Must match every `id` in `test.csv` exactly |
| `prediction_up` | float | 0.5 | Final up score (used for scoring) |
| `prediction_down` | float | 0.5 | Final down score (used for scoring) |
| `prediction_up_seed42` | float | 0.5 | Up score from seed 42 |
| `prediction_down_seed42` | float | 0.5 | Down score from seed 42 |
| `prediction_up_seed43` | float | 0.5 | Up score from seed 43 |
| `prediction_down_seed43` | float | 0.5 | Down score from seed 43 |
| `prediction_up_seed44` | float | 0.5 | Up score from seed 44 |
| `prediction_down_seed44` | float | 0.5 | Down score from seed 44 |
| `reasoning_trace_seed42` | string | "" | Full LLM output text for seed 42 |
| `reasoning_trace_seed43` | string | "" | Full LLM output text for seed 43 |
| `reasoning_trace_seed44` | string | "" | Full LLM output text for seed 44 |
| `tokens_used` | int | 0 | Total tokens (input + output) across all 3 calls |
| `model_name` | string | "" | Model identifier |

**Zip contents**: `submission.csv` + `prompt.txt` (the prompt template used)

#### Track B -- Agentic tool-use (`sample_submission_track_b.csv`)

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | string | -- | Must match every `id` in `test.csv` exactly |
| `prediction_up` | float | 0.5 | Predicted up score, float in [0, 1] |
| `prediction_down` | float | 0.5 | Predicted down score, float in [0, 1] |
| `reasoning_trace` | string | "" | Full agent trace (all steps) |
| `tokens_used` | int | 0 | Total tokens (input + output) |
| `num_tool_calls` | int | 0 | Number of tool calls made for this row (max 250) |
| `prompt_tokens` | int | 0 | Number of tokens in the system prompt (max 16,384) |
| `num_distinct_tools` | int | 0 | Number of distinct tool definitions (max 100) |
| `model_name` | string | "" | Model identifier |

**Zip contents**: `submission.csv` + `tools/` folder containing `.py` tool definition files + `prompt.txt`

#### Track C -- Fine-tuning (`sample_submission_track_c.csv`)

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | string | -- | Must match every `id` in `test.csv` exactly |
| `prediction_up` | float | 0.5 | Predicted up score, float in [0, 1] |
| `prediction_down` | float | 0.5 | Predicted down score, float in [0, 1] |
| `reasoning_trace` | string | "" | Full model output text |
| `tokens_used` | int | 0 | Total tokens (new tokens generated) |
| `model_name` | string | "" | Model identifier (e.g., "Qwen3-4B-Thinking-2507-lora") |

**Zip contents**: `submission.csv` + `prompt.txt`

---

## Row ID Format

Every row ID identifies the perturbation and target gene:

```
{perturbation}_{gene}
```

Examples:

- `Aars_Actb` -- knockdown of *Aars*, target gene *Actb*
- `Stat1_Irf1` -- knockdown of *Stat1*, target gene *Irf1*

---

## Evaluation

The competition metric is the **average of two micro AUROCs** (differential expression vs. direction), derived from your ternary labels and the two prediction columns:

```
score = (micro_AUROC_DE + micro_AUROC_DIR) / 2
```

- **DE (differential expression) AUROC**: Labels are binarized as **1** if the true label is `up` or `down`, and **0** if `none`. The model score for each row is **`prediction_up + prediction_down`** (a single scalar in [0, 1] used as the DE-positive score).
- **DIR (direction) AUROC**: Restrict to rows where the true label is **`up` or `down`**. Labels are binarized as **1** for `up` and **0** for `down`. The model score for each row is **`prediction_up / (prediction_up + prediction_down)`** (conditional probability of up given DE).

A random baseline scores approximately **0.5**. A perfect model scores **1.0**.

### Leaderboard Metrics

The primary ranking metric is the score above. Additionally, the leaderboard will display:

- **Total tokens used** (all tracks) -- sum of `tokens_used` across all rows in the submission
- **Total tool calls** (Track B only) -- sum of `num_tool_calls` across all rows

These secondary metrics are for transparency and do not affect ranking.

---

## Useful Context

- Gene names follow mouse nomenclature (e.g., *Aars*, *Actb*, *Stat1*).
- The cell type is **mouse bone marrow-derived macrophages (BMDMs)**.
- The perturbation mechanism is **CRISPRi** (transcriptional repression / knockdown).
- Fold-change values are **shrunken log2 fold-changes** (not raw).
- **Sampling**: For each perturbation, up to **9** top differentially expressed genes (DEGs) from the split's gene pool are selected (by FDR), labeled **`up` or `down`** according to the sign of logFC, plus balanced negative genes labeled **`none`** (~11 per perturbation on average, ~20 rows per perturbation). Negative sampling matches each gene's negative count to its positive count across KOs, then tops up each KO to have at least as many negatives as positives. Negatives are restricted to |LFC| < 0.25. **482** perturbations are retained (those with at least 9 DEGs globally). Gene pools are **disjoint** across splits (60/20/20).

---

*Data derived from CropFlow CRISPRi Perturb-seq. Format inspired by [PerturbQA](https://github.com/Genentech/PerturbQA) (Wu et al., ICLR 2025).*
