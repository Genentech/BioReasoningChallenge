## Dataset Description

This competition uses data derived from a genome-wide CRISPRi Perturb-seq screen in **mouse bone marrow-derived macrophages (BMDMs)**, processed through the CropFlow differential expression pipeline. Each row represents a (perturbation, target gene) pair and is associated with one of two binary prediction tasks:

- **Differential Expression (DE)** -- Does CRISPRi knockdown of a perturbation gene cause statistically significant differential expression of a target gene?
- **Direction of Change (Dir)** -- For pairs that *are* differentially expressed, is the target gene upregulated or downregulated?

Ground-truth labels were determined using a **5% FDR** threshold and a **|shrunken log2 fold-change| >= log2(1.5)** threshold.

### Splits

Data is split along the **perturbation axis**: every (perturbation, gene) pair for a given perturbation belongs to exactly one split. This means the test set contains perturbations that are entirely unseen during training.

| Split | Perturbations | Total Rows | DE Rows | Dir Rows |
|-------|--------------|------------|---------|----------|
| Train | 75 | 675 | 525 | 150 |
| Test (Public + Private) | 422 | 3,798 | 2,954 | 844 |

The public leaderboard is scored on a **validation** subset (225 perturbations) of the test file. The private leaderboard uses the remaining **test** subset (197 perturbations). You submit predictions for all rows in `test.csv`; Kaggle handles the Public/Private split automatically.

---

## Files

### train.csv

Training data **with** ground-truth labels. Use this to build and validate your model.

| Column | Description |
|--------|-------------|
| `id` | Unique row identifier: `{task}_{perturbation}_{gene}` |
| `pert` | Name of the perturbed (knocked-down) gene |
| `gene` | Name of the target gene whose expression may change |
| `task` | `de` (differential expression) or `dir` (direction of change) |
| `label` | Ground-truth binary label (see below) |

**Label definitions:**

- **DE task** (`task == "de"`): `1` = the target gene is differentially expressed upon knockdown of the perturbation gene; `0` = no significant change.
- **Dir task** (`task == "dir"`): `1` = the target gene is upregulated; `0` = the target gene is downregulated. Dir rows only exist for pairs that are DE-positive.

### test.csv

Test data **without** labels. Submit your predictions for every row in this file.

| Column | Description |
|--------|-------------|
| `id` | Unique row identifier (same format as train) |
| `pert` | Perturbed gene |
| `gene` | Target gene |
| `task` | `de` or `dir` |

### Submission Files (per track)

Each track has its own sample submission file with track-specific metadata columns. The `id` and `prediction` columns are common to all tracks and are the only columns used for scoring. All other columns are required metadata for auditability and leaderboard display.

Submissions are uploaded as a **zip file** containing `submission.csv` (the filled-in sample submission) plus any additional required files (see per-track details below).

#### Track A -- Prompt-only (`sample_submission_track_a.csv`)

Three calls per question using seeds 42, 43, and 44. The final `prediction` is the average of the three per-seed predictions.

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | string | -- | Must match every `id` in `test.csv` exactly |
| `prediction` | float | 0.5 | Final prediction: average of the 3 seed predictions |
| `prediction_seed42` | float | 0.5 | Prediction from seed 42 |
| `prediction_seed43` | float | 0.5 | Prediction from seed 43 |
| `prediction_seed44` | float | 0.5 | Prediction from seed 44 |
| `reasoning_trace_seed42` | string | "" | Full LLM output text for seed 42 |
| `reasoning_trace_seed43` | string | "" | Full LLM output text for seed 43 |
| `reasoning_trace_seed44` | string | "" | Full LLM output text for seed 44 |
| `tokens_used` | int | 0 | Total tokens (input + output) across all 3 calls |

**Zip contents**: `submission.csv` + `prompt.txt` (the prompt template used)

#### Track B -- Agentic tool-use (`sample_submission_track_b.csv`)

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | string | -- | Must match every `id` in `test.csv` exactly |
| `prediction` | float | 0.5 | Predicted probability, float in [0, 1] |
| `reasoning_trace` | string | "" | Full agent trace (all steps) |
| `tokens_used` | int | 0 | Total tokens (input + output) |
| `num_tool_calls` | int | 0 | Number of tool calls made for this row |

**Zip contents**: `submission.csv` + `tools/` folder containing `.py` tool definition files + `prompt.txt`

#### Track C -- Fine-tuning (`sample_submission_track_c.csv`)

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | string | -- | Must match every `id` in `test.csv` exactly |
| `prediction` | float | 0.5 | Predicted probability, float in [0, 1] |
| `reasoning_trace` | string | "" | Full model output text |
| `tokens_used` | int | 0 | Total tokens (new tokens generated) |
| `model_name` | string | "" | Model identifier (e.g., "Qwen3-4B-Thinking-2507-lora") |

**Zip contents**: `submission.csv` + `prompt.txt`

---

## Row ID Format

Every row ID encodes three pieces of information separated by underscores:

```
{task}_{perturbation}_{gene}
```

Examples:

- `de_Aars_Actb` -- DE task, knockdown of *Aars*, target gene *Actb*
- `dir_Stat1_Irf1` -- Direction task, knockdown of *Stat1*, target gene *Irf1*

---

## Evaluation

The competition metric is the **mean of per-gene macro AUROC** across both tasks:

```
score = ( macro_AUROC_DE + macro_AUROC_Dir ) / 2
```

**Per-gene macro AUROC**: for each target gene, compute the AUROC over all perturbations that include that gene, then average across genes. Genes that have only one label class in the scored split are excluded from the average.

A random baseline (predicting 0.5 for everything) scores approximately **0.5**. A perfect model scores **1.0**.

### Leaderboard Metrics

The primary ranking metric is the AUROC score above. Additionally, the leaderboard will display:

- **Total tokens used** (all tracks) -- sum of `tokens_used` across all rows in the submission
- **Total tool calls** (Track B only) -- sum of `num_tool_calls` across all rows

These secondary metrics are for transparency and do not affect ranking.

---

## Useful Context

- Gene names follow mouse nomenclature (e.g., *Aars*, *Actb*, *Stat1*).
- The cell type is **mouse bone marrow-derived macrophages (BMDMs)**.
- The perturbation mechanism is **CRISPRi** (transcriptional repression / knockdown).
- Fold-change values are **shrunken log2 fold-changes** (not raw).
- Each perturbation has 2 top DE genes (by adjusted p-value) selected as positives and 5 randomly sampled non-DE genes as negatives, yielding 7 total DE rows per perturbation. Dir rows are a subset corresponding to the DE-positive pairs only.
- The dataset covers **497 unique perturbations** and **2,496 unique target genes**.

---

*Data derived from CropFlow CRISPRi Perturb-seq. Format inspired by [PerturbQA](https://github.com/Genentech/PerturbQA) (Wu et al., ICLR 2025).*
