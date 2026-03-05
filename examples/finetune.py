"""
Track C -- Fine-tune Qwen3-4B-Thinking-2507 with LoRA on competition data.

Converts train.csv into chat-format examples, runs LoRA fine-tuning via
trl's SFTTrainer, then merges and saves the adapter for serving with vLLM.

Install the training dependencies (separate from serving -- see README):
    uv sync --extra train

Usage:
    uv run --extra train python examples/finetune.py                       # defaults
    uv run --extra train python examples/finetune.py --epochs 5 --lr 2e-4  # custom

After training, switch to the serve environment and start vLLM:
    uv sync --extra serve
    uv run --extra serve vllm serve outputs/finetuned_model/ \\
        --host 0.0.0.0 --port 8000 \\
        --max-model-len 4096

Then run inference:
    uv run --extra serve python examples/track_c_finetune.py \\
        --api-base http://localhost:8000/v1 \\
        --model outputs/finetuned_model/

NOTE: The train and serve extras use incompatible transformers versions
(>=5.3 vs <5) and cannot be installed simultaneously. After fine-tuning,
run the tokenizer patch described in README.md before serving.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "train.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "finetuned_model"
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


def check_dependencies() -> None:
    """Exit early with a helpful message if heavy dependencies are missing."""
    missing = []
    for mod in ("torch", "transformers", "peft", "trl", "datasets"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(
            f"Missing dependencies: {', '.join(missing)}\n"
            f"Install them with:\n"
            f"  pip install torch transformers peft trl datasets accelerate\n",
            file=sys.stderr,
        )
        sys.exit(1)


def build_chat_examples(train_csv: Path) -> list[dict]:
    """
    Convert train.csv rows into chat-format training examples.

    Each example is a dict with a "messages" key containing a list of
    {"role": ..., "content": ...} dicts suitable for SFTTrainer.
    """
    import pandas as pd

    # Delayed import so format_prompt is available
    sys.path.insert(0, str(ROOT))
    from mlgenx import format_prompt

    df = pd.read_csv(train_csv)
    examples = []

    answer_map = {
        ("de", 0): "A) Knockdown of {pert} does not impact {gene}.",
        ("de", 1): "B) Knockdown of {pert} results in differential expression of {gene}.",
        ("dir", 0): "A) Knockdown of {pert} results in down-regulation of {gene}.",
        ("dir", 1): "B) Knockdown of {pert} results in up-regulation of {gene}.",
    }

    for _, row in df.iterrows():
        task = row["task"]
        label = int(row["label"])
        pert = row["pert"]
        gene = row["gene"]

        user_prompt = format_prompt(pert, gene, task)
        answer_template = answer_map.get((task, label), f"Label: {label}")
        assistant_answer = answer_template.format(pert=pert, gene=gene)

        examples.append({
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert molecular biologist. Answer the "
                        "question about gene expression with the correct choice."
                    ),
                },
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": f"<answer>{'A' if label == 0 else 'B'}</answer>\n{assistant_answer}"},
            ]
        })

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track C: Fine-tune Qwen3-4B-Thinking-2507 with LoRA"
    )
    parser.add_argument(
        "--model-id", default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Directory to save the merged fine-tuned model",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--target-modules", nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="LoRA target modules",
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Save adapter only (don't merge into base model)",
    )
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    check_dependencies()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    print(f"Model:       {args.model_id}")
    print(f"Train CSV:   {args.train_csv}")
    print(f"Output:      {args.output_dir}")
    print(f"LoRA rank:   {args.lora_rank}")
    print(f"Epochs:      {args.epochs}")
    print(f"LR:          {args.lr}")
    print(f"Batch size:  {args.batch_size} (grad accum: {args.grad_accum_steps})")
    print(f"BF16:        {args.bf16}")
    print()

    # -- Load tokenizer and model --
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.use_cache = False

    # -- LoRA config --
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )

    # -- Build dataset --
    print("Building training dataset from train.csv...")
    chat_examples = build_chat_examples(args.train_csv)
    dataset = Dataset.from_list(chat_examples)
    print(f"  {len(dataset)} training examples")

    # -- Training config --
    training_dir = args.output_dir / "training_run"
    sft_config = SFTConfig(
        output_dir=str(training_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        seed=args.seed,
        max_length=args.max_seq_len,
        report_to="none",
    )

    # -- Train --
    print("Starting LoRA fine-tuning...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()

    # -- Save --
    if args.no_merge:
        save_path = args.output_dir / "adapter"
        print(f"Saving LoRA adapter to {save_path}...")
        trainer.model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
    else:
        print("Merging LoRA adapter into base model...")
        merged = trainer.model.merge_and_unload()
        save_path = args.output_dir
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving merged model to {save_path}...")
        merged.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))

    print()
    print("Done! To serve the model with vLLM:")
    print(f"  vllm serve {save_path} --host 0.0.0.0 --port 8000 --max-model-len 4096")
    print()
    print("Then run inference:")
    print(f"  python examples/track_c_finetune.py --api-base http://localhost:8000/v1 --model {save_path}")


if __name__ == "__main__":
    main()
