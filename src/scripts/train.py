"""LoRA fine-tuning script — runs INSIDE the SageMaker training container.

Loaded via the HuggingFace PyTorch DLC. Receives:
  - Training data at /opt/ml/input/data/training/train.jsonl
  - Eval data at /opt/ml/input/data/eval/eval.jsonl (optional)
  - Hyperparameters via CLI args (SageMaker converts HyperParameters to args)

Writes the trained LoRA adapter to /opt/ml/model — SageMaker tars it up and
uploads to the output S3 URI automatically.

Expected JSONL schema per line:
    {"prompt": "...", "completion": "..."}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Argument parsing — these are provided by SageMaker as CLI args
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # SageMaker conventions (env vars point to these paths)
    parser.add_argument("--train_data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--eval_data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL"))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

    # Base model
    parser.add_argument("--hf_model_id", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,v_proj,k_proj,o_proj",
        help="Comma-separated attention modules to adapt.",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--flash_attention_2", action="store_true", default=False)

    # Metadata (passed through, useful for logging)
    parser.add_argument("--tenant_id", type=str, default="")
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--version", type=str, default="")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {e}") from e
    return records


def find_jsonl(data_dir: str) -> Path | None:
    if not data_dir:
        return None
    d = Path(data_dir)
    if not d.is_dir():
        return None
    # Prefer train.jsonl/eval.jsonl by convention, fall back to any .jsonl
    for candidate in ("train.jsonl", "eval.jsonl"):
        p = d / candidate
        if p.is_file():
            return p
    matches = list(d.glob("*.jsonl"))
    return matches[0] if matches else None


def build_dataset(
    records: list[dict], tokenizer, max_seq_length: int
) -> Dataset:
    """Format {prompt, completion} into a single training sequence and tokenize."""

    def format_example(rec: dict) -> str:
        prompt = rec.get("prompt", "").strip()
        completion = rec.get("completion", "").strip()
        if not prompt or not completion:
            raise ValueError(
                "Each training record must have non-empty 'prompt' and 'completion' fields"
            )
        # Simple instruction template; tenants can customize via their data.
        return f"### Instruction:\n{prompt}\n\n### Response:\n{completion}{tokenizer.eos_token}"

    texts = [format_example(r) for r in records]
    ds = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    return ds.map(tokenize, batched=True, remove_columns=["text"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    logger.info("Starting LoRA fine-tuning job")
    logger.info("tenant=%s task=%s version=%s", args.tenant_id, args.task, args.version)
    logger.info("Base model: %s", args.hf_model_id)

    # Locate training data
    train_path = find_jsonl(args.train_data_dir)
    if not train_path:
        raise FileNotFoundError(f"No training JSONL found in {args.train_data_dir}")
    logger.info("Training data: %s", train_path)

    eval_path = find_jsonl(args.eval_data_dir) if args.eval_data_dir else None
    if eval_path:
        logger.info("Eval data: %s", eval_path)

    # Load base model + tokenizer
    logger.info("Loading tokenizer...")
    tokenizer_kwargs: dict = {}
    if args.hf_token:
        tokenizer_kwargs["token"] = args.hf_token
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model...")
    model_kwargs: dict = {
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "device_map": "auto",
    }
    if args.flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if args.hf_token:
        model_kwargs["token"] = args.hf_token

    base_model = AutoModelForCausalLM.from_pretrained(args.hf_model_id, **model_kwargs)
    base_model.config.use_cache = False

    # Wrap with LoRA
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, peft_cfg)
    model.print_trainable_parameters()

    # Build datasets
    logger.info("Building datasets...")
    train_records = load_jsonl(train_path)
    if not train_records:
        raise ValueError(f"No training records found in {train_path}")
    train_ds = build_dataset(train_records, tokenizer, args.max_seq_length)

    eval_ds = None
    if eval_path:
        eval_records = load_jsonl(eval_path)
        if eval_records:
            eval_ds = build_dataset(eval_records, tokenizer, args.max_seq_length)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy="epoch" if eval_ds is not None else "no",
        bf16=args.bf16,
        report_to=[],
        remove_unused_columns=False,
        # Only save the LoRA adapter, not the base model
        save_total_limit=1,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save ONLY the LoRA adapter (PeftModel.save_pretrained saves adapter_config.json + adapter_model.safetensors)
    logger.info("Saving adapter to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Write a small metadata file so the evaluation pipeline can identify this artifact
    metadata = {
        "tenant_id": args.tenant_id,
        "task": args.task,
        "version": args.version,
        "base_model": args.hf_model_id,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "epochs": args.epochs,
        "num_train_examples": len(train_records),
    }
    with (Path(args.output_dir) / "adapter_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training complete. Adapter saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
