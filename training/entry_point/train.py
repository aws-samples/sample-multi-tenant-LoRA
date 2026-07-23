"""LoRA fine-tuning entry point executed inside the HuggingFace training DLC.

SageMaker invokes this script as::

    python train.py --<hp_name> <value> ...

and places channel data at the paths advertised through the standard
SageMaker environment variables (``SM_CHANNEL_TRAINING``, ``SM_CHANNEL_MODEL``,
``SM_MODEL_DIR``, ...). After training, the adapter weights and tokenizer are
packaged into ``adapter.tar.gz`` and uploaded to the S3 URI supplied via the
``adapter_output_s3_uri`` hyperparameter so the inference component can load
them directly.

This script is intentionally dependency-light: it relies only on the
HuggingFace PyTorch training DLC image plus the extras listed in
``requirements.txt`` next to it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess  # nosec - controlled invocation of `aws s3 cp`
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("lora_trainer")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


@dataclass
class TrainArgs:
    hf_model_id: str
    peft_type: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    epochs: int
    per_device_train_batch_size: int
    learning_rate: float
    max_seq_length: int
    adapter_output_s3_uri: str


def _parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_id", type=str, required=True)
    parser.add_argument("--peft_type", type=str, default="LORA")
    parser.add_argument("--lora_r", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--lora_dropout", type=float, required=True)
    parser.add_argument("--target_modules", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--adapter_output_s3_uri", type=str, required=True)
    args, _ = parser.parse_known_args()

    return TrainArgs(
        hf_model_id=args.hf_model_id,
        peft_type=args.peft_type,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        adapter_output_s3_uri=args.adapter_output_s3_uri,
    )


# ---------------------------------------------------------------------------
# Environment helpers (SageMaker channels)
# ---------------------------------------------------------------------------


def _channel_dir(name: str) -> Optional[str]:
    return os.environ.get(f"SM_CHANNEL_{name.upper()}")


def _sm_model_dir() -> str:
    return os.environ.get("SM_MODEL_DIR", "/opt/ml/model")


def _resolve_model_source(hf_model_id: str) -> str:
    """Return the local or remote model identifier to pass into from_pretrained.

    If a ``model`` channel is mounted, use that directory (base weights shipped
    via S3). Otherwise fall back to the HF hub id.
    """
    model_channel = _channel_dir("model")
    if model_channel and any(Path(model_channel).iterdir()):
        logger.info("Using base model weights from channel: %s", model_channel)
        return model_channel
    logger.info("Using HuggingFace hub id: %s", hf_model_id)
    return hf_model_id


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


_INPUT_FIELDS = ("instruction", "input", "prompt")
_OUTPUT_FIELDS = ("output", "completion", "response")


def _extract_pair(record: Dict) -> Dict[str, str]:
    for f in _INPUT_FIELDS:
        if f in record:
            input_text = record[f]
            break
    else:
        raise ValueError(f"record missing input field: {list(record.keys())}")
    for f in _OUTPUT_FIELDS:
        if f in record:
            output_text = record[f]
            break
    else:
        raise ValueError(f"record missing output field: {list(record.keys())}")
    return {"input": str(input_text), "output": str(output_text)}


def _load_jsonl_dataset(path: Path):
    from datasets import Dataset  # imported inside the container

    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc
            records.append(_extract_pair(record))
    if not records:
        raise RuntimeError(f"no records parsed from {path}")
    return Dataset.from_list(records)


def _find_jsonl(channel: str, filename: str) -> Path:
    channel_dir = _channel_dir(channel)
    if not channel_dir:
        raise RuntimeError(f"SageMaker channel {channel!r} is not mounted")
    candidate = Path(channel_dir) / filename
    if not candidate.is_file():
        raise RuntimeError(f"{filename} not found under {channel_dir}")
    return candidate


# ---------------------------------------------------------------------------
# Tokenisation and training
# ---------------------------------------------------------------------------


def _tokenize_fn(tokenizer, max_seq_length: int):
    prompt_template = "### Instruction:\n{input}\n\n### Response:\n{output}"

    def _tokenize(batch):
        texts = [
            prompt_template.format(input=i, output=o)
            for i, o in zip(batch["input"], batch["output"])
        ]
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    return _tokenize


def _train(args: TrainArgs) -> Path:
    # Imports deferred so argparse failures surface first.
    import torch
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    model_src = _resolve_model_source(args.hf_model_id)

    logger.info("Loading tokenizer from %s", model_src)
    tokenizer = AutoTokenizer.from_pretrained(model_src, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model from %s", model_src)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        model_src, torch_dtype=dtype
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_path = _find_jsonl("training", "train.jsonl")
    train_dataset = _load_jsonl_dataset(train_path)

    eval_dataset = None
    training_dir = _channel_dir("training")
    val_path = Path(training_dir) / "validation.jsonl" if training_dir else None
    if val_path and val_path.is_file():
        eval_dataset = _load_jsonl_dataset(val_path)

    tokenize = _tokenize_fn(tokenizer, args.max_seq_length)
    train_dataset = train_dataset.map(
        tokenize, batched=True, remove_columns=train_dataset.column_names
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            tokenize, batched=True, remove_columns=eval_dataset.column_names
        )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = Path(_sm_model_dir())
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    logger.info("Starting LoRA fine-tune")
    trainer.train()

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving adapter to %s", adapter_dir)
    # save_pretrained on a PeftModel writes only the adapter weights.
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return adapter_dir


# ---------------------------------------------------------------------------
# Packaging and upload
# ---------------------------------------------------------------------------


def _package_adapter(adapter_dir: Path) -> Path:
    tmp_tarball = Path(tempfile.gettempdir()) / "adapter.tar.gz"
    logger.info("Packaging adapter directory %s -> %s", adapter_dir, tmp_tarball)
    with tarfile.open(tmp_tarball, "w:gz") as tar:
        for entry in adapter_dir.iterdir():
            tar.add(entry, arcname=entry.name)
    return tmp_tarball


def _upload_adapter(tarball: Path, s3_uri: str) -> None:
    if not s3_uri.startswith("s3://"):
        raise RuntimeError(f"adapter_output_s3_uri must be s3://, got {s3_uri!r}")
    if not s3_uri.endswith("/"):
        s3_uri = s3_uri + "/"
    destination = f"{s3_uri}adapter.tar.gz"
    logger.info("Uploading adapter tarball to %s", destination)
    # Use the AWS CLI bundled in the training DLC; falls back to boto3 if
    # unavailable.
    try:
        subprocess.run(  # nosec - fixed command, untrusted args quoted by argv
            ["aws", "s3", "cp", str(tarball), destination],
            check=True,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("aws cli upload failed (%s); falling back to boto3", exc)

    import boto3

    bucket, _, key = destination[len("s3://") :].partition("/")
    boto3.client("s3").upload_file(str(tarball), bucket, key)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        stream=sys.stdout,
    )

    args = _parse_args()
    logger.info("Training args: %s", args)

    adapter_dir = _train(args)
    tarball = _package_adapter(adapter_dir)
    _upload_adapter(tarball, args.adapter_output_s3_uri)
    shutil.copy(tarball, Path(_sm_model_dir()) / "adapter.tar.gz")
    logger.info("Training entry point completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
