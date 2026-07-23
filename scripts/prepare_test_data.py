"""Prepare real training data for the end-to-end test.

Uses a small public sentiment dataset (stanfordnlp/sentiment_treebank) —
a well-known benchmark, 5-class sentiment classification. Each tenant gets
a subset that simulates per-customer data partitions.

Run:
    python scripts/prepare_test_data.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

random.seed(42)

OUT_DIR = Path(__file__).parent.parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


LABEL_MAP = {0: "negative", 1: "negative", 2: "neutral", 3: "positive", 4: "positive"}


def to_record(text: str, label: int) -> dict:
    return {
        "prompt": f"Classify the sentiment as positive, negative, or neutral.\n\nText: {text}",
        "completion": LABEL_MAP[label],
    }


def main() -> None:
    print("Loading SST-5 sentiment dataset...")
    ds = load_dataset("SetFit/sst5", split="train").shuffle(seed=42)

    # Give tenant-alpha the first 100, tenant-beta the next 100
    # This simulates per-customer data partitions
    partitions = {
        "tenant-alpha": ds.select(range(100)),
        "tenant-beta": ds.select(range(100, 200)),
    }

    for tenant_id, data in partitions.items():
        tenant_dir = OUT_DIR / tenant_id / "sentiment"
        tenant_dir.mkdir(parents=True, exist_ok=True)

        # 80/20 train/eval split
        split_point = int(0.8 * len(data))
        train_data = data.select(range(split_point))
        eval_data = data.select(range(split_point, len(data)))

        train_file = tenant_dir / "train.jsonl"
        with train_file.open("w") as f:
            for row in train_data:
                record = to_record(row["text"], row["label"])
                f.write(json.dumps(record) + "\n")

        eval_file = tenant_dir / "eval.jsonl"
        with eval_file.open("w") as f:
            for row in eval_data:
                record = to_record(row["text"], row["label"])
                f.write(json.dumps(record) + "\n")

        print(
            f"  {tenant_id}: train={len(train_data)} eval={len(eval_data)} -> {tenant_dir}"
        )


if __name__ == "__main__":
    main()
