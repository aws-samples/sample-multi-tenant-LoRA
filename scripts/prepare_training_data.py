#!/usr/bin/env python3
"""Prepare training data from raw Jira ITSM data for multi-tenant LoRA training.

This script reads the raw issues.json and produces JSONL files suitable for
the multi-tenant LoRA platform. It creates two classification tasks:

1. type-classifier: Predict the issue type from the ticket text
2. priority-classifier: Predict the priority level from the ticket text

Tenants are split by product area (derived from labels/components) to simulate
a real multi-tenant scenario where different teams/products get their own
fine-tuned adapter.

Output structure:
    prepared_data/
    ├── tenant-support/
    │   ├── type-classifier/
    │   │   ├── train.jsonl
    │   │   └── eval.jsonl
    │   └── priority-classifier/
    │       ├── train.jsonl
    │       └── eval.jsonl
    └── tenant-partner/
        ├── type-classifier/
        │   ├── train.jsonl
        │   └── eval.jsonl
        └── priority-classifier/
            ├── train.jsonl
            └── eval.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def extract_text_from_adf(adf: dict | None) -> str:
    """Extract plain text from Atlassian Document Format (ADF)."""
    if not adf or "content" not in adf:
        return ""
    parts: list[str] = []
    _walk_adf(adf["content"], parts)
    return " ".join(parts).strip()


def _walk_adf(nodes: list[dict], parts: list[str]) -> None:
    """Recursively walk ADF nodes to extract text."""
    for node in nodes:
        if node.get("type") == "text":
            text = node.get("text", "").strip()
            if text:
                parts.append(text)
        if "content" in node:
            _walk_adf(node["content"], parts)


def build_prompt(summary: str, description: str, task: str) -> str:
    """Build a classification prompt from ticket text."""
    ticket_text = summary
    if description and description != summary:
        # Truncate long descriptions to keep training efficient
        desc_truncated = description[:500]
        ticket_text = f"{summary}\n\nDetails: {desc_truncated}"

    if task == "type-classifier":
        return (
            f"Classify the following support ticket into one of these categories: "
            f"Support Request, Partner, Licensing, Question, Improvement, Security, Incident, Task.\n\n"
            f"Ticket: {ticket_text}\n\n"
            f"Category:"
        )
    elif task == "priority-classifier":
        return (
            f"Classify the priority of the following support ticket. "
            f"Choose one: Critical, Major, Minor, Trivial.\n\n"
            f"Ticket: {ticket_text}\n\n"
            f"Priority:"
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def assign_tenant(issue: dict) -> str:
    """Assign an issue to a tenant based on issue type grouping.

    Two tenants:
    - tenant-support: General support (Support Request, Question, Improvement,
      Task, Security, Incident) — the bulk of tickets
    - tenant-partner: Partner and licensing related tickets
    """
    issue_type = issue["fields"]["issuetype"]["name"]
    if issue_type in ("Partner", "Licensing"):
        return "tenant-partner"
    else:
        # Support Request, Question, Improvement, Task, Security, Incident
        return "tenant-support"


def main() -> None:
    random.seed(42)

    # Load raw data
    data_dir = Path("data")
    output_dir = Path("prepared_data")
    output_dir.mkdir(exist_ok=True)

    print("Loading issues.json...")
    with open(data_dir / "issues.json", "r") as f:
        issues = json.load(f)
    print(f"  Loaded {len(issues)} issues")

    # Group issues by tenant
    tenant_issues: dict[str, list[dict]] = {}
    skipped = 0

    for issue in issues:
        fields = issue["fields"]
        summary = fields.get("summary", "").strip()
        description = extract_text_from_adf(fields.get("description"))

        # Skip issues with no usable text
        combined = f"{summary} {description}".strip()
        if len(combined) < 15:
            skipped += 1
            continue

        tenant_id = assign_tenant(issue)
        if tenant_id not in tenant_issues:
            tenant_issues[tenant_id] = []
        tenant_issues[tenant_id].append(issue)

    print(f"  Skipped {skipped} issues with insufficient text")
    print(f"  Tenant distribution:")
    for tid, issues_list in sorted(tenant_issues.items()):
        print(f"    {tid}: {len(issues_list)} issues")

    # Generate training data for each tenant and task
    tasks = ["type-classifier", "priority-classifier"]
    eval_ratio = 0.15  # 15% for evaluation

    for tenant_id, tenant_data in tenant_issues.items():
        # Shuffle for random train/eval split
        random.shuffle(tenant_data)
        split_idx = max(1, int(len(tenant_data) * (1 - eval_ratio)))
        train_issues = tenant_data[:split_idx]
        eval_issues = tenant_data[split_idx:]

        for task in tasks:
            task_dir = output_dir / tenant_id / task
            task_dir.mkdir(parents=True, exist_ok=True)

            # Build training examples
            train_examples = []
            for issue in train_issues:
                fields = issue["fields"]
                summary = fields.get("summary", "").strip()
                description = extract_text_from_adf(fields.get("description"))
                prompt = build_prompt(summary, description, task)

                if task == "type-classifier":
                    completion = fields["issuetype"]["name"]
                elif task == "priority-classifier":
                    completion = fields["priority"]["name"]
                else:
                    continue

                train_examples.append({
                    "prompt": prompt,
                    "completion": f" {completion}"
                })

            eval_examples = []
            for issue in eval_issues:
                fields = issue["fields"]
                summary = fields.get("summary", "").strip()
                description = extract_text_from_adf(fields.get("description"))
                prompt = build_prompt(summary, description, task)

                if task == "type-classifier":
                    completion = fields["issuetype"]["name"]
                elif task == "priority-classifier":
                    completion = fields["priority"]["name"]
                else:
                    continue

                eval_examples.append({
                    "prompt": prompt,
                    "completion": f" {completion}"
                })

            # Write JSONL files
            train_path = task_dir / "train.jsonl"
            with open(train_path, "w") as f:
                for ex in train_examples:
                    f.write(json.dumps(ex) + "\n")

            eval_path = task_dir / "eval.jsonl"
            with open(eval_path, "w") as f:
                for ex in eval_examples:
                    f.write(json.dumps(ex) + "\n")

            print(f"  {tenant_id}/{task}: {len(train_examples)} train, {len(eval_examples)} eval")

    print(f"\nData written to: {output_dir}/")
    print("\nNext steps:")
    print("  1. Update config/config.yaml with your AWS account details")
    print("  2. Run: multi-tenant-lora bootstrap")
    print("  3. Upload data for each tenant/task")
    print("  4. Run: multi-tenant-lora run-all --version 1")


if __name__ == "__main__":
    main()
