#!/usr/bin/env python3
"""Prepare response-generation training data from ITSM conversation threads.

Extracts (ticket context + customer message) → agent reply pairs from the
issue comments. This trains adapters that can draft support responses.

Tenant split:
- tenant-support: General support tickets (Support Request, Question, etc.)
- tenant-partner: Partner & licensing tickets

Each tenant's adapter learns that tenant's communication style and domain.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from collections import defaultdict


# Agent domains — these are the support staff
AGENT_DOMAINS = {"appfire.fake", "achievers.fake"}

# Minimum lengths for quality filtering
MIN_CUSTOMER_MSG_LEN = 20
MIN_AGENT_REPLY_LEN = 50
MAX_AGENT_REPLY_LEN = 1500  # Truncate very long replies


def extract_text_from_adf(adf: dict | None) -> str:
    """Extract plain text from Atlassian Document Format."""
    if not adf or "content" not in adf:
        return ""
    parts: list[str] = []
    _walk_adf(adf["content"], parts)
    return " ".join(parts).strip()


def _walk_adf(nodes: list[dict], parts: list[str]) -> None:
    for node in nodes:
        if node.get("type") == "text":
            text = node.get("text", "").strip()
            if text:
                parts.append(text)
        if "content" in node:
            _walk_adf(node["content"], parts)


def is_agent(email: str) -> bool:
    """Check if an email belongs to a support agent."""
    if not email or "@" not in email:
        return False
    domain = email.split("@")[-1]
    return domain in AGENT_DOMAINS


def is_boilerplate(text: str) -> bool:
    """Filter out automated/boilerplate messages."""
    boilerplate_markers = [
        "Customer info:",
        "Customer Identifier",
        "Customer Rank",
        "Auto re-assignment",
        "Did this resolve your issue?",
        "User submitted No",
        "User submitted Yes",
        "No response submitted",
        "👋 Hi! I'm your virtual service agent",
        "Tip for Accuracy:",
        "[Automatic AI Triage]",
        "Please note this request has been created from",
        "How to find SEN",
        "How to find the Entitlement number",
    ]
    for marker in boilerplate_markers:
        if marker in text:
            return True
    # Very short agent messages are usually not useful
    if len(text) < MIN_AGENT_REPLY_LEN:
        return True
    return False


def build_response_prompt(summary: str, description: str, conversation_history: list[dict], customer_msg: str) -> str:
    """Build a prompt for response generation."""
    # Ticket context
    ticket_context = summary
    if description and description != summary:
        ticket_context += f"\n\nDescription: {description[:400]}"

    # Recent conversation (last 2 exchanges max for context)
    history_text = ""
    if conversation_history:
        recent = conversation_history[-4:]  # Last 4 messages
        history_lines = []
        for msg in recent:
            role = "Agent" if msg["is_agent"] else "Customer"
            history_lines.append(f"{role}: {msg['text'][:300]}")
        history_text = "\n".join(history_lines)

    prompt = f"""You are a technical support agent. Based on the ticket and conversation, write a helpful response to the customer.

Ticket: {ticket_context}

"""
    if history_text:
        prompt += f"Conversation history:\n{history_text}\n\n"

    prompt += f"Customer's latest message: {customer_msg}\n\nAgent response:"
    return prompt


def assign_tenant(issue: dict) -> str:
    """Assign issue to tenant based on type."""
    issue_type = issue["fields"]["issuetype"]["name"]
    if issue_type in ("Partner", "Licensing"):
        return "tenant-partner"
    return "tenant-support"


def extract_training_pairs(issue: dict) -> list[dict]:
    """Extract (prompt, completion) pairs from an issue's comment thread."""
    fields = issue["fields"]
    summary = fields.get("summary", "").strip()
    description = extract_text_from_adf(fields.get("description"))
    comments = fields.get("comment", {}).get("comments", [])

    if len(comments) < 2:
        return []

    # Build conversation timeline
    timeline = []
    for c in comments:
        author_email = c.get("author", {}).get("emailAddress", "")
        body = extract_text_from_adf(c.get("body", {}))
        if not body:
            continue
        timeline.append({
            "text": body,
            "is_agent": is_agent(author_email),
            "author": c.get("author", {}).get("displayName", "Unknown"),
        })

    # Find customer→agent pairs
    pairs = []
    conversation_history = []

    for i in range(len(timeline) - 1):
        current = timeline[i]
        next_msg = timeline[i + 1]

        # We want: customer message followed by agent reply
        if not current["is_agent"] and next_msg["is_agent"]:
            customer_msg = current["text"]
            agent_reply = next_msg["text"]

            # Quality filters
            if len(customer_msg) < MIN_CUSTOMER_MSG_LEN:
                conversation_history.append(current)
                continue
            if is_boilerplate(agent_reply):
                conversation_history.append(current)
                continue

            # Truncate long replies
            if len(agent_reply) > MAX_AGENT_REPLY_LEN:
                agent_reply = agent_reply[:MAX_AGENT_REPLY_LEN]

            prompt = build_response_prompt(
                summary, description, conversation_history, customer_msg
            )
            pairs.append({
                "prompt": prompt,
                "completion": f" {agent_reply}"
            })

        conversation_history.append(current)

    return pairs


def main() -> None:
    random.seed(42)

    data_dir = Path("data")
    output_dir = Path("prepared_data")
    output_dir.mkdir(exist_ok=True)

    print("Loading issues.json...")
    with open(data_dir / "issues.json", "r") as f:
        issues = json.load(f)
    print(f"  Loaded {len(issues)} issues")

    # Extract training pairs grouped by tenant
    tenant_pairs: dict[str, list[dict]] = defaultdict(list)
    issues_with_pairs = 0

    for issue in issues:
        pairs = extract_training_pairs(issue)
        if pairs:
            issues_with_pairs += 1
            tenant_id = assign_tenant(issue)
            tenant_pairs[tenant_id].extend(pairs)

    print(f"  Issues with usable conversation pairs: {issues_with_pairs}/{len(issues)}")
    print(f"  Training pairs by tenant:")
    for tid, pairs in sorted(tenant_pairs.items()):
        print(f"    {tid}: {len(pairs)} pairs")

    # Write train/eval splits
    eval_ratio = 0.15
    task_name = "response-generation"

    for tenant_id, pairs in tenant_pairs.items():
        random.shuffle(pairs)
        split_idx = max(1, int(len(pairs) * (1 - eval_ratio)))
        train_pairs = pairs[:split_idx]
        eval_pairs = pairs[split_idx:]

        task_dir = output_dir / tenant_id / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        with open(task_dir / "train.jsonl", "w") as f:
            for p in train_pairs:
                f.write(json.dumps(p) + "\n")

        with open(task_dir / "eval.jsonl", "w") as f:
            for p in eval_pairs:
                f.write(json.dumps(p) + "\n")

        print(f"  {tenant_id}/{task_name}: {len(train_pairs)} train, {len(eval_pairs)} eval")

    # Print metadata only — never echo real customer ticket content here.
    # The training data is extracted from production support tickets and may
    # contain PII (names, emails, account IDs). Sample output is intentionally
    # restricted to counts and field-length stats.
    print("\n=== Sample training pair stats (tenant-support) ===")
    if tenant_pairs["tenant-support"]:
        sample = tenant_pairs["tenant-support"][0]
        print(f"  prompt length:     {len(sample['prompt'])} chars")
        print(f"  completion length: {len(sample['completion'])} chars")
        print("  (content suppressed — may contain customer PII)")

    print(f"\nData written to: {output_dir}/")


if __name__ == "__main__":
    main()
