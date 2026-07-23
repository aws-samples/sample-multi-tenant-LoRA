# Codebase Notes

> Read this first before browsing the repo.

## Two implementations live here — use `src/` only

This project contains **two parallel implementations** from different iterations:

### ✅ `src/multi_tenant_lora/` — the production code

This is the implementation that was built, tested end-to-end on a real AWS
account, and documented in `README.md`, `SETUP_GUIDE.md`, and the blog post.

**This is the one customers should use.** It uses:

- SageMaker Python SDK v3 `SFTTrainer(training_type=LORA)` for training
- SageMaker JumpStart hub for the base model (no HuggingFace dependency)
- JumpStart's SFT-LoRA `RecipeCollection` hosting configs for inference
- A single `config/config.yaml` drives everything
- CLI entry point: `multi-tenant-lora ...`

Install and use:

```bash
pip install -e .
multi-tenant-lora preflight
multi-tenant-lora bootstrap
multi-tenant-lora run-all --version 1
```

Modules:
- `config.py` — YAML loader, schema, S3 path helpers
- `storage.py` — S3 bucket + data uploads
- `training.py` — per-tenant LoRA training via SDK v3
- `inference.py` — shared endpoint + adapter components
- `evaluation.py` — eval, compare, gate, publish metrics
- `mlops.py` — CloudWatch alarms + MLflow
- `scaling.py` — endpoint auto-scaling
- `cli.py` — CLI entry point (10 commands)

### ❌ `common/`, `training/`, `inference/`, `evaluation/`, `orchestration/` — legacy

These are from an earlier, more fragmented iteration. **Do not build on this.**
Tests still pass against it (they're still in `tests/`), but the working
pipeline uses `src/multi_tenant_lora/` instead.

You can safely delete these directories if you want — the only thing you'll
lose is a handful of older tests.

## Why two exist

The `src/`-based implementation replaces the flat-module one. The flat
structure works fine but was spread across too many small files for a
platform this size. Consolidating into 8 cohesive modules under `src/`
made the end-to-end flow easier to follow, test, and ship.

## Test suite

All 42 tests currently pass, but they exercise the legacy structure:

```bash
pytest tests/ -v      # 42 passed
```

When you migrate to a DynamoDB tenant registry (see `SCALING_TENANTS.md`)
or rewrite tests against `src/`, drop the legacy directories at that point.

## Author note

First deliverable pass done against a real account in us-west-2. Training
jobs and inference endpoint both verified working. See the blog's "What we
learned from building this" section for gotchas.
