# Multi-Tenant LoRA Fine-Tuning & Inference on Amazon SageMaker AI

A production-grade, reusable implementation of the architecture described in the
[Multi-Tenant LoRA SageMaker blog](../Multi_Tenant_LoRA_SageMaker_Blog_rewritten.md).

**One base foundation model serves hundreds of tenants, each with their own
LoRA adapter, from a single SageMaker endpoint.** Every tenant is isolated at
the training and IAM layers.

## Why this exists

The blog describes the architecture. This project is the turnkey implementation —
drop in a config file, plug in your tenant data, run three commands, and you have:

1. Per-tenant LoRA training jobs on SageMaker (native SDK v3 `SFTTrainer`)
2. A shared inference endpoint hosting the base model
3. Per-tenant adapter inference components for routing
4. Automated evaluation + regression gating
5. CloudWatch alarms per tenant/task
6. Endpoint auto-scaling

## What customers customize (one file)

Everything lives in `config/config.yaml`. The code itself never changes:

```yaml
base_model:
  hf_model_id: meta-textgeneration-llama-3-2-1b-instruct  # any JumpStart model
  name: llama-3-2-1b

training:
  instance_type: ml.g5.2xlarge
  lora: { r: 16, alpha: 32 }
  hyperparameters:
    epochs: 3
    learning_rate: 2.0e-4

inference:
  instance_type: ml.g5.12xlarge  # scales with base model size
  option:
    max_loras: 30         # hot in GPU
    max_cpu_loras: 70     # warm in CPU

tenants:
  - id: acme-corp
    tasks: [{ name: ticket-classifier }]
  - id: globex
    lora_overrides: { r: 8 }
    tasks: [{ name: ticket-classifier }, { name: severity-predictor }]
```

## Quickstart

**For first-time setup, see [SETUP_GUIDE.md](SETUP_GUIDE.md)** for IAM roles,
service quotas, and AWS prerequisites.

Once the AWS setup is done, customers only edit `config/config.yaml` and run:

```bash
# One-time, after editing config.yaml
multi-tenant-lora preflight          # verify AWS access
multi-tenant-lora bootstrap          # create the S3 bucket

# Upload each tenant's data (one call per tenant+task)
multi-tenant-lora upload-data \
    --tenant acme-corp --task ticket-classifier \
    --train-file ./data/acme/train.jsonl \
    --eval-file ./data/acme/eval.jsonl

# One command runs the whole pipeline: train → deploy → evaluate → alarm
multi-tenant-lora run-all --version 1

# Test inference
multi-tenant-lora invoke \
    --tenant acme-corp --task ticket-classifier --version 1 \
    --prompt "Printer is offline on floor 3"

# Tear down when done (stops GPU billing)
multi-tenant-lora teardown --confirm
```

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  TRAINING LAYER (per tenant, strict isolation)                  │
  │                                                                 │
  │   tenant-alpha ┐                                               │
  │                ├── SFTTrainer(LORA) → ml.g5.2xlarge            │
  │   tenant-beta  ┘    ↓                                          │
  │                    adapter.tar.gz →                             │
  │                    s3://.../tenants/<id>/adapters/<task>/v1/    │
  └─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  INFERENCE LAYER (shared endpoint, logical isolation)           │
  │                                                                 │
  │   ┌──────────────────────────────────────────┐                  │
  │   │ SageMaker Endpoint (ml.g5.12xlarge)      │                  │
  │   │                                          │                  │
  │   │  Base Inference Component (FM, shared)   │                  │
  │   │  ├─ Adapter Component: alpha-v1          │                  │
  │   │  ├─ Adapter Component: beta-v1           │                  │
  │   │  └─ Adapter Component: ...               │                  │
  │   └──────────────────────────────────────────┘                  │
  │                                                                 │
  │   Invocation:                                                   │
  │     InvokeEndpoint(InferenceComponentName="alpha-v1", ...)      │
  └─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  EVALUATION + MLOPS                                             │
  │                                                                 │
  │   eval.jsonl → invoke adapter → compute accuracy                │
  │   → compare to previous version                                 │
  │   → APPROVE or BLOCK                                            │
  │   → publish CloudWatch metric                                   │
  │   → alarm on regression                                         │
  └─────────────────────────────────────────────────────────────────┘
```

## Project layout

```
multi_tenant_lora_sagemaker/
├── config/
│   ├── config.example.yaml        # Copy to config.yaml and customize
│   └── config.yaml                # Your working config
├── src/
│   ├── multi_tenant_lora/         # The platform code
│   │   ├── config.py              # Config schema + S3 path helpers
│   │   ├── storage.py             # S3 bucket + data uploads
│   │   ├── training.py            # SFTTrainer-based per-tenant training
│   │   ├── inference.py           # Shared endpoint + adapter components
│   │   ├── evaluation.py          # Eval + gate + CloudWatch publish
│   │   ├── mlops.py               # CloudWatch alarms + MLflow
│   │   ├── scaling.py             # Endpoint auto-scaling
│   │   └── cli.py                 # CLI entry point
│   └── scripts/                   # (Legacy) custom training script
├── tests/                         # Unit tests (pytest + moto)
├── scripts/
│   └── prepare_test_data.py       # Download + partition a public dataset
├── iam/                           # IAM role policies for setup
├── SETUP_GUIDE.md                 # First-time AWS account setup
├── SCALING_TENANTS.md             # Migration path for 1,000+ tenants
├── pyproject.toml
└── requirements.txt
```

## CLI reference

| Command | Purpose |
|---------|---------|
| `bootstrap` | Create the platform S3 bucket with encryption + versioning |
| `upload-data` | Upload a tenant's train/eval JSONL files |
| `train` | Submit LoRA training jobs (`--tenant` optional, defaults to all) |
| `deploy-endpoint` | Create the shared inference endpoint + base component |
| `deploy-adapter` | Register an adapter as an inference component |
| `evaluate` | Run evaluation + regression gating for a specific adapter |
| `invoke` | Send one prompt to a tenant's adapter (smoke test) |
| `create-alarms` | Create CloudWatch alarms for every tenant+task |
| `list-adapters` | List deployed adapter components |
| `teardown` | Delete endpoint + components (requires `--confirm`) |

Run any command with `--help` for details.

## Tenant isolation — how it works

| Layer | Mechanism | Enforcement |
|-------|-----------|-------------|
| Training data | IAM-scoped S3 prefixes | Per-tenant prefix, training job role |
| Training compute | Separate SageMaker jobs | Natural isolation from the job model |
| Adapter artifacts | Tenant-scoped S3 paths | `tenants/<id>/adapters/<task>/v<n>/` |
| Inference routing | `InferenceComponentName` | Embeds tenant_id as prefix |
| IAM invoke | Name-prefix condition | `sagemaker:InferenceComponentName` |
| Observability | CloudWatch dimensions | `TenantId` + `Task` dimensions |

## Cost guardrails

- Training job stopping condition: capped at `max_runtime_seconds`
- Inference auto-scaling: configurable min/max instance counts
- `teardown` command deletes everything billable (endpoint + components)
- S3 data is retained unless manually deleted — adapter artifacts are small (typically <50 MB each)

## Testing

```bash
# Unit tests (uses moto to mock AWS)
pytest tests/ -v
```

## Migration to another AWS account

The entire platform is account-portable:

1. Copy `config/config.yaml`
2. Update `aws.region`, `aws.execution_role_arn`, and `storage.platform_bucket`
3. Ensure the execution role has S3 access to the new bucket
4. Run `bootstrap`

No code changes required.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

