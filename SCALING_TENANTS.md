# Scaling Tenant Management Beyond config.yaml

## When to migrate

The shipped platform stores tenants inline in `config/config.yaml`. This is the
right trade-off when you have **up to ~50 tenants**. Past that, a YAML file
starts to hurt:

| Problem | Symptom |
|---|---|
| No concurrent edits | Two operators racing to add tenants overwrite each other's changes |
| No audit trail | "Who added tenant-X and when?" has no answer |
| No atomic operations | You can't reliably update 3 tenants in one transaction |
| No querying | "Give me all premium-tier tenants on `ticket-classifier`" requires a script |
| Merge conflicts | Git PRs for tenant changes become toil |
| Slow startup | Parsing and validating thousands of tenants on every CLI invocation |

Move to a proper tenant registry once any of the following is true:

- You have more than 50 tenants
- More than one operator is onboarding/offboarding tenants
- You need per-tenant SLAs or tier-based routing
- Tenant changes need to trigger automated workflows (retraining, alerts)
- You need compliance-grade audit logs for tenant metadata changes

## Migration options, ranked

### Option 1: DynamoDB tenant registry (recommended default)

**Best for most SaaS customers up to 100,000+ tenants.**

A single DynamoDB table with one item per tenant:

```
Table: multi-tenant-lora-tenants

PK (tenant_id)   | display_name  | tier      | tasks                    | lora_overrides | status    | created_at
-----------------+---------------+-----------+--------------------------+----------------+-----------+-----------
tenant-alpha     | Tenant Alpha  | premium   | [ticket-classifier]      | {}             | active    | 2026-05-01
tenant-beta      | Tenant Beta   | standard  | [ticket-classifier]      | {r: 8}         | active    | 2026-05-02
tenant-omega     | Tenant Omega  | enterprise| [ticket, severity]       | {r: 32}        | disabled  | 2026-04-15
```

GSI: `status-index` (to query active/disabled tenants without a full scan).

Why DynamoDB:

- Serverless, no infrastructure to manage
- Millisecond reads, unlimited scale
- On-demand billing matches variable tenant counts
- IAM conditions let you enforce per-tenant access at the row level
- DynamoDB Streams trigger Lambda when tenants are added/updated → wire to retraining, provisioning, notifications
- Native backup and point-in-time recovery for compliance
- Cross-region global tables for multi-region platforms

### Option 2: Aurora PostgreSQL / RDS

**Best when you already run a relational database for your SaaS platform.**

```sql
CREATE TABLE tenants (
    tenant_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    tier TEXT NOT NULL,
    lora_overrides JSONB DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE tenant_tasks (
    tenant_id TEXT REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    task_name TEXT NOT NULL,
    description TEXT,
    PRIMARY KEY (tenant_id, task_name)
);

CREATE INDEX idx_tenants_tier ON tenants(tier);
CREATE INDEX idx_tenants_status ON tenants(status);
```

Why Aurora/RDS:

- ACID transactions across tenant + task updates
- SQL analytics: joins with customer, billing, usage tables
- Schema evolution via migrations
- Familiar tooling (pgAdmin, dbeaver, psql)

Cost: ~$60/month minimum for a db.t4g.medium — worth it if you already operate RDS.

### Option 3: S3-backed tenant manifests

**Best for small platforms that want zero new services.**

One JSON file per tenant at `s3://platform-bucket/tenants/<tenant_id>/manifest.json`:

```json
{
  "tenant_id": "tenant-alpha",
  "display_name": "Tenant Alpha",
  "tier": "premium",
  "tasks": [{"name": "ticket-classifier"}],
  "lora_overrides": {},
  "status": "active",
  "created_at": "2026-05-01T00:00:00Z"
}
```

Pros:

- No new AWS service to provision
- S3 versioning + object locking give you an audit trail for free
- Cross-region replication comes built-in

Cons:

- Listing all tenants is slow (S3 LIST is not indexed)
- No atomic multi-tenant updates
- No GSI-equivalent for querying by tier/status

Good for up to ~500 tenants and read-heavy workloads.

### Option 4: Event-driven onboarding (SaaS scale)

**Best for true SaaS with automated tenant onboarding.**

```
Customer signup API
       │
       ▼
    Lambda ────▶ DynamoDB (Tenants table)
                       │
                       ▼ DynamoDB Streams
                    Lambda ────▶ Step Functions
                                        │
                                        ├──▶ SageMaker Training Job
                                        ├──▶ S3 data bootstrap
                                        ├──▶ Deploy adapter IC
                                        └──▶ SNS notification
```

No one edits tenant lists by hand. Tenants get created by a signup API,
streams fire the provisioning workflow, and Step Functions orchestrates
training, adapter deployment, and evaluation. config.yaml stays
platform-only forever.

This is the right pattern for SaaS platforms with thousands of tenants
that self-onboard.

## Refactoring the platform to use a tenant registry

If you choose Option 1 (DynamoDB), the code change is contained to three
places:

### 1. Replace the inline tenants list with a backend pointer

```yaml
# config/config.yaml
tenants:
  registry: dynamodb
  table_name: multi-tenant-lora-tenants
  # Remove the inline list — tenants now live in the DynamoDB table
```

### 2. Introduce a `TenantRegistry` abstraction

```python
# src/multi_tenant_lora/tenant_registry.py

from typing import Protocol, Iterator
from .config import TenantConfig

class TenantRegistry(Protocol):
    """Contract every registry backend must satisfy."""

    def get(self, tenant_id: str) -> TenantConfig: ...
    def list_all(self, status: str = "active") -> Iterator[TenantConfig]: ...
    def add(self, tenant: TenantConfig) -> None: ...
    def update(self, tenant_id: str, updates: dict) -> None: ...
    def remove(self, tenant_id: str) -> None: ...


class DynamoDBTenantRegistry:
    def __init__(self, table_name: str, session):
        self._table = session.resource("dynamodb").Table(table_name)

    def get(self, tenant_id: str) -> TenantConfig:
        resp = self._table.get_item(Key={"tenant_id": tenant_id})
        if "Item" not in resp:
            raise KeyError(tenant_id)
        return TenantConfig.model_validate(resp["Item"])

    def list_all(self, status: str = "active") -> Iterator[TenantConfig]:
        paginator = self._table.meta.client.get_paginator("query")
        for page in paginator.paginate(
            TableName=self._table.name,
            IndexName="status-index",
            KeyConditionExpression="#s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": status},
        ):
            for item in page["Items"]:
                yield TenantConfig.model_validate(item)

    # ... add, update, remove
```

### 3. Swap the call sites

Everywhere the platform reads `config.tenants`, it now goes through the
registry:

```python
# training.py — before:
for tenant in self.config.tenants:
    for task in tenant.tasks:
        self.submit(tenant.id, task.name, version)

# training.py — after:
for tenant in self.registry.list_all(status="active"):
    for task in tenant.tasks:
        self.submit(tenant.id, task.name, version)
```

The CLI gets new commands for tenant lifecycle:

```bash
multi-tenant-lora tenant add \
    --id acme-corp \
    --display-name "Acme Corporation" \
    --tier premium \
    --task ticket-classifier \
    --task severity-predictor

multi-tenant-lora tenant list --status active --tier premium
multi-tenant-lora tenant get --id acme-corp
multi-tenant-lora tenant update --id acme-corp --lora-override r=32
multi-tenant-lora tenant disable --id acme-corp
multi-tenant-lora tenant import --from tenants.csv
```

## Which option should I pick?

| If you... | Pick |
|---|---|
| Have fewer than 50 tenants, single operator | Keep `config.yaml` |
| Have 50–1,000 tenants, multiple operators | DynamoDB |
| Already run a relational DB for the SaaS | Aurora/RDS |
| Need compliance-grade audit + zero new services | S3 manifests + versioning |
| Run true SaaS with self-onboarding | Event-driven + DynamoDB + Step Functions |

DynamoDB is the right default for almost everyone past the config.yaml
threshold. S3 and Aurora are niche. Event-driven is a maturity milestone
for SaaS platforms, not a starting point.

## What stays the same

The underlying SageMaker architecture is unchanged:

- Per-tenant training jobs with IAM isolation
- Shared inference endpoint + per-tenant adapter components
- Evaluation + gating pipeline
- CloudWatch alarms per tenant

The tenant registry only controls **where the list of tenants lives** —
it doesn't change how they're trained or served.
