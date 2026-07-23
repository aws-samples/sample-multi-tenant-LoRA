# Setup Guide — Deploying to a New AWS Account

This guide walks you through the one-time AWS setup before you can run the
multi-tenant LoRA platform. Once these steps are done, customers only edit
`config/config.yaml` and run one command.

## Prerequisites checklist

- [ ] AWS account with admin or SageMaker permissions
- [ ] Python 3.10 or newer
- [ ] AWS CLI installed and configured (`aws configure`)
- [ ] S3 bucket you own (or let the tool auto-create one)
- [ ] Service quota for the GPU instance types you'll use (see below)

## Step 1: Configure AWS credentials

The tool uses the default AWS credentials chain. Any of these works:

```bash
# Option A: AWS CLI profile (recommended for dev)
aws configure
aws configure set region us-west-2

# Option B: Environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-west-2

# Option C: SSO
aws sso login
```

Verify it works:

```bash
aws sts get-caller-identity
```

## Step 2: Create the SageMaker execution role

SageMaker training and inference jobs assume an IAM role to read your data
from S3 and write model artifacts back. The tool includes the exact policies
you need.

> ⚠️ **Security note — do not attach `AmazonSageMakerFullAccess`.** Earlier
> versions of this guide suggested attaching the AWS-managed
> `AmazonSageMakerFullAccess` policy. That policy contains wildcard actions
> across many services and is not least-privilege. The threat model
> (`.threatmodel/threat-model.md`, threat **T17**) explicitly calls this out
> as an abuse path. The instructions below build a scoped inline policy
> instead.

### Option A: use an existing role

If you already have a role named `AmazonSageMaker-ExecutionRole-*`, you can
reuse it — but verify it does **not** have `AmazonSageMakerFullAccess`
attached, and that it has the permissions in
`iam/execution_role_policy.json` plus the scoped SageMaker policy below.

### Option B: create a new role (recommended)

Replace `PLATFORM_BUCKET_NAME` in the policy with the bucket name you plan
to use (e.g., `multi-tenant-lora-platform-<account-id>-<region>`). Also
substitute `REGION`, `ACCOUNT`, and your platform naming prefix
(e.g. `mtl-`) in the inline SageMaker policy.

```bash
# 1) Create the role with trust policy
aws iam create-role \
    --role-name MultiTenantLoRAExecutionRole \
    --assume-role-policy-document file://iam/trust_policy.json

# 2) Attach the scoped platform permissions (S3, ECR, Logs, CloudWatch)
aws iam put-role-policy \
    --role-name MultiTenantLoRAExecutionRole \
    --policy-name MultiTenantLoRAPlatform \
    --policy-document file://iam/execution_role_policy.json

# 3) Attach a scoped SageMaker policy in place of AmazonSageMakerFullAccess.
#    Save the JSON below to /tmp/sagemaker-scoped.json first, with REGION,
#    ACCOUNT, and the "mtl-" prefix replaced for your environment.
aws iam put-role-policy \
    --role-name MultiTenantLoRAExecutionRole \
    --policy-name MultiTenantLoRASageMakerScoped \
    --policy-document file:///tmp/sagemaker-scoped.json

# 4) Get the role ARN for config.yaml
aws iam get-role --role-name MultiTenantLoRAExecutionRole \
    --query 'Role.Arn' --output text
```

#### Scoped SageMaker policy (replacement for AmazonSageMakerFullAccess)

This grants only what the platform actually uses: training-job
create/describe/stop, model registry writes, endpoint + endpoint-config +
inference-component management, model package group reads, and
`InvokeEndpoint`. Replace `REGION`, `ACCOUNT`, and the `mtl-` naming
prefix with values that match `common/naming.py` in your deployment.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "TrainingJobs",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:StopTrainingJob",
        "sagemaker:ListTrainingJobs",
        "sagemaker:AddTags"
      ],
      "Resource": "arn:aws:sagemaker:REGION:ACCOUNT:training-job/mtl-*"
    },
    {
      "Sid": "ModelsAndEndpoints",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateModel",
        "sagemaker:DescribeModel",
        "sagemaker:DeleteModel",
        "sagemaker:CreateEndpointConfig",
        "sagemaker:DescribeEndpointConfig",
        "sagemaker:DeleteEndpointConfig",
        "sagemaker:CreateEndpoint",
        "sagemaker:DescribeEndpoint",
        "sagemaker:UpdateEndpoint",
        "sagemaker:DeleteEndpoint",
        "sagemaker:CreateInferenceComponent",
        "sagemaker:DescribeInferenceComponent",
        "sagemaker:UpdateInferenceComponent",
        "sagemaker:DeleteInferenceComponent",
        "sagemaker:ListInferenceComponents"
      ],
      "Resource": [
        "arn:aws:sagemaker:REGION:ACCOUNT:model/mtl-*",
        "arn:aws:sagemaker:REGION:ACCOUNT:endpoint-config/mtl-*",
        "arn:aws:sagemaker:REGION:ACCOUNT:endpoint/mtl-*",
        "arn:aws:sagemaker:REGION:ACCOUNT:inference-component/mtl-*"
      ]
    },
    {
      "Sid": "ModelRegistry",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateModelPackage",
        "sagemaker:DescribeModelPackage",
        "sagemaker:ListModelPackages",
        "sagemaker:UpdateModelPackage",
        "sagemaker:CreateModelPackageGroup",
        "sagemaker:DescribeModelPackageGroup"
      ],
      "Resource": [
        "arn:aws:sagemaker:REGION:ACCOUNT:model-package/mtl-*",
        "arn:aws:sagemaker:REGION:ACCOUNT:model-package-group/mtl-*"
      ]
    },
    {
      "Sid": "Invoke",
      "Effect": "Allow",
      "Action": [
        "sagemaker:InvokeEndpoint",
        "sagemaker:InvokeEndpointWithResponseStream"
      ],
      "Resource": "arn:aws:sagemaker:REGION:ACCOUNT:endpoint/mtl-*"
    },
    {
      "Sid": "PassRoleToSelf",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::ACCOUNT:role/MultiTenantLoRAExecutionRole",
      "Condition": {
        "StringEquals": {
          "iam:PassedToService": "sagemaker.amazonaws.com"
        }
      }
    }
  ]
}
```

> 🔐 **Tenant isolation reminder.** This shared execution role can read and
> write **all** tenant prefixes inside `PLATFORM_BUCKET_NAME`. Anyone who
> can assume the role (operators, training jobs, endpoints) can reach every
> tenant's data. If your trust model requires per-tenant isolation, follow
> mitigation **M1** in `.threatmodel/threat-model.md` and create a separate
> role per tenant with S3 conditions like
> `s3:prefix = "tenants/<tenant-id>/*"`. Treat the shared-role pattern as
> appropriate **only** when operators are explicitly trusted with
> cross-tenant access.

## Step 3: Request service quotas

New AWS accounts have 0 quota for GPU instances. You need to request a
quota increase for the instance types you plan to use.

Check your current quota:

```bash
aws service-quotas list-service-quotas --service-code sagemaker \
    --query 'Quotas[?contains(QuotaName, `ml.g5`)].[QuotaName,Value]' \
    --output table
```

Request an increase at:
https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas

You need quota for (at minimum):

| Instance | Used For | Typical Request |
|----------|----------|----------------:|
| `ml.g5.2xlarge` for training jobs | Per-tenant LoRA training | 2 or more |
| `ml.g5.4xlarge` for endpoints | Shared inference endpoint | 1 |

Quota approval typically takes 24-48 hours.

## Step 4: Install the platform

```bash
# Clone or copy the multi_tenant_lora_sagemaker directory
cd multi_tenant_lora_sagemaker

# Create a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -e .
```

## Step 5: Configure the platform

Copy the example config and edit it for your account:

```bash
cp config/config.example.yaml config/config.yaml
```

Edit `config/config.yaml`:

```yaml
aws:
  region: us-west-2                             # your region
  execution_role_arn: arn:aws:iam::<YOUR-ACCOUNT>:role/MultiTenantLoRAExecutionRole

storage:
  platform_bucket: null                         # leave null to auto-create

base_model:
  hf_model_id: meta-textgeneration-llama-3-2-1b-instruct  # any JumpStart model
  name: llama-3-2-1b-instruct

training:
  instance_type: ml.g5.2xlarge
  # ...

inference:
  instance_type: ml.g5.4xlarge                  # must match an SFT-LoRA recipe
  # ...

tenants:                                        # YOUR tenants
  - id: customer-one
    tasks:
      - name: ticket-classifier
  - id: customer-two
    tasks:
      - name: ticket-classifier
```

## Step 6: Run preflight

This catches misconfigurations before you spend any money:

```bash
multi-tenant-lora preflight
```

It verifies credentials, the execution role, JumpStart access, and flags
missing pieces. Fix any red X's before proceeding.

## Step 7: Prepare your tenant data

For each tenant, create a JSONL file with this format (one record per line):

```jsonl
{"prompt": "Classify the ticket: Printer offline on floor 3", "completion": "hardware"}
{"prompt": "Classify the ticket: Cannot access shared drive", "completion": "network"}
```

Upload each tenant's data:

```bash
multi-tenant-lora upload-data \
    --tenant customer-one \
    --task ticket-classifier \
    --train-file ./my-data/customer-one/train.jsonl \
    --eval-file  ./my-data/customer-one/eval.jsonl
```

## Step 8: Run the full pipeline

Once data is uploaded, a single command trains, deploys, and evaluates
everything:

```bash
multi-tenant-lora run-all --version 1
```

This takes about 30–60 minutes total depending on tenant count and data size.
When it completes, you have:

- All tenant LoRA adapters trained
- Shared inference endpoint running
- Each tenant's adapter registered as an inference component
- Evaluation results (APPROVE/BLOCK decisions)
- CloudWatch alarms wired up

## Step 9: Invoke

Test that inference works:

```bash
multi-tenant-lora invoke \
    --tenant customer-one \
    --task ticket-classifier \
    --version 1 \
    --prompt "Printer is jammed on the 4th floor"
```

## Tear down

When you're done or want to avoid idle GPU charges:

```bash
multi-tenant-lora teardown --confirm
```

This deletes the endpoint, inference components, and endpoint configs.
Training artifacts in S3 are **not** deleted — remove the bucket manually
if you want a full cleanup.

## Routine operations

| Task | Command |
|------|---------|
| Train a new adapter version for one tenant | `multi-tenant-lora train --tenant X --task Y --version 2` |
| Deploy that new version | `multi-tenant-lora deploy-adapter --tenant X --task Y --version 2` |
| Evaluate + gate it | `multi-tenant-lora evaluate --tenant X --task Y --version 2` |
| List deployed adapters | `multi-tenant-lora list-adapters` |
| Full rebuild | `multi-tenant-lora run-all --version N` |

## Troubleshooting

### "The current AWS identity is not a role"
You set `aws.execution_role_arn: null` but aren't running in a SageMaker
Studio / notebook context. Fill in the role ARN explicitly.

### `CannotStartContainerError` on base inference component
The tool uses JumpStart's validated SFT-LoRA recipe hosting configs. If the
model you chose doesn't have a LoRA recipe for your `instance_type`, the
generic config runs and may fail. Check with:
```bash
aws sagemaker describe-hub-content \
    --hub-name SageMakerPublicHub \
    --hub-content-type Model \
    --hub-content-name <model-id> \
    --query 'HubContentDocument' --output text | jq '.RecipeCollection[] |
    select(.Peft == "LORA") | .HostingConfigs[].InstanceType'
```

### `ResourceLimitExceeded` on training or inference
You don't have quota for the instance type in this region. Request one via
Service Quotas.

### Inference component stays `Failed` with no logs
The container died before bootstrap finished, usually due to insufficient
memory. Bump `inference.instance_type` to a larger size.
