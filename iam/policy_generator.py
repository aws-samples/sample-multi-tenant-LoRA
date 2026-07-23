"""Generate per-tenant IAM policy JSON documents.

Two policies are produced per tenant:

* **training-role** — attached to the IAM role that SageMaker Training Jobs
  assume. Grants read access to the tenant's ``training-data/`` prefix and the
  shared ``base-model/`` prefix, and write access to the tenant's
  ``adapters/`` and ``training-output/`` prefixes.
* **invoke-role** — attached to the IAM role the tenant's runtime clients use
  to call ``invoke_endpoint``. Constrained to the tenant's adapter inference
  components by name via the ``sagemaker:InferenceComponentName`` condition
  key.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from multi_tenant_lora_sagemaker.common.naming import adapter_component_name
from multi_tenant_lora_sagemaker.config.loader import AppConfig
from multi_tenant_lora_sagemaker.config.schemas import PlatformConfig, TenantConfig

logger = logging.getLogger(__name__)


def _training_s3_resources(
    platform: PlatformConfig, tenant: TenantConfig
) -> Dict[str, List[str]]:
    bucket = platform.data_bucket
    tenant_id = tenant.tenant_id
    read_resources = [
        f"arn:aws:s3:::{bucket}/{tenant_id}/training-data/*",
        f"arn:aws:s3:::{bucket}/{platform.base_model_prefix}/*",
    ]
    list_resources = [f"arn:aws:s3:::{bucket}"]
    write_resources = [
        f"arn:aws:s3:::{bucket}/{tenant_id}/adapters/*",
        f"arn:aws:s3:::{bucket}/{tenant_id}/training-output/*",
        f"arn:aws:s3:::{bucket}/{tenant_id}/code/*",
    ]
    return {"read": read_resources, "list": list_resources, "write": write_resources}


def generate_training_role_policy(
    platform: PlatformConfig, tenant: TenantConfig
) -> Dict[str, Any]:
    """Return the JSON policy document for a tenant's training role."""

    resources = _training_s3_resources(platform, tenant)
    statements: List[Dict[str, Any]] = [
        {
            "Sid": "ReadTrainingAndBaseModel",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:GetObjectVersion"],
            "Resource": resources["read"],
        },
        {
            "Sid": "ListBucketScopedToTenant",
            "Effect": "Allow",
            "Action": ["s3:ListBucket"],
            "Resource": resources["list"],
            "Condition": {
                "StringLike": {
                    "s3:prefix": [
                        f"{tenant.tenant_id}/training-data/*",
                        f"{tenant.tenant_id}/training-output/*",
                        f"{tenant.tenant_id}/adapters/*",
                        f"{tenant.tenant_id}/code/*",
                        f"{platform.base_model_prefix}/*",
                    ]
                }
            },
        },
        {
            "Sid": "WriteAdaptersAndOutputs",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:PutObjectAcl",
                "s3:AbortMultipartUpload",
            ],
            "Resource": resources["write"],
        },
        {
            "Sid": "CloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams",
            ],
            "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*",
        },
        {
            "Sid": "ECRPull",
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
            ],
            "Resource": "*",
        },
    ]

    if tenant.kms_key_arn:
        statements.append(
            {
                "Sid": "KmsForTenantKey",
                "Effect": "Allow",
                "Action": [
                    "kms:Encrypt",
                    "kms:Decrypt",
                    "kms:ReEncrypt*",
                    "kms:GenerateDataKey*",
                    "kms:DescribeKey",
                ],
                "Resource": tenant.kms_key_arn,
            }
        )

    return {"Version": "2012-10-17", "Statement": statements}


def generate_adapter_invoke_policy(
    platform: PlatformConfig, tenant: TenantConfig
) -> Dict[str, Any]:
    """Return the JSON policy document for a tenant's runtime invocation role.

    The ``sagemaker:InferenceComponentName`` condition key restricts the role
    to invoking inference components that belong to this tenant. Component
    names follow the ``{tenant}-{task}-v{version}`` convention.
    """
    region = platform.region
    account = platform.aws_account_id
    endpoint_arn = (
        f"arn:aws:sagemaker:{region}:{account}:endpoint/{platform.endpoint_name}"
    )
    allowed_component_names = sorted(
        {
            adapter_component_name(tenant.tenant_id, task.name, task.current_version)
            for task in tenant.tasks
        }
        | {f"{tenant.tenant_id}-{task.name}-v*" for task in tenant.tasks}
    )

    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeOwnAdapterComponents",
                "Effect": "Allow",
                "Action": [
                    "sagemaker:InvokeEndpoint",
                    "sagemaker:InvokeEndpointWithResponseStream",
                ],
                "Resource": endpoint_arn,
                "Condition": {
                    "StringLike": {
                        "sagemaker:InferenceComponentName": allowed_component_names
                    }
                },
            }
        ],
    }


def write_policy_templates(
    app_config: AppConfig, output_dir: Path
) -> List[Path]:
    """Write all per-tenant policy documents to ``output_dir``.

    Returns the list of files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for tenant in app_config.tenants.tenants:
        train_doc = generate_training_role_policy(app_config.platform, tenant)
        invoke_doc = generate_adapter_invoke_policy(app_config.platform, tenant)
        train_path = output_dir / f"{tenant.tenant_id}-training-policy.json"
        invoke_path = output_dir / f"{tenant.tenant_id}-invoke-policy.json"
        train_path.write_text(json.dumps(train_doc, indent=2))
        invoke_path.write_text(json.dumps(invoke_doc, indent=2))
        written.extend([train_path, invoke_path])
        logger.info(
            "Wrote IAM policy templates for tenant=%s: %s, %s",
            tenant.tenant_id,
            train_path,
            invoke_path,
        )
    return written
