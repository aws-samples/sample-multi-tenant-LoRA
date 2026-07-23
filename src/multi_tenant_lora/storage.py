"""S3 storage utilities for the multi-tenant platform.

Handles:
  - Creating the platform bucket (on-demand)
  - Uploading base model weights (shared, read-only)
  - Uploading per-tenant training/eval data
  - Listing adapter artifacts produced by training jobs

All S3 paths follow the convention encoded in config.PlatformConfig — callers
should never construct S3 URIs directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from .config import PlatformConfig

logger = logging.getLogger(__name__)


class StorageManager:
    """Thin wrapper over S3 + STS for platform-managed storage."""

    def __init__(self, config: PlatformConfig, session: boto3.Session | None = None):
        self.config = config
        self._session = session or boto3.Session(region_name=config.aws.region)
        self._s3 = self._session.client("s3")
        self._sts = self._session.client("sts")

    # -------------------------------------------------------------------
    # Bucket lifecycle
    # -------------------------------------------------------------------

    def ensure_bucket(self) -> str:
        """Ensure the platform bucket exists. Returns the bucket name."""
        bucket = self.config.storage.platform_bucket
        if not bucket:
            account_id = self._sts.get_caller_identity()["Account"]
            bucket = f"multi-tenant-lora-platform-{account_id}-{self.config.aws.region}"
            # Persist back so subsequent calls see the same name
            self.config.storage.platform_bucket = bucket

        try:
            self._s3.head_bucket(Bucket=bucket)
            logger.info("Platform bucket already exists: %s", bucket)
            return bucket
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code not in ("404", "NoSuchBucket"):
                raise

        # Create the bucket
        region = self.config.aws.region
        logger.info("Creating platform bucket %s in %s", bucket, region)
        if region == "us-east-1":
            # us-east-1 doesn't accept a LocationConstraint
            self._s3.create_bucket(Bucket=bucket)
        else:
            self._s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )

        # Secure defaults
        self._s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        self._s3.put_bucket_encryption(
            Bucket=bucket,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"},
                        "BucketKeyEnabled": True,
                    }
                ]
            },
        )
        # Versioning on tenant adapters lets customers roll back
        self._s3.put_bucket_versioning(
            Bucket=bucket,
            VersioningConfiguration={"Status": "Enabled"},
        )
        return bucket

    # -------------------------------------------------------------------
    # Data uploads
    # -------------------------------------------------------------------

    def upload_file(self, local_path: str | Path, s3_key: str) -> str:
        """Upload a single file to the platform bucket. Returns s3:// URI."""
        bucket = self.config.require_bucket()
        self._s3.upload_file(str(local_path), bucket, s3_key)
        uri = f"s3://{bucket}/{s3_key}"
        logger.info("Uploaded %s -> %s", local_path, uri)
        return uri

    def upload_training_data(
        self,
        tenant_id: str,
        task: str,
        train_file: str | Path,
        eval_file: str | Path | None = None,
    ) -> dict[str, str]:
        """Upload a tenant's training (and optional eval) JSONL files."""
        prefix = self.config.tenant_prefix(tenant_id)
        result = {}
        train_key = f"{prefix}/training-data/{task}/train.jsonl"
        result["train"] = self.upload_file(train_file, train_key)
        if eval_file is not None:
            eval_key = f"{prefix}/eval-data/{task}/eval.jsonl"
            result["eval"] = self.upload_file(eval_file, eval_key)
        return result

    # -------------------------------------------------------------------
    # Adapter discovery
    # -------------------------------------------------------------------

    def find_adapter_artifact(
        self, tenant_id: str, task: str, version: str
    ) -> str | None:
        """Return the S3 URI of the LoRA adapter for a given (tenant, task, version).

        SDK v3 SFTTrainer writes artifacts to a nested path:
          s3://bucket/tenants/<id>/adapters/<task>/v<n>/<job>/output/model/checkpoints/hf/
        with adapter_config.json and adapter_model.safetensors.

        We return the S3 URI to the directory containing these files, suitable
        for passing to a SageMaker inference component's ArtifactUrl.

        Returns None if no adapter is found — training may still be running.
        """
        bucket = self.config.require_bucket()
        version_prefix = f"{self.config.tenant_prefix(tenant_id)}/adapters/{task}/v{version}/"
        # Look for adapter_model.safetensors anywhere under the version prefix
        paginator = self._s3.get_paginator("list_objects_v2")
        best_key: str | None = None
        for page in paginator.paginate(Bucket=bucket, Prefix=version_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("adapter_model.safetensors"):
                    best_key = key
                    break
            if best_key:
                break
        if not best_key:
            return None
        # Return the DIRECTORY containing the adapter (strip file name)
        parent = best_key.rsplit("/", 1)[0] + "/"
        return f"s3://{bucket}/{parent}"

    def list_tenant_adapters(self, tenant_id: str) -> list[dict]:
        """List all adapter artifacts for a tenant."""
        bucket = self.config.require_bucket()
        prefix = f"{self.config.tenant_prefix(tenant_id)}/adapters/"
        paginator = self._s3.get_paginator("list_objects_v2")
        results: list[dict] = []
        seen_versions = set()
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("adapter_model.safetensors"):
                    continue
                # Parse: tenants/<t>/adapters/<task>/v<ver>/...
                parts = key.split("/")
                if len(parts) < 5 or not parts[4].startswith("v"):
                    continue
                task = parts[3]
                version = parts[4][1:]
                key_id = (task, version)
                if key_id in seen_versions:
                    continue
                seen_versions.add(key_id)
                parent = key.rsplit("/", 1)[0] + "/"
                results.append(
                    {
                        "tenant_id": tenant_id,
                        "task": task,
                        "version": version,
                        "uri": f"s3://{bucket}/{parent}",
                        "size_bytes": obj.get("Size", 0),
                        "last_modified": obj.get("LastModified"),
                    }
                )
        return results
