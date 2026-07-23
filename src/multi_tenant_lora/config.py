"""Configuration loader for the multi-tenant LoRA platform.

Reads config.yaml and validates it with pydantic. Provides typed access
to every setting the platform needs. Customers edit config.yaml only —
never the Python code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import boto3
import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class AWSConfig(BaseModel):
    region: str = "us-east-1"
    execution_role_arn: str | None = None


class StorageConfig(BaseModel):
    platform_bucket: str | None = None
    base_model_prefix: str = "base-model"
    tenant_data_prefix: str = "tenants"


class BaseModelConfig(BaseModel):
    hf_model_id: str
    hf_token_secret_name: str | None = None
    name: str


class LoRAConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    task_type: str = "CAUSAL_LM"


class TrainingConfig(BaseModel):
    image_uri: str | None = None
    transformers_version: str = "4.49.0"
    pytorch_version: str = "2.5.1"
    python_version: str = "py311"
    instance_type: str = "ml.g5.2xlarge"
    instance_count: int = 1
    volume_size_gb: int = 100
    max_runtime_seconds: int = 7200
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class AutoscalingConfig(BaseModel):
    enabled: bool = True
    target_invocations_per_instance: int = 10
    scale_in_cooldown_seconds: int = 300
    scale_out_cooldown_seconds: int = 60


class InferenceOptionConfig(BaseModel):
    rolling_batch: str = "lmi-dist"
    enable_lora: bool = True
    max_loras: int = 30
    max_cpu_loras: int = 70
    dtype: str = "fp16"
    max_model_len: int = 4096
    max_rolling_batch_size: int = 16
    tensor_parallel_degree: str | int = "max"
    trust_remote_code: bool = True


class InferenceConfig(BaseModel):
    image_uri: str | None = None
    djl_version: str = "0.33.0"
    endpoint_name: str = "multi-tenant-lora-endpoint"
    base_component_name: str = "base-foundation-model"
    instance_type: str = "ml.g5.12xlarge"
    initial_instance_count: int = 1
    min_instance_count: int = 1
    max_instance_count: int = 4
    engine: str = "Python"
    option: InferenceOptionConfig = Field(default_factory=InferenceOptionConfig)
    autoscaling: AutoscalingConfig = Field(default_factory=AutoscalingConfig)


class CloudWatchConfig(BaseModel):
    alarm_namespace: str = "MultiTenantLoRA"
    alarm_threshold: float = 0.85
    sns_topic_arn: str | None = None


class MLflowConfig(BaseModel):
    enabled: bool = False
    tracking_server_arn: str | None = None


class EvaluationConfig(BaseModel):
    accuracy_regression_threshold: float = 0.02
    min_absolute_accuracy: float = 0.70
    eval_sample_size: int = 100
    temperature: float = 0.0
    max_new_tokens: int = 100
    cloudwatch: CloudWatchConfig = Field(default_factory=CloudWatchConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


class TaskConfig(BaseModel):
    name: str
    description: str = ""

    @field_validator("name")
    @classmethod
    def _validate_task_name(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError(
                f"Task name must be non-empty and contain only alphanumerics, '-', '_': {v!r}"
            )
        return v


class TenantConfig(BaseModel):
    id: str
    display_name: str = ""
    lora_overrides: dict[str, Any] = Field(default_factory=dict)
    tasks: list[TaskConfig] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _validate_tenant_id(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError(
                f"Tenant id must be non-empty and contain only alphanumerics, '-', '_': {v!r}"
            )
        return v


class DataResidencyConfig(BaseModel):
    tenant_regions: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class PlatformConfig(BaseModel):
    """Top-level validated config. Loaded from config.yaml."""

    aws: AWSConfig = Field(default_factory=AWSConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    base_model: BaseModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    tenants: list[TenantConfig] = Field(default_factory=list)
    data_residency: DataResidencyConfig = Field(default_factory=DataResidencyConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PlatformConfig":
        """Load config from a YAML file and validate it.

        Args:
            path: Path to config.yaml.

        Returns:
            Validated PlatformConfig instance.

        Raises:
            FileNotFoundError: If path does not exist.
            pydantic.ValidationError: If config is invalid.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    # -------------------------------------------------------------------
    # Convenience accessors — keep these here so they can be mocked in tests
    # and so callers never have to string-concat S3 URIs directly.
    # -------------------------------------------------------------------

    def require_bucket(self) -> str:
        """Return the platform bucket name, raising if unset."""
        if not self.storage.platform_bucket:
            raise ValueError(
                "storage.platform_bucket is not set. "
                "Set it in config.yaml or run `python -m multi_tenant_lora bootstrap` to auto-create one."
            )
        return self.storage.platform_bucket

    def require_execution_role(self) -> str:
        """Resolve the IAM role ARN to use for SageMaker jobs."""
        if self.aws.execution_role_arn:
            return self.aws.execution_role_arn
        # Fall back to SageMaker's default execution-role resolver. Only works
        # when run inside a SageMaker Studio/Notebook context or when a
        # default AmazonSageMaker-ExecutionRole is present in the account.
        try:
            import sagemaker
            return sagemaker.get_execution_role()
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                "aws.execution_role_arn is not set and no default SageMaker role "
                "could be resolved. Set it explicitly in config.yaml."
            ) from e

    def base_model_s3_prefix(self) -> str:
        return f"{self.storage.base_model_prefix}/{self.base_model.name}"

    def base_model_s3_uri(self) -> str:
        return f"s3://{self.require_bucket()}/{self.base_model_s3_prefix()}/"

    def tenant_prefix(self, tenant_id: str) -> str:
        return f"{self.storage.tenant_data_prefix}/{tenant_id}"

    def tenant_training_data_uri(self, tenant_id: str, task: str) -> str:
        return f"s3://{self.require_bucket()}/{self.tenant_prefix(tenant_id)}/training-data/{task}/"

    def tenant_eval_data_uri(self, tenant_id: str, task: str) -> str:
        return f"s3://{self.require_bucket()}/{self.tenant_prefix(tenant_id)}/eval-data/{task}/"

    def tenant_adapter_output_uri(self, tenant_id: str, task: str, version: str) -> str:
        return (
            f"s3://{self.require_bucket()}/"
            f"{self.tenant_prefix(tenant_id)}/adapters/{task}/v{version}/"
        )

    def inference_component_name(self, tenant_id: str, task: str, version: str) -> str:
        """Generate a deterministic, tenant-scoped inference component name.

        This name is used both as the SageMaker resource identifier AND as the
        runtime routing key. The naming convention embeds tenant_id first so
        IAM policies can restrict InvokeEndpoint by prefix.
        """
        return f"{tenant_id}-{task}-v{version}"

    def tenant_region(self, tenant_id: str) -> str:
        """Return the region a tenant's training/inference should run in.

        Falls back to the platform-default region if no override is set.
        """
        return self.data_residency.tenant_regions.get(tenant_id, self.aws.region)

    def lora_for_tenant(self, tenant_id: str) -> LoRAConfig:
        """Merge platform-default LoRA config with any per-tenant overrides."""
        tenant = self.get_tenant(tenant_id)
        if not tenant.lora_overrides:
            return self.training.lora
        merged = self.training.lora.model_dump()
        merged.update(tenant.lora_overrides)
        return LoRAConfig.model_validate(merged)

    def get_tenant(self, tenant_id: str) -> TenantConfig:
        for t in self.tenants:
            if t.id == tenant_id:
                return t
        raise KeyError(f"Unknown tenant: {tenant_id!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_bucket_name(region: str, account_id: str | None = None) -> str:
    """Build a deterministic, account-scoped bucket name when customers want
    the platform to auto-provision storage."""
    if account_id is None:
        account_id = boto3.client("sts").get_caller_identity()["Account"]
    return f"multi-tenant-lora-platform-{account_id}-{region}"
