"""Pydantic schemas for the four YAML configuration files.

These schemas are the single source of truth for what a customer may configure.
The downstream training, inference, and evaluation modules must only read
validated ``AppConfig`` objects — never raw dicts.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Common validators
# ---------------------------------------------------------------------------

_TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_TASK_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
_AWS_REGION_RE = re.compile(r"^[a-z]{2}-[a-z]+-\d$")


def _validate_tenant_id(value: str) -> str:
    if not _TENANT_ID_RE.match(value):
        raise ValueError(
            f"tenant_id {value!r} must match {_TENANT_ID_RE.pattern} "
            "(lowercase, digits, hyphen; max 63 chars)"
        )
    return value


def _validate_task_name(value: str) -> str:
    if not _TASK_NAME_RE.match(value):
        raise ValueError(
            f"task name {value!r} must match {_TASK_NAME_RE.pattern} "
            "(lowercase, digits, hyphen; max 41 chars)"
        )
    return value


# ---------------------------------------------------------------------------
# platform.yaml
# ---------------------------------------------------------------------------


class MLflowConfig(BaseModel):
    """SageMaker managed MLflow tracking server configuration.

    The ``tracking_server_arn`` is passed verbatim to
    ``mlflow.set_tracking_uri`` by the ``sagemaker-mlflow`` plugin.
    """

    model_config = ConfigDict(extra="forbid")

    tracking_server_arn: str = Field(
        ...,
        description="ARN of the SageMaker managed MLflow tracking server.",
    )
    tracking_server_version: str = Field(
        "3.0",
        description=(
            "Tracking server version (2.13, 2.16, or 3.0). The MLflow "
            "client pinned in requirements.txt must match."
        ),
    )


class PlatformConfig(BaseModel):
    """Platform-wide settings shared by all tenants."""

    model_config = ConfigDict(extra="forbid")

    aws_account_id: str = Field(
        ...,
        description="12-digit AWS account ID that owns all resources.",
    )
    region: str = Field(
        ...,
        description="AWS region (e.g. us-east-1, eu-west-1).",
    )
    data_bucket: str = Field(
        ...,
        description=(
            "Name of the S3 bucket holding training data, adapters, and "
            "evaluation datasets for all tenants."
        ),
    )
    base_model_prefix: str = Field(
        "base-model",
        description=(
            "Top-level S3 prefix under ``data_bucket`` where read-only base "
            "model weights are stored. Shared across tenants."
        ),
    )
    endpoint_name: str = Field(
        ...,
        description="Name of the shared multi-adapter SageMaker endpoint.",
    )
    endpoint_variant_name: str = Field(
        "AllTraffic",
        description="Production variant name on the endpoint.",
    )
    base_inference_component_name: str = Field(
        ...,
        description=(
            "Name of the base inference component hosting the shared "
            "foundation model. Adapter components attach to this by name."
        ),
    )
    mlflow: Optional[MLflowConfig] = Field(
        None,
        description=(
            "Optional SageMaker managed MLflow configuration. When omitted "
            "experiment tracking is disabled."
        ),
    )
    cloudwatch_namespace: str = Field(
        "MultiTenantLoRA",
        description="CloudWatch metric namespace for adapter accuracy.",
    )
    regression_alarm_sns_topic_arn: Optional[str] = Field(
        None,
        description=(
            "SNS topic ARN for regression alarm notifications. When omitted "
            "alarms are created without notification actions."
        ),
    )

    @field_validator("aws_account_id")
    @classmethod
    def _validate_account_id(cls, value: str) -> str:
        if not re.fullmatch(r"\d{12}", value):
            raise ValueError("aws_account_id must be a 12-digit string")
        return value

    @field_validator("region")
    @classmethod
    def _validate_region(cls, value: str) -> str:
        if not _AWS_REGION_RE.match(value):
            raise ValueError(f"region {value!r} is not a valid AWS region code")
        return value


# ---------------------------------------------------------------------------
# tenants.yaml
# ---------------------------------------------------------------------------


class TenantTaskConfig(BaseModel):
    """A single task (e.g. ticket-type-classifier) owned by one tenant."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Task identifier, unique within tenant.")
    base_model_id: str = Field(
        ...,
        description=(
            "Reference to an entry in base_models.yaml. Determines the "
            "training container, inference container, and LoRA target "
            "modules."
        ),
    )
    training_data_s3_prefix: str = Field(
        ...,
        description=(
            "S3 prefix (relative to the tenant's training-data/ folder) "
            "containing train.jsonl and optional validation.jsonl."
        ),
    )
    evaluation_data_s3_key: str = Field(
        ...,
        description=(
            "S3 key (relative to the tenant's evaluation-data/ folder) "
            "for the eval JSONL used by the evaluate_adapter step."
        ),
    )
    current_version: int = Field(
        1,
        ge=1,
        description="Latest approved adapter version for this task.",
    )
    regression_threshold: float = Field(
        0.02,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum allowed drop in accuracy compared to the previous "
            "version before the deployment is gated."
        ),
    )
    accuracy_floor: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional absolute accuracy floor for the CloudWatch alarm. "
            "Falls back to the blog default of 0.85 when unset."
        ),
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _validate_task_name(value)


class TenantConfig(BaseModel):
    """Per-tenant registry entry."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(..., description="Unique tenant identifier.")
    training_role_arn: str = Field(
        ...,
        description=(
            "IAM role assumed by SageMaker Training Jobs for this tenant. "
            "Must be scoped to the tenant's S3 prefix."
        ),
    )
    invoke_role_arn: Optional[str] = Field(
        None,
        description=(
            "Optional IAM role used by the tenant's runtime clients to "
            "invoke the shared endpoint. Informational only; used when "
            "generating IAM policy templates."
        ),
    )
    kms_key_arn: Optional[str] = Field(
        None,
        description=(
            "Optional per-tenant KMS CMK ARN for SSE-KMS on training "
            "inputs/outputs and inference traffic."
        ),
    )
    tasks: List[TenantTaskConfig] = Field(
        ...,
        min_length=1,
        description="List of tasks owned by this tenant.",
    )

    @field_validator("tenant_id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_tenant_id(value)

    @model_validator(mode="after")
    def _validate_unique_task_names(self) -> "TenantConfig":
        names = [t.name for t in self.tasks]
        if len(set(names)) != len(names):
            raise ValueError(
                f"tenant {self.tenant_id!r} has duplicate task names: {names}"
            )
        return self


class TenantsConfig(BaseModel):
    """Wrapper around the list of tenants in tenants.yaml."""

    model_config = ConfigDict(extra="forbid")

    tenants: List[TenantConfig] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_unique_tenant_ids(self) -> "TenantsConfig":
        ids = [t.tenant_id for t in self.tenants]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate tenant_id values: {ids}")
        return self

    def get(self, tenant_id: str) -> TenantConfig:
        for t in self.tenants:
            if t.tenant_id == tenant_id:
                return t
        raise KeyError(f"tenant {tenant_id!r} not found in tenants.yaml")


# ---------------------------------------------------------------------------
# base_models.yaml
# ---------------------------------------------------------------------------


class LoRAHyperparameters(BaseModel):
    """Defaults for LoRA hyperparameters per base model.

    These can be overridden at training launch time via CLI flags.
    """

    model_config = ConfigDict(extra="forbid")

    lora_r: int = Field(16, ge=1, le=256)
    lora_alpha: int = Field(32, ge=1, le=512)
    lora_dropout: float = Field(0.05, ge=0.0, le=0.5)
    target_modules: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "List of Transformer module names to adapt, e.g. "
            "['q_proj', 'v_proj', 'k_proj', 'o_proj']. Must match the "
            "base model's architecture."
        ),
    )
    epochs: int = Field(3, ge=1, le=50)
    per_device_train_batch_size: int = Field(4, ge=1, le=256)
    learning_rate: float = Field(2e-4, gt=0.0, le=1.0)
    max_seq_length: int = Field(2048, ge=128, le=32768)


class BaseModelEntry(BaseModel):
    """Catalog entry for a supported base model."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., description="Stable identifier, e.g. llama-3-1-8b-instruct.")
    hf_model_id: str = Field(
        ...,
        description=(
            "HuggingFace model id or S3 URI pointing to base model weights. "
            "Used as HF_MODEL_ID inside the LMI container."
        ),
    )
    training_framework: str = Field(
        "huggingface",
        description=(
            "SageMaker framework name for image_uris.retrieve(). "
            "Currently only 'huggingface' is supported."
        ),
    )
    training_framework_version: str = Field(
        ...,
        description="Transformers version, e.g. '4.46.1'.",
    )
    training_pytorch_version: str = Field(
        ...,
        description="PyTorch version, e.g. '2.3.0'.",
    )
    training_py_version: str = Field(
        "py311",
        description="Python version tag, e.g. 'py310', 'py311', 'py312'.",
    )
    inference_framework: str = Field(
        "djl-lmi",
        description=(
            "Framework for image_uris.retrieve() for inference. "
            "'djl-lmi' is required for multi-adapter inference."
        ),
    )
    inference_framework_version: str = Field(
        ...,
        description="DJL-LMI version, e.g. '0.29.0'.",
    )
    default_hyperparameters: LoRAHyperparameters = Field(
        ...,
        description="Default LoRA hyperparameters for this base model.",
    )
    lmi_env: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Additional environment variables injected into the LMI base "
            "inference component (OPTION_ENABLE_LORA, OPTION_MAX_LORAS, "
            "OPTION_MAX_CPU_LORAS, OPTION_DTYPE, OPTION_MAX_MODEL_LEN, ...)."
        ),
    )


class BaseModelsConfig(BaseModel):
    """Wrapper around the catalog of base models."""

    model_config = ConfigDict(extra="forbid")

    models: List[BaseModelEntry] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_unique_model_ids(self) -> "BaseModelsConfig":
        ids = [m.model_id for m in self.models]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate base model_id values: {ids}")
        return self

    def get(self, model_id: str) -> BaseModelEntry:
        for m in self.models:
            if m.model_id == model_id:
                return m
        raise KeyError(f"base model {model_id!r} not found in base_models.yaml")


# ---------------------------------------------------------------------------
# instances.yaml
# ---------------------------------------------------------------------------


class TrainingInstanceProfile(BaseModel):
    """Per-model-size training instance profile."""

    model_config = ConfigDict(extra="forbid")

    instance_type: str = Field(..., description="e.g. ml.g5.xlarge")
    instance_count: int = Field(1, ge=1, le=64)
    volume_size_in_gb: int = Field(50, ge=10, le=16384)
    max_runtime_in_seconds: int = Field(3600, ge=60, le=432000)


class InstanceProfile(BaseModel):
    """Inference instance profile + LMI adapter capacity hints."""

    model_config = ConfigDict(extra="forbid")

    instance_type: str = Field(..., description="e.g. ml.g5.12xlarge")
    initial_instance_count: int = Field(1, ge=1, le=64)
    max_instance_count: int = Field(
        4,
        ge=1,
        le=64,
        description="Upper bound used when configuring autoscaling.",
    )
    max_hot_adapters: int = Field(
        30,
        ge=1,
        description="OPTION_MAX_LORAS — adapters kept in GPU memory.",
    )
    max_cpu_adapters: int = Field(
        70,
        ge=1,
        description="OPTION_MAX_CPU_LORAS — adapters kept in CPU memory.",
    )


class InstancesConfig(BaseModel):
    """Wrapper around instance profiles.

    ``training[model_id]`` is looked up first by exact base-model id, falling
    back to ``training["default"]``; inference is keyed by profile name.
    """

    model_config = ConfigDict(extra="forbid")

    training: Dict[str, TrainingInstanceProfile] = Field(
        ...,
        description="Mapping of base model_id (or 'default') to training profile.",
    )
    inference: Dict[str, InstanceProfile] = Field(
        ...,
        description="Mapping of logical profile name to inference profile.",
    )
    default_inference_profile: str = Field(
        ...,
        description="Key in ``inference`` to use when no override is given.",
    )

    @model_validator(mode="after")
    def _validate_defaults(self) -> "InstancesConfig":
        if "default" not in self.training:
            raise ValueError("instances.training must define a 'default' entry")
        if self.default_inference_profile not in self.inference:
            raise ValueError(
                f"default_inference_profile {self.default_inference_profile!r} "
                "is not defined in instances.inference"
            )
        return self

    def training_profile_for(self, model_id: str) -> TrainingInstanceProfile:
        return self.training.get(model_id, self.training["default"])

    def inference_profile(self, name: Optional[str] = None) -> InstanceProfile:
        key = name or self.default_inference_profile
        if key not in self.inference:
            raise KeyError(
                f"inference profile {key!r} not found in instances.yaml"
            )
        return self.inference[key]
