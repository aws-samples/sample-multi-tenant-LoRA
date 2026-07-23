"""Per-tenant LoRA training using SageMaker SDK v3 SFTTrainer.

SDK v3 introduced native LoRA support via SFTTrainer(training_type=LORA).
It handles:
  - Container selection (the SageMaker-managed SFT container)
  - LoRA adapter config (via typed FineTuningOptions object)
  - Model packaging (fine-tuned model is published to a ModelPackageGroup)

Each tenant gets its own ModelPackageGroup so adapter versioning is
naturally isolated per tenant.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.exceptions import ClientError
from sagemaker.core.shapes.shapes import StoppingCondition
from sagemaker.train.common import TrainingType
from sagemaker.train.sft_trainer import SFTTrainer

from .config import PlatformConfig, TenantConfig

logger = logging.getLogger(__name__)


# Mapping from our config-friendly hyperparameter names to the names
# expected by the SageMaker-managed SFT container's FineTuningOptions.
# We use our names in config.yaml so customers have a clean API; we translate
# here when submitting to the SDK.
HYPERPARAMETER_NAME_MAP = {
    "lora_r": "lora_rank",
    "lora_alpha": "lora_alpha",
    "lora_dropout": "lora_dropout",
    "epochs": "max_epochs",
    "per_device_train_batch_size": "global_batch_size",
    "learning_rate": "learning_rate",
    "lr_scheduler_type": "lr_scheduler",
    "warmup_ratio": "lr_warmup_steps_ratio",
    "max_seq_length": "dataset_max_len",
    "logging_steps": "logging_steps",
    "weight_decay": "weight_decay",
    "seed": "seed",
    "train_val_split_ratio": "train_val_split_ratio",
}


@dataclass
class TrainingJob:
    """Handle for a submitted training job."""

    tenant_id: str
    task: str
    version: str
    model_package_group: str
    training_dataset: str
    validation_dataset: str | None
    sagemaker_job: Any = None

    @property
    def job_name(self) -> str:
        if self.sagemaker_job is not None:
            return getattr(self.sagemaker_job, "training_job_name", "")
        return ""

    @property
    def status(self) -> str:
        if self.sagemaker_job is None:
            return "NotSubmitted"
        try:
            self.sagemaker_job.refresh()
            return getattr(self.sagemaker_job, "training_job_status", "Unknown")
        except Exception:  # noqa: BLE001
            return "Unknown"

    @property
    def output_model_package_arn(self) -> str | None:
        if self.sagemaker_job is None:
            return None
        return getattr(self.sagemaker_job, "output_model_package_arn", None)


class TrainingPipeline:
    """Submit and monitor per-tenant LoRA training jobs via SDK v3."""

    def __init__(
        self,
        config: PlatformConfig,
        session: boto3.Session | None = None,
    ):
        self.config = config
        self._session = session or boto3.Session(region_name=config.aws.region)
        self._sm = self._session.client("sagemaker")

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def submit(
        self,
        tenant_id: str,
        task: str,
        version: str,
        *,
        wait: bool = False,
        hyperparameter_overrides: dict[str, Any] | None = None,
    ) -> TrainingJob:
        """Submit a LoRA fine-tuning job for one (tenant, task, version)."""
        tenant = self.config.get_tenant(tenant_id)
        self._validate_task(tenant, task)

        training_dataset = self.config.tenant_training_data_uri(tenant_id, task)
        eval_uri = self.config.tenant_eval_data_uri(tenant_id, task)
        validation_dataset = eval_uri if self._prefix_exists(eval_uri) else None

        model_package_group = self._ensure_model_package_group(tenant_id, task)
        output_path = self.config.tenant_adapter_output_uri(tenant_id, task, version)

        stopping_condition = StoppingCondition(
            max_runtime_in_seconds=self.config.training.max_runtime_seconds
        )

        logger.info(
            "Submitting LoRA training: tenant=%s task=%s v%s base=%s",
            tenant_id, task, version, self.config.base_model.hf_model_id,
        )

        trainer = SFTTrainer(
            model=self.config.base_model.hf_model_id,
            training_type=TrainingType.LORA,
            model_package_group=model_package_group,
            s3_output_path=output_path,
            accept_eula=True,
            stopping_condition=stopping_condition,
            role=self.config.require_execution_role(),
        )

        # Apply hyperparameters to the SDK's typed FineTuningOptions object
        self._apply_hyperparameters(
            trainer=trainer,
            tenant_id=tenant_id,
            overrides=hyperparameter_overrides or {},
        )

        # Tags
        trainer.tags = [
            {"Key": "TenantId", "Value": tenant_id},
            {"Key": "Task", "Value": task},
            {"Key": "Version", "Value": version},
            {"Key": "Platform", "Value": "multi-tenant-lora"},
        ]

        # Use a deterministic base job name — SageMaker appends a timestamp
        trainer.base_job_name = self._make_job_name(tenant_id, task, version)

        # Submit
        sagemaker_job = trainer.train(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            wait=wait,
        )

        return TrainingJob(
            tenant_id=tenant_id,
            task=task,
            version=version,
            model_package_group=model_package_group,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            sagemaker_job=sagemaker_job,
        )

    def submit_all_tenants(
        self,
        version: str,
        *,
        wait: bool = False,
    ) -> list[TrainingJob]:
        """Submit a job for every (tenant, task) pair in config."""
        jobs: list[TrainingJob] = []
        for tenant in self.config.tenants:
            for task in tenant.tasks:
                jobs.append(
                    self.submit(tenant.id, task.name, version, wait=False)
                )
        if wait:
            self.wait_for_all(jobs)
        return jobs

    def wait_for_all(self, jobs: list[TrainingJob], poll_seconds: int = 30) -> None:
        """Block until all submitted jobs reach a terminal state."""
        remaining = list(jobs)
        while remaining:
            still_running = []
            for job in remaining:
                status = job.status
                if status in {"Completed", "Failed", "Stopped"}:
                    logger.info(
                        "tenant=%s task=%s v%s -> %s",
                        job.tenant_id, job.task, job.version, status,
                    )
                else:
                    still_running.append(job)
            remaining = still_running
            if remaining:
                # nosemgrep: arbitrary-sleep - polling interval for SageMaker training job status
                time.sleep(poll_seconds)

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    @staticmethod
    def _validate_task(tenant: TenantConfig, task: str) -> None:
        if not any(t.name == task for t in tenant.tasks):
            raise ValueError(
                f"Task {task!r} is not declared for tenant {tenant.id!r}. "
                f"Declared tasks: {[t.name for t in tenant.tasks]}"
            )

    @staticmethod
    def _make_job_name(tenant_id: str, task: str, version: str) -> str:
        ts = time.strftime("%Y%m%d-%H%M%S")
        raw = f"mtl-{tenant_id}-{task}-v{version}-{ts}"
        sanitized = "".join(c if (c.isalnum() or c == "-") else "-" for c in raw)
        return sanitized[:63]

    def _ensure_model_package_group(self, tenant_id: str, task: str) -> str:
        group_name = f"mtl-{tenant_id}-{task}"[:63]
        try:
            self._sm.describe_model_package_group(ModelPackageGroupName=group_name)
            return group_name
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise
        self._sm.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription=(
                f"LoRA adapters for tenant={tenant_id} task={task}"
            ),
            Tags=[
                {"Key": "TenantId", "Value": tenant_id},
                {"Key": "Task", "Value": task},
                {"Key": "Platform", "Value": "multi-tenant-lora"},
            ],
        )
        logger.info("Created ModelPackageGroup: %s", group_name)
        return group_name

    def _apply_hyperparameters(
        self,
        trainer: SFTTrainer,
        tenant_id: str,
        overrides: dict[str, Any],
    ) -> None:
        """Apply LoRA + training hyperparameters to the trainer's typed options.

        SFTTrainer auto-populates trainer.hyperparameters with a typed
        FineTuningOptions object based on the model. We set individual
        attributes on it, validating types per the SDK's specs.
        """
        lora = self.config.lora_for_tenant(tenant_id)

        # Build the merged hyperparameter dict using our canonical names,
        # then translate to SDK names below
        raw: dict[str, Any] = {
            "lora_r": lora.r,
            "lora_alpha": lora.alpha,
            "lora_dropout": lora.dropout,
        }
        raw.update(self.config.training.hyperparameters)
        raw.update(overrides)

        opts = trainer.hyperparameters
        for our_name, value in raw.items():
            sdk_name = HYPERPARAMETER_NAME_MAP.get(our_name, our_name)
            if not hasattr(opts, sdk_name):
                logger.warning(
                    "Skipping unknown hyperparameter %r (mapped to %r) — not in SDK schema",
                    our_name, sdk_name,
                )
                continue
            try:
                # Respect the SDK's type validation. Use int for ints, float for floats.
                setattr(opts, sdk_name, self._coerce_hyperparam_type(opts, sdk_name, value))
            except (ValueError, AttributeError) as e:
                logger.warning("Skipping hyperparam %s=%r: %s", sdk_name, value, e)

    @staticmethod
    def _coerce_hyperparam_type(opts: Any, name: str, value: Any) -> Any:
        """Coerce value to the type expected by FineTuningOptions._specs[name]."""
        specs = getattr(opts, "_specs", {})
        spec = specs.get(name, {})
        if isinstance(spec, dict):
            expected = spec.get("type", "string")
            if expected == "integer":
                return int(value)
            if expected == "float":
                return float(value)
            if expected == "boolean" or expected == "bool":
                return bool(value) if not isinstance(value, str) else value.lower() == "true"
            return str(value)
        return value

    def _prefix_exists(self, s3_uri: str) -> bool:
        assert s3_uri.startswith("s3://")
        _, _, rest = s3_uri.partition("s3://")
        bucket, _, key = rest.partition("/")
        s3 = self._session.client("s3")
        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
        except ClientError:
            return False
        return "Contents" in resp
