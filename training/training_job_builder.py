"""Build a SageMaker training job spec from validated config.

Uses ``sagemaker-core`` (SDK v3) classes for strong typing. The entry point
script in ``training/entry_point/train.py`` performs the actual LoRA fine-tune
inside the HuggingFace PyTorch training DLC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from multi_tenant_lora_sagemaker.common.naming import (
    adapter_s3_prefix,
    base_model_s3_uri,
    training_data_s3_uri,
    training_job_name,
    training_output_s3_uri,
)
from multi_tenant_lora_sagemaker.config.loader import AppConfig
from multi_tenant_lora_sagemaker.config.schemas import (
    BaseModelEntry,
    LoRAHyperparameters,
    TenantConfig,
    TenantTaskConfig,
    TrainingInstanceProfile,
)

logger = logging.getLogger(__name__)


_ENTRY_POINT_DIR = Path(__file__).parent / "entry_point"
_ENTRY_POINT_SCRIPT = "train.py"


@dataclass(frozen=True)
class TrainingJobSpec:
    """Typed description of a training job ready to submit."""

    job_name: str
    tenant_id: str
    task: str
    version: int
    role_arn: str
    training_image: str
    input_data_config: List[Any]  # sagemaker_core.shapes.Channel list
    output_data_config: Any  # sagemaker_core.shapes.OutputDataConfig
    resource_config: Any  # sagemaker_core.shapes.ResourceConfig
    stopping_condition: Any  # sagemaker_core.shapes.StoppingCondition
    hyper_parameters: Dict[str, str]
    environment: Dict[str, str]
    algorithm_specification: Any  # sagemaker_core.shapes.AlgorithmSpecification
    tags: List[Dict[str, str]]


def _retrieve_huggingface_training_image(
    base_model: BaseModelEntry, region: str
) -> str:
    """Look up the current HuggingFace training DLC via sagemaker (v2).

    ``sagemaker-core`` does not yet expose ``image_uris.retrieve``, so we
    delegate to the v2 SDK which is installed alongside per requirements.txt.
    This is the documented substitution pattern.
    """
    from sagemaker import image_uris

    uri = image_uris.retrieve(
        framework=base_model.training_framework,
        region=region,
        version=base_model.training_framework_version,
        base_framework_version=f"pytorch{base_model.training_pytorch_version}",
        py_version=base_model.training_py_version,
        instance_type="ml.g5.xlarge",  # any GPU type picks the GPU DLC
        image_scope="training",
    )
    logger.info("Resolved training DLC: %s", uri)
    return uri


def _resolve_lora_hyperparameters(
    base_model: BaseModelEntry, overrides: Optional[Dict[str, Any]] = None
) -> LoRAHyperparameters:
    """Merge user-supplied hyperparameter overrides into the base model's defaults."""
    if not overrides:
        return base_model.default_hyperparameters
    data = base_model.default_hyperparameters.model_dump()
    data.update(overrides)
    return LoRAHyperparameters.model_validate(data)


def _stringify_hyperparameters(
    base_model: BaseModelEntry, lora_hp: LoRAHyperparameters
) -> Dict[str, str]:
    """SageMaker hyperparameters are stringly-typed on the wire."""
    return {
        "hf_model_id": base_model.hf_model_id,
        "peft_type": "LORA",
        "lora_r": str(lora_hp.lora_r),
        "lora_alpha": str(lora_hp.lora_alpha),
        "lora_dropout": str(lora_hp.lora_dropout),
        "target_modules": ",".join(lora_hp.target_modules),
        "epochs": str(lora_hp.epochs),
        "per_device_train_batch_size": str(lora_hp.per_device_train_batch_size),
        "learning_rate": str(lora_hp.learning_rate),
        "max_seq_length": str(lora_hp.max_seq_length),
        # sagemaker_sdk_script_mode convention:
        "sagemaker_program": _ENTRY_POINT_SCRIPT,
        "sagemaker_submit_directory": "/opt/ml/input/data/code/",
    }


def entry_point_dir() -> Path:
    """Return the local path to the entry point directory.

    Customers upload this directory (``train.py`` + ``requirements.txt``) to
    S3 and pass the resulting URI as the ``code`` input channel. The builder
    does that automatically; this helper is exposed for tests.
    """
    if not _ENTRY_POINT_DIR.is_dir():
        raise RuntimeError(
            f"training entry-point directory not found at {_ENTRY_POINT_DIR}"
        )
    return _ENTRY_POINT_DIR


class TrainingJobBuilder:
    """Builds :class:`TrainingJobSpec` objects and submits them via sagemaker-core."""

    def __init__(self, app_config: AppConfig):
        self._config = app_config

    # ------------------------------------------------------------------
    # Spec construction
    # ------------------------------------------------------------------

    def build_spec(
        self,
        tenant_id: str,
        task_name: str,
        version: int,
        code_s3_uri: str,
        hyperparameter_overrides: Optional[Dict[str, Any]] = None,
        instance_type_override: Optional[str] = None,
    ) -> TrainingJobSpec:
        """Build a spec for training one adapter version.

        Parameters
        ----------
        code_s3_uri:
            S3 URI of the uploaded entry-point tarball (``sourcedir.tar.gz``
            or the ``entry_point/`` folder prefix). Passed as the ``code``
            channel to the training job.
        hyperparameter_overrides:
            Optional partial LoRAHyperparameters dict that overrides the base
            model defaults.
        instance_type_override:
            Optional override of the training instance type selected by
            ``instances.yaml``.
        """

        from sagemaker_core import shapes

        tenant: TenantConfig = self._config.tenants.get(tenant_id)
        task: TenantTaskConfig = next(
            (t for t in tenant.tasks if t.name == task_name), None
        )
        if task is None:
            raise KeyError(
                f"tenant {tenant_id!r} has no task named {task_name!r}"
            )

        base_model: BaseModelEntry = self._config.base_models.get(task.base_model_id)
        lora_hp = _resolve_lora_hyperparameters(
            base_model, hyperparameter_overrides
        )

        training_profile: TrainingInstanceProfile = (
            self._config.instances.training_profile_for(base_model.model_id)
        )

        training_image = _retrieve_huggingface_training_image(
            base_model, self._config.platform.region
        )

        job_name = training_job_name(tenant_id, task_name, version)
        train_uri = training_data_s3_uri(
            self._config.platform, tenant_id, task_name
        )
        output_uri = training_output_s3_uri(
            self._config.platform, tenant_id, task_name, version
        )
        adapter_upload_uri = adapter_s3_prefix(
            self._config.platform, tenant_id, task_name, version
        )
        base_model_uri = (
            base_model.hf_model_id
            if base_model.hf_model_id.startswith("s3://")
            else base_model_s3_uri(self._config.platform, base_model.hf_model_id)
        )

        input_data_config = [
            shapes.Channel(
                channel_name="training",
                content_type="application/jsonlines",
                input_mode="File",
                data_source=shapes.DataSource(
                    s3_data_source=shapes.S3DataSource(
                        s3_data_type="S3Prefix",
                        s3_uri=train_uri,
                        s3_data_distribution_type="FullyReplicated",
                    )
                ),
            ),
            shapes.Channel(
                channel_name="code",
                content_type="application/x-tar",
                input_mode="File",
                data_source=shapes.DataSource(
                    s3_data_source=shapes.S3DataSource(
                        s3_data_type="S3Prefix",
                        s3_uri=code_s3_uri,
                        s3_data_distribution_type="FullyReplicated",
                    )
                ),
            ),
        ]

        # Attach an optional base-model channel when weights live in our S3
        # bucket. When hf_model_id is a HuggingFace hub id, the entry-point
        # trainer will download it from the hub directly.
        if base_model_uri.startswith("s3://"):
            input_data_config.append(
                shapes.Channel(
                    channel_name="model",
                    content_type="application/x-tar",
                    input_mode="File",
                    data_source=shapes.DataSource(
                        s3_data_source=shapes.S3DataSource(
                            s3_data_type="S3Prefix",
                            s3_uri=base_model_uri,
                            s3_data_distribution_type="FullyReplicated",
                        )
                    ),
                )
            )

        output_data_config = shapes.OutputDataConfig(
            s3_output_path=output_uri,
            kms_key_id=tenant.kms_key_arn or "",
        )

        instance_type = instance_type_override or training_profile.instance_type
        resource_config = shapes.ResourceConfig(
            instance_type=instance_type,
            instance_count=training_profile.instance_count,
            volume_size_in_gb=training_profile.volume_size_in_gb,
            volume_kms_key_id=tenant.kms_key_arn or "",
        )
        stopping_condition = shapes.StoppingCondition(
            max_runtime_in_seconds=training_profile.max_runtime_in_seconds
        )

        algorithm_specification = shapes.AlgorithmSpecification(
            training_image=training_image,
            training_input_mode="File",
        )

        hyper_parameters = _stringify_hyperparameters(base_model, lora_hp)
        # The entry-point script uses this to upload adapter.tar.gz next to
        # the training outputs, which the inference component consumes.
        hyper_parameters["adapter_output_s3_uri"] = adapter_upload_uri

        environment = {
            "TENANT_ID": tenant_id,
            "TASK_NAME": task_name,
            "ADAPTER_VERSION": str(version),
        }

        tags = [
            {"Key": "TenantId", "Value": tenant_id},
            {"Key": "Task", "Value": task_name},
            {"Key": "AdapterVersion", "Value": str(version)},
            {"Key": "ManagedBy", "Value": "multi-tenant-lora-sagemaker"},
        ]

        return TrainingJobSpec(
            job_name=job_name,
            tenant_id=tenant_id,
            task=task_name,
            version=version,
            role_arn=tenant.training_role_arn,
            training_image=training_image,
            input_data_config=input_data_config,
            output_data_config=output_data_config,
            resource_config=resource_config,
            stopping_condition=stopping_condition,
            hyper_parameters=hyper_parameters,
            environment=environment,
            algorithm_specification=algorithm_specification,
            tags=tags,
        )

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(
        self,
        spec: TrainingJobSpec,
        wait: bool = False,
        session_profile: Optional[str] = None,
    ):
        """Create the SageMaker Training Job via sagemaker-core.

        Returns the ``sagemaker_core.resources.TrainingJob`` resource handle.
        """
        from sagemaker_core import shapes
        from sagemaker_core.resources import TrainingJob

        from multi_tenant_lora_sagemaker.common.sessions import build_boto_session

        boto_session = build_boto_session(
            region=self._config.platform.region, profile=session_profile
        )

        logger.info(
            "Submitting training job %s for tenant=%s task=%s v=%d",
            spec.job_name,
            spec.tenant_id,
            spec.task,
            spec.version,
        )
        job = TrainingJob.create(
            training_job_name=spec.job_name,
            algorithm_specification=spec.algorithm_specification,
            role_arn=spec.role_arn,
            input_data_config=spec.input_data_config,
            output_data_config=spec.output_data_config,
            resource_config=spec.resource_config,
            stopping_condition=spec.stopping_condition,
            hyper_parameters=spec.hyper_parameters,
            environment=spec.environment,
            tags=[shapes.Tag(key=t["Key"], value=t["Value"]) for t in spec.tags],
            session=boto_session,
            region=self._config.platform.region,
        )
        if wait:
            job.wait(logs=True)
        return job
