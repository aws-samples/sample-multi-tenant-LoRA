"""Pure-Python orchestrator for the train -> evaluate -> compare -> deploy loop.

We deliberately avoid Step Functions or SageMaker Pipelines here: the
no-IaC constraint in the task description means we implement the DAG in
Python and submit each step synchronously to SageMaker. For customers who later
want a managed orchestrator, the building blocks (TrainingJobBuilder,
AdapterComponentManager, evaluate_adapter, compare_and_gate, ...) are all
independently invokable so they can be wrapped in Step Functions states or
SageMaker Pipelines steps without code changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from multi_tenant_lora_sagemaker.common.naming import training_data_s3_uri
from multi_tenant_lora_sagemaker.config.loader import AppConfig
from multi_tenant_lora_sagemaker.evaluation.compare_and_gate import (
    CompareResult,
    compare_and_gate,
)
from multi_tenant_lora_sagemaker.evaluation.create_regression_alarms import (
    ensure_regression_alarm,
)
from multi_tenant_lora_sagemaker.evaluation.evaluate_adapter import (
    EvaluationResult,
    evaluate_adapter,
)
from multi_tenant_lora_sagemaker.evaluation.publish_metrics import publish_accuracy
from multi_tenant_lora_sagemaker.evaluation.register_experiment import (
    register_experiment,
)
from multi_tenant_lora_sagemaker.inference.deploy_adapter import (
    AdapterComponentManager,
)
from multi_tenant_lora_sagemaker.training.code_uploader import upload_entry_point
from multi_tenant_lora_sagemaker.training.data_validator import (
    validate_training_data,
)
from multi_tenant_lora_sagemaker.training.training_job_builder import TrainingJobBuilder

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    tenant_id: str
    task: str
    version: int
    training_job_name: str
    evaluation: Optional[EvaluationResult] = None
    gate: Optional[CompareResult] = None
    adapter_component_name: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    alarm_arn: Optional[str] = None
    steps_executed: list = field(default_factory=list)
    steps_skipped: list = field(default_factory=list)


def _training_hyperparameter_overrides(
    overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return dict(overrides) if overrides else {}


def run_pipeline(
    app_config: AppConfig,
    tenant_id: str,
    task: str,
    version: int,
    *,
    skip_data_validation: bool = False,
    skip_training: bool = False,
    skip_evaluation: bool = False,
    skip_deploy_on_block: bool = True,
    hyperparameter_overrides: Optional[Dict[str, Any]] = None,
    instance_type_override: Optional[str] = None,
    profile: Optional[str] = None,
) -> PipelineResult:
    """Run train -> evaluate -> compare -> deploy for one tenant/task/version.

    Each step can be skipped via the ``skip_*`` flags for scenarios like
    re-evaluating an existing adapter or just (re-)deploying a previously
    approved version.
    """

    platform = app_config.platform
    from multi_tenant_lora_sagemaker.common.naming import training_job_name

    result = PipelineResult(
        tenant_id=tenant_id,
        task=task,
        version=version,
        training_job_name=training_job_name(tenant_id, task, version),
    )

    # ---------- 1. Train ----------
    if not skip_training:
        if not skip_data_validation:
            logger.info("Step 1a/5: validating training data")
            validate_training_data(
                training_data_prefix=training_data_s3_uri(platform, tenant_id, task),
                region=platform.region,
                profile=profile,
            )
            result.steps_executed.append("validate_data")
        else:
            result.steps_skipped.append("validate_data")

        logger.info("Step 1b/5: uploading training entry point")
        code_s3_uri = upload_entry_point(
            platform=platform,
            tenant_id=tenant_id,
            task=task,
            version=version,
            region=platform.region,
            profile=profile,
        )

        logger.info("Step 1c/5: submitting training job")
        builder = TrainingJobBuilder(app_config)
        spec = builder.build_spec(
            tenant_id=tenant_id,
            task_name=task,
            version=version,
            code_s3_uri=code_s3_uri,
            hyperparameter_overrides=_training_hyperparameter_overrides(
                hyperparameter_overrides
            ),
            instance_type_override=instance_type_override,
        )
        job = builder.submit(spec, wait=True, session_profile=profile)
        logger.info(
            "Training job %s reached status %s",
            spec.job_name,
            job.training_job_status,
        )
        result.steps_executed.append("train")
    else:
        result.steps_skipped.append("train")

    # At this point the adapter tarball should be at
    # s3://{bucket}/{tenant}/adapters/{task}/v{version}/adapter.tar.gz.

    # ---------- 2. Deploy adapter component (temporary — needed for evaluation) ----------
    logger.info("Step 2/5: deploying adapter inference component")
    adapter_mgr = AdapterComponentManager(app_config, profile=profile)
    adapter_plan = adapter_mgr.plan(tenant_id=tenant_id, task=task, version=version)
    adapter_mgr.deploy(adapter_plan, wait=True)
    result.adapter_component_name = adapter_plan.component_name
    result.steps_executed.append("deploy_adapter")

    # ---------- 3. Evaluate ----------
    if skip_evaluation:
        result.steps_skipped.append("evaluate")
        return result

    logger.info("Step 3/5: evaluating adapter")
    evaluation = evaluate_adapter(
        app_config=app_config,
        tenant_id=tenant_id,
        task=task,
        version=version,
        profile=profile,
    )
    result.evaluation = evaluation
    result.steps_executed.append("evaluate")

    # ---------- 4. Compare, publish, MLflow log, alarm ----------
    logger.info("Step 4/5: comparing against previous baseline")
    gate = compare_and_gate(
        app_config=app_config,
        tenant_id=tenant_id,
        task=task,
        new_accuracy=evaluation.accuracy,
        profile=profile,
    )
    result.gate = gate

    publish_accuracy(
        platform=platform,
        tenant_id=tenant_id,
        task=task,
        version=version,
        accuracy=evaluation.accuracy,
        profile=profile,
    )
    result.steps_executed.append("publish_metrics")

    result.alarm_arn = ensure_regression_alarm(
        app_config=app_config,
        tenant_id=tenant_id,
        task=task,
        profile=profile,
    )
    result.steps_executed.append("ensure_alarm")

    hp = (hyperparameter_overrides or {}).copy()
    hp.setdefault("base_model_id", app_config.tenants.get(tenant_id).tasks[0].base_model_id)
    result.mlflow_run_id = register_experiment(
        app_config=app_config,
        tenant_id=tenant_id,
        task=task,
        version=version,
        accuracy=evaluation.accuracy,
        hyperparameters=hp,
    )
    if result.mlflow_run_id:
        result.steps_executed.append("mlflow_log")

    # ---------- 5. Gate decision ----------
    if gate.decision == "BLOCK" and skip_deploy_on_block:
        logger.warning(
            "Gate BLOCKED; leaving adapter component deployed for debugging but "
            "refusing to promote version %d. Reason: %s",
            version,
            gate.reason,
        )
        return result

    logger.info(
        "Step 5/5: gate=%s; pipeline complete. Promote version %d by updating "
        "tenants.yaml (current_version: %d).",
        gate.decision,
        version,
        version,
    )
    return result
