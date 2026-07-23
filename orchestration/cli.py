"""Top-level command group: ``python -m multi_tenant_lora_sagemaker.orchestration.cli ...``."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Optional

import click

from multi_tenant_lora_sagemaker.common.logging_setup import configure_logging
from multi_tenant_lora_sagemaker.config.loader import load_config
from multi_tenant_lora_sagemaker.evaluation.compare_and_gate import compare_and_gate
from multi_tenant_lora_sagemaker.evaluation.create_regression_alarms import (
    ensure_regression_alarm,
)
from multi_tenant_lora_sagemaker.evaluation.evaluate_adapter import evaluate_adapter
from multi_tenant_lora_sagemaker.evaluation.publish_metrics import publish_accuracy
from multi_tenant_lora_sagemaker.inference.deploy_adapter import (
    AdapterComponentManager,
)
from multi_tenant_lora_sagemaker.inference.launch_endpoint import BaseEndpointManager
from multi_tenant_lora_sagemaker.inference.invoke import MultiTenantClient
from multi_tenant_lora_sagemaker.inference.autoscaling import apply_endpoint_autoscaling
from multi_tenant_lora_sagemaker.orchestration.pipeline import run_pipeline
from multi_tenant_lora_sagemaker.training.code_uploader import upload_entry_point
from multi_tenant_lora_sagemaker.training.data_validator import validate_training_data
from multi_tenant_lora_sagemaker.training.training_job_builder import TrainingJobBuilder
from multi_tenant_lora_sagemaker.common.naming import training_data_s3_uri

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------


def _config_option(f):
    return click.option(
        "--config-dir",
        required=True,
        type=click.Path(
            exists=True, file_okay=False, dir_okay=True, resolve_path=True
        ),
        help="Directory with the four YAML configuration files.",
    )(f)


def _profile_option(f):
    return click.option(
        "--profile", default=None, help="AWS named profile to use."
    )(f)


# ---------------------------------------------------------------------------
# Command group
# ---------------------------------------------------------------------------


@click.group(help="Multi-tenant LoRA fine-tuning and inference orchestration CLI.")
def cli() -> None:
    configure_logging()


# ---- train ----------------------------------------------------------------


@cli.command("train")
@_config_option
@click.option("--tenant", "tenant_id", required=True)
@click.option("--task", required=True)
@click.option("--version", required=True, type=int)
@click.option("--instance-type", default=None)
@click.option("--wait/--no-wait", default=True)
@click.option("--skip-data-validation/--no-skip-data-validation", default=False)
@_profile_option
def train(
    config_dir: str,
    tenant_id: str,
    task: str,
    version: int,
    instance_type: Optional[str],
    wait: bool,
    skip_data_validation: bool,
    profile: Optional[str],
) -> None:
    app_config = load_config(config_dir)
    platform = app_config.platform
    if not skip_data_validation:
        validate_training_data(
            training_data_prefix=training_data_s3_uri(platform, tenant_id, task),
            region=platform.region,
            profile=profile,
        )
    code_s3_uri = upload_entry_point(
        platform=platform,
        tenant_id=tenant_id,
        task=task,
        version=version,
        region=platform.region,
        profile=profile,
    )
    builder = TrainingJobBuilder(app_config)
    spec = builder.build_spec(
        tenant_id=tenant_id,
        task_name=task,
        version=version,
        code_s3_uri=code_s3_uri,
        instance_type_override=instance_type,
    )
    builder.submit(spec, wait=wait, session_profile=profile)
    click.echo(f"Training job submitted: {spec.job_name}")


# ---- deploy-endpoint ------------------------------------------------------


@cli.command("deploy-endpoint")
@_config_option
@click.option("--base-model-id", required=True)
@click.option("--inference-profile", default=None)
@click.option("--execution-role-arn", required=True)
@click.option("--wait/--no-wait", default=True)
@_profile_option
def deploy_endpoint(
    config_dir: str,
    base_model_id: str,
    inference_profile: Optional[str],
    execution_role_arn: str,
    wait: bool,
    profile: Optional[str],
) -> None:
    app_config = load_config(config_dir)
    manager = BaseEndpointManager(app_config, profile=profile)
    plan = manager.build_plan(
        base_model_id=base_model_id,
        inference_profile_name=inference_profile,
    )
    manager.deploy(plan, execution_role_arn=execution_role_arn, wait=wait)
    click.echo(
        f"Endpoint {plan.endpoint_name!r} InService with base component "
        f"{plan.base_inference_component_name!r}"
    )


# ---- deploy-adapter -------------------------------------------------------


@cli.command("deploy-adapter")
@_config_option
@click.option("--tenant", "tenant_id", required=True)
@click.option("--task", required=True)
@click.option("--version", required=True, type=int)
@click.option("--wait/--no-wait", default=True)
@_profile_option
def deploy_adapter(
    config_dir: str,
    tenant_id: str,
    task: str,
    version: int,
    wait: bool,
    profile: Optional[str],
) -> None:
    app_config = load_config(config_dir)
    manager = AdapterComponentManager(app_config, profile=profile)
    plan = manager.plan(tenant_id=tenant_id, task=task, version=version)
    manager.deploy(plan, wait=wait)
    click.echo(f"Adapter component {plan.component_name!r} is InService")


# ---- autoscale ------------------------------------------------------------


@cli.command("autoscale")
@_config_option
@click.option("--target-value", type=float, required=True)
@click.option(
    "--metric-type",
    type=click.Choice(
        [
            "SageMakerVariantInvocationsPerInstance",
            "SageMakerInferenceComponentInvocationsPerCopy",
        ]
    ),
    default="SageMakerVariantInvocationsPerInstance",
)
@click.option("--scale-in-cooldown", type=int, default=300)
@click.option("--scale-out-cooldown", type=int, default=60)
@click.option("--inference-profile", default=None)
@_profile_option
def autoscale(
    config_dir: str,
    target_value: float,
    metric_type: str,
    scale_in_cooldown: int,
    scale_out_cooldown: int,
    inference_profile: Optional[str],
    profile: Optional[str],
) -> None:
    app_config = load_config(config_dir)
    arn = apply_endpoint_autoscaling(
        app_config=app_config,
        target_value=target_value,
        predefined_metric_type=metric_type,
        scale_in_cooldown=scale_in_cooldown,
        scale_out_cooldown=scale_out_cooldown,
        inference_profile_name=inference_profile,
        profile=profile,
    )
    click.echo(f"Autoscaling policy: {arn}")


# ---- evaluate -------------------------------------------------------------


@cli.command("evaluate")
@_config_option
@click.option("--tenant", "tenant_id", required=True)
@click.option("--task", required=True)
@click.option("--version", required=True, type=int)
@_profile_option
def evaluate(
    config_dir: str,
    tenant_id: str,
    task: str,
    version: int,
    profile: Optional[str],
) -> None:
    app_config = load_config(config_dir)
    result = evaluate_adapter(
        app_config=app_config,
        tenant_id=tenant_id,
        task=task,
        version=version,
        profile=profile,
    )
    click.echo(json.dumps(asdict(result), indent=2))


# ---- compare --------------------------------------------------------------


@cli.command("compare")
@_config_option
@click.option("--tenant", "tenant_id", required=True)
@click.option("--task", required=True)
@click.option("--accuracy", type=float, required=True)
@_profile_option
def compare(
    config_dir: str,
    tenant_id: str,
    task: str,
    accuracy: float,
    profile: Optional[str],
) -> None:
    app_config = load_config(config_dir)
    gate = compare_and_gate(
        app_config=app_config,
        tenant_id=tenant_id,
        task=task,
        new_accuracy=accuracy,
        profile=profile,
    )
    click.echo(json.dumps(asdict(gate), indent=2))


# ---- publish-metric -------------------------------------------------------


@cli.command("publish-metric")
@_config_option
@click.option("--tenant", "tenant_id", required=True)
@click.option("--task", required=True)
@click.option("--version", required=True, type=int)
@click.option("--accuracy", type=float, required=True)
@_profile_option
def publish_metric(
    config_dir: str,
    tenant_id: str,
    task: str,
    version: int,
    accuracy: float,
    profile: Optional[str],
) -> None:
    app_config = load_config(config_dir)
    publish_accuracy(
        platform=app_config.platform,
        tenant_id=tenant_id,
        task=task,
        version=version,
        accuracy=accuracy,
        profile=profile,
    )
    click.echo("Accuracy published to CloudWatch.")


# ---- create-alarms --------------------------------------------------------


@cli.command("create-alarms")
@_config_option
@click.option("--tenant", "tenant_id", default=None)
@click.option("--task", default=None)
@_profile_option
def create_alarms(
    config_dir: str,
    tenant_id: Optional[str],
    task: Optional[str],
    profile: Optional[str],
) -> None:
    app_config = load_config(config_dir)
    tenants = (
        [app_config.tenants.get(tenant_id)]
        if tenant_id
        else app_config.tenants.tenants
    )
    for tenant in tenants:
        for task_cfg in tenant.tasks:
            if task and task_cfg.name != task:
                continue
            arn = ensure_regression_alarm(
                app_config=app_config,
                tenant_id=tenant.tenant_id,
                task=task_cfg.name,
                profile=profile,
            )
            click.echo(arn)


# ---- invoke ---------------------------------------------------------------


@cli.command("invoke")
@_config_option
@click.option("--tenant", "tenant_id", required=True)
@click.option("--task", required=True)
@click.option("--version", type=int, default=None)
@click.option("--input-text", required=True)
@click.option("--max-new-tokens", type=int, default=50)
@click.option("--temperature", type=float, default=0.0)
@_profile_option
def invoke(
    config_dir: str,
    tenant_id: str,
    task: str,
    version: Optional[int],
    input_text: str,
    max_new_tokens: int,
    temperature: float,
    profile: Optional[str],
) -> None:
    app_config = load_config(config_dir)
    client = MultiTenantClient(app_config, profile=profile)
    result = client.invoke(
        tenant_id=tenant_id,
        task=task,
        input_text=input_text,
        parameters={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        version=version,
    )
    click.echo(json.dumps(result.raw, indent=2))


# ---- pipeline -------------------------------------------------------------


@cli.command("pipeline")
@_config_option
@click.option("--tenant", "tenant_id", required=True)
@click.option("--task", required=True)
@click.option("--version", required=True, type=int)
@click.option("--instance-type", default=None)
@click.option("--skip-training/--no-skip-training", default=False)
@click.option("--skip-evaluation/--no-skip-evaluation", default=False)
@click.option("--skip-data-validation/--no-skip-data-validation", default=False)
@_profile_option
def pipeline(
    config_dir: str,
    tenant_id: str,
    task: str,
    version: int,
    instance_type: Optional[str],
    skip_training: bool,
    skip_evaluation: bool,
    skip_data_validation: bool,
    profile: Optional[str],
) -> None:
    app_config = load_config(config_dir)
    result = run_pipeline(
        app_config=app_config,
        tenant_id=tenant_id,
        task=task,
        version=version,
        skip_training=skip_training,
        skip_evaluation=skip_evaluation,
        skip_data_validation=skip_data_validation,
        instance_type_override=instance_type,
        profile=profile,
    )
    payload: dict = {
        "tenant_id": result.tenant_id,
        "task": result.task,
        "version": result.version,
        "training_job_name": result.training_job_name,
        "adapter_component_name": result.adapter_component_name,
        "mlflow_run_id": result.mlflow_run_id,
        "alarm_arn": result.alarm_arn,
        "steps_executed": result.steps_executed,
        "steps_skipped": result.steps_skipped,
    }
    if result.evaluation is not None:
        payload["evaluation"] = asdict(result.evaluation)
    if result.gate is not None:
        payload["gate"] = asdict(result.gate)
    click.echo(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    cli()
