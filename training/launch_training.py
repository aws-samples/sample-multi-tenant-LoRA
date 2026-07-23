"""CLI: launch a LoRA training job for one tenant/task/version.

Usage::

    python -m multi_tenant_lora_sagemaker.training.launch_training \\
        --config-dir ./config \\
        --tenant tenant-a \\
        --task type-classifier \\
        --version 4 \\
        [--wait] [--skip-data-validation] [--profile my-aws-profile] \\
        [--instance-type ml.g5.2xlarge]
"""

from __future__ import annotations

import logging
from typing import Optional

import click

from multi_tenant_lora_sagemaker.common.logging_setup import configure_logging
from multi_tenant_lora_sagemaker.common.naming import training_data_s3_uri
from multi_tenant_lora_sagemaker.config.loader import load_config
from multi_tenant_lora_sagemaker.training.code_uploader import upload_entry_point
from multi_tenant_lora_sagemaker.training.data_validator import (
    DataValidationError,
    validate_training_data,
)
from multi_tenant_lora_sagemaker.training.training_job_builder import TrainingJobBuilder

logger = logging.getLogger(__name__)


@click.command("launch-training")
@click.option(
    "--config-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    help="Directory containing platform.yaml, tenants.yaml, base_models.yaml, instances.yaml.",
)
@click.option("--tenant", "tenant_id", required=True, help="Tenant id.")
@click.option("--task", required=True, help="Task name as listed in tenants.yaml.")
@click.option("--version", required=True, type=int, help="Adapter version (integer).")
@click.option(
    "--instance-type",
    default=None,
    help="Optional override of the training instance type from instances.yaml.",
)
@click.option(
    "--wait/--no-wait",
    default=False,
    help="Block until the training job reaches a terminal state.",
)
@click.option(
    "--skip-data-validation/--no-skip-data-validation",
    default=False,
    help="Skip the pre-flight JSONL validation step.",
)
@click.option("--profile", default=None, help="AWS named profile to use.")
def cli(
    config_dir: str,
    tenant_id: str,
    task: str,
    version: int,
    instance_type: Optional[str],
    wait: bool,
    skip_data_validation: bool,
    profile: Optional[str],
) -> None:
    configure_logging()
    app_config = load_config(config_dir)
    platform = app_config.platform

    if not skip_data_validation:
        train_prefix = training_data_s3_uri(platform, tenant_id, task)
        try:
            validate_training_data(
                training_data_prefix=train_prefix,
                region=platform.region,
                profile=profile,
            )
        except DataValidationError as exc:
            click.echo(f"Data validation failed: {exc}", err=True)
            raise SystemExit(2)

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
    job = builder.submit(spec, wait=wait, session_profile=profile)

    click.echo(f"Submitted training job: {spec.job_name}")
    if wait:
        logger.info("Training job reached terminal status.")


if __name__ == "__main__":
    cli()
