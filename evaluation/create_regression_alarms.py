"""Create a CloudWatch alarm per tenant+task that fires when accuracy regresses.

The alarm watches the ``AdapterAccuracy`` metric published by
:mod:`publish_metrics`. Threshold is the per-task ``accuracy_floor`` when set,
otherwise the blog default of 0.85.
"""

from __future__ import annotations

import logging
from typing import Optional

import boto3
import click

from multi_tenant_lora_sagemaker.common.logging_setup import configure_logging
from multi_tenant_lora_sagemaker.common.sessions import default_boto_config
from multi_tenant_lora_sagemaker.config.loader import AppConfig, load_config

logger = logging.getLogger(__name__)

_DEFAULT_FLOOR = 0.85
_DEFAULT_PERIOD_SECONDS = 86400


def ensure_regression_alarm(
    app_config: AppConfig,
    tenant_id: str,
    task: str,
    profile: Optional[str] = None,
) -> str:
    """Create or update the alarm for ``(tenant_id, task)`` and return its ARN."""

    tenant = app_config.tenants.get(tenant_id)
    task_cfg = next((t for t in tenant.tasks if t.name == task), None)
    if task_cfg is None:
        raise KeyError(f"tenant {tenant_id!r} has no task named {task!r}")

    platform = app_config.platform
    session = boto3.Session(region_name=platform.region, profile_name=profile)
    cw = session.client("cloudwatch", config=default_boto_config())

    alarm_name = f"{tenant_id}-{task}-accuracy-regression"
    threshold = task_cfg.accuracy_floor if task_cfg.accuracy_floor is not None else _DEFAULT_FLOOR

    logger.info(
        "Ensuring alarm %s (threshold=%.4f, namespace=%s)",
        alarm_name,
        threshold,
        platform.cloudwatch_namespace,
    )

    put_kwargs = dict(
        AlarmName=alarm_name,
        AlarmDescription=(
            f"Accuracy regression guard for tenant={tenant_id} task={task}. "
            "Triggered when adapter accuracy falls below the configured floor."
        ),
        MetricName="AdapterAccuracy",
        Namespace=platform.cloudwatch_namespace,
        Dimensions=[
            {"Name": "TenantId", "Value": tenant_id},
            {"Name": "Task", "Value": task},
        ],
        Statistic="Average",
        Period=_DEFAULT_PERIOD_SECONDS,
        EvaluationPeriods=1,
        Threshold=threshold,
        ComparisonOperator="LessThanThreshold",
        TreatMissingData="notBreaching",
    )
    if platform.regression_alarm_sns_topic_arn:
        put_kwargs["AlarmActions"] = [platform.regression_alarm_sns_topic_arn]

    cw.put_metric_alarm(**put_kwargs)
    described = cw.describe_alarms(AlarmNames=[alarm_name])
    alarms = described.get("MetricAlarms", [])
    if not alarms:
        raise RuntimeError(
            f"alarm {alarm_name} was not returned after creation"
        )
    return alarms[0]["AlarmArn"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("create-regression-alarms")
@click.option(
    "--config-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
)
@click.option(
    "--tenant",
    "tenant_id",
    default=None,
    help="Limit to a single tenant; omit to cover every tenant in tenants.yaml.",
)
@click.option(
    "--task",
    default=None,
    help="Limit to a single task; omit to cover every task of the selected tenant(s).",
)
@click.option("--profile", default=None)
def cli(
    config_dir: str,
    tenant_id: Optional[str],
    task: Optional[str],
    profile: Optional[str],
) -> None:
    configure_logging()
    app_config = load_config(config_dir)

    tenants = (
        [app_config.tenants.get(tenant_id)] if tenant_id else app_config.tenants.tenants
    )
    created = 0
    for tenant in tenants:
        for task_cfg in tenant.tasks:
            if task and task_cfg.name != task:
                continue
            arn = ensure_regression_alarm(
                app_config,
                tenant_id=tenant.tenant_id,
                task=task_cfg.name,
                profile=profile,
            )
            click.echo(f"Alarm ready: {arn}")
            created += 1
    click.echo(f"Ensured {created} alarms.")


if __name__ == "__main__":
    cli()
