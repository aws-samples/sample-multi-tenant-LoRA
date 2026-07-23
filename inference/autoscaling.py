"""Apply a target-tracking autoscaling policy to the shared endpoint.

Uses Application Auto Scaling via boto3 (sagemaker-core does not yet wrap this
service). The policy targets ``SageMakerVariantInvocationsPerInstance`` as in
the blog; customers can also target ``SageMakerInferenceComponentInvocationsPerCopy``
by switching ``predefined_metric_type`` in the CLI.
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


def apply_endpoint_autoscaling(
    app_config: AppConfig,
    target_value: float,
    predefined_metric_type: str = "SageMakerVariantInvocationsPerInstance",
    scale_in_cooldown: int = 300,
    scale_out_cooldown: int = 60,
    policy_name: str = "multi-tenant-lora-invocations-scaling",
    inference_profile_name: Optional[str] = None,
    profile: Optional[str] = None,
) -> str:
    """Register and attach a target-tracking policy to the endpoint variant.

    Returns the created/updated policy ARN.
    """

    platform = app_config.platform
    instance_profile = app_config.instances.inference_profile(inference_profile_name)

    session = boto3.Session(region_name=platform.region, profile_name=profile)
    aas = session.client("application-autoscaling", config=default_boto_config())

    resource_id = f"endpoint/{platform.endpoint_name}/variant/{platform.endpoint_variant_name}"
    logger.info(
        "Registering scalable target for %s (min=%d, max=%d)",
        resource_id,
        instance_profile.initial_instance_count,
        instance_profile.max_instance_count,
    )
    aas.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=instance_profile.initial_instance_count,
        MaxCapacity=instance_profile.max_instance_count,
    )

    logger.info(
        "Applying target-tracking policy %s (metric=%s, target=%s)",
        policy_name,
        predefined_metric_type,
        target_value,
    )
    response = aas.put_scaling_policy(
        PolicyName=policy_name,
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": target_value,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": predefined_metric_type,
            },
            "ScaleInCooldown": scale_in_cooldown,
            "ScaleOutCooldown": scale_out_cooldown,
        },
    )
    return response["PolicyARN"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("autoscale-endpoint")
@click.option(
    "--config-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
)
@click.option("--target-value", type=float, required=True)
@click.option(
    "--metric-type",
    default="SageMakerVariantInvocationsPerInstance",
    type=click.Choice(
        [
            "SageMakerVariantInvocationsPerInstance",
            "SageMakerInferenceComponentInvocationsPerCopy",
        ]
    ),
)
@click.option("--scale-in-cooldown", type=int, default=300)
@click.option("--scale-out-cooldown", type=int, default=60)
@click.option("--inference-profile", default=None)
@click.option("--profile", default=None)
def cli(
    config_dir: str,
    target_value: float,
    metric_type: str,
    scale_in_cooldown: int,
    scale_out_cooldown: int,
    inference_profile: Optional[str],
    profile: Optional[str],
) -> None:
    configure_logging()
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
    click.echo(f"Autoscaling policy applied: {arn}")


if __name__ == "__main__":
    cli()
