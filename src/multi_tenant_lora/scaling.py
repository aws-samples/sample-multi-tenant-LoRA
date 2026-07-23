"""Endpoint auto-scaling for the shared inference endpoint.

Configures Application Auto Scaling against the SageMaker variant using
InvocationsPerInstance as the target metric. When scaling out, each new
instance loads the base model and begins serving adapters independently —
the adapter cache hierarchy (hot in GPU, warm in CPU, cold on disk) operates
per-instance.
"""

from __future__ import annotations

import logging

import boto3
from botocore.exceptions import ClientError

from .config import PlatformConfig

logger = logging.getLogger(__name__)


class ScalingManager:
    """Manage Application Auto Scaling for the shared inference endpoint."""

    def __init__(
        self,
        config: PlatformConfig,
        session: boto3.Session | None = None,
    ):
        self.config = config
        self._session = session or boto3.Session(region_name=config.aws.region)
        self._aas = self._session.client("application-autoscaling")

    def configure(self) -> None:
        """Apply the auto-scaling config from config.yaml."""
        if not self.config.inference.autoscaling.enabled:
            logger.info("Auto-scaling disabled in config — skipping")
            return

        resource_id = self._resource_id()
        scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
        service_namespace = "sagemaker"
        cfg = self.config.inference

        # 1. Register the scalable target
        self._aas.register_scalable_target(
            ServiceNamespace=service_namespace,
            ResourceId=resource_id,
            ScalableDimension=scalable_dimension,
            MinCapacity=cfg.min_instance_count,
            MaxCapacity=cfg.max_instance_count,
        )

        # 2. Attach a target-tracking policy on invocations per instance
        policy_name = f"{cfg.endpoint_name}-target-tracking"
        self._aas.put_scaling_policy(
            PolicyName=policy_name,
            ServiceNamespace=service_namespace,
            ResourceId=resource_id,
            ScalableDimension=scalable_dimension,
            PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": float(cfg.autoscaling.target_invocations_per_instance),
                "PredefinedMetricSpecification": {
                    "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
                },
                "ScaleInCooldown": cfg.autoscaling.scale_in_cooldown_seconds,
                "ScaleOutCooldown": cfg.autoscaling.scale_out_cooldown_seconds,
            },
        )
        logger.info(
            "Auto-scaling configured for %s: %d-%d instances, target=%d inv/min",
            cfg.endpoint_name,
            cfg.min_instance_count,
            cfg.max_instance_count,
            cfg.autoscaling.target_invocations_per_instance,
        )

    def deregister(self) -> None:
        """Remove the scalable target (used during teardown)."""
        try:
            self._aas.deregister_scalable_target(
                ServiceNamespace="sagemaker",
                ResourceId=self._resource_id(),
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            )
        except ClientError as e:
            logger.warning("Error deregistering scalable target: %s", e)

    def _resource_id(self) -> str:
        return f"endpoint/{self.config.inference.endpoint_name}/variant/AllTraffic"
