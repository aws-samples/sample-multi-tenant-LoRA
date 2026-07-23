"""MLOps utilities: CloudWatch alarms, SNS notifications, MLflow tracking.

This module ties together the CloudWatch metrics published during evaluation
with operator-facing notifications and experiment tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.exceptions import ClientError

from .config import PlatformConfig

logger = logging.getLogger(__name__)


@dataclass
class AlarmInfo:
    alarm_name: str
    tenant_id: str
    task: str
    threshold: float


class MLOpsManager:
    """Manage CloudWatch alarms and (optionally) MLflow experiment tracking."""

    def __init__(
        self,
        config: PlatformConfig,
        session: boto3.Session | None = None,
    ):
        self.config = config
        self._session = session or boto3.Session(region_name=config.aws.region)
        self._cloudwatch = self._session.client("cloudwatch")

    # =====================================================================
    # CloudWatch alarms — one per (tenant, task) pair
    # =====================================================================

    def create_accuracy_alarm(
        self,
        tenant_id: str,
        task: str,
        threshold: float | None = None,
    ) -> AlarmInfo:
        """Create (or update) a regression alarm for a tenant's task."""
        cw_cfg = self.config.evaluation.cloudwatch
        threshold = threshold if threshold is not None else cw_cfg.alarm_threshold
        alarm_name = f"{cw_cfg.alarm_namespace}-{tenant_id}-{task}-accuracy"

        alarm_actions: list[str] = []
        if cw_cfg.sns_topic_arn:
            alarm_actions.append(cw_cfg.sns_topic_arn)

        self._cloudwatch.put_metric_alarm(
            AlarmName=alarm_name,
            AlarmDescription=(
                f"Fires when adapter accuracy for {tenant_id}/{task} "
                f"drops below {threshold:.2f}"
            ),
            Namespace=cw_cfg.alarm_namespace,
            MetricName="AdapterAccuracy",
            Dimensions=[
                {"Name": "TenantId", "Value": tenant_id},
                {"Name": "Task", "Value": task},
            ],
            Statistic="Average",
            Period=86400,  # 1 day — matches typical retraining cadence
            EvaluationPeriods=1,
            Threshold=threshold,
            ComparisonOperator="LessThanThreshold",
            TreatMissingData="notBreaching",
            AlarmActions=alarm_actions,
        )
        logger.info("CloudWatch alarm created/updated: %s", alarm_name)
        return AlarmInfo(
            alarm_name=alarm_name,
            tenant_id=tenant_id,
            task=task,
            threshold=threshold,
        )

    def create_alarms_for_all_tenants(self) -> list[AlarmInfo]:
        """Create alarms for every (tenant, task) pair in config."""
        results: list[AlarmInfo] = []
        for tenant in self.config.tenants:
            for task in tenant.tasks:
                results.append(self.create_accuracy_alarm(tenant.id, task.name))
        return results

    def delete_accuracy_alarm(self, tenant_id: str, task: str) -> None:
        alarm_name = (
            f"{self.config.evaluation.cloudwatch.alarm_namespace}"
            f"-{tenant_id}-{task}-accuracy"
        )
        try:
            self._cloudwatch.delete_alarms(AlarmNames=[alarm_name])
        except ClientError as e:
            logger.warning("Failed to delete alarm %s: %s", alarm_name, e)

    # =====================================================================
    # MLflow (optional)
    # =====================================================================

    def log_adapter_experiment(
        self,
        tenant_id: str,
        task: str,
        version: str,
        accuracy: float,
        hyperparameters: dict[str, Any],
    ) -> None:
        """Log an adapter training run to SageMaker managed MLflow.

        Gracefully no-ops when MLflow is disabled or the dependency is missing.
        """
        mlflow_cfg = self.config.evaluation.mlflow
        if not mlflow_cfg.enabled or not mlflow_cfg.tracking_server_arn:
            logger.debug("MLflow not enabled — skipping experiment log")
            return

        try:
            import mlflow  # type: ignore
        except ImportError:
            logger.warning("mlflow not installed — install it to enable tracking")
            return

        mlflow.set_tracking_uri(mlflow_cfg.tracking_server_arn)
        mlflow.set_experiment(f"{tenant_id}/{task}")
        with mlflow.start_run(run_name=f"v{version}"):
            mlflow.log_params(
                {
                    "tenant_id": tenant_id,
                    "task": task,
                    "version": version,
                    **hyperparameters,
                }
            )
            mlflow.log_metric("accuracy", accuracy)
