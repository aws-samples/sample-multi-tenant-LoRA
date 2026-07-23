"""Publish adapter accuracy metrics to CloudWatch and read the previous value.

Metrics are published under the namespace configured in ``platform.yaml``
(default ``MultiTenantLoRA``) with dimensions ``TenantId`` and ``Task`` so the
alarm module can wire an alarm to each per-tenant / per-task series.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Optional

import boto3

from multi_tenant_lora_sagemaker.common.sessions import default_boto_config
from multi_tenant_lora_sagemaker.config.schemas import PlatformConfig

logger = logging.getLogger(__name__)

_METRIC_NAME = "AdapterAccuracy"
_VERSION_METRIC = "AdapterAccuracyByVersion"


def publish_accuracy(
    platform: PlatformConfig,
    tenant_id: str,
    task: str,
    version: int,
    accuracy: float,
    profile: Optional[str] = None,
    timestamp: Optional[_dt.datetime] = None,
) -> None:
    """Emit two CloudWatch datapoints per call.

    ``AdapterAccuracy`` is keyed by ``(TenantId, Task)`` so alarms operate
    independently per tenant+task. ``AdapterAccuracyByVersion`` adds the
    ``Version`` dimension for dashboards.
    """

    session = boto3.Session(region_name=platform.region, profile_name=profile)
    cw = session.client("cloudwatch", config=default_boto_config())
    ts = timestamp or _dt.datetime.now(tz=_dt.timezone.utc)

    logger.info(
        "Publishing accuracy=%.4f for tenant=%s task=%s v=%d",
        accuracy,
        tenant_id,
        task,
        version,
    )
    cw.put_metric_data(
        Namespace=platform.cloudwatch_namespace,
        MetricData=[
            {
                "MetricName": _METRIC_NAME,
                "Dimensions": [
                    {"Name": "TenantId", "Value": tenant_id},
                    {"Name": "Task", "Value": task},
                ],
                "Value": accuracy,
                "Unit": "None",
                "Timestamp": ts,
            },
            {
                "MetricName": _VERSION_METRIC,
                "Dimensions": [
                    {"Name": "TenantId", "Value": tenant_id},
                    {"Name": "Task", "Value": task},
                    {"Name": "Version", "Value": str(version)},
                ],
                "Value": accuracy,
                "Unit": "None",
                "Timestamp": ts,
            },
        ],
    )


def read_previous_accuracy(
    platform: PlatformConfig,
    tenant_id: str,
    task: str,
    lookback_days: int = 30,
    profile: Optional[str] = None,
) -> Optional[float]:
    """Return the most recent ``AdapterAccuracy`` value for this tenant+task, if any."""

    session = boto3.Session(region_name=platform.region, profile_name=profile)
    cw = session.client("cloudwatch", config=default_boto_config())
    end = _dt.datetime.now(tz=_dt.timezone.utc)
    start = end - _dt.timedelta(days=lookback_days)

    response = cw.get_metric_statistics(
        Namespace=platform.cloudwatch_namespace,
        MetricName=_METRIC_NAME,
        Dimensions=[
            {"Name": "TenantId", "Value": tenant_id},
            {"Name": "Task", "Value": task},
        ],
        StartTime=start,
        EndTime=end,
        Period=3600,
        Statistics=["Average"],
    )
    datapoints = sorted(
        response.get("Datapoints", []), key=lambda d: d["Timestamp"]
    )
    if not datapoints:
        logger.info(
            "No previous accuracy metric found for tenant=%s task=%s", tenant_id, task
        )
        return None
    return float(datapoints[-1]["Average"])
