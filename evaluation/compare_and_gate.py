"""Regression gate: compare a new adapter version against the previous baseline.

The previous baseline is sourced from CloudWatch (the last value published by
``publish_accuracy``). If no prior metric exists, the new version is approved
automatically — this is the expected behaviour for a tenant's first version.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from multi_tenant_lora_sagemaker.config.loader import AppConfig
from multi_tenant_lora_sagemaker.evaluation.publish_metrics import (
    read_previous_accuracy,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompareResult:
    decision: str  # "APPROVE" or "BLOCK"
    reason: str
    new_accuracy: float
    previous_accuracy: Optional[float]
    delta: Optional[float]


def compare_and_gate(
    app_config: AppConfig,
    tenant_id: str,
    task: str,
    new_accuracy: float,
    profile: Optional[str] = None,
) -> CompareResult:
    """Return APPROVE if the new accuracy is within the tenant's regression threshold."""

    tenant = app_config.tenants.get(tenant_id)
    task_cfg = next((t for t in tenant.tasks if t.name == task), None)
    if task_cfg is None:
        raise KeyError(f"tenant {tenant_id!r} has no task named {task!r}")

    previous = read_previous_accuracy(
        app_config.platform,
        tenant_id=tenant_id,
        task=task,
        profile=profile,
    )
    if previous is None:
        logger.info(
            "No prior accuracy baseline for tenant=%s task=%s; approving.",
            tenant_id,
            task,
        )
        return CompareResult(
            decision="APPROVE",
            reason="no previous baseline",
            new_accuracy=new_accuracy,
            previous_accuracy=None,
            delta=None,
        )

    delta = new_accuracy - previous
    threshold = task_cfg.regression_threshold
    if delta < -threshold:
        reason = (
            f"accuracy regressed by {abs(delta):.4f} "
            f"(threshold={threshold:.4f}, prev={previous:.4f}, new={new_accuracy:.4f})"
        )
        logger.warning("BLOCK decision: %s", reason)
        return CompareResult(
            decision="BLOCK",
            reason=reason,
            new_accuracy=new_accuracy,
            previous_accuracy=previous,
            delta=delta,
        )

    reason = (
        f"delta={delta:+.4f} within threshold {threshold:.4f} "
        f"(prev={previous:.4f}, new={new_accuracy:.4f})"
    )
    logger.info("APPROVE decision: %s", reason)
    return CompareResult(
        decision="APPROVE",
        reason=reason,
        new_accuracy=new_accuracy,
        previous_accuracy=previous,
        delta=delta,
    )
