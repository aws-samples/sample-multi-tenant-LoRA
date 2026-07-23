"""Log one adapter training + evaluation run to SageMaker managed MLflow.

When ``platform.yaml`` does not define an ``mlflow`` block this function is a
no-op, so the rest of the package still works for customers who do not use the
managed MLflow tracking server.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from multi_tenant_lora_sagemaker.config.loader import AppConfig

logger = logging.getLogger(__name__)


def register_experiment(
    app_config: AppConfig,
    tenant_id: str,
    task: str,
    version: int,
    accuracy: float,
    hyperparameters: Dict[str, Any],
    training_data_hash: Optional[str] = None,
    adapter_s3_uri: Optional[str] = None,
    run_tags: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Log the run to MLflow and return the run id (or None when MLflow is disabled)."""

    mlflow_config = app_config.platform.mlflow
    if mlflow_config is None:
        logger.info("MLflow tracking is disabled; skipping experiment registration.")
        return None

    # Imports deferred so customers without MLflow deps can still use the
    # rest of the package.
    import mlflow

    logger.info(
        "Setting MLflow tracking URI to %s", mlflow_config.tracking_server_arn
    )
    mlflow.set_tracking_uri(mlflow_config.tracking_server_arn)
    mlflow.set_experiment(f"{tenant_id}/{task}")

    with mlflow.start_run(run_name=f"v{version}") as run:
        mlflow.set_tags(
            {
                "tenant_id": tenant_id,
                "task": task,
                "version": str(version),
                **(run_tags or {}),
            }
        )
        params = {"tenant_id": tenant_id, "task": task, "version": version}
        params.update({k: str(v) for k, v in hyperparameters.items()})
        if training_data_hash:
            params["training_data_hash"] = training_data_hash
        if adapter_s3_uri:
            params["adapter_s3_uri"] = adapter_s3_uri
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        return run.info.run_id
