"""Create / update / delete adapter inference components for a specific tenant.

Each call produces a component named ``{tenant}-{task}-v{version}`` as mandated
by Layer 3 of the blog's isolation design. The component points at
``s3://{bucket}/{tenant}/adapters/{task}/v{version}/adapter.tar.gz`` via the
``Container.ArtifactUrl`` field and inherits compute resources from the base
inference component.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import click

from multi_tenant_lora_sagemaker.common.logging_setup import configure_logging
from multi_tenant_lora_sagemaker.common.naming import (
    adapter_component_name,
    adapter_tarball_s3_uri,
)
from multi_tenant_lora_sagemaker.common.sessions import build_boto_session
from multi_tenant_lora_sagemaker.config.loader import AppConfig, load_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdapterDeployment:
    component_name: str
    endpoint_name: str
    base_inference_component_name: str
    adapter_s3_uri: str


class AdapterComponentManager:
    """Thin wrapper around the SageMaker InferenceComponent API for adapters."""

    def __init__(self, app_config: AppConfig, profile: Optional[str] = None):
        self._config = app_config
        self._profile = profile
        self._boto_session = build_boto_session(
            region=app_config.platform.region, profile=profile
        )

    def plan(
        self, tenant_id: str, task: str, version: int
    ) -> AdapterDeployment:
        # Ensures the tenant/task pair exists in the registry.
        tenant = self._config.tenants.get(tenant_id)
        if not any(t.name == task for t in tenant.tasks):
            raise KeyError(
                f"tenant {tenant_id!r} has no task named {task!r}"
            )
        platform = self._config.platform
        return AdapterDeployment(
            component_name=adapter_component_name(tenant_id, task, version),
            endpoint_name=platform.endpoint_name,
            base_inference_component_name=platform.base_inference_component_name,
            adapter_s3_uri=adapter_tarball_s3_uri(platform, tenant_id, task, version),
        )

    def deploy(self, deployment: AdapterDeployment, wait: bool = True) -> None:
        """Create or update the adapter inference component."""
        from botocore.exceptions import ClientError
        from sagemaker_core import shapes
        from sagemaker_core.resources import InferenceComponent

        region = self._config.platform.region

        specification = shapes.InferenceComponentSpecification(
            base_inference_component_name=deployment.base_inference_component_name,
            container=shapes.InferenceComponentContainerSpecification(
                artifact_url=deployment.adapter_s3_uri,
            ),
        )

        try:
            existing = InferenceComponent.get(
                inference_component_name=deployment.component_name,
                session=self._boto_session,
                region=region,
            )
            logger.info(
                "Updating existing adapter component %s", deployment.component_name
            )
            existing.update(specification=specification)
            if wait:
                existing.wait_for_status(target_status="InService")
            return
        except ClientError as exc:
            if "ValidationException" not in str(exc) and "not found" not in str(exc).lower():
                raise

        logger.info(
            "Creating adapter component %s -> %s",
            deployment.component_name,
            deployment.adapter_s3_uri,
        )
        ic = InferenceComponent.create(
            inference_component_name=deployment.component_name,
            endpoint_name=deployment.endpoint_name,
            variant_name=self._config.platform.endpoint_variant_name,
            specification=specification,
            session=self._boto_session,
            region=region,
        )
        if wait:
            ic.wait_for_status(target_status="InService")

    def delete(
        self, tenant_id: str, task: str, version: int, wait: bool = True
    ) -> None:
        """Delete a specific adapter inference component."""
        from botocore.exceptions import ClientError
        from sagemaker_core.resources import InferenceComponent

        component = adapter_component_name(tenant_id, task, version)
        region = self._config.platform.region
        try:
            ic = InferenceComponent.get(
                inference_component_name=component,
                session=self._boto_session,
                region=region,
            )
        except ClientError as exc:
            if "ValidationException" in str(exc) or "not found" in str(exc).lower():
                logger.info("Component %s not found; nothing to delete", component)
                return
            raise

        ic.delete()
        if wait:
            ic.wait_for_delete()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group("adapter")
def cli() -> None:
    pass


@cli.command("deploy")
@click.option(
    "--config-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
)
@click.option("--tenant", "tenant_id", required=True)
@click.option("--task", required=True)
@click.option("--version", required=True, type=int)
@click.option("--profile", default=None)
@click.option("--wait/--no-wait", default=True)
def deploy(
    config_dir: str,
    tenant_id: str,
    task: str,
    version: int,
    profile: Optional[str],
    wait: bool,
) -> None:
    configure_logging()
    app_config = load_config(config_dir)
    manager = AdapterComponentManager(app_config, profile=profile)
    deployment = manager.plan(tenant_id=tenant_id, task=task, version=version)
    manager.deploy(deployment, wait=wait)
    click.echo(
        f"Adapter component {deployment.component_name!r} is InService "
        f"(artifact={deployment.adapter_s3_uri})."
    )


@cli.command("delete")
@click.option(
    "--config-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
)
@click.option("--tenant", "tenant_id", required=True)
@click.option("--task", required=True)
@click.option("--version", required=True, type=int)
@click.option("--profile", default=None)
@click.option("--wait/--no-wait", default=True)
def delete(
    config_dir: str,
    tenant_id: str,
    task: str,
    version: int,
    profile: Optional[str],
    wait: bool,
) -> None:
    configure_logging()
    app_config = load_config(config_dir)
    manager = AdapterComponentManager(app_config, profile=profile)
    manager.delete(tenant_id=tenant_id, task=task, version=version, wait=wait)
    click.echo(
        f"Adapter component for tenant={tenant_id} task={task} "
        f"version={version} deleted."
    )


if __name__ == "__main__":
    cli()
