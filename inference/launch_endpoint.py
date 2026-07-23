"""Deploy the shared base inference component that all tenants attach adapters to.

The endpoint is created idempotently: if one already exists it is reused and
the base inference component is created/updated in place. All adapter
inference components are deployed separately via
:mod:`multi_tenant_lora_sagemaker.inference.deploy_adapter`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import click

from multi_tenant_lora_sagemaker.common.logging_setup import configure_logging
from multi_tenant_lora_sagemaker.common.sessions import build_boto_session
from multi_tenant_lora_sagemaker.config.loader import AppConfig, load_config
from multi_tenant_lora_sagemaker.config.schemas import BaseModelEntry, InstanceProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaseEndpointPlan:
    endpoint_name: str
    endpoint_config_name: str
    base_inference_component_name: str
    variant_name: str
    inference_image: str
    hf_model_id: str
    instance_profile: InstanceProfile
    environment: dict


class BaseEndpointManager:
    """Creates / updates the shared endpoint + base inference component."""

    def __init__(self, app_config: AppConfig, profile: Optional[str] = None):
        self._config = app_config
        self._profile = profile
        self._boto_session = build_boto_session(
            region=app_config.platform.region, profile=profile
        )

    # ------------------------------------------------------------------
    # Image URI lookup
    # ------------------------------------------------------------------

    def _retrieve_lmi_image(self, base_model: BaseModelEntry) -> str:
        from sagemaker import image_uris

        uri = image_uris.retrieve(
            framework=base_model.inference_framework,
            region=self._config.platform.region,
            version=base_model.inference_framework_version,
        )
        logger.info("Resolved inference DLC: %s", uri)
        return uri

    # ------------------------------------------------------------------
    # Plan building
    # ------------------------------------------------------------------

    def build_plan(
        self,
        base_model_id: str,
        inference_profile_name: Optional[str] = None,
    ) -> BaseEndpointPlan:
        base_model = self._config.base_models.get(base_model_id)
        profile = self._config.instances.inference_profile(inference_profile_name)
        platform = self._config.platform

        environment = {
            "HF_MODEL_ID": base_model.hf_model_id,
            "OPTION_ENABLE_LORA": "true",
            "OPTION_MAX_LORAS": str(profile.max_hot_adapters),
            "OPTION_MAX_CPU_LORAS": str(profile.max_cpu_adapters),
        }
        # Per-base-model LMI knobs (dtype, tensor parallelism, max seq len, ...)
        # layer on top of the isolation defaults above.
        for k, v in base_model.lmi_env.items():
            environment.setdefault(k, str(v))

        return BaseEndpointPlan(
            endpoint_name=platform.endpoint_name,
            endpoint_config_name=f"{platform.endpoint_name}-config",
            base_inference_component_name=platform.base_inference_component_name,
            variant_name=platform.endpoint_variant_name,
            inference_image=self._retrieve_lmi_image(base_model),
            hf_model_id=base_model.hf_model_id,
            instance_profile=profile,
            environment=environment,
        )

    # ------------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------------

    def deploy(
        self,
        plan: BaseEndpointPlan,
        execution_role_arn: str,
        wait: bool = True,
    ) -> None:
        """Create the endpoint (+ endpoint config + base inference component) if missing.

        The operations are idempotent: existing resources are left untouched so
        repeated ``deploy`` calls converge to the same state.
        """
        from botocore.exceptions import ClientError
        from sagemaker_core import shapes
        from sagemaker_core.resources import Endpoint, EndpointConfig, InferenceComponent

        region = self._config.platform.region

        # ---- EndpointConfig (empty; inference components hold the model) ----
        try:
            EndpointConfig.get(
                endpoint_config_name=plan.endpoint_config_name,
                session=self._boto_session,
                region=region,
            )
            logger.info("EndpointConfig %s already exists", plan.endpoint_config_name)
        except ClientError as exc:
            if "ValidationException" not in str(exc) and "not found" not in str(exc).lower():
                raise
            logger.info("Creating EndpointConfig %s", plan.endpoint_config_name)
            EndpointConfig.create(
                endpoint_config_name=plan.endpoint_config_name,
                production_variants=[
                    shapes.ProductionVariant(
                        variant_name=plan.variant_name,
                        instance_type=plan.instance_profile.instance_type,
                        initial_instance_count=plan.instance_profile.initial_instance_count,
                        # No model_name here: inference components own the models.
                    )
                ],
                execution_role_arn=execution_role_arn,
                session=self._boto_session,
                region=region,
            )

        # ---- Endpoint ----
        endpoint: Optional[Endpoint] = None
        try:
            endpoint = Endpoint.get(
                endpoint_name=plan.endpoint_name,
                session=self._boto_session,
                region=region,
            )
            logger.info(
                "Endpoint %s already exists (status=%s)",
                plan.endpoint_name,
                endpoint.endpoint_status,
            )
        except ClientError as exc:
            if "ValidationException" not in str(exc) and "not found" not in str(exc).lower():
                raise
            logger.info("Creating Endpoint %s", plan.endpoint_name)
            endpoint = Endpoint.create(
                endpoint_name=plan.endpoint_name,
                endpoint_config_name=plan.endpoint_config_name,
                session=self._boto_session,
                region=region,
            )
        if wait and endpoint is not None:
            endpoint.wait_for_status(target_status="InService")

        # ---- Base InferenceComponent ----
        try:
            InferenceComponent.get(
                inference_component_name=plan.base_inference_component_name,
                session=self._boto_session,
                region=region,
            )
            logger.info(
                "Base inference component %s already exists",
                plan.base_inference_component_name,
            )
            return
        except ClientError as exc:
            if "ValidationException" not in str(exc) and "not found" not in str(exc).lower():
                raise

        logger.info(
            "Creating base inference component %s",
            plan.base_inference_component_name,
        )
        specification = shapes.InferenceComponentSpecification(
            container=shapes.InferenceComponentContainerSpecification(
                image=plan.inference_image,
                environment=plan.environment,
            ),
            compute_resource_requirements=shapes.InferenceComponentComputeResourceRequirements(
                number_of_accelerator_devices_required=1,
                min_memory_required_in_mb=1024,
            ),
            startup_parameters=shapes.InferenceComponentStartupParameters(
                model_data_download_timeout_in_seconds=1800,
                container_startup_health_check_timeout_in_seconds=1800,
            ),
        )
        runtime_config = shapes.InferenceComponentRuntimeConfig(copy_count=1)
        ic = InferenceComponent.create(
            inference_component_name=plan.base_inference_component_name,
            endpoint_name=plan.endpoint_name,
            variant_name=plan.variant_name,
            specification=specification,
            runtime_config=runtime_config,
            session=self._boto_session,
            region=region,
        )
        if wait:
            ic.wait_for_status(target_status="InService")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("deploy-endpoint")
@click.option(
    "--config-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
)
@click.option("--base-model-id", required=True, help="Base model id from base_models.yaml.")
@click.option(
    "--inference-profile",
    default=None,
    help="Inference profile name from instances.yaml (defaults to the configured default).",
)
@click.option(
    "--execution-role-arn",
    required=True,
    help="IAM role ARN that SageMaker will assume when pulling images and weights.",
)
@click.option("--profile", default=None, help="AWS named profile.")
@click.option("--wait/--no-wait", default=True)
def cli(
    config_dir: str,
    base_model_id: str,
    inference_profile: Optional[str],
    execution_role_arn: str,
    profile: Optional[str],
    wait: bool,
) -> None:
    configure_logging()
    app_config = load_config(config_dir)
    manager = BaseEndpointManager(app_config, profile=profile)
    plan = manager.build_plan(
        base_model_id=base_model_id,
        inference_profile_name=inference_profile,
    )
    manager.deploy(plan, execution_role_arn=execution_role_arn, wait=wait)
    click.echo(
        f"Endpoint {plan.endpoint_name!r} is InService with base inference "
        f"component {plan.base_inference_component_name!r}."
    )


if __name__ == "__main__":
    cli()
