"""Inference platform: shared base model endpoint + per-tenant adapter components.

Uses SageMaker JumpStart as the source of the base model. JumpStart provides:
  - Pre-packaged base model artifacts in S3 (no HF auth needed, no download)
  - The right LMI container URI per region/instance type
  - Validated default environment variables per instance

Every model in the JumpStart Public Hub exposes a HubContentDocument that
contains ready-to-deploy HostingArtifactUri + HostingEcrUri fields. We read
those at runtime instead of hard-coding image tags or building serving.properties.

On top of the JumpStart base, we register one adapter inference component per
tenant using the LoRA artifacts that SFTTrainer produced during training.
"""

from __future__ import annotations

import json
import logging
import time

import boto3
from botocore.exceptions import ClientError

from .config import PlatformConfig
from .storage import StorageManager

logger = logging.getLogger(__name__)


class InferencePlatform:
    """Manage the shared inference endpoint and all adapter components."""

    def __init__(
        self,
        config: PlatformConfig,
        session: boto3.Session | None = None,
    ):
        self.config = config
        self._session = session or boto3.Session(region_name=config.aws.region)
        self._sm = self._session.client("sagemaker")
        self._sm_runtime = self._session.client("sagemaker-runtime")
        self._storage = StorageManager(config, session=self._session)

    # =====================================================================
    # Shared-endpoint lifecycle
    # =====================================================================

    def deploy_base_endpoint(self) -> str:
        """Create the shared endpoint + base inference component (if not present).

        Looks up the JumpStart model spec for self.config.base_model.hf_model_id
        and uses SageMaker's pre-packaged artifacts + LMI container.
        """
        endpoint_name = self.config.inference.endpoint_name
        base_component_name = self.config.inference.base_component_name

        jumpstart = self._get_jumpstart_spec()

        model_name = self._ensure_base_model(jumpstart)
        endpoint_config_name = self._ensure_endpoint_config(endpoint_name)
        self._create_or_update_endpoint(endpoint_name, endpoint_config_name)
        self._wait_for_endpoint(endpoint_name, target_status="InService")
        self._ensure_base_inference_component(
            endpoint_name=endpoint_name,
            component_name=base_component_name,
            model_name=model_name,
            jumpstart=jumpstart,
        )
        logger.info("Base endpoint ready: %s", endpoint_name)
        return endpoint_name

    def delete_endpoint(self) -> None:
        """Delete endpoint, all components, and configs."""
        endpoint_name = self.config.inference.endpoint_name
        try:
            paginator = self._sm.get_paginator("list_inference_components")
            for page in paginator.paginate(EndpointNameEquals=endpoint_name):
                for ic in page.get("InferenceComponents", []):
                    try:
                        self._sm.delete_inference_component(
                            InferenceComponentName=ic["InferenceComponentName"]
                        )
                    except ClientError as e:
                        logger.warning("Delete IC %s: %s", ic["InferenceComponentName"], e)
        except ClientError as e:
            logger.warning("List inference components failed: %s", e)

        try:
            self._sm.delete_endpoint(EndpointName=endpoint_name)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise

        try:
            paginator = self._sm.get_paginator("list_endpoint_configs")
            for page in paginator.paginate(NameContains=endpoint_name):
                for ec in page.get("EndpointConfigs", []):
                    try:
                        self._sm.delete_endpoint_config(
                            EndpointConfigName=ec["EndpointConfigName"]
                        )
                    except ClientError:
                        pass
        except ClientError:
            pass

    # =====================================================================
    # Per-tenant adapter lifecycle
    # =====================================================================

    def deploy_adapter(
        self,
        tenant_id: str,
        task: str,
        version: str,
        adapter_s3_uri: str | None = None,
    ) -> str:
        """Register an adapter as an inference component on the shared endpoint."""
        if adapter_s3_uri is None:
            adapter_s3_uri = self._storage.find_adapter_artifact(tenant_id, task, version)
            if not adapter_s3_uri:
                raise FileNotFoundError(
                    f"No adapter found for tenant={tenant_id} task={task} "
                    f"version={version}. Training may still be running."
                )

        component_name = self.config.inference_component_name(tenant_id, task, version)
        endpoint_name = self.config.inference.endpoint_name
        base_component = self.config.inference.base_component_name

        logger.info(
            "Deploying adapter component: %s -> %s", component_name, adapter_s3_uri
        )

        try:
            self._sm.create_inference_component(
                InferenceComponentName=component_name,
                EndpointName=endpoint_name,
                Specification={
                    "BaseInferenceComponentName": base_component,
                    "Container": {"ArtifactUrl": adapter_s3_uri},
                },
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceInUse":
                logger.info("Component exists — updating: %s", component_name)
                self._sm.update_inference_component(
                    InferenceComponentName=component_name,
                    Specification={
                        "BaseInferenceComponentName": base_component,
                        "Container": {"ArtifactUrl": adapter_s3_uri},
                    },
                )
            else:
                raise

        self._wait_for_inference_component(component_name, target_status="InService")
        return component_name

    def delete_adapter(self, tenant_id: str, task: str, version: str) -> None:
        component_name = self.config.inference_component_name(tenant_id, task, version)
        try:
            self._sm.delete_inference_component(InferenceComponentName=component_name)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise

    def list_adapters(self) -> list[dict]:
        endpoint_name = self.config.inference.endpoint_name
        try:
            resp = self._sm.list_inference_components(EndpointNameEquals=endpoint_name)
        except ClientError:
            return []
        base = self.config.inference.base_component_name
        return [
            {
                "name": ic["InferenceComponentName"],
                "status": ic.get("InferenceComponentStatus", "Unknown"),
                "created": ic.get("CreationTime"),
            }
            for ic in resp.get("InferenceComponents", [])
            if ic["InferenceComponentName"] != base
        ]

    # =====================================================================
    # Invocation
    # =====================================================================

    def invoke(
        self,
        tenant_id: str,
        task: str,
        version: str,
        prompt: str,
        *,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """Send a prompt to a tenant's adapter. Returns the generated text."""
        component_name = self.config.inference_component_name(tenant_id, task, version)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        }
        resp = self._sm_runtime.invoke_endpoint(
            EndpointName=self.config.inference.endpoint_name,
            InferenceComponentName=component_name,
            Body=json.dumps(payload),
            ContentType="application/json",
        )
        body = resp["Body"].read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return body
        if isinstance(parsed, dict):
            return (
                parsed.get("generated_text")
                or parsed.get("generation")
                or parsed.get("output")
                or body
            )
        if isinstance(parsed, list) and parsed:
            first = parsed[0]
            if isinstance(first, dict):
                return first.get("generated_text") or first.get("generation") or body
        return body

    # =====================================================================
    # JumpStart spec lookup
    # =====================================================================

    def _get_jumpstart_spec(self) -> dict:
        """Resolve the JumpStart hosting spec for our target instance type.

        Prefers the SFT/LoRA recipe's HostingConfig when available — this is
        the SageMaker-validated config for multi-adapter serving, including
        the right container version, entrypoint, and env vars.

        Falls back to the generic hosting spec if no LoRA recipe is found.
        """
        model_id = self.config.base_model.hf_model_id
        instance_type = self.config.inference.instance_type
        hub_name = "SageMakerPublicHub"

        logger.info(
            "Resolving JumpStart spec for model=%s instance=%s",
            model_id, instance_type,
        )

        resp = self._sm.describe_hub_content(
            HubName=hub_name,
            HubContentType="Model",
            HubContentName=model_id,
        )
        doc = json.loads(resp["HubContentDocument"])

        # 1. Prefer the SFT LoRA recipe's HostingConfig if it matches our instance
        lora_hosting = self._find_lora_hosting_config(doc, instance_type)
        if lora_hosting:
            logger.info("Using validated SFT-LoRA HostingConfig for %s", instance_type)
            return self._spec_from_recipe_hosting(lora_hosting, doc)

        # 2. Fall back to the generic inference config
        logger.info("No SFT-LoRA recipe for %s — using generic inference config", instance_type)
        return self._spec_from_generic_config(doc, instance_type)

    @staticmethod
    def _find_lora_hosting_config(doc: dict, instance_type: str) -> dict | None:
        """Scan RecipeCollection for an SFT+LoRA recipe whose HostingConfigs
        include our target instance type."""
        for recipe in doc.get("RecipeCollection", []):
            if (
                recipe.get("CustomizationTechnique") == "SFT"
                and recipe.get("Peft") == "LORA"
                and recipe.get("Type") == "FineTuning"
            ):
                for hc in recipe.get("HostingConfigs", []):
                    if hc.get("InstanceType") == instance_type:
                        return hc
        return None

    def _spec_from_recipe_hosting(self, hc: dict, doc: dict) -> dict:
        """Build a deployment spec from an SFT-LoRA recipe HostingConfig."""
        env = dict(hc.get("Environment", {}))

        # Layer in our own config knobs — but NEVER override the recipe's
        # ENTRYPOINT, ASYNC_MODE, or ROLLING_BATCH. These are what make the
        # container actually start.
        cfg_opts = self.config.inference.option
        user_overrides = {
            "OPTION_MAX_LORAS": str(cfg_opts.max_loras),
            "OPTION_MAX_CPU_LORAS": str(cfg_opts.max_cpu_loras),
        }
        env.update(user_overrides)

        resources = hc.get("ComputeResourceRequirements", {})
        return {
            "ImageUri": hc.get("EcrAddress"),
            "ArtifactUri": doc.get("HostingArtifactUri"),
            "ArtifactCompression": doc.get("HostingArtifactCompressionType", "None"),
            "Environment": env,
            "MinMemoryMb": resources.get("MinMemoryRequiredInMb", 32768),
            "NumAccelerators": resources.get("NumberOfAcceleratorDevicesRequired", 1),
        }

    def _spec_from_generic_config(self, doc: dict, instance_type: str) -> dict:
        """Fallback: use the generic InferenceConfigComponents spec."""
        rankings = (
            doc.get("InferenceConfigRankings", {})
            .get("overall", {})
            .get("Rankings", [])
        )
        components = doc.get("InferenceConfigComponents", {})
        if not components:
            raise ValueError(
                f"No InferenceConfigComponents and no matching SFT-LoRA recipe "
                f"for {self.config.base_model.hf_model_id} on {instance_type}. "
                f"Set inference.image_uri explicitly in config.yaml."
            )
        config_name = next(
            (r for r in rankings if r in components),
            next(iter(components)),
        )
        cfg = components[config_name]
        logger.info("Using JumpStart config: %s", config_name)

        variants = cfg.get("HostingInstanceTypeVariants", {}).get("Variants", {})
        variant = variants.get(instance_type)
        if variant is None:
            family = instance_type.split(".")[1] if "." in instance_type else ""
            variant = variants.get(family, {})
        variant_props = variant.get("Properties", {})

        image_uri = variant_props.get("ImageUri") or cfg.get("HostingEcrUri")
        if not image_uri:
            raise ValueError(
                f"No ImageUri for {self.config.base_model.hf_model_id} on {instance_type}. "
                f"Set inference.image_uri explicitly in config.yaml."
            )

        default_envs: dict[str, str] = {}
        for env in cfg.get("InferenceEnvironmentVariables", []):
            name = env.get("Name")
            default = env.get("Default")
            if name and default is not None:
                default_envs[name] = str(default)
        env_overrides = variant_props.get("EnvironmentVariables", {})
        merged_env = {**default_envs, **{k: str(v) for k, v in env_overrides.items()}}
        merged_env.update(self._build_lora_env())

        resources = variant_props.get("ResourceRequirements", {})
        return {
            "ImageUri": image_uri,
            "ArtifactUri": doc.get("HostingArtifactUri"),
            "ArtifactCompression": doc.get("HostingArtifactCompressionType", "None"),
            "Environment": merged_env,
            "MinMemoryMb": resources.get("MinMemoryMb", 8192),
            "NumAccelerators": resources.get("NumAccelerators", 1),
        }

    def _build_lora_env(self) -> dict[str, str]:
        """Multi-LoRA env vars from our config — layered on top of JumpStart defaults."""
        opts = self.config.inference.option
        return {
            "OPTION_ENABLE_LORA": "true" if opts.enable_lora else "false",
            "OPTION_MAX_LORAS": str(opts.max_loras),
            "OPTION_MAX_CPU_LORAS": str(opts.max_cpu_loras),
        }

    # =====================================================================
    # Internal helpers
    # =====================================================================

    def _ensure_base_model(self, jumpstart: dict) -> str:
        """Create the base SageMaker Model object (idempotent)."""
        model_name = f"{self.config.inference.base_component_name}-model"[:63]
        try:
            self._sm.describe_model(ModelName=model_name)
            logger.info("Base model already exists: %s", model_name)
            return model_name
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise

        logger.info("Creating base model: %s", model_name)
        primary_container: dict = {
            "Image": jumpstart["ImageUri"],
            "Environment": jumpstart["Environment"],
        }
        if jumpstart.get("ArtifactUri"):
            primary_container["ModelDataSource"] = {
                "S3DataSource": {
                    "S3Uri": jumpstart["ArtifactUri"],
                    "S3DataType": "S3Prefix",
                    "CompressionType": jumpstart.get("ArtifactCompression", "None"),
                    "ModelAccessConfig": {"AcceptEula": True},
                }
            }
        self._sm.create_model(
            ModelName=model_name,
            PrimaryContainer=primary_container,
            ExecutionRoleArn=self.config.require_execution_role(),
            EnableNetworkIsolation=False,
        )
        return model_name

    def _ensure_endpoint_config(self, endpoint_name: str) -> str:
        endpoint_config_name = f"{endpoint_name}-config"[:63]
        try:
            self._sm.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            return endpoint_config_name
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise

        self._sm.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ExecutionRoleArn=self.config.require_execution_role(),
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "InstanceType": self.config.inference.instance_type,
                    "InitialInstanceCount": self.config.inference.initial_instance_count,
                    "ModelDataDownloadTimeoutInSeconds": 3600,
                    "ContainerStartupHealthCheckTimeoutInSeconds": 3600,
                    "RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
                },
            ],
        )
        return endpoint_config_name

    def _create_or_update_endpoint(
        self, endpoint_name: str, endpoint_config_name: str
    ) -> None:
        try:
            resp = self._sm.describe_endpoint(EndpointName=endpoint_name)
            status = resp["EndpointStatus"]
            if status == "InService":
                return
            if status in {"Failed", "RollingBack"}:
                self._sm.delete_endpoint(EndpointName=endpoint_name)
                self._wait_for_endpoint_deletion(endpoint_name)
                self._sm.create_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config_name,
                )
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise
            self._sm.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
            )

    def _ensure_base_inference_component(
        self,
        endpoint_name: str,
        component_name: str,
        model_name: str,
        jumpstart: dict,
    ) -> None:
        try:
            resp = self._sm.describe_inference_component(
                InferenceComponentName=component_name
            )
            status = resp.get("InferenceComponentStatus")
            if status == "InService":
                logger.info("Base inference component already InService: %s", component_name)
                return
            if status == "Failed":
                # Delete and recreate so we don't inherit a broken config
                self._sm.delete_inference_component(InferenceComponentName=component_name)
                self._wait_for_ic_deletion(component_name)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise

        logger.info("Creating base inference component: %s", component_name)
        self._sm.create_inference_component(
            InferenceComponentName=component_name,
            EndpointName=endpoint_name,
            VariantName="AllTraffic",
            Specification={
                "ModelName": model_name,
                "ComputeResourceRequirements": {
                    "NumberOfAcceleratorDevicesRequired": jumpstart["NumAccelerators"],
                    "MinMemoryRequiredInMb": jumpstart["MinMemoryMb"],
                },
                "StartupParameters": {
                    "ModelDataDownloadTimeoutInSeconds": 3600,
                    "ContainerStartupHealthCheckTimeoutInSeconds": 3600,
                },
            },
            RuntimeConfig={"CopyCount": 1},
        )
        self._wait_for_inference_component(component_name, target_status="InService")

    def _wait_for_endpoint(
        self, endpoint_name: str, target_status: str, poll_seconds: int = 30
    ) -> None:
        while True:
            resp = self._sm.describe_endpoint(EndpointName=endpoint_name)
            status = resp["EndpointStatus"]
            logger.info("Endpoint %s: %s", endpoint_name, status)
            if status == target_status:
                return
            if status in {"Failed", "RollingBack"}:
                raise RuntimeError(
                    f"Endpoint {endpoint_name} entered failure state: {status}. "
                    f"Reason: {resp.get('FailureReason', 'unknown')}"
                )
            # nosemgrep: arbitrary-sleep - polling interval for SageMaker endpoint status
            time.sleep(poll_seconds)

    def _wait_for_endpoint_deletion(self, endpoint_name: str, poll_seconds: int = 15) -> None:
        while True:
            try:
                self._sm.describe_endpoint(EndpointName=endpoint_name)
                # nosemgrep: arbitrary-sleep - polling interval for SageMaker endpoint deletion
                time.sleep(poll_seconds)
            except ClientError as e:
                if e.response["Error"]["Code"] == "ValidationException":
                    return
                raise

    def _wait_for_ic_deletion(self, component_name: str, poll_seconds: int = 15) -> None:
        while True:
            try:
                self._sm.describe_inference_component(InferenceComponentName=component_name)
                # nosemgrep: arbitrary-sleep - polling interval for SageMaker inference component deletion
                time.sleep(poll_seconds)
            except ClientError as e:
                if e.response["Error"]["Code"] == "ValidationException":
                    return
                raise

    def _wait_for_inference_component(
        self, component_name: str, target_status: str, poll_seconds: int = 20
    ) -> None:
        while True:
            resp = self._sm.describe_inference_component(
                InferenceComponentName=component_name
            )
            status = resp.get("InferenceComponentStatus", "Unknown")
            logger.info("Inference component %s: %s", component_name, status)
            if status == target_status:
                return
            if status in {"Failed", "Deleting"}:
                raise RuntimeError(
                    f"Inference component {component_name} entered state: {status}. "
                    f"Reason: {resp.get('FailureReason', 'unknown')}"
                )
            # nosemgrep: arbitrary-sleep - polling interval for SageMaker inference component status
            time.sleep(poll_seconds)
