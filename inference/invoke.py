"""Thin client library for invoking per-tenant adapter inference components.

Application code that serves traffic should use :class:`MultiTenantClient`.
It enforces the naming convention (Layer 3 isolation) and exposes a small,
typed interface. The actual HTTP call to the SageMaker runtime is still a
``boto3.client('sagemaker-runtime').invoke_endpoint`` call; sagemaker-core v3
wraps the same operation through ``Endpoint.invoke`` but we keep a plain boto3
path here so that this module has minimal startup cost in latency-sensitive
application runtimes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

import boto3

from multi_tenant_lora_sagemaker.common.naming import adapter_component_name
from multi_tenant_lora_sagemaker.common.sessions import default_boto_config
from multi_tenant_lora_sagemaker.config.loader import AppConfig
from multi_tenant_lora_sagemaker.config.schemas import TenantConfig

logger = logging.getLogger(__name__)


class InvocationError(RuntimeError):
    """Raised when invoking the endpoint fails or returns malformed output."""


@dataclass(frozen=True)
class InvocationResult:
    raw: Dict[str, Any]
    generated_text: Optional[str]


class MultiTenantClient:
    """Per-tenant-aware client over the shared multi-adapter endpoint."""

    def __init__(
        self,
        app_config: AppConfig,
        profile: Optional[str] = None,
    ):
        self._config = app_config
        session = boto3.Session(
            region_name=app_config.platform.region, profile_name=profile
        )
        self._runtime = session.client(
            "sagemaker-runtime", config=default_boto_config()
        )

    # ------------------------------------------------------------------

    def _resolve_tenant(self, tenant_id: str) -> TenantConfig:
        return self._config.tenants.get(tenant_id)

    def _resolve_version(
        self, tenant: TenantConfig, task: str, version: Optional[int]
    ) -> int:
        task_cfg = next((t for t in tenant.tasks if t.name == task), None)
        if task_cfg is None:
            raise KeyError(
                f"tenant {tenant.tenant_id!r} has no task named {task!r}"
            )
        return version if version is not None else task_cfg.current_version

    # ------------------------------------------------------------------

    def invoke(
        self,
        tenant_id: str,
        task: str,
        input_text: str,
        parameters: Optional[Mapping[str, Any]] = None,
        version: Optional[int] = None,
    ) -> InvocationResult:
        """Invoke the adapter inference component for one tenant/task.

        Parameters
        ----------
        parameters:
            Dict merged into the inference request (``max_new_tokens``,
            ``temperature``, ``top_p``, ...).
        version:
            When omitted, uses the tenant's ``current_version`` from
            ``tenants.yaml``.
        """
        tenant = self._resolve_tenant(tenant_id)
        resolved_version = self._resolve_version(tenant, task, version)
        component = adapter_component_name(tenant_id, task, resolved_version)
        body: Dict[str, Any] = {"inputs": input_text}
        if parameters:
            body["parameters"] = dict(parameters)

        logger.debug(
            "Invoking endpoint=%s component=%s",
            self._config.platform.endpoint_name,
            component,
        )
        response = self._runtime.invoke_endpoint(
            EndpointName=self._config.platform.endpoint_name,
            InferenceComponentName=component,
            Body=json.dumps(body).encode("utf-8"),
            ContentType="application/json",
            Accept="application/json",
        )
        raw_bytes = response["Body"].read()
        try:
            payload = json.loads(raw_bytes)
        except json.JSONDecodeError as exc:
            raise InvocationError(
                f"invalid JSON in response: {raw_bytes[:256]!r}"
            ) from exc
        generated_text = self._extract_generated_text(payload)
        return InvocationResult(raw=payload, generated_text=generated_text)

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_generated_text(payload: Union[Dict[str, Any], list]) -> Optional[str]:
        """Best-effort extraction across the slight variations emitted by LMI."""
        if isinstance(payload, list) and payload:
            payload = payload[0]
        if not isinstance(payload, dict):
            return None
        if "generated_text" in payload and isinstance(
            payload["generated_text"], str
        ):
            return payload["generated_text"]
        if "generated_texts" in payload and isinstance(
            payload["generated_texts"], list
        ) and payload["generated_texts"]:
            return payload["generated_texts"][0]
        # TGI-style
        if "outputs" in payload and isinstance(payload["outputs"], list):
            out = payload["outputs"][0] if payload["outputs"] else None
            if isinstance(out, dict) and isinstance(out.get("generated_text"), str):
                return out["generated_text"]
        return None
