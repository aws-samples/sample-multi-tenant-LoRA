"""YAML loader that produces a validated ``AppConfig`` object."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import ValidationError

from multi_tenant_lora_sagemaker.config.schemas import (
    BaseModelsConfig,
    InstancesConfig,
    PlatformConfig,
    TenantsConfig,
)

logger = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised when configuration files are missing or fail validation."""


@dataclass(frozen=True)
class AppConfig:
    """Aggregated, validated configuration bundle consumed by the rest of the package."""

    platform: PlatformConfig
    tenants: TenantsConfig
    base_models: BaseModelsConfig
    instances: InstancesConfig


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"config file {path} is empty")
    if not isinstance(data, dict):
        raise ConfigError(
            f"config file {path} must contain a YAML mapping at the top level, "
            f"got {type(data).__name__}"
        )
    return data


def _load_model(
    path: Path, model_cls: type, *, unwrap_key: Optional[str] = None
) -> Any:
    data = _read_yaml(path)
    if unwrap_key is not None and set(data.keys()) == {unwrap_key}:
        # Allow either a single top-level list or a wrapped mapping.
        data = {unwrap_key: data[unwrap_key]}
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            f"validation error in {path}:\n{exc}"
        ) from exc


def load_config(config_dir: Union[str, Path]) -> AppConfig:
    """Load all four YAML files from ``config_dir`` and return a typed bundle.

    The directory must contain:

    * ``platform.yaml``
    * ``tenants.yaml``
    * ``base_models.yaml``
    * ``instances.yaml``

    Every file is validated against its Pydantic schema; a ``ConfigError`` is
    raised with a detailed message if anything is invalid.
    """

    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise ConfigError(f"config directory not found: {config_dir}")

    logger.info("Loading configuration from %s", config_dir)

    platform = _load_model(config_dir / "platform.yaml", PlatformConfig)
    tenants = _load_model(config_dir / "tenants.yaml", TenantsConfig)
    base_models = _load_model(config_dir / "base_models.yaml", BaseModelsConfig)
    instances = _load_model(config_dir / "instances.yaml", InstancesConfig)

    # Cross-file validation: every tenant task must reference a known base model.
    known_model_ids = {m.model_id for m in base_models.models}
    for tenant in tenants.tenants:
        for task in tenant.tasks:
            if task.base_model_id not in known_model_ids:
                raise ConfigError(
                    f"tenant {tenant.tenant_id!r} task {task.name!r} "
                    f"references unknown base_model_id {task.base_model_id!r}"
                )

    logger.info(
        "Loaded %d tenants, %d base models, %d inference profiles",
        len(tenants.tenants),
        len(base_models.models),
        len(instances.inference),
    )
    return AppConfig(
        platform=platform,
        tenants=tenants,
        base_models=base_models,
        instances=instances,
    )
