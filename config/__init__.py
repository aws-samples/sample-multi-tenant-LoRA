"""Configuration layer.

The configuration is split across four YAML files:

* ``platform.yaml``  – account-wide settings (region, bucket, MLflow server, ...)
* ``tenants.yaml``   – registry of tenants, their tasks, IAM roles, data paths
* ``base_models.yaml`` – catalog of supported base models + LoRA compatibility
* ``instances.yaml`` – instance-type profiles for training and inference

All files are loaded and validated with Pydantic. Customers edit the YAML; the
rest of the package consumes typed config objects only.
"""

from multi_tenant_lora_sagemaker.config.loader import (
    AppConfig,
    load_config,
)
from multi_tenant_lora_sagemaker.config.schemas import (
    BaseModelEntry,
    BaseModelsConfig,
    InstanceProfile,
    InstancesConfig,
    LoRAHyperparameters,
    MLflowConfig,
    PlatformConfig,
    TenantConfig,
    TenantsConfig,
    TrainingInstanceProfile,
)

__all__ = [
    "AppConfig",
    "load_config",
    "BaseModelEntry",
    "BaseModelsConfig",
    "InstanceProfile",
    "InstancesConfig",
    "LoRAHyperparameters",
    "MLflowConfig",
    "PlatformConfig",
    "TenantConfig",
    "TenantsConfig",
    "TrainingInstanceProfile",
]
