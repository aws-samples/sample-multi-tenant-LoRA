"""Multi-tenant LoRA fine-tuning and inference on Amazon SageMaker AI.

A production-grade, parameterized implementation of per-tenant LoRA adapters
served from a shared SageMaker endpoint. Customers plug in their data and tenant
list via config/config.yaml — everything else is handled by the platform.

Public API:
    PlatformConfig — load and validate config.yaml
    TrainingPipeline — per-tenant training jobs with strict IAM isolation
    InferencePlatform — shared endpoint + per-tenant adapter components
    EvaluationPipeline — test, compare, gate every adapter version
    ScalingManager — auto-scale the shared inference endpoint

CLI:
    python -m multi_tenant_lora --help
"""

from .config import PlatformConfig, TenantConfig
from .training import TrainingPipeline
from .inference import InferencePlatform
from .evaluation import EvaluationPipeline
from .mlops import MLOpsManager
from .scaling import ScalingManager

__all__ = [
    "PlatformConfig",
    "TenantConfig",
    "TrainingPipeline",
    "InferencePlatform",
    "EvaluationPipeline",
    "MLOpsManager",
    "ScalingManager",
]

__version__ = "0.1.0"
