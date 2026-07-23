"""IAM policy template generator.

The blog assumes customers apply tenant-scoped IAM roles manually (no IaC).
This module renders the exact JSON policy documents required for each role
from the validated :class:`AppConfig` so customers can hand them to their
security team or apply them with ``aws iam put-role-policy``.
"""

from multi_tenant_lora_sagemaker.iam.policy_generator import (
    generate_adapter_invoke_policy,
    generate_training_role_policy,
    write_policy_templates,
)

__all__ = [
    "generate_adapter_invoke_policy",
    "generate_training_role_policy",
    "write_policy_templates",
]
