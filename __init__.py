"""Multi-tenant LoRA fine-tuning and inference on Amazon SageMaker AI.

This package implements the architecture described in
``Multi_Tenant_LoRA_SageMaker_Blog_rewritten.md``: per-tenant LoRA training jobs
with IAM-scoped isolation, a shared multi-adapter inference endpoint with
per-tenant adapter inference components, and an evaluation/MLOps pipeline that
treats every adapter version as a testable artifact.

No infrastructure-as-code (CDK/CloudFormation/Terraform) is used; all resources
are managed through the SageMaker SDK v3 (``sagemaker-core``) with a
``boto3`` fallback where v3 does not yet expose the API.
"""

__version__ = "0.1.0"
