"""Unit tests for the per-tenant IAM policy generator.

Validates the two invariants the blog relies on:

* training-role can read the tenant's training-data prefix AND the shared
  base-model prefix, and write only to its own prefixes.
* invoke-role can call ``sagemaker:InvokeEndpoint`` but only for inference
  components whose names start with ``{tenant-id}-``.
"""

from __future__ import annotations

from pathlib import Path

from multi_tenant_lora_sagemaker.config.loader import load_config
from multi_tenant_lora_sagemaker.iam.policy_generator import (
    generate_adapter_invoke_policy,
    generate_training_role_policy,
    write_policy_templates,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "config"


def _load():
    return load_config(EXAMPLES_DIR)


def test_training_policy_scopes_to_tenant_prefix():
    app = _load()
    tenant = app.tenants.get("tenant-a")
    doc = generate_training_role_policy(app.platform, tenant)
    statements = doc["Statement"]
    read_stmt = next(s for s in statements if s["Sid"] == "ReadTrainingAndBaseModel")
    # Read must include tenant training prefix AND shared base model.
    assert any(
        "tenant-a/training-data/" in r for r in read_stmt["Resource"]
    )
    assert any("base-model/" in r for r in read_stmt["Resource"])
    # Read must NOT include another tenant's prefix.
    assert not any("tenant-b/" in r for r in read_stmt["Resource"])

    write_stmt = next(s for s in statements if s["Sid"] == "WriteAdaptersAndOutputs")
    for r in write_stmt["Resource"]:
        assert r.startswith("arn:aws:s3:::")
        assert "tenant-a/" in r
        assert "tenant-b" not in r


def test_invoke_policy_conditions_on_component_name():
    app = _load()
    tenant = app.tenants.get("tenant-a")
    doc = generate_adapter_invoke_policy(app.platform, tenant)
    stmt = doc["Statement"][0]
    assert "sagemaker:InvokeEndpoint" in stmt["Action"]
    allowed = stmt["Condition"]["StringLike"]["sagemaker:InferenceComponentName"]
    # Condition must only allow names that start with the tenant's id.
    assert all(name.startswith("tenant-a-") for name in allowed)
    # Endpoint ARN must reference the shared endpoint from platform.yaml.
    assert stmt["Resource"].endswith(f":endpoint/{app.platform.endpoint_name}")


def test_write_policy_templates_creates_one_file_per_role_per_tenant(tmp_path: Path):
    app = _load()
    written = write_policy_templates(app, tmp_path)
    # 2 tenants × 2 roles = 4 files.
    assert len(written) == 4
    names = {p.name for p in written}
    assert "tenant-a-training-policy.json" in names
    assert "tenant-a-invoke-policy.json" in names
    assert "tenant-b-training-policy.json" in names
    assert "tenant-b-invoke-policy.json" in names
