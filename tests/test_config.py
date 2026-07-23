"""Unit tests for PlatformConfig."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from multi_tenant_lora.config import PlatformConfig


def test_loads_valid_config(config_yaml: Path) -> None:
    cfg = PlatformConfig.from_yaml(config_yaml)
    assert cfg.aws.region == "us-east-1"
    assert cfg.base_model.name == "llama-3.1-8b"
    assert [t.id for t in cfg.tenants] == ["tenant-a", "tenant-b"]


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        PlatformConfig.from_yaml("/nonexistent/path.yaml")


def test_rejects_invalid_tenant_id(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        """
base_model: {hf_model_id: x, name: y}
tenants:
  - id: "bad id with spaces"
    tasks: [{name: t}]
"""
    )
    with pytest.raises(ValidationError):
        PlatformConfig.from_yaml(p)


def test_rejects_invalid_task_name(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        """
base_model: {hf_model_id: x, name: y}
tenants:
  - id: tenant-a
    tasks: [{name: "bad/task"}]
"""
    )
    with pytest.raises(ValidationError):
        PlatformConfig.from_yaml(p)


def test_lora_for_tenant_merges_overrides(config_yaml: Path) -> None:
    cfg = PlatformConfig.from_yaml(config_yaml)
    a = cfg.lora_for_tenant("tenant-a")
    b = cfg.lora_for_tenant("tenant-b")
    # tenant-b overrides r=8, everything else stays default
    assert a.r == 16
    assert b.r == 8
    assert b.alpha == 32  # inherited from platform default


def test_s3_uri_generation(config_yaml: Path) -> None:
    cfg = PlatformConfig.from_yaml(config_yaml)
    assert (
        cfg.tenant_training_data_uri("tenant-a", "classifier")
        == "s3://test-bucket/tenants/tenant-a/training-data/classifier/"
    )
    assert (
        cfg.tenant_adapter_output_uri("tenant-a", "classifier", "1")
        == "s3://test-bucket/tenants/tenant-a/adapters/classifier/v1/"
    )


def test_inference_component_name_embeds_tenant_id(config_yaml: Path) -> None:
    cfg = PlatformConfig.from_yaml(config_yaml)
    name = cfg.inference_component_name("tenant-a", "classifier", "1")
    assert name == "tenant-a-classifier-v1"
    # Tenant ID comes first — enables IAM prefix-based isolation
    assert name.startswith("tenant-a-")


def test_unknown_tenant_raises(config_yaml: Path) -> None:
    cfg = PlatformConfig.from_yaml(config_yaml)
    with pytest.raises(KeyError):
        cfg.get_tenant("nonexistent")


def test_tenant_region_falls_back_to_default(config_yaml: Path) -> None:
    cfg = PlatformConfig.from_yaml(config_yaml)
    # No residency override — returns platform default
    assert cfg.tenant_region("tenant-a") == "us-east-1"


def test_require_bucket_errors_when_unset(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        """
base_model: {hf_model_id: x, name: y}
"""
    )
    cfg = PlatformConfig.from_yaml(p)
    with pytest.raises(ValueError, match="platform_bucket"):
        cfg.require_bucket()
