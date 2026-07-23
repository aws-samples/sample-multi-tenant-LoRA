"""Unit tests for the YAML configuration loader and schemas.

These tests do not require AWS credentials or network access; they validate
that the Pydantic schemas accept the example configuration and reject the
common classes of errors we care about.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from multi_tenant_lora_sagemaker.config.loader import ConfigError, load_config

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "config"


def test_example_config_loads_successfully():
    app = load_config(EXAMPLES_DIR)
    assert app.platform.endpoint_name == "itsm-multi-tenant-endpoint"
    assert len(app.tenants.tenants) == 2
    assert len(app.base_models.models) == 2
    assert "default" in app.instances.training
    assert (
        app.instances.inference_profile().instance_type == "ml.g5.12xlarge"
    )


def test_unknown_base_model_is_rejected(tmp_path: Path):
    _copy_examples(tmp_path)
    tenants_path = tmp_path / "tenants.yaml"
    tenants_path.write_text(
        tenants_path.read_text().replace(
            "llama-3-1-8b-instruct", "this-model-does-not-exist"
        )
    )
    with pytest.raises(ConfigError, match="unknown base_model_id"):
        load_config(tmp_path)


def test_invalid_tenant_id_is_rejected(tmp_path: Path):
    _copy_examples(tmp_path)
    tenants_path = tmp_path / "tenants.yaml"
    tenants_path.write_text(
        tenants_path.read_text().replace("tenant_id: tenant-a", "tenant_id: INVALID!")
    )
    with pytest.raises(ConfigError, match="tenant_id"):
        load_config(tmp_path)


def test_missing_default_training_profile_is_rejected(tmp_path: Path):
    _copy_examples(tmp_path)
    instances = tmp_path / "instances.yaml"
    lines = instances.read_text().splitlines()
    # Remove the 'default:' block (first key under training).
    keep = []
    skip = False
    for line in lines:
        if skip and (line.startswith("    ") or not line.strip()):
            continue
        if line.strip() == "default:":
            skip = True
            continue
        skip = False
        keep.append(line)
    instances.write_text("\n".join(keep))
    with pytest.raises(ConfigError, match="training must define a 'default' entry"):
        load_config(tmp_path)


def test_duplicate_tenant_ids_rejected(tmp_path: Path):
    _copy_examples(tmp_path)
    tenants_path = tmp_path / "tenants.yaml"
    tenants_path.write_text(tenants_path.read_text().replace("tenant-b", "tenant-a"))
    with pytest.raises(ConfigError, match="duplicate tenant_id"):
        load_config(tmp_path)


def _copy_examples(dest: Path) -> None:
    for name in ("platform.yaml", "tenants.yaml", "base_models.yaml", "instances.yaml"):
        shutil.copy(EXAMPLES_DIR / name, dest / name)
