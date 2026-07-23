"""Unit tests for the adapter packager.

These tests exercise the pure-Python tar packaging and validation logic
without needing any AWS access.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from multi_tenant_lora_sagemaker.inference.adapter_packager import (
    AdapterPackagingError,
    package_adapter,
)


def _write_valid_adapter(dir_path: Path) -> None:
    (dir_path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 16, "target_modules": ["q_proj"]})
    )
    (dir_path / "adapter_model.safetensors").write_bytes(b"fake-weights")


def test_package_adapter_produces_tarball_with_root_level_files(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_valid_adapter(adapter_dir)

    output = package_adapter(adapter_dir, tmp_path / "adapter.tar.gz")
    assert output.is_file()
    with tarfile.open(output, "r:gz") as tar:
        names = tar.getnames()
    # The LMI container expects PEFT files at the root of the archive.
    assert "adapter_config.json" in names
    assert "adapter_model.safetensors" in names
    assert not any(n.startswith("adapter/") for n in names)


def test_package_adapter_rejects_missing_config(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake")
    with pytest.raises(AdapterPackagingError, match="missing required files"):
        package_adapter(adapter_dir, tmp_path / "adapter.tar.gz")


def test_package_adapter_rejects_missing_weights(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")
    with pytest.raises(AdapterPackagingError, match="no weight file"):
        package_adapter(adapter_dir, tmp_path / "adapter.tar.gz")
