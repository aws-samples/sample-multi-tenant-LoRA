"""Shared pytest fixtures."""
import logging
import sys
from pathlib import Path

import pytest

# Silence noisy loggers during tests
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def config_yaml(tmp_path: Path) -> Path:
    """Minimal valid config YAML for tests."""
    content = """
aws:
  region: us-east-1
  execution_role_arn: arn:aws:iam::123456789012:role/SageMakerExecutionRole
storage:
  platform_bucket: test-bucket
base_model:
  hf_model_id: meta-llama/Meta-Llama-3.1-8B-Instruct
  name: llama-3.1-8b
training:
  instance_type: ml.g5.xlarge
  hyperparameters:
    epochs: 1
tenants:
  - id: tenant-a
    tasks:
      - name: classifier
  - id: tenant-b
    lora_overrides:
      r: 8
    tasks:
      - name: classifier
"""
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p
