"""End-to-end integration test SCAFFOLD.

This test is deliberately skipped unless the user provides a live config
directory through the ``MULTI_TENANT_LORA_INT_CONFIG`` environment variable
and sets ``MULTI_TENANT_LORA_RUN_INT=1``. When both are present it drives the
full train -> evaluate -> compare -> deploy pipeline against real AWS
infrastructure — the customer supplies their own account, bucket, IAM roles
and tenant data. It is NOT run by default CI.

Usage::

    export MULTI_TENANT_LORA_INT_CONFIG=/path/to/my/config
    export MULTI_TENANT_LORA_RUN_INT=1
    export MULTI_TENANT_LORA_INT_TENANT=tenant-a
    export MULTI_TENANT_LORA_INT_TASK=type-classifier
    export MULTI_TENANT_LORA_INT_VERSION=1
    pytest multi_tenant_lora_sagemaker/tests/test_integration_scaffold.py -s
"""

from __future__ import annotations

import os

import pytest

from multi_tenant_lora_sagemaker.config.loader import load_config
from multi_tenant_lora_sagemaker.orchestration.pipeline import run_pipeline


@pytest.mark.skipif(
    os.environ.get("MULTI_TENANT_LORA_RUN_INT") != "1",
    reason="Integration tests are disabled. Set MULTI_TENANT_LORA_RUN_INT=1 to enable.",
)
def test_end_to_end_pipeline_against_real_aws():
    config_dir = os.environ.get("MULTI_TENANT_LORA_INT_CONFIG")
    if not config_dir:
        pytest.skip("Set MULTI_TENANT_LORA_INT_CONFIG to run this test.")

    tenant_id = os.environ.get("MULTI_TENANT_LORA_INT_TENANT", "tenant-a")
    task = os.environ.get("MULTI_TENANT_LORA_INT_TASK", "type-classifier")
    version = int(os.environ.get("MULTI_TENANT_LORA_INT_VERSION", "1"))

    app = load_config(config_dir)
    result = run_pipeline(
        app_config=app,
        tenant_id=tenant_id,
        task=task,
        version=version,
    )
    assert result.evaluation is not None
    assert result.gate is not None
    # Do not fail the build on BLOCK — that is a legitimate pipeline
    # outcome. The test only fails if the pipeline itself errors out.
