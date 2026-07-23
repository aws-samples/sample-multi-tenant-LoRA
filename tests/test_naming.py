"""Unit tests for naming helpers (Layer 3 isolation enforcement)."""

from __future__ import annotations

import pytest

from multi_tenant_lora_sagemaker.common.naming import (
    adapter_component_name,
    adapter_s3_prefix,
    adapter_tarball_s3_uri,
    base_model_s3_uri,
    evaluation_data_s3_uri,
    training_data_s3_uri,
    training_job_name,
    training_output_s3_uri,
)
from multi_tenant_lora_sagemaker.config.schemas import PlatformConfig


@pytest.fixture()
def platform() -> PlatformConfig:
    return PlatformConfig(
        aws_account_id="123456789012",
        region="us-east-1",
        data_bucket="platform-data",
        endpoint_name="itsm-multi-tenant-endpoint",
        base_inference_component_name="itsm-base-model",
    )


def test_adapter_component_name_embeds_tenant_and_version():
    assert (
        adapter_component_name("tenant-a", "type-classifier", 3)
        == "tenant-a-type-classifier-v3"
    )


def test_adapter_component_name_rejects_invalid():
    with pytest.raises(ValueError):
        adapter_component_name("TENANT_WITH_UNDERSCORE", "task", 1)


def test_training_job_name_suffix():
    assert (
        training_job_name("tenant-b", "severity", 2)
        == "tenant-b-severity-v2-train"
    )


def test_s3_helpers_match_blog_layout(platform: PlatformConfig):
    assert (
        training_data_s3_uri(platform, "tenant-a", "type-classifier")
        == "s3://platform-data/tenant-a/training-data/type-classifier/"
    )
    assert (
        adapter_s3_prefix(platform, "tenant-a", "type-classifier", 3)
        == "s3://platform-data/tenant-a/adapters/type-classifier/v3/"
    )
    assert (
        adapter_tarball_s3_uri(platform, "tenant-a", "type-classifier", 3)
        == "s3://platform-data/tenant-a/adapters/type-classifier/v3/adapter.tar.gz"
    )
    assert (
        training_output_s3_uri(platform, "tenant-a", "type-classifier", 3)
        == "s3://platform-data/tenant-a/training-output/type-classifier/v3/"
    )
    assert (
        evaluation_data_s3_uri(platform, "tenant-a", "type-classifier/eval.jsonl")
        == "s3://platform-data/tenant-a/evaluation-data/type-classifier/eval.jsonl"
    )
    assert (
        base_model_s3_uri(platform, "llama-3.1-8b-instruct/")
        == "s3://platform-data/base-model/llama-3.1-8b-instruct/"
    )
