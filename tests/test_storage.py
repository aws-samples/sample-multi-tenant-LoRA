"""Unit tests for StorageManager — uses moto to mock S3."""
from __future__ import annotations

from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from multi_tenant_lora.config import PlatformConfig
from multi_tenant_lora.storage import StorageManager


@pytest.fixture
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def test_ensure_bucket_creates_when_missing(config_yaml: Path, aws_credentials) -> None:
    cfg = PlatformConfig.from_yaml(config_yaml)
    # Change to a bucket that doesn't exist
    cfg.storage.platform_bucket = "new-bucket-abc"

    with mock_aws():
        mgr = StorageManager(cfg)
        bucket = mgr.ensure_bucket()
        assert bucket == "new-bucket-abc"

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.head_bucket(Bucket="new-bucket-abc")

        # Secure defaults applied
        enc = s3.get_bucket_encryption(Bucket=bucket)
        rules = enc["ServerSideEncryptionConfiguration"]["Rules"]
        assert rules[0]["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"] == "AES256"


def test_ensure_bucket_idempotent(config_yaml: Path, aws_credentials) -> None:
    cfg = PlatformConfig.from_yaml(config_yaml)

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        mgr = StorageManager(cfg)
        bucket1 = mgr.ensure_bucket()
        bucket2 = mgr.ensure_bucket()
        assert bucket1 == bucket2 == "test-bucket"


def test_upload_training_data(config_yaml: Path, aws_credentials, tmp_path: Path) -> None:
    cfg = PlatformConfig.from_yaml(config_yaml)

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        mgr = StorageManager(cfg)

        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"prompt": "a", "completion": "b"}\n')
        eval_file = tmp_path / "eval.jsonl"
        eval_file.write_text('{"prompt": "c", "completion": "d"}\n')

        result = mgr.upload_training_data(
            tenant_id="tenant-a", task="classifier",
            train_file=train_file, eval_file=eval_file,
        )
        assert "train" in result
        assert "eval" in result
        assert result["train"].endswith("/tenants/tenant-a/training-data/classifier/train.jsonl")
        assert result["eval"].endswith("/tenants/tenant-a/eval-data/classifier/eval.jsonl")

        # Verify the objects actually exist
        s3.head_object(Bucket="test-bucket", Key="tenants/tenant-a/training-data/classifier/train.jsonl")
        s3.head_object(Bucket="test-bucket", Key="tenants/tenant-a/eval-data/classifier/eval.jsonl")


def test_find_adapter_artifact_returns_none_when_absent(
    config_yaml: Path, aws_credentials,
) -> None:
    cfg = PlatformConfig.from_yaml(config_yaml)
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        mgr = StorageManager(cfg)
        assert mgr.find_adapter_artifact("tenant-a", "classifier", "1") is None


def test_find_adapter_artifact_returns_uri_when_present(
    config_yaml: Path, aws_credentials,
) -> None:
    cfg = PlatformConfig.from_yaml(config_yaml)
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        # SDK v3 writes artifacts under a nested job path
        key = (
            "tenants/tenant-a/adapters/classifier/v1/"
            "mtl-tenant-a-classifier-v1-20260101/output/model/checkpoints/hf/"
            "adapter_model.safetensors"
        )
        s3.put_object(Bucket="test-bucket", Key=key, Body=b"fake-archive")

        mgr = StorageManager(cfg)
        uri = mgr.find_adapter_artifact("tenant-a", "classifier", "1")
        assert uri is not None
        # Should point to the directory containing the adapter
        assert uri.endswith("/hf/")
        assert "adapter_model.safetensors" not in uri
