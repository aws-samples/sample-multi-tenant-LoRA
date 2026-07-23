"""Unit tests for the regression gate decision logic."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from multi_tenant_lora_sagemaker.config.loader import load_config
from multi_tenant_lora_sagemaker.evaluation.compare_and_gate import compare_and_gate

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "config"


def _app():
    return load_config(EXAMPLES_DIR)


def test_first_run_approves_when_no_baseline_exists():
    app = _app()
    with mock.patch(
        "multi_tenant_lora_sagemaker.evaluation.compare_and_gate.read_previous_accuracy",
        return_value=None,
    ):
        result = compare_and_gate(
            app_config=app,
            tenant_id="tenant-a",
            task="type-classifier",
            new_accuracy=0.90,
        )
    assert result.decision == "APPROVE"
    assert result.previous_accuracy is None


def test_block_when_regression_exceeds_threshold():
    app = _app()
    with mock.patch(
        "multi_tenant_lora_sagemaker.evaluation.compare_and_gate.read_previous_accuracy",
        return_value=0.92,
    ):
        result = compare_and_gate(
            app_config=app,
            tenant_id="tenant-a",
            task="type-classifier",
            new_accuracy=0.85,  # drop of 0.07 > threshold 0.02
        )
    assert result.decision == "BLOCK"
    assert result.previous_accuracy == 0.92
    assert result.delta is not None
    assert result.delta == pytest.approx(-0.07)


def test_approve_when_within_threshold():
    app = _app()
    with mock.patch(
        "multi_tenant_lora_sagemaker.evaluation.compare_and_gate.read_previous_accuracy",
        return_value=0.90,
    ):
        result = compare_and_gate(
            app_config=app,
            tenant_id="tenant-a",
            task="type-classifier",
            new_accuracy=0.89,  # drop of 0.01 <= threshold 0.02
        )
    assert result.decision == "APPROVE"
