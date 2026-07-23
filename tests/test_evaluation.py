"""Unit tests for EvaluationPipeline gating logic."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from multi_tenant_lora.config import PlatformConfig
from multi_tenant_lora.evaluation import EvaluationPipeline


@pytest.fixture
def pipeline(config_yaml: Path):
    cfg = PlatformConfig.from_yaml(config_yaml)
    ep = EvaluationPipeline.__new__(EvaluationPipeline)
    ep.config = cfg
    ep._inference = MagicMock()
    ep._cloudwatch = MagicMock()
    ep._s3 = MagicMock()
    ep._session = MagicMock()
    return ep


def test_gate_approves_when_above_threshold_and_no_prev(pipeline) -> None:
    decision, reason = pipeline._gate(0.95, None, None)
    assert decision == "APPROVE"


def test_gate_blocks_below_minimum(pipeline) -> None:
    decision, reason = pipeline._gate(0.50, None, None)
    assert decision == "BLOCK"
    assert "below minimum" in reason


def test_gate_blocks_on_regression(pipeline) -> None:
    # default threshold is 0.02 — a drop of 0.05 should block
    decision, reason = pipeline._gate(0.80, 0.85, -0.05)
    assert decision == "BLOCK"
    assert "regressed" in reason


def test_gate_approves_within_regression_threshold(pipeline) -> None:
    # 0.01 drop is within the 0.02 threshold
    decision, reason = pipeline._gate(0.84, 0.85, -0.01)
    assert decision == "APPROVE"


def test_gate_approves_on_improvement(pipeline) -> None:
    decision, reason = pipeline._gate(0.90, 0.85, 0.05)
    assert decision == "APPROVE"


def test_is_match_case_insensitive(pipeline) -> None:
    assert pipeline._is_match("SEVERITY HIGH", "severity high")
    assert pipeline._is_match("High", "HIGH")


def test_is_match_accepts_substring_of_prediction(pipeline) -> None:
    # Model may emit extra text — expected just needs to appear
    assert pipeline._is_match("The answer is: urgent", "urgent")


def test_is_match_rejects_unrelated(pipeline) -> None:
    assert not pipeline._is_match("low", "high")
