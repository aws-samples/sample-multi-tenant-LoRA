"""Evaluation pipeline — every adapter version is a testable artifact.

Implements the blog's "integration test with regression alarms" pattern:
  1. Run the newly trained adapter against a held-out eval dataset
  2. Compute accuracy vs. expected answers
  3. Compare to the previous version's accuracy
  4. BLOCK or APPROVE deployment based on configurable thresholds
  5. Publish metric + fire CloudWatch alarm on regression
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from .config import PlatformConfig
from .inference import InferencePlatform

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    tenant_id: str
    task: str
    version: str
    accuracy: float
    correct: int
    total: int
    decision: str            # "APPROVE" | "BLOCK"
    reason: str
    previous_accuracy: float | None
    delta: float | None


class EvaluationPipeline:
    """Evaluate adapter versions and gate production deployment."""

    def __init__(
        self,
        config: PlatformConfig,
        inference: InferencePlatform | None = None,
        session: boto3.Session | None = None,
    ):
        self.config = config
        self._session = session or boto3.Session(region_name=config.aws.region)
        self._inference = inference or InferencePlatform(config, session=self._session)
        self._cloudwatch = self._session.client("cloudwatch")
        self._s3 = self._session.client("s3")

    # -------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------

    def evaluate_and_gate(
        self,
        tenant_id: str,
        task: str,
        version: str,
        *,
        eval_data_uri: str | None = None,
    ) -> EvaluationResult:
        """Run the full eval+gate cycle for a single adapter version.

        Args:
            tenant_id, task, version: Adapter to evaluate.
            eval_data_uri: Optional override. Defaults to the tenant's standard
                eval-data prefix from config.

        Returns:
            EvaluationResult with APPROVE or BLOCK decision.
        """
        eval_data_uri = eval_data_uri or self.config.tenant_eval_data_uri(tenant_id, task)
        records = self._load_eval_dataset(eval_data_uri)
        if not records:
            raise ValueError(
                f"No eval records found at {eval_data_uri}. "
                f"Upload an eval.jsonl before running evaluation."
            )

        # Cap at configured sample size
        records = records[: self.config.evaluation.eval_sample_size]
        logger.info(
            "Evaluating tenant=%s task=%s version=%s on %d records",
            tenant_id, task, version, len(records),
        )

        correct = 0
        failures = []
        for i, rec in enumerate(records):
            prompt = rec["prompt"]
            expected = rec["completion"]
            try:
                predicted = self._inference.invoke(
                    tenant_id=tenant_id,
                    task=task,
                    version=version,
                    prompt=prompt,
                    max_new_tokens=self.config.evaluation.max_new_tokens,
                    temperature=self.config.evaluation.temperature,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Inference failed on example %d: %s", i, e)
                failures.append({"index": i, "error": str(e)})
                continue

            if self._is_match(predicted, expected):
                correct += 1

        total = len(records)
        accuracy = correct / total if total else 0.0

        # Compare to previous
        previous_accuracy = self._get_previous_accuracy(tenant_id, task, version)
        delta = (accuracy - previous_accuracy) if previous_accuracy is not None else None

        # Gate decision
        decision, reason = self._gate(accuracy, previous_accuracy, delta)

        result = EvaluationResult(
            tenant_id=tenant_id,
            task=task,
            version=version,
            accuracy=accuracy,
            correct=correct,
            total=total,
            decision=decision,
            reason=reason,
            previous_accuracy=previous_accuracy,
            delta=delta,
        )

        # Persist + publish
        self._record_result(result)
        self._publish_metric(result)

        logger.info(
            "Evaluation result: %s accuracy=%.3f (prev=%s delta=%s) decision=%s",
            result.version,
            result.accuracy,
            f"{previous_accuracy:.3f}" if previous_accuracy is not None else "n/a",
            f"{delta:+.3f}" if delta is not None else "n/a",
            result.decision,
        )
        return result

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    def _load_eval_dataset(self, s3_uri: str) -> list[dict]:
        bucket, key_prefix = self._parse_s3_uri(s3_uri)
        resp = self._s3.list_objects_v2(Bucket=bucket, Prefix=key_prefix)
        records: list[dict] = []
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".jsonl"):
                continue
            body = self._s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
            for line_num, line in enumerate(body.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line %d of %s", line_num, key)
        return records

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple[str, str]:
        assert uri.startswith("s3://"), f"Invalid S3 URI: {uri}"
        _, _, rest = uri.partition("s3://")
        bucket, _, key = rest.partition("/")
        return bucket, key

    @staticmethod
    def _is_match(predicted: str, expected: str) -> bool:
        """Case- and whitespace-insensitive match. The prediction must contain
        the expected label — this handles cases where the model emits extra text."""
        p = EvaluationPipeline._normalize(predicted)
        e = EvaluationPipeline._normalize(expected)
        return e in p or p == e

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _gate(
        self,
        accuracy: float,
        previous_accuracy: float | None,
        delta: float | None,
    ) -> tuple[str, str]:
        cfg = self.config.evaluation
        if accuracy < cfg.min_absolute_accuracy:
            return "BLOCK", (
                f"Accuracy {accuracy:.3f} below minimum "
                f"{cfg.min_absolute_accuracy:.3f}"
            )
        if delta is not None and delta < -cfg.accuracy_regression_threshold:
            return "BLOCK", (
                f"Accuracy regressed by {abs(delta):.3f} "
                f"(threshold: {cfg.accuracy_regression_threshold:.3f})"
            )
        return "APPROVE", "Accuracy meets thresholds"

    def _get_previous_accuracy(
        self, tenant_id: str, task: str, current_version: str
    ) -> float | None:
        """Look up the most recent past accuracy for this tenant+task."""
        bucket = self.config.require_bucket()
        prefix = f"{self.config.tenant_prefix(tenant_id)}/evaluations/{task}/"
        try:
            resp = self._s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        except ClientError:
            return None
        candidates = []
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            # Skip the current version when looking up "previous"
            if f"/v{current_version}/" in key:
                continue
            candidates.append(obj)
        if not candidates:
            return None
        # Pick the most recent one
        latest = max(candidates, key=lambda o: o["LastModified"])
        body = self._s3.get_object(Bucket=bucket, Key=latest["Key"])["Body"].read()
        try:
            data = json.loads(body)
            return float(data.get("accuracy"))
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def _record_result(self, result: EvaluationResult) -> None:
        bucket = self.config.require_bucket()
        key = (
            f"{self.config.tenant_prefix(result.tenant_id)}/evaluations/"
            f"{result.task}/v{result.version}/result.json"
        )
        body = json.dumps(
            {
                "tenant_id": result.tenant_id,
                "task": result.task,
                "version": result.version,
                "accuracy": result.accuracy,
                "correct": result.correct,
                "total": result.total,
                "decision": result.decision,
                "reason": result.reason,
                "previous_accuracy": result.previous_accuracy,
                "delta": result.delta,
                "recorded_at": int(time.time()),
            },
            indent=2,
        ).encode("utf-8")
        self._s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")

    def _publish_metric(self, result: EvaluationResult) -> None:
        """Publish the accuracy metric to CloudWatch for alarm monitoring."""
        cw = self.config.evaluation.cloudwatch
        try:
            self._cloudwatch.put_metric_data(
                Namespace=cw.alarm_namespace,
                MetricData=[
                    {
                        "MetricName": "AdapterAccuracy",
                        "Dimensions": [
                            {"Name": "TenantId", "Value": result.tenant_id},
                            {"Name": "Task", "Value": result.task},
                        ],
                        "Value": result.accuracy,
                        "Unit": "None",
                    }
                ],
            )
        except ClientError as e:
            logger.warning("Failed to publish CloudWatch metric: %s", e)
