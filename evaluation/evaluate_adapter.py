"""Run a tenant evaluation JSONL through a deployed adapter and compute accuracy.

The evaluation dataset must be a JSON Lines file with one record per line::

    {"input": "...", "expected": "exact-expected-completion"}

Accuracy is exact-match of the model's ``generated_text`` against ``expected``
after stripping leading/trailing whitespace, matching the blog's definition.
Customers can extend :func:`_score_record` to add fuzzy scoring.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import boto3

from multi_tenant_lora_sagemaker.common.naming import evaluation_data_s3_uri
from multi_tenant_lora_sagemaker.common.sessions import default_boto_config
from multi_tenant_lora_sagemaker.config.loader import AppConfig
from multi_tenant_lora_sagemaker.inference.invoke import MultiTenantClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    tenant_id: str
    task: str
    version: int
    accuracy: float
    correct: int
    total: int
    dataset_s3_uri: str


def _load_eval_records(s3_uri: str, region: str) -> List[Dict[str, Any]]:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected an s3:// URI, got {s3_uri!r}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    s3 = boto3.client("s3", region_name=region, config=default_boto_config())
    response = s3.get_object(Bucket=bucket, Key=key)
    records: List[Dict[str, Any]] = []
    for line_no, raw in enumerate(
        response["Body"].iter_lines(), start=1
    ):
        if not raw:
            continue
        try:
            record = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid JSON on line {line_no} of {s3_uri}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise RuntimeError(
                f"line {line_no} of {s3_uri} is not a JSON object"
            )
        if "input" not in record or "expected" not in record:
            raise RuntimeError(
                f"line {line_no} of {s3_uri} missing 'input' or 'expected' field"
            )
        records.append(record)
    if not records:
        raise RuntimeError(f"evaluation dataset {s3_uri} is empty")
    return records


def _score_record(generated_text: Optional[str], expected: str) -> bool:
    if generated_text is None:
        return False
    return generated_text.strip() == expected.strip()


def evaluate_adapter(
    app_config: AppConfig,
    tenant_id: str,
    task: str,
    version: int,
    inference_parameters: Optional[Dict[str, Any]] = None,
    profile: Optional[str] = None,
) -> EvaluationResult:
    """Invoke the deployed adapter over the tenant's eval dataset and return accuracy."""

    tenant = app_config.tenants.get(tenant_id)
    task_cfg = next((t for t in tenant.tasks if t.name == task), None)
    if task_cfg is None:
        raise KeyError(f"tenant {tenant_id!r} has no task named {task!r}")

    dataset_uri = evaluation_data_s3_uri(
        app_config.platform, tenant_id, task_cfg.evaluation_data_s3_key
    )
    logger.info("Evaluating adapter v%d on %s", version, dataset_uri)
    records = _load_eval_records(dataset_uri, app_config.platform.region)

    client = MultiTenantClient(app_config, profile=profile)
    parameters = inference_parameters or {
        "max_new_tokens": 50,
        "temperature": 0.0,
    }

    correct = 0
    total = len(records)
    for i, record in enumerate(records, start=1):
        result = client.invoke(
            tenant_id=tenant_id,
            task=task,
            input_text=record["input"],
            parameters=parameters,
            version=version,
        )
        if _score_record(result.generated_text, record["expected"]):
            correct += 1
        if i % 50 == 0 or i == total:
            logger.info("Progress: %d/%d (correct=%d)", i, total, correct)

    accuracy = correct / total if total else 0.0
    return EvaluationResult(
        tenant_id=tenant_id,
        task=task,
        version=version,
        accuracy=accuracy,
        correct=correct,
        total=total,
        dataset_s3_uri=dataset_uri,
    )
