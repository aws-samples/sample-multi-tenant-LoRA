"""Validate tenant training data before launching a training job.

Expected layout in S3 (paths produced by :func:`common.naming.training_data_s3_uri`)::

    s3://<bucket>/<tenant_id>/training-data/<task>/train.jsonl
    s3://<bucket>/<tenant_id>/training-data/<task>/validation.jsonl   (optional)

Each record in the JSONL files must be a JSON object with at least an
``instruction`` or ``input`` field and a ``output`` field. This matches what
the entry-point trainer consumes. Customers can extend the schema in their own
config but these two fields are always required.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from multi_tenant_lora_sagemaker.common.sessions import default_boto_config

logger = logging.getLogger(__name__)


class DataValidationError(RuntimeError):
    """Raised when tenant training data fails pre-flight validation."""


@dataclass(frozen=True)
class DataValidationResult:
    train_records: int
    validation_records: Optional[int]
    train_s3_uri: str
    validation_s3_uri: Optional[str]


_REQUIRED_INPUT_FIELDS: Tuple[str, ...] = ("instruction", "input", "prompt")
_REQUIRED_OUTPUT_FIELDS: Tuple[str, ...] = ("output", "completion", "response")


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise DataValidationError(f"expected an s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _list_s3_objects(
    s3_client, bucket: str, prefix: str
) -> List[Tuple[str, int]]:
    paginator = s3_client.get_paginator("list_objects_v2")
    results: List[Tuple[str, int]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            results.append((obj["Key"], obj["Size"]))
    return results


def _iter_jsonl_records(
    s3_client, bucket: str, key: str, max_records: int = 100
) -> Iterable[Any]:
    """Yield up to ``max_records`` records from a JSONL object in S3.

    Uses a ranged GET to avoid streaming multi-GB files.
    """
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        raise DataValidationError(
            f"failed to read s3://{bucket}/{key}: {exc}"
        ) from exc

    count = 0
    body = obj["Body"]
    try:
        for raw_line in body.iter_lines():
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"invalid JSON on line {count + 1} of s3://{bucket}/{key}: {exc}"
                ) from exc
            yield record
            count += 1
            if count >= max_records:
                return
    finally:
        body.close()


def _validate_record(record: Any, index: int, source: str) -> None:
    if not isinstance(record, dict):
        raise DataValidationError(
            f"record {index} in {source} is not a JSON object"
        )
    if not any(f in record for f in _REQUIRED_INPUT_FIELDS):
        raise DataValidationError(
            f"record {index} in {source} is missing an input field "
            f"(one of {_REQUIRED_INPUT_FIELDS})"
        )
    if not any(f in record for f in _REQUIRED_OUTPUT_FIELDS):
        raise DataValidationError(
            f"record {index} in {source} is missing an output field "
            f"(one of {_REQUIRED_OUTPUT_FIELDS})"
        )


def validate_training_data(
    training_data_prefix: str,
    region: str,
    profile: Optional[str] = None,
    sample_records: int = 100,
) -> DataValidationResult:
    """Validate the training data layout and a sample of records.

    * Ensures ``train.jsonl`` exists and is non-empty.
    * Parses and type-checks the first ``sample_records`` records.
    * Reports the presence and record count of an optional ``validation.jsonl``.

    Returns a :class:`DataValidationResult` when validation passes.
    """

    logger.info("Validating training data at %s", training_data_prefix)
    if not training_data_prefix.endswith("/"):
        training_data_prefix = training_data_prefix + "/"

    bucket, prefix = _parse_s3_uri(training_data_prefix)
    s3 = boto3.client("s3", region_name=region, config=default_boto_config())

    objects = _list_s3_objects(s3, bucket, prefix)
    objects_by_name = {key.rsplit("/", 1)[-1]: (key, size) for key, size in objects}

    if "train.jsonl" not in objects_by_name:
        raise DataValidationError(
            f"train.jsonl not found under {training_data_prefix}"
        )
    train_key, train_size = objects_by_name["train.jsonl"]
    if train_size == 0:
        raise DataValidationError(
            f"train.jsonl at s3://{bucket}/{train_key} is empty"
        )

    train_records = 0
    for idx, record in enumerate(
        _iter_jsonl_records(s3, bucket, train_key, max_records=sample_records)
    ):
        _validate_record(record, idx, f"s3://{bucket}/{train_key}")
        train_records += 1

    validation_records: Optional[int] = None
    validation_uri: Optional[str] = None
    if "validation.jsonl" in objects_by_name:
        val_key, val_size = objects_by_name["validation.jsonl"]
        validation_uri = f"s3://{bucket}/{val_key}"
        if val_size == 0:
            raise DataValidationError(
                f"validation.jsonl at {validation_uri} is empty"
            )
        validation_records = 0
        for idx, record in enumerate(
            _iter_jsonl_records(s3, bucket, val_key, max_records=sample_records)
        ):
            _validate_record(record, idx, validation_uri)
            validation_records += 1

    logger.info(
        "Validation passed: train sampled %d records; validation %s",
        train_records,
        f"sampled {validation_records} records" if validation_records is not None else "absent",
    )
    return DataValidationResult(
        train_records=train_records,
        validation_records=validation_records,
        train_s3_uri=f"s3://{bucket}/{train_key}",
        validation_s3_uri=validation_uri,
    )
