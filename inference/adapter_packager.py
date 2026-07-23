"""Package a trained adapter directory into the ``adapter.tar.gz`` format expected by SageMaker.

SageMaker Multi-Adapter Inference expects a gzipped tar archive containing, at
its root, the PEFT adapter files (``adapter_config.json``, ``adapter_model.safetensors`` /
``adapter_model.bin``, optional tokenizer files). When the training entry point
runs inside the DLC it produces this archive automatically, but this module is
useful for:

* Re-packaging an adapter produced outside SageMaker.
* Validating an archive before handing it to the inference component.
"""

from __future__ import annotations

import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Tuple

import boto3

from multi_tenant_lora_sagemaker.common.sessions import default_boto_config

logger = logging.getLogger(__name__)

# Files the LMI container expects inside adapter.tar.gz. ``adapter_model.*``
# may be either ``.safetensors`` (preferred) or ``.bin`` depending on PEFT
# version.
_REQUIRED_FILES: Tuple[str, ...] = ("adapter_config.json",)
_WEIGHT_CANDIDATES: Tuple[str, ...] = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)


class AdapterPackagingError(RuntimeError):
    """Raised when the adapter directory is missing required PEFT files."""


def _collect_files(adapter_dir: Path) -> Iterable[Path]:
    for entry in sorted(adapter_dir.iterdir()):
        if entry.is_file():
            yield entry
        elif entry.is_dir():
            # PEFT may emit a ``default`` subdirectory; archive it verbatim.
            yield entry


def package_adapter(
    adapter_dir: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Validate and tar a PEFT adapter directory into ``adapter.tar.gz``.

    Returns the path to the created tarball.
    """

    adapter_dir = Path(adapter_dir).resolve()
    if not adapter_dir.is_dir():
        raise AdapterPackagingError(f"not a directory: {adapter_dir}")

    present = {p.name for p in adapter_dir.iterdir()}
    missing = [f for f in _REQUIRED_FILES if f not in present]
    if missing:
        raise AdapterPackagingError(
            f"adapter directory {adapter_dir} is missing required files: {missing}"
        )
    if not any(c in present for c in _WEIGHT_CANDIDATES):
        raise AdapterPackagingError(
            f"adapter directory {adapter_dir} has no weight file "
            f"(expected one of {_WEIGHT_CANDIDATES})"
        )

    if output_path is None:
        output_path = Path(tempfile.gettempdir()) / "adapter.tar.gz"
    output_path = Path(output_path)
    logger.info("Packaging adapter %s -> %s", adapter_dir, output_path)
    with tarfile.open(output_path, "w:gz") as tar:
        for entry in _collect_files(adapter_dir):
            tar.add(entry, arcname=entry.name)
    return output_path


def upload_adapter_tarball(
    tarball: Path,
    s3_uri: str,
    region: str,
    profile: Optional[str] = None,
) -> None:
    """Upload a packaged adapter tarball to the exact S3 URI the inference component expects."""

    if not s3_uri.startswith("s3://") or not s3_uri.endswith("adapter.tar.gz"):
        raise ValueError(
            f"s3_uri {s3_uri!r} must be a full s3://.../adapter.tar.gz URI"
        )
    bucket, _, key = s3_uri[len("s3://") :].partition("/")
    s3 = boto3.Session(region_name=region, profile_name=profile).client(
        "s3", config=default_boto_config()
    )
    logger.info("Uploading %s to %s", tarball, s3_uri)
    s3.upload_file(str(tarball), bucket, key)
