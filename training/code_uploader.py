"""Upload the ``entry_point/`` directory as a sourcedir tarball to S3.

The builder passes the resulting ``s3://.../sourcedir.tar.gz`` URI as the
``code`` channel to the training job. The HuggingFace DLC then executes
``train.py`` via the standard SageMaker script-mode bootstrap driven by the
``sagemaker_program`` / ``sagemaker_submit_directory`` hyperparameters the
builder emits.
"""

from __future__ import annotations

import hashlib
import logging
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional

import boto3

from multi_tenant_lora_sagemaker.common.sessions import default_boto_config
from multi_tenant_lora_sagemaker.config.schemas import PlatformConfig
from multi_tenant_lora_sagemaker.training.training_job_builder import entry_point_dir

logger = logging.getLogger(__name__)


def _make_tarball(source_dir: Path, output_path: Path) -> None:
    logger.debug("Creating sourcedir tarball %s", output_path)
    with tarfile.open(output_path, "w:gz") as tar:
        for entry in sorted(source_dir.iterdir()):
            tar.add(entry, arcname=entry.name)


def upload_entry_point(
    platform: PlatformConfig,
    tenant_id: str,
    task: str,
    version: int,
    region: str,
    profile: Optional[str] = None,
) -> str:
    """Tar and upload ``entry_point/`` to ``s3://{bucket}/{tenant}/code/{task}/v{ver}/sourcedir.tar.gz``."""

    source_dir = entry_point_dir()
    tmp_tarball = Path(tempfile.gettempdir()) / (
        f"sourcedir-{tenant_id}-{task}-v{version}-{int(time.time())}.tar.gz"
    )
    _make_tarball(source_dir, tmp_tarball)

    digest = hashlib.sha256(tmp_tarball.read_bytes()).hexdigest()[:12]
    key = (
        f"{tenant_id}/code/{task}/v{version}/sourcedir-{digest}.tar.gz"
    )
    s3 = boto3.Session(region_name=region, profile_name=profile).client(
        "s3", config=default_boto_config()
    )
    logger.info(
        "Uploading training source dir to s3://%s/%s", platform.data_bucket, key
    )
    s3.upload_file(str(tmp_tarball), platform.data_bucket, key)
    return f"s3://{platform.data_bucket}/{key}"
