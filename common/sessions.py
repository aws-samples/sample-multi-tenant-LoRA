"""AWS session helpers.

All boto3 and sagemaker-core sessions used by the package flow through these
helpers so that region, profile and retry configuration are centralised.
"""

from __future__ import annotations

import logging
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)


_DEFAULT_RETRIES = {"max_attempts": 10, "mode": "standard"}


def build_boto_session(
    region: str, profile: Optional[str] = None
) -> boto3.session.Session:
    """Return a ``boto3.Session`` configured for the target region/profile."""
    logger.debug(
        "Creating boto3 session (region=%s, profile=%s)", region, profile
    )
    return boto3.session.Session(region_name=region, profile_name=profile)


def default_boto_config() -> BotoConfig:
    """Shared ``botocore.Config`` with sensible retry defaults."""
    return BotoConfig(retries=_DEFAULT_RETRIES, user_agent_extra="multi-tenant-lora-sagemaker/0.1")


def build_sagemaker_core_session(
    region: str, profile: Optional[str] = None
):
    """Return a ``sagemaker_core.helper.session_helper.Session`` bound to a boto3 session.

    Importing ``sagemaker_core`` is deferred so that config-only operations
    (which do not need the SDK) stay fast and work even when the SDK is not
    installed.
    """
    from sagemaker_core.helper.session_helper import Session as SmCoreSession

    boto_session = build_boto_session(region=region, profile=profile)
    return SmCoreSession(boto_session=boto_session)
