"""Centralised logging configuration.

All CLIs and long-running scripts call :func:`configure_logging` once at startup
so that output is consistent and controllable through a single env var
(``MULTI_TENANT_LORA_LOG_LEVEL``, default ``INFO``).
"""

from __future__ import annotations

import logging
import os
import sys


_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"


def configure_logging(level: str | int | None = None) -> None:
    """Configure the root logger for this package.

    Safe to call multiple times; duplicate handlers are avoided.
    """

    if level is None:
        level = os.environ.get("MULTI_TENANT_LORA_LOG_LEVEL", "INFO")

    root = logging.getLogger()
    if getattr(root, "_multi_tenant_lora_configured", False):
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet a few chatty AWS libraries to WARNING by default.
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    root._multi_tenant_lora_configured = True  # type: ignore[attr-defined]
