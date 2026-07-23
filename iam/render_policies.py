"""CLI helper: render IAM policy templates for every tenant in tenants.yaml."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from multi_tenant_lora_sagemaker.common.logging_setup import configure_logging
from multi_tenant_lora_sagemaker.config.loader import load_config
from multi_tenant_lora_sagemaker.iam.policy_generator import write_policy_templates

logger = logging.getLogger(__name__)


@click.command("render-iam-policies")
@click.option(
    "--config-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
    help="Directory in which to write the JSON policy files.",
)
def cli(config_dir: str, output_dir: str) -> None:
    configure_logging()
    app_config = load_config(config_dir)
    paths = write_policy_templates(app_config, Path(output_dir))
    for p in paths:
        click.echo(str(p))


if __name__ == "__main__":
    cli()
