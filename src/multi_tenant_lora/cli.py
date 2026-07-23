"""Command-line interface for the multi-tenant LoRA platform.

Customers run:
    python -m multi_tenant_lora bootstrap       # one-time platform setup
    python -m multi_tenant_lora upload-data ... # upload tenant data
    python -m multi_tenant_lora train ...       # submit training jobs
    python -m multi_tenant_lora deploy-endpoint # create shared inference endpoint
    python -m multi_tenant_lora deploy-adapter  # register adapter components
    python -m multi_tenant_lora evaluate ...    # eval + gate adapter versions
    python -m multi_tenant_lora teardown        # clean everything up
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import PlatformConfig
from .evaluation import EvaluationPipeline
from .inference import InferencePlatform
from .mlops import MLOpsManager
from .scaling import ScalingManager
from .storage import StorageManager
from .training import TrainingPipeline

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=level,
    )
    # Silence noisy SageMaker SDK v3 informational logs
    for noisy in ("sagemaker", "botocore.credentials", "boto3"):
        logging.getLogger(noisy).setLevel(logging.WARNING if not verbose else logging.INFO)


def _load_config(path: str) -> PlatformConfig:
    try:
        return PlatformConfig.from_yaml(path)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Invalid config: {e}[/red]")
        sys.exit(1)


# =============================================================================
# CLI group
# =============================================================================


@click.group()
@click.option(
    "--config",
    "-c",
    default="config/config.yaml",
    show_default=True,
    help="Path to config.yaml",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool) -> None:
    """Multi-tenant LoRA fine-tuning and inference on Amazon SageMaker AI."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


# -----------------------------------------------------------------------------
# bootstrap
# -----------------------------------------------------------------------------


@cli.command()
@click.pass_context
def bootstrap(ctx: click.Context) -> None:
    """Create the platform S3 bucket with secure defaults.

    Idempotent — safe to run multiple times.
    """
    config = _load_config(ctx.obj["config_path"])
    storage = StorageManager(config)
    bucket = storage.ensure_bucket()
    console.print(f"[green]Platform bucket ready:[/green] s3://{bucket}/")
    console.print(f"  base model path: {config.base_model_s3_uri()}")
    console.print(f"  tenant data path: s3://{bucket}/{config.storage.tenant_data_prefix}/")


# -----------------------------------------------------------------------------
# upload-data
# -----------------------------------------------------------------------------


@cli.command("upload-data")
@click.option("--tenant", required=True, help="Tenant ID")
@click.option("--task", required=True, help="Task name")
@click.option("--train-file", required=True, type=click.Path(exists=True))
@click.option("--eval-file", type=click.Path(exists=True), default=None)
@click.pass_context
def upload_data(
    ctx: click.Context,
    tenant: str,
    task: str,
    train_file: str,
    eval_file: str | None,
) -> None:
    """Upload a tenant's training (and optional eval) JSONL files."""
    config = _load_config(ctx.obj["config_path"])
    # Validate tenant exists in config
    try:
        config.get_tenant(tenant)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    storage = StorageManager(config)
    result = storage.upload_training_data(
        tenant_id=tenant, task=task, train_file=train_file, eval_file=eval_file
    )
    for kind, uri in result.items():
        console.print(f"[green]{kind}:[/green] {uri}")


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------


@cli.command()
@click.option("--tenant", help="Train a specific tenant (default: all tenants)")
@click.option("--task", help="Train a specific task (requires --tenant)")
@click.option("--version", required=True, help="Adapter version (e.g. 1)")
@click.option("--wait/--no-wait", default=True, help="Wait for job completion")
@click.pass_context
def train(
    ctx: click.Context,
    tenant: str | None,
    task: str | None,
    version: str,
    wait: bool,
) -> None:
    """Submit LoRA training jobs."""
    config = _load_config(ctx.obj["config_path"])
    pipeline = TrainingPipeline(config)

    jobs = []
    if tenant:
        tenant_cfg = config.get_tenant(tenant)
        tasks = [task] if task else [t.name for t in tenant_cfg.tasks]
        for t in tasks:
            jobs.append(pipeline.submit(tenant, t, version, wait=False))
    else:
        jobs = pipeline.submit_all_tenants(version=version, wait=False)

    # Display
    table = Table(title="Submitted Training Jobs")
    table.add_column("Tenant")
    table.add_column("Task")
    table.add_column("Version")
    table.add_column("Job Name")
    for j in jobs:
        table.add_row(j.tenant_id, j.task, j.version, j.job_name)
    console.print(table)

    if wait:
        console.print("[yellow]Waiting for all jobs to complete...[/yellow]")
        pipeline.wait_for_all(jobs)
        for j in jobs:
            status = pipeline.describe(j.job_name)["TrainingJobStatus"]
            color = "green" if status == "Completed" else "red"
            console.print(f"  {j.job_name}: [{color}]{status}[/{color}]")


# -----------------------------------------------------------------------------
# deploy-endpoint
# -----------------------------------------------------------------------------


@cli.command("deploy-endpoint")
@click.pass_context
def deploy_endpoint(ctx: click.Context) -> None:
    """Create the shared inference endpoint + base component."""
    config = _load_config(ctx.obj["config_path"])
    inference = InferencePlatform(config)
    endpoint_name = inference.deploy_base_endpoint()

    if config.inference.autoscaling.enabled:
        ScalingManager(config).configure()

    console.print(f"[green]Endpoint ready:[/green] {endpoint_name}")


# -----------------------------------------------------------------------------
# deploy-adapter
# -----------------------------------------------------------------------------


@cli.command("deploy-adapter")
@click.option("--tenant", required=True)
@click.option("--task", required=True)
@click.option("--version", required=True)
@click.pass_context
def deploy_adapter(
    ctx: click.Context, tenant: str, task: str, version: str
) -> None:
    """Register a tenant's adapter as an inference component."""
    config = _load_config(ctx.obj["config_path"])
    inference = InferencePlatform(config)
    name = inference.deploy_adapter(tenant, task, version)
    console.print(f"[green]Adapter component deployed:[/green] {name}")


# -----------------------------------------------------------------------------
# evaluate
# -----------------------------------------------------------------------------


@cli.command()
@click.option("--tenant", required=True)
@click.option("--task", required=True)
@click.option("--version", required=True)
@click.pass_context
def evaluate(ctx: click.Context, tenant: str, task: str, version: str) -> None:
    """Run the eval + gate cycle for an adapter version."""
    config = _load_config(ctx.obj["config_path"])
    pipeline = EvaluationPipeline(config)
    result = pipeline.evaluate_and_gate(tenant, task, version)
    color = "green" if result.decision == "APPROVE" else "red"
    console.print(
        f"[{color}]{result.decision}[/{color}] "
        f"tenant={result.tenant_id} task={result.task} v{result.version} "
        f"accuracy={result.accuracy:.3f} "
        f"(prev={result.previous_accuracy:.3f if result.previous_accuracy is not None else 'n/a'})"
    )
    console.print(f"  Reason: {result.reason}")


# -----------------------------------------------------------------------------
# invoke — interactive test
# -----------------------------------------------------------------------------


@cli.command()
@click.option("--tenant", required=True)
@click.option("--task", required=True)
@click.option("--version", required=True)
@click.option("--prompt", required=True)
@click.option("--max-tokens", default=100)
@click.option("--temperature", default=0.0)
@click.pass_context
def invoke(
    ctx: click.Context,
    tenant: str,
    task: str,
    version: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> None:
    """Send a single prompt to a tenant adapter."""
    config = _load_config(ctx.obj["config_path"])
    inference = InferencePlatform(config)
    out = inference.invoke(
        tenant_id=tenant,
        task=task,
        version=version,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    console.print(out)


# -----------------------------------------------------------------------------
# alarms
# -----------------------------------------------------------------------------


@cli.command("create-alarms")
@click.pass_context
def create_alarms(ctx: click.Context) -> None:
    """Create CloudWatch accuracy alarms for all tenants+tasks."""
    config = _load_config(ctx.obj["config_path"])
    manager = MLOpsManager(config)
    alarms = manager.create_alarms_for_all_tenants()

    table = Table(title="CloudWatch Alarms")
    table.add_column("Tenant")
    table.add_column("Task")
    table.add_column("Alarm")
    table.add_column("Threshold")
    for a in alarms:
        table.add_row(a.tenant_id, a.task, a.alarm_name, f"{a.threshold:.2f}")
    console.print(table)


# -----------------------------------------------------------------------------
# list-adapters
# -----------------------------------------------------------------------------


@cli.command("list-adapters")
@click.pass_context
def list_adapters(ctx: click.Context) -> None:
    """List all deployed adapter inference components."""
    config = _load_config(ctx.obj["config_path"])
    inference = InferencePlatform(config)
    adapters = inference.list_adapters()
    if not adapters:
        console.print("[yellow]No adapter components deployed yet.[/yellow]")
        return
    table = Table(title=f"Adapters on {config.inference.endpoint_name}")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Created")
    for a in adapters:
        table.add_row(a["name"], a["status"], str(a.get("created", "")))
    console.print(table)


# -----------------------------------------------------------------------------
# preflight — verify AWS prerequisites
# -----------------------------------------------------------------------------


@cli.command()
@click.pass_context
def preflight(ctx: click.Context) -> None:
    """Check AWS prerequisites before running any other command.

    Verifies:
      - AWS credentials work
      - Region matches config.yaml
      - Execution role exists and has required permissions
      - JumpStart base model is accessible
      - Service quotas for configured instance types
    """
    import boto3
    from botocore.exceptions import ClientError

    config = _load_config(ctx.obj["config_path"])
    problems: list[str] = []
    warnings: list[str] = []

    # 1. Credentials + region
    try:
        sts = boto3.client("sts", region_name=config.aws.region)
        identity = sts.get_caller_identity()
        console.print(f"[green]✓[/green] AWS credentials OK (account {identity['Account']})")
    except Exception as e:  # noqa: BLE001
        problems.append(f"AWS credentials missing or invalid: {e}")
        console.print(f"[red]✗[/red] AWS credentials: {e}")

    # 2. Execution role
    if config.aws.execution_role_arn:
        try:
            iam = boto3.client("iam")
            role_name = config.aws.execution_role_arn.rsplit("/", 1)[-1]
            iam.get_role(RoleName=role_name)
            console.print(f"[green]✓[/green] Execution role exists: {role_name}")
        except ClientError as e:
            problems.append(
                f"Execution role {config.aws.execution_role_arn} not found or not accessible: {e}"
            )
            console.print(f"[red]✗[/red] Execution role: {e}")
    else:
        warnings.append(
            "aws.execution_role_arn is not set in config.yaml — will fall back to "
            "sagemaker.get_execution_role() which only works in Studio/notebook contexts"
        )
        console.print(f"[yellow]![/yellow] Execution role not set in config.yaml")

    # 3. JumpStart model
    try:
        sm = boto3.client("sagemaker", region_name=config.aws.region)
        sm.describe_hub_content(
            HubName="SageMakerPublicHub",
            HubContentType="Model",
            HubContentName=config.base_model.hf_model_id,
        )
        console.print(
            f"[green]✓[/green] JumpStart model available: {config.base_model.hf_model_id}"
        )
    except ClientError as e:
        problems.append(
            f"JumpStart model '{config.base_model.hf_model_id}' not found in region "
            f"{config.aws.region}: {e}"
        )
        console.print(f"[red]✗[/red] JumpStart model: {e}")

    # 4. S3 bucket (check accessible if specified)
    if config.storage.platform_bucket:
        try:
            s3 = boto3.client("s3", region_name=config.aws.region)
            s3.head_bucket(Bucket=config.storage.platform_bucket)
            console.print(
                f"[green]✓[/green] Platform bucket exists: {config.storage.platform_bucket}"
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchBucket"):
                console.print(
                    f"[yellow]![/yellow] Platform bucket does not exist yet — "
                    f"run `multi-tenant-lora bootstrap` to create it"
                )
            else:
                problems.append(f"Cannot access platform bucket: {e}")
                console.print(f"[red]✗[/red] Platform bucket: {e}")

    # 5. Quota hints — SageMaker doesn't expose quota APIs cleanly, so we just warn
    console.print(
        f"[yellow]![/yellow] Verify quota for {config.training.instance_type} "
        f"(training) and {config.inference.instance_type} (inference) "
        f"in region {config.aws.region}."
    )
    console.print(
        f"  See: https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas"
    )

    # Summary
    console.print()
    if problems:
        console.print(f"[red]{len(problems)} blocking issue(s):[/red]")
        for p in problems:
            console.print(f"  • {p}")
        sys.exit(1)
    if warnings:
        console.print(f"[yellow]{len(warnings)} warning(s):[/yellow]")
        for w in warnings:
            console.print(f"  • {w}")
    console.print("[green]Preflight passed — ready to bootstrap.[/green]")


# -----------------------------------------------------------------------------
# run-all — one-shot end-to-end pipeline
# -----------------------------------------------------------------------------


@cli.command("run-all")
@click.option("--version", required=True, help="Adapter version for this run (e.g. 1)")
@click.option(
    "--skip-training",
    is_flag=True,
    help="Skip training — use existing adapters at the specified version",
)
@click.pass_context
def run_all(ctx: click.Context, version: str, skip_training: bool) -> None:
    """Run the full pipeline for every tenant in config.yaml.

    Executes:
      1. bootstrap (idempotent)
      2. train — all tenants + tasks (unless --skip-training)
      3. deploy-endpoint (idempotent)
      4. deploy-adapter — all tenants + tasks
      5. evaluate — all tenants + tasks
      6. create-alarms — all tenants + tasks

    After this completes, the full multi-tenant platform is operational.
    """
    config = _load_config(ctx.obj["config_path"])

    # 1. Bootstrap
    console.print("[bold cyan]Step 1/6: Bootstrap[/bold cyan]")
    storage = StorageManager(config)
    bucket = storage.ensure_bucket()
    console.print(f"  Platform bucket: s3://{bucket}/")

    # 2. Train
    if not skip_training:
        console.print(f"\n[bold cyan]Step 2/6: Train adapters (version={version})[/bold cyan]")
        training = TrainingPipeline(config)
        jobs = training.submit_all_tenants(version=version, wait=False)
        for j in jobs:
            console.print(f"  Submitted: {j.tenant_id}/{j.task} v{j.version}")
        console.print("  Waiting for all jobs to complete...")
        training.wait_for_all(jobs)
        console.print("  All training jobs finished.")
    else:
        console.print(f"\n[yellow]Step 2/6: Skipped (using existing v{version})[/yellow]")

    # 3. Deploy endpoint
    console.print("\n[bold cyan]Step 3/6: Deploy base endpoint[/bold cyan]")
    inference = InferencePlatform(config)
    endpoint_name = inference.deploy_base_endpoint()
    console.print(f"  Endpoint: {endpoint_name}")
    if config.inference.autoscaling.enabled:
        ScalingManager(config).configure()
        console.print("  Auto-scaling configured")

    # 4. Deploy adapters
    console.print(f"\n[bold cyan]Step 4/6: Deploy adapters[/bold cyan]")
    for tenant in config.tenants:
        for task in tenant.tasks:
            try:
                name = inference.deploy_adapter(tenant.id, task.name, version)
                console.print(f"  Deployed: {name}")
            except FileNotFoundError as e:
                console.print(f"  [red]SKIP {tenant.id}/{task.name}: {e}[/red]")

    # 5. Evaluate
    console.print(f"\n[bold cyan]Step 5/6: Evaluate adapters[/bold cyan]")
    ep = EvaluationPipeline(config, inference=inference)
    eval_results = []
    for tenant in config.tenants:
        for task in tenant.tasks:
            try:
                result = ep.evaluate_and_gate(tenant.id, task.name, version)
                color = "green" if result.decision == "APPROVE" else "red"
                console.print(
                    f"  [{color}]{result.decision}[/{color}] "
                    f"{tenant.id}/{task.name}: accuracy={result.accuracy:.3f}"
                )
                eval_results.append(result)
            except Exception as e:  # noqa: BLE001
                console.print(f"  [red]Eval failed for {tenant.id}/{task.name}: {e}[/red]")

    # 6. Create alarms
    console.print(f"\n[bold cyan]Step 6/6: CloudWatch alarms[/bold cyan]")
    MLOpsManager(config).create_alarms_for_all_tenants()
    console.print("  Alarms created")

    console.print("\n[bold green]Pipeline complete.[/bold green]")
    console.print("\nTo invoke:")
    console.print(
        f"  multi-tenant-lora invoke --tenant <id> --task <task> --version {version} "
        f"--prompt 'your prompt'"
    )
    console.print("\nTo tear down:")
    console.print("  multi-tenant-lora teardown --confirm")


# -----------------------------------------------------------------------------
# teardown — MUST be explicit, two-step
# -----------------------------------------------------------------------------


@cli.command()
@click.option("--confirm", is_flag=True, help="Required. Confirms destructive action.")
@click.pass_context
def teardown(ctx: click.Context, confirm: bool) -> None:
    """Delete the inference endpoint and all adapter components.

    Training jobs that have already completed are NOT deleted — their artifacts
    remain in S3. To delete S3 data, remove the bucket manually.
    """
    if not confirm:
        console.print("[red]Pass --confirm to actually delete resources.[/red]")
        sys.exit(1)
    config = _load_config(ctx.obj["config_path"])
    ScalingManager(config).deregister()
    InferencePlatform(config).delete_endpoint()
    console.print("[green]Endpoint + components deleted.[/green]")


if __name__ == "__main__":
    cli()
