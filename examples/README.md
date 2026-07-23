# Examples

The files in this directory are **templates only**. They exist so new
customers can see the shape of the configuration and training/eval data. They
contain placeholder AWS account IDs, bucket names and IAM role ARNs and
will not work unmodified against any real AWS account.

## Contents

- `config/platform.yaml` – platform-wide settings (account, region, bucket,
  shared endpoint name, optional MLflow + SNS configuration).
- `config/tenants.yaml` – starter registry with two synthetic tenants and one
  task each. Replace `tenant-a`, `tenant-b`, account ids, role ARNs, etc.
  with your own values.
- `config/base_models.yaml` – catalog of two supported base models
  (Llama 3.1 8B Instruct, Mistral 7B Instruct v0.3) with current HuggingFace
  training DLC versions and LMI inference settings for multi-adapter
  serving.
- `config/instances.yaml` – training and inference instance profiles,
  including the `multi-tenant-production` profile that defaults to
  `ml.g5.12xlarge` as described in the reference blog.
- `sample_train.jsonl` – three illustrative training records in the expected
  `{"instruction", "input", "output"}` schema. Do not use these for real
  training.
- `sample_eval.jsonl` – three illustrative eval records in the
  `{"input", "expected"}` schema consumed by `evaluate_adapter`.

## How to adapt

1. Copy `examples/config/` to your own repo and rename it (e.g. to
   `./config/` in your project root).
2. Replace every placeholder account id, bucket name, IAM role ARN, and
   tenant id.
3. Upload your real training JSONL files to
   `s3://<data_bucket>/<tenant_id>/training-data/<task>/train.jsonl`.
4. Upload your real evaluation JSONL to
   `s3://<data_bucket>/<tenant_id>/evaluation-data/<task>/eval.jsonl`.
5. Run `python -m multi_tenant_lora_sagemaker.orchestration.cli --help`
   to see all available commands.
