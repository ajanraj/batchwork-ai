# Machine output

Use explicit root `--json` for one buffered document and root `--jsonl` for streaming records.
Redirected defaults are JSON for bounded commands and JSONL for `run` and `results`, but agents
should not rely on TTY detection.

Every record has `schema_version: 1` and a stable `type`. Reject unknown versions before reading
payload fields. Ignore unknown additive fields. Keys are snake_case; timestamps are UTC RFC
3339; provider-owned nested JSON remains provider-shaped.

## Envelope types

- `job`: accepted job and route-complete identity.
- `snapshot`: one normalized provider state.
- `result`: one streamed result item.
- `job_list`: buffered local registry jobs.
- `result_list`: buffered results.
- `run`: buffered job, terminal snapshot, and results.
- `paths`: effective config and registry paths.
- `config_validation`: config validity without credential reads.
- `config_view`: normalized non-secret profile settings.
- `registry_check`: schema and integrity state.
- `registry_prune_plan`: non-mutating local prune preview.
- `registry_change`: forget, committed prune, or reset; remote jobs unchanged.
- `image_manifest`: portable materialized-image entries.
- `error`: one structured expected failure on stderr.

Validate each success against the published schema-v1 definition and require the command's
expected type:

- `submit` → `job`;
- `status`, `wait`, or `cancel` → `snapshot`;
- `results` → buffered `result_list` or streamed `result` records;
- `list` → buffered `job_list` or streamed `job` records;
- `run` → buffered `run`, or ordered streamed `job`, `snapshot`, then zero or more `result`;
- `config path` → `paths`; `config validate` → `config_validation`; `config show` →
  `config_view`;
- `registry check` → `registry_check`;
- prune preview → `registry_prune_plan`;
- `forget`, `prune --yes`, or `registry reset --backup` → `registry_change`.

Stop rather than infer when the schema or expected type cannot be validated.

For `registry_prune_plan`, parse `candidate_records` as a nonnegative integer, including zero,
and require `committed: false` and `remote_jobs_changed: false`. Never treat a nonzero preview
as committed deletion.

Accept reset only as `registry reset --backup`. For reset success, require `backup_path` and
`remote_jobs_changed: false`. When `records_count_known` is false, `changed_records` must be
absent; never infer or invent it. Require `remote_jobs_changed: false` on every
`registry_prune_plan` and `registry_change` before reporting cleanup success.

The normative schema and generated credential-free example for every envelope and error code
are published at:

- `https://batchwork.ajanraj.com/schemas/batchwork-cli-v1.schema.json`
- `https://batchwork.ajanraj.com/docs/reference/cli-machine-schema`

## Stream handling

For JSONL, process only complete lines and retain lines emitted before later failure. The first
`job` record from `run` is recovery identity. Empty streamed lists produce no stdout lines.
Buffered JSON is transactional and remains empty until one complete valid envelope is ready.

Capture stdout, stderr, and process status separately. Never parse progress or human output.
When relaying records, preserve provider-owned nested fields and unknown additive fields.

Result replay starts at the beginning. Deduplicate using canonical job identity plus
`result.custom_id`, never output order.
