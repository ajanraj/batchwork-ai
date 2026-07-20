---
name: batchwork-ai
description: >-
  Use this skill when the user names Batchwork, the batchwork CLI, or an
  existing Batchwork job and needs to install or configure it; submit,
  monitor, retrieve, cancel, resume, or adopt provider-native text,
  embedding, or image batches; parse Batchwork JSON or JSONL; troubleshoot
  Batchwork exits; or recover interrupted or partial Batchwork work. Do not
  use for ordinary synchronous model calls, Python SDK implementation, generic
  shell JSON parsing, provider APIs without Batchwork context, or general
  workflow orchestration.
license: MIT
compatibility: Requires Batchwork >=0.2,<0.3 with machine schema_version 1; provider operations require network access and environment-supplied credentials.
metadata:
  version: "0.1.0"
---

# Operate Batchwork safely

Invoke the installed `batchwork` executable directly. Operate one provider-native text,
embedding, or image batch at a time. Use lifecycle commands or the transparent `run`
composition; do not create a planner, scheduler, checkpoint store, retry engine, or
multi-batch orchestration layer.

Activate only when Batchwork, `batchwork`, or an existing Batchwork job is explicit in the
request or established context. Do not activate for generic batching, provider APIs, JSONL,
or asynchronous work.

## Before execution

1. Distinguish explanation from execution, read-only work from mutation, and foreground
   `run` from submit-and-resume.
2. Prefer an installed `batchwork`. If it is absent or outside `>=0.2,<0.3`, do not install
   or upgrade automatically. Ask for authorization, or offer
   `uvx --from batchwork-ai batchwork` for an approved one-shot invocation.
   If the executable is missing, its version is outside that range, or root/relevant command
   help differs from this contract, stop all operational work. Do not invoke provider
   operations before an authorized installation or upgrade restores compatibility.
3. Inspect root and relevant command help without requiring credentials. Help does not prove
   schema compatibility: validate `schema_version == 1` on every machine envelope before
   reading its payload.
4. Resolve modality, source, full `provider/model`, profile, credential-variable name, and
   endpoint by documented precedence. Standard environment-variable names and provider
   built-in endpoints are valid resolved defaults. Never inspect secret values to infer an
   account or route.
5. Clarify only unresolved or ambiguous execution facts. Never silently change provider,
   model, profile, endpoint, modality, source, or output destination.

Read [commands](references/commands.md) before selecting a command or selector. Read
[configuration and secrets](references/configuration-and-secrets.md) before configuration,
profile, route, credential, header, or endpoint work.

## Authorization

A current direct user request authorizes that exact ordinary submission, run, status, wait,
or results operation after deterministic resolution. Do not ask twice.

Always obtain explicit authorization before:

- installing or upgrading Batchwork;
- adding `--allow-large-batch` after reporting the measured soft gate;
- cancellation not already requested by the user;
- `--save`, `forget`, `prune --yes`, or `registry reset --backup`;
- introducing a custom base URL;
- replacing or destructively changing local files or configuration.

Changing provider, model, profile, endpoint, workload, or destination beyond the user's
request also requires approval.

## Non-negotiable invariants

- Global controls are root-only: `batchwork --json --quiet status JOB`, never place `--json`,
  `--jsonl`, `--human`, profile, config, registry, quiet, or progress after the command.
- Automation selects machine output and never parses human tables or prose.
- Capture stdout, stderr, and the original exit status separately. stdout is primary data;
  stderr is progress, diagnostics, and the single machine error envelope.
- Registered jobs use immutable local `record_id`; unregistered jobs use
  `provider_reference` in `provider:provider-job-id` form. Never retain an alias as canonical
  machine identity.
- Never infer a provider from a job-ID shape or available credentials.
- Timeout, interruption, termination, broken pipe, and local failures never cancel remote
  work.
- Never retry submission, provider upload, batch creation, cancellation, registry mutation,
  or any failed CLI invocation automatically. The CLI owns bounded safe-read retries.
- `error.retryable` permits a later user-directed invocation after conditions change; it does
  not authorize a retry loop.
- Never blindly resubmit when `submission_outcome` is `unknown`.
- Credentials and secret headers remain environment-only. Never print, persist, or copy
  resolved values.
- A custom base URL receives credentials and workload data. Never introduce one silently.
- Image files are written only through explicit `--output-dir`; never overwrite, clear, or
  silently suffix a target.
- Preserve every complete output record and recoverable identity even when exit status is
  nonzero.
- Reject unknown `schema_version` values. Preserve unknown additive fields when relaying
  records.

## Safe operating algorithm

1. Inspect non-secret state with `config path`, `config validate`, and `config show`. Never
   dump the environment.
2. Preflight the complete local workload. stdin and unknown extensions require `--format`.
   Preserve explicit `custom_id` values. Convenience flags are defaults beneath non-null
   record values.
3. Use `--provider-options-file` for nontrivial JSON. Preserve exact key spelling. Do not
   silently remove unsupported options or canonical/provider collisions.
4. Apply the authorization gate above.
5. Choose an execution shape that preserves separate streams and the original status:
   - Use root `--jsonl --quiet` for streaming `run` or `results` only when complete stdout
     lines are available incrementally.
   - Otherwise use root `--json --quiet` with `submit`, then one `wait`, then `results`.
     These are lifecycle primitives, not a polling loop.
   - If stream separation or original status is unavailable, provide commands instead of
     executing.
6. Capture identity immediately. From a `job` envelope retain `job.record_id` when present,
   otherwise `job.provider_reference`. Subsequent envelopes expose the canonical selector in
   top-level `job`.
7. Parse complete JSON or JSONL records structurally. Validate schema version and `type`
   before payload fields. Keep output emitted before a later error.
8. Interpret the process exit together with output. `status` exit 0 means observation
   succeeded, not that the job completed. Exit 6 may accompany useful terminal or partial
   results.
9. Follow structured `error.recovery.command` only after checking it preserves route-complete
   identity and does not cross an authorization boundary.
10. On replay, deduplicate by canonical job identity plus `custom_id`, never by output order.

Read [machine output](references/machine-output.md) before parsing envelopes. Read
[failures and recovery](references/failures-and-recovery.md) after any nonzero exit, partial
stream, unknown submission outcome, timeout, or registry failure.

## Command selection

- `submit text|embeddings|images`: create one job, record identity, and return.
- `run text|embeddings|images`: submit, wait, and retrieve available results.
- `status JOB`: one current snapshot; no wait.
- `wait JOB`: poll until terminal or local timeout; no remote cancellation on timeout.
- `results JOB`: one status refresh and one terminal retrieval attempt; no wait.
- `cancel JOB`: one cancellation request unless already terminal; requires explicit current
  authorization.
- `list`: local cached records only; no provider scan.
- `forget`: remove one local record only; guarded.
- `prune`: preview by default; `--yes` commits local deletion and is guarded.
- `config path|validate|show`: non-secret inspection.
- `registry check`: local integrity inspection.
- `registry reset --backup`: preserve a recovery set, then reset; guarded.

For exact options and selector rules, read [commands](references/commands.md). For JSON,
JSONL, CSV, text, defaults, volume gates, and provider options, read
[inputs and options](references/inputs-and-options.md).

## Configuration edits

Edit the user TOML only when explicitly requested. First resolve the effective path and show
non-secret effective settings. Make the smallest change, preserve unrelated profiles and
restrictive ownership/permissions, and never write a resolved secret. Do not add a custom
base URL without authorization. Run `config validate` and `config show`; if validation
fails, restore the original content and stop.

Create a new POSIX config as a current-user-owned regular non-symlink file with mode `0600`,
subject to any stricter existing policy. It must never be group- or world-writable.

The bundled [configuration example](assets/config.example.toml) contains environment-variable
names only. The [provider-options example](assets/provider-options.example.json) is the inner
selected-provider object; available keys depend on the implemented provider documentation.

## Images

Machine results preserve normalized inline data or URLs without writing files. Add
`--output-dir` only when the user explicitly requests materialization. Verify the target is
absent or an empty non-symlink directory before remote work. Do not clean, overwrite, or
choose an alternate target. Preserve completed files and `manifest.json` after partial
failure. Read [image materialization](references/image-materialization.md) before downloading
Batchwork image results.
